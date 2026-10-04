"""Paired endogenous mechanism diagnostics; isolated from frozen earlier experiments."""

import asyncio
import copy
import itertools
import json
import math
import os
import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from behavioral_market.agents.portfolio_facts import portfolio_facts
from behavioral_market.agents.state import population
from behavioral_market.data.archive import digest, write_json
from behavioral_market.data.calendar import sessions
from behavioral_market.environments.endogenous_event import EndogenousEventEnvironment
from behavioral_market.llm.client import MockLLMClient, OpenAICompatibleClient, RequestPacer
from behavioral_market.llm.schemas import OrderDecision, hold
from behavioral_market.simulation.event_runner import git_state, llm_config, source_hash

from .two_event_data import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "two-event-diagnostic-v1"
SYSTEM = """You are one investor in a simulated continuous double auction. Decide only from
the supplied information available at decision_at. Never use remembered future historical
outcomes. Prices and cash are quoted PER LOT in quote_currency. quantity is an integer
number of lots; lot_size states the underlying asset amount. All displayed prices use this
same lot unit. Limit orders may not fill. No margin or short selling. Respect the stated
price band, maximum quantity, available cash and inventory. Your outstanding orders from
earlier rounds have been cancelled. Other agents may disagree. There is no instructed
direction of trade. Private targets, if provided, are subjective portfolio preferences,
not knowledge of true value. Use the supplied experience to reconsider your decision.
Return ONLY a JSON object with action (buy/sell/hold), quantity (integer), limit_price
(positive number or null for hold), rationale (short string), sentiment
(strongly_bearish/bearish/neutral/bullish/strongly_bullish). For hold set quantity=0 and
limit_price=null. Do not add fields."""

LOT_RULES = """Prices and cash are quoted PER LOT in quote_currency. quantity is an integer
number of lots; lot_size states the underlying asset amount. All displayed prices use this
same lot unit."""
ASSET_RULES = """All displayed market prices and limit_price are in quote_currency per ONE
underlying asset unit (one share or one BTC). quantity is an integer number of lots, and
lot_size is the underlying units per lot. Order notional is quantity * lot_size * limit_price.
Respect price_tick. Computed account facts give current weights and allocation gaps."""
assert LOT_RULES in SYSTEM


def system_prompt(config):
    basis = config.get("price_basis", "lot")
    if basis not in {"lot", "underlying"}:
        raise ValueError("unsupported observation price basis")
    return SYSTEM.replace(LOT_RULES, ASSET_RULES) if basis == "underlying" else SYSTEM


def prompt_version(config):
    if config.get("price_basis", "lot") == "lot" and not config.get("computed_portfolio_facts"):
        return PROMPT_VERSION
    return (
        PROMPT_VERSION
        + "-"
        + config.get("price_basis", "lot")
        + f"-facts{int(bool(config.get('computed_portfolio_facts')))}"
    )


def code_hash():
    return digest((source_hash() + PROMPT_VERSION).encode() + Path(__file__).read_bytes())


def round_schedule(spec):
    """Same decision opportunities for news and no-news; true intraday news cutoffs."""
    result = []
    for day, row in sessions(spec["start"], spec["end"], spec["calendar"]).iterrows():
        times = {row.market_open}
        times.update(
            pd.Timestamp(event["at"])
            for event in spec["events"]
            if row.market_open < pd.Timestamp(event["at"]) < row.market_close
        )
        result.extend((day, time.isoformat()) for time in sorted(times))
    return result


def visible_events(spec, cutoff, condition):
    if condition == "no_news":
        return []
    if condition != "news":
        raise ValueError("unknown news condition")
    return [
        {"published_at": event["at"], "text": event["text"]}
        for event in spec["events"]
        if pd.Timestamp(event["at"]) <= pd.Timestamp(cutoff)
    ]


def lot_history(bars, spec):
    """Discard future rows BEFORE scaling; do not truncate fractional crypto volume."""
    result = bars.loc[bars.symbol.eq(spec["symbol"]) & (bars.session_date < spec["start"])].copy(
        deep=True
    )
    if result.empty or spec["lot_size"] <= 0:
        raise ValueError("positive lot size and pre-start data required")
    for column in ("open", "high", "low", "close"):
        if "asset_price_tick" in spec:
            tick = Decimal(str(spec["asset_price_tick"]))
            if not tick.is_finite() or tick <= 0:
                raise ValueError("invalid asset price grid")
            lot = Decimal(str(spec["lot_size"]))
            result[column] = result[column].map(
                lambda value, tick=tick, lot=lot: float(
                    (Decimal(str(value)) / tick).to_integral_value() * tick * lot
                )
            )
        else:
            result[column] = result[column].astype(float) * spec["lot_size"]
    result["volume"] = result.volume.astype(float) / spec["lot_size"]
    return result.sort_values("session_date")


def setup(config, spec, bars, seed, liquidity):
    history = lot_history(bars, spec)
    price = float(history.close.iloc[-1])
    shares = math.floor(config["initial_stock_value"] / price)
    if shares < 1:
        raise ValueError("initial portfolio cannot hold one lot")
    agents = population(
        {
            "experiment": {"seed": seed},
            "population": {
                "n_agents": config["n_agents"],
                "initial_cash": config["initial_cash"],
                "initial_shares": shares,
                "initial_fundamental_share": 0.5,
            },
        },
        "vanilla_llm",
        price,
    )
    n = config["n_agents"]
    if n < 2:
        raise ValueError("at least two participants required")
    # Stratification guarantees different preferences without claiming human estimation.
    targets = [round(0.2 + 0.6 * i / (n - 1), 6) for i in range(n)]
    random.Random(seed + 177).shuffle(targets)
    background = None
    if liquidity == "finite" or liquidity in config.get("liquidity_profiles", {}):
        spec_lp = {**config["liquidity"], **config.get("liquidity_profiles", {}).get(liquidity, {})}
        if any(
            not math.isfinite(spec_lp.get(name, 1)) or spec_lp.get(name, 1) <= 0
            for name in ("wealth_multiple", "depth_multiple")
        ):
            raise ValueError("liquidity resources and depth must be finite and positive")
        max_quantity = math.floor(
            float(agents[0].portfolio.total_equity) * config["max_order_equity_fraction"] / price
        )
        background = {
            "initial_cash": config["initial_cash"] * n * spec_lp["wealth_multiple"],
            "initial_shares": int(shares * n * spec_lp["wealth_multiple"]),
            "half_spread_bps": spec_lp["half_spread_bps"],
            "level_spacing_bps": spec_lp["level_spacing_bps"],
            "level_quantities": [
                max(
                    1,
                    math.floor(
                        max_quantity * n / spec_lp["levels"] * spec_lp.get("depth_multiple", 1)
                    ),
                )
            ]
            * spec_lp["levels"],
            "inventory_skew_bps": spec_lp["inventory_skew_bps"],
        }
        if "asset_price_tick" in spec:
            background["tick_size"] = str(
                Decimal(str(spec["asset_price_tick"])) * Decimal(str(spec["lot_size"]))
            )
    elif liquidity != "none":
        raise ValueError("unknown liquidity condition")
    environment = EndogenousEventEnvironment(
        history, spec["start"], spec["symbol"], agents, background
    )
    return agents, targets, environment


def context(config, spec, agent, target, environment, cutoff, condition, treatment, previous):
    price = float(environment.exchange.market_price)
    closes = environment.history.close.astype(float)
    opening = environment._opening
    state = {
        "agent_id": agent.persona.agent_id,
        "portfolio": {
            "cash": float(agent.portfolio.cash),
            "lots": agent.portfolio.position,
            "equity": float(agent.portfolio.total_equity),
            "average_cost_per_lot": float(agent.portfolio.avg_cost),
        },
        "memory": agent.memory[-config["memory_rounds"] :],
    }
    if treatment == "private_targets":
        state["subjective_target_asset_weight"] = target
    elif treatment != "vanilla":
        raise ValueError("unknown treatment")
    if config.get("computed_portfolio_facts"):
        state["computed_account_facts"] = portfolio_facts(
            agent.portfolio,
            price,
            spec["lot_size"],
            target if treatment == "private_targets" else None,
        )
    multiplier = spec["lot_size"] if config.get("price_basis", "lot") == "underlying" else 1.0
    book = environment.book_context()
    for key in ("best_bid", "best_ask", "spread", "last_trade"):
        if book[key] is not None:
            book[key] = round(book[key] / multiplier, 6)
    if multiplier != 1:
        state["portfolio"]["average_cost_per_asset_unit"] = (
            state["portfolio"].pop("average_cost_per_lot") / multiplier
        )
    payload = {
        "decision_at": cutoff,
        "asset": spec["symbol"],
        "quote_currency": spec["quote_currency"],
        "lot_size": spec["lot_size"],
        "order_notional_multiplier": multiplier,
        "agent_state": state,
        "observation": {
            "last_close": round(float(closes.iloc[-1]) / multiplier, 6),
            "last_trade": round(price / multiplier, 6),
            "completed_close_history": (closes.tail(6) / multiplier).round(6).tolist(),
            "completed_history_through": str(environment.history.session_date.iloc[-1]),
            "book": book,
            "previous_round_agent_actions": previous,
            "news": visible_events(spec, cutoff, condition),
        },
        "constraints": {
            "min_price": round(opening * config["price_band"][0] / multiplier, 6),
            "max_price": round(opening * config["price_band"][1] / multiplier, 6),
            "max_quantity": math.floor(
                float(agent.portfolio.total_equity) * config["max_order_equity_fraction"] / price
            ),
        },
    }
    if "asset_price_tick" in spec:
        payload["constraints"]["price_tick"] = float(
            Decimal(str(spec["asset_price_tick"]))
            * Decimal(str(spec["lot_size"]))
            / Decimal(str(multiplier))
        )
    return payload


def rule_decision(payload):
    """Transparent preference/momentum baseline; deliberately no signed news shock."""
    state, obs, constraints = (
        payload["agent_state"],
        payload["observation"],
        payload["constraints"],
    )
    price, book = obs["last_trade"], obs["book"]
    multiplier = Decimal(str(payload.get("order_notional_multiplier", 1)))
    weight = state["portfolio"]["lots"] * price * float(multiplier) / state["portfolio"]["equity"]
    history = obs["completed_close_history"]
    momentum = history[-1] / history[0] - 1
    target = state.get("subjective_target_asset_weight", 0.5)
    score = target - weight + 0.5 * momentum
    if abs(score) < 0.01 or constraints["max_quantity"] == 0:
        return hold("rule: allocation gap below threshold")
    side = "buy" if score > 0 else "sell"
    quote = book["best_ask" if side == "buy" else "best_bid"]
    limit = quote if quote is not None else price * (1.006 if side == "buy" else 0.994)
    limit = round(min(constraints["max_price"], max(constraints["min_price"], limit)), 6)
    if "price_tick" in constraints:
        tick = Decimal(str(constraints["price_tick"]))
        limit = float((Decimal(str(limit)) / tick).to_integral_value() * tick)
    cap = (
        int(Decimal(str(state["portfolio"]["cash"])) // (Decimal(str(limit)) * multiplier))
        if side == "buy"
        else state["portfolio"]["lots"]
    )
    quantity = min(cap, constraints["max_quantity"])
    if quantity <= 0:
        return hold("rule: portfolio bound")
    return OrderDecision(
        action=side,
        quantity=quantity,
        limit_price=limit,
        rationale="rule: private allocation gap plus lagged momentum; news ignored",
        sentiment="bullish" if side == "buy" else "bearish",
    )


def validate_order(decision, payload):
    """Invalid proposals become explicitly REJECTED, not successful HOLD decisions."""
    if decision.action == "hold":
        return None
    constraints = payload["constraints"]
    if "price_tick" in constraints and (
        Decimal(str(decision.limit_price)) % Decimal(str(constraints["price_tick"])) != 0
    ):
        return "price_tick"
    if not constraints["min_price"] <= decision.limit_price <= constraints["max_price"]:
        return "price_band"
    if decision.quantity > constraints["max_quantity"]:
        return "maximum_quantity"
    portfolio = payload["agent_state"]["portfolio"]
    if decision.action == "buy" and (
        Decimal(str(decision.limit_price))
        * decision.quantity
        * Decimal(str(payload.get("order_notional_multiplier", 1)))
        > Decimal(str(portfolio["cash"]))
    ):
        return "insufficient_cash"
    if decision.action == "sell" and decision.quantity > portfolio["lots"]:
        return "insufficient_inventory"
    return None


def exchange_decision(decision, payload):
    if decision.action == "hold":
        return decision
    multiplier = Decimal(str(payload.get("order_notional_multiplier", 1)))
    return decision.model_copy(
        update={"limit_price": float(Decimal(str(decision.limit_price)) * multiplier)}
    )


def append_json(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


async def run_case(config, spec, bars, case, output, manifest_hash, client=None, provider="rule"):
    seed, treatment, condition, liquidity = case
    identity = {
        "scenario": spec["scenario_id"],
        "seed": seed,
        "treatment": treatment,
        "event_condition": condition,
        "liquidity": liquidity,
        "provider": provider,
    }
    conf_hash = digest(json.dumps([config, identity], sort_keys=True).encode())
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if any(
            old.get(key) != value
            for key, value in {
                "config_hash": conf_hash,
                "data_manifest_hash": manifest_hash,
                "source_hash": code_hash(),
            }.items()
        ):
            raise RuntimeError("existing output has different inputs; use a new experiment_id")
        if old["status"] == "complete":
            return json.loads((output / "metrics.json").read_text())
        # Keep failed/interrupted attempts reviewable; never append a rerun to their logs.
        attempt = output / "attempts" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        attempt.mkdir(parents=True)
        for item in list(output.iterdir()):
            if item.name != "attempts":
                item.rename(attempt / item.name)
    output.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    metadata = {
        **identity,
        "status": "running",
        "config_hash": conf_hash,
        "data_manifest_hash": manifest_hash,
        "source_hash": code_hash(),
        "git_commit": commit,
        "dirty_worktree": dirty,
        "started_at": datetime.now(UTC).isoformat(),
        "prompt_version": prompt_version(config),
        "interpretation": config["interpretation"],
    }
    write_json(metadata_path, metadata)
    write_json(output / "config.json", config)
    agents, targets, env = setup(config, spec, bars, seed, liquidity)
    initial = float(env.exchange.market_price)
    write_json(
        output / "initial_state.json",
        {
            "price_per_lot": initial,
            "lot_size": spec["lot_size"],
            "private_targets": targets,
            "participants": [a.context() for a in agents],
            "total_cash_including_background": str(env.initial_cash),
            "total_lots_including_background": env.initial_shares,
        },
    )
    schedule = round_schedule(spec)
    actions, trades, days = [], [], []
    previous = {key: 0 for key in ("buy", "sell", "hold", "rejected")}
    try:
        for index, (day, cutoff) in enumerate(schedule):
            if env._session is None:
                env.start_session(day)
            # Cancel prior-round participant orders before reading cash/inventory limits.
            for order in list(env.exchange.book.resting_orders()):
                if order.agent_id != "BACKGROUND_LP":
                    env.exchange.cancel_order(order.order_id, order.agent_id)
            trade_start = len(env._trades)
            env.requote_background()
            payloads = [
                context(config, spec, agent, target, env, cutoff, condition, treatment, previous)
                for agent, target in zip(agents, targets, strict=True)
            ]

            async def decide(agent, payload, cutoff=cutoff):
                if provider == "rule":
                    return rule_decision(payload), {"status": "rule"}
                # Paired pre-news histories reuse identical responses when payloads match.
                call_identity = {
                    key: value for key, value in identity.items() if key != "event_condition"
                }
                return await client.decide(
                    system_prompt(config),
                    json.dumps(payload, sort_keys=True),
                    OrderDecision,
                    {**call_identity, "decision_at": cutoff, "agent": agent.persona.agent_id},
                )

            outcomes = await asyncio.gather(
                *(decide(a, p) for a, p in zip(agents, payloads, strict=True))
            )
            if any(meta.get("fallback") for _, meta in outcomes):
                raise RuntimeError("invalid model response: run fails instead of using fallback")
            arrival = list(range(len(agents)))
            random.Random(seed * 100003 + index).shuffle(arrival)
            previous = {key: 0 for key in ("buy", "sell", "hold", "rejected")}
            rows = {}
            submitted = {}
            for i in arrival:
                agent, (decision, meta), payload = agents[i], outcomes[i], payloads[i]
                rejection = validate_order(decision, payload)
                internal_decision = exchange_decision(decision, payload)
                # The exchange uses exact Decimal accounting; displayed float cash is
                # only an observation and must never bypass the final portfolio check.
                if (
                    rejection is None
                    and decision.action == "buy"
                    and (
                        Decimal(str(internal_decision.limit_price)) * decision.quantity
                        > agent.portfolio.cash
                    )
                ):
                    rejection = "insufficient_cash"
                if rejection is None:
                    env.submit(agent, internal_decision)
                    submitted[i] = env._orders[-1].get("order")
                previous["rejected" if rejection else decision.action] += 1
                row = {
                    "decision_at": cutoff,
                    "agent_id": agent.persona.agent_id,
                    "payload": payload,
                    "decision": decision.model_dump(),
                    "exchange_limit_price_per_lot": internal_decision.limit_price,
                    "rejection": rejection,
                    "response_metadata": meta,
                }
                rows[i] = row
                actions.append(row)
            new_trades = env._trades[trade_start:]
            for trade in new_trades:
                trades.append({**trade, "decision_at": cutoff})
            for i, agent in enumerate(agents):
                order = submitted.get(i)
                execution = {
                    "filled_lots": order.quantity - order.remaining_quantity if order else 0,
                    "status_at_round_end": (
                        order.status.value
                        if order
                        else ("rejected" if rows[i]["rejection"] else "hold")
                    ),
                    "equity": float(agent.portfolio.total_equity),
                    "return_since_start": float(agent.portfolio.total_equity) / agent.initial_equity
                    - 1,
                }
                rows[i]["execution"] = execution
                append_json(output / "decisions.jsonl", rows[i])
                agent.memory.append(
                    {"decision_at": cutoff, "decision": outcomes[i][0].model_dump(), **execution}
                )
            if index + 1 == len(schedule) or schedule[index + 1][0] != day:
                close, volume, _, _ = env.finish_session(agents)
                days.append({"session_date": day, "close_per_lot": close, "volume_lots": volume})
        write_json(output / "daily.json", days)
        write_json(output / "trades.json", trades)
        write_json(output / "background_orders.json", env.background_orders)
        counts = {
            side: sum(r["decision"]["action"] == side for r in actions)
            for side in ("buy", "sell", "hold")
        }
        accepted_lots = sum(r["decision"]["quantity"] for r in actions if not r["rejection"])
        filled_lots = sum(r["execution"]["filled_lots"] for r in actions)
        prices = pd.Series([initial, *[row["close_per_lot"] for row in days]])
        metrics = {
            **identity,
            "rounds": len(schedule),
            "decisions": len(actions),
            "proposed_actions": counts,
            "rejected_orders": sum(r["rejection"] is not None for r in actions),
            "trades": len(trades),
            "trade_volume_lots": sum(t["quantity"] for t in trades),
            "agent_fill_ratio": filled_lots / accepted_lots if accepted_lots else None,
            "return": prices.iloc[-1] / initial - 1,
            "close_max_drawdown": float((1 - prices / prices.cummax()).max()),
            "zero_volume_sessions": sum(row["volume_lots"] == 0 for row in days),
            "background_trade_share": (
                sum("BACKGROUND_LP" in (t["buyer_id"], t["seller_id"]) for t in trades)
                / len(trades)
                if trades
                else None
            ),
            "cash_conservation": str(
                sum(p.cash for p in env.exchange.portfolios.values()) - env.initial_cash
            ),
            "lot_conservation": sum(p.position for p in env.exchange.portfolios.values())
            - env.initial_shares,
            "agent_mean_return": sum(
                float(a.portfolio.total_equity) / a.initial_equity - 1 for a in agents
            )
            / len(agents),
            "background_final_cash": str(env.background.portfolio.cash) if env.background else None,
            "background_final_lots": env.background.portfolio.position if env.background else None,
        }
        write_json(output / "metrics.json", metrics)
        metadata.update(status="complete", completed_at=datetime.now(UTC).isoformat())
        write_json(metadata_path, metadata)
        return metrics
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__)
        write_json(metadata_path, metadata)
        raise


async def run_suite(config, provider, root=ROOT, cache_only=False):
    config = copy.deepcopy(config)
    data_config = json.loads((root / config["data_config"]).read_text())
    directory = root / "data/processed" / data_config["study_id"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_hash"] != digest(json.dumps(data_config, sort_keys=True).encode()):
        raise ValueError("data configuration drift")
    verify_manifest(manifest, root)
    output = root / "outputs" / data_config["study_id"] / config["experiment_id"] / provider
    provider_spec = None
    live = provider not in {"rule", "mock"}
    if live:
        if git_state()[1]:
            raise RuntimeError("freeze and commit validated code before live calls")
        provider_spec = llm_config(
            json.loads((root / config["provider_config"]).read_text()), provider
        )
        overrides = config.get("provider_overrides", {})
        if set(overrides) - {
            "thinking",
            "require_no_reasoning",
            "max_tokens",
            "reasoning_effort",
            "allowed_openai_params",
            "extra_body",
            "chat_template_kwargs",
        }:
            raise ValueError("unsupported pilot provider override")
        provider_spec.update(overrides)
        # Freeze full provider configuration in run identity, without credentials.
        config["resolved_provider_config"] = provider_spec
        load_dotenv(root / ".env")
    capacity = config.get("request_concurrency", 12)
    if type(capacity) is not int or not 1 <= capacity <= 32:
        raise ValueError("request_concurrency must be between one and 32")
    shared_semaphore, pacer = asyncio.Semaphore(capacity), RequestPacer(4)
    count = config.get("pair_concurrency", 1)
    if type(count) is not int or count < 1 or count > 8:
        raise ValueError("pair_concurrency must be between one and eight")
    pair_semaphore = asyncio.Semaphore(count)
    bars_by_scenario = {
        spec["scenario_id"]: pd.read_parquet(directory / f"{spec['scenario_id']}_daily.parquet")
        for spec in config["scenarios"]
    }
    if live and not cache_only:
        verifier = OpenAICompatibleClient(
            provider_spec,
            os.getenv(provider_spec["api_key_env"], ""),
            output / "registry",
            root / ".cache" / config["experiment_id"] / provider,
        )
        try:
            await verifier.verify_model()
        finally:
            await verifier.close()

    async def run_pair(spec, seed, treatment, liquidity):
        results = []
        async with pair_semaphore:
            # Only disjoint pairs overlap. Each pair completes news before no-news,
            # so identical pre-news responses are written before their paired reuse.
            for condition in config["event_conditions"]:
                results.append(await run_one(spec, seed, treatment, liquidity, condition))
        return results

    async def run_one(spec, seed, treatment, liquidity, condition):
        destination = output / spec["symbol"] / f"seed{seed}" / treatment / liquidity / condition
        client = None
        if live:
            client = OpenAICompatibleClient(
                provider_spec,
                os.getenv(provider_spec["api_key_env"], ""),
                destination / "transport",
                root / ".cache" / config["experiment_id"] / provider,
                semaphore=shared_semaphore,
                pacer=pacer,
                cache_only=cache_only,
                cache_namespace=prompt_version(config),
            )
        elif provider == "mock":
            client = MockLLMClient()
        try:
            result = await run_case(
                config,
                spec,
                bars_by_scenario[spec["scenario_id"]],
                (seed, treatment, condition, liquidity),
                destination,
                digest(manifest_path.read_bytes()),
                client,
                provider,
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "scenario",
                            "seed",
                            "treatment",
                            "event_condition",
                            "liquidity",
                            "return",
                            "trades",
                            "rejected_orders",
                        )
                    }
                ),
                flush=True,
            )
            return result
        finally:
            if live:
                await client.close()

    tasks = [
        asyncio.create_task(run_pair(spec, seed, treatment, liquidity))
        for spec in config["scenarios"]
        for seed, treatment, liquidity in itertools.product(
            config["seeds"], config["treatments"], config["liquidity_conditions"]
        )
    ]
    try:
        paired_rows = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    rows = [row for pair in paired_rows for row in pair]
    write_json(output / "summary.json", rows)
    return rows
