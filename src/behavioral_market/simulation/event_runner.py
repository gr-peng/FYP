import argparse
import asyncio
import copy
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from behavioral_market.agents.policy import enforce_portfolio, rule_order, rule_style, signal
from behavioral_market.agents.state import population
from behavioral_market.data.archive import digest, write_json
from behavioral_market.data.calendar import sessions
from behavioral_market.data.features import features
from behavioral_market.data.schemas import MarketEvent
from behavioral_market.environments.endogenous_event import EndogenousEventEnvironment
from behavioral_market.environments.historical_replay import HistoricalReplayEnvironment
from behavioral_market.environments.observations import available_fundamentals, build_observation
from behavioral_market.evaluation.metrics import run_metrics
from behavioral_market.llm.client import MockLLMClient, OpenAICompatibleClient, RequestPacer
from behavioral_market.llm.prompts import PROMPT_VERSION, prompts
from behavioral_market.llm.schemas import OrderDecision, StyleDecision
from behavioral_market.population.mean_field import compute_mean_field
from behavioral_market.simulation.finite_rule_pair import (
    LEGACY_METHOD,
    FiniteRulePair,
    return_columns,
)
from behavioral_market.simulation.finite_rule_pair import (
    PROMPT_VERSION as FINITE_PROMPT_VERSION,
)
from behavioral_market.simulation.finite_rule_pair import (
    settings as return_settings,
)
from behavioral_market.simulation.hybrid_review import append_event, review_batch, validate_client

ROOT = Path(__file__).resolve().parents[3]
TREATMENTS = ("rule_abm", "vanilla_llm", "low_herding", "high_herding", "hybrid_style_llm")
LIVE_PROVIDERS = ("siliconflow", "aigc_relay")


def llm_config(config, provider):
    common = {k: v for k, v in config["llm"].items() if k != "providers"}
    return {**common, **config["llm"]["providers"][provider]}


def git_state(root=ROOT):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    )
    return commit, dirty


def load_dataset(config):
    directory = ROOT / config["data"]["dataset_dir"]
    manifest_path = ROOT / config["data"].get(
        "manifest", "data/manifests/meta_compute_2026_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["processed_files"]:
        if digest((ROOT / record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError("processed dataset hash mismatch")
    bars = pd.read_parquet(directory / "market_daily.parquet")
    events = [
        MarketEvent.model_validate(row)
        for row in json.loads((directory / "events.json").read_text())
    ]
    sec_by_symbol = directory / "sec_companyfacts_by_symbol.json"
    sec = directory / "sec_companyfacts.json"
    facts = (
        json.loads(sec_by_symbol.read_text()).get(config["data"]["primary_symbol"])
        if sec_by_symbol.exists()
        else (json.loads(sec.read_text()) if sec.exists() else None)
    )
    return (
        bars,
        events,
        facts,
        digest(manifest_path.read_bytes()),
    )


def output_path(config, mode, treatment, provider):
    seed = config["experiment"]["seed"]
    shock = config["fundamental_shock"]["magnitude"]
    condition = event_condition(config)
    count = config["population"]["n_agents"]
    root = ROOT / "outputs/meta_compute_2026" / provider
    batch = config["experiment"].get("output_batch")
    if batch:
        root = root / batch / config["data"]["primary_symbol"]
        suffix = f"n{count}_seed{seed}_shock{shock:g}_{condition}"
    else:
        suffix = f"n{count}_seed{seed}_shock{shock:g}_e2{int(condition == 'e1_e2')}"
    if config["behavior"].get("mean_field_visibility", "visible") == "hidden":
        suffix += "_mean_field_hidden"
    method, _ = return_settings(config["behavior"])
    if method != LEGACY_METHOD:
        suffix += "_" + method
    if "rule_style_adaptation" in config["behavior"]:
        suffix += "_rule_style_" + rule_adaptation(config["behavior"], treatment, method)
    return root / mode / treatment / suffix


def output_root(config, provider):
    root = ROOT / "outputs/meta_compute_2026" / provider
    batch = config["experiment"].get("output_batch")
    return root / batch if batch else root


def event_condition(config):
    condition = config["events"].get("condition")
    if condition is None:
        condition = "e1_e2" if config["events"].get("include_secondary", True) else "e1_only"
    if condition not in {"no_event", "e1_only", "e1_e2"}:
        raise ValueError("events.condition must be no_event, e1_only or e1_e2")
    return condition


def source_hash():
    paths = sorted((ROOT / "src/behavioral_market").rglob("*.py"))
    return digest(b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in paths))


def rule_adaptation(behavior, treatment, return_method):
    mode = behavior.get("rule_style_adaptation", "adaptive")
    if mode not in ("fixed", "adaptive"):
        raise ValueError("rule_style_adaptation must be fixed or adaptive")
    if "rule_style_adaptation" in behavior and (
        treatment != "rule_abm" or return_method == LEGACY_METHOD
    ):
        raise ValueError("explicit rule_style_adaptation requires finite rule_abm")
    return mode


async def run_experiment(
    config,
    mode,
    treatment,
    provider="siliconflow",
    output=None,
    bars=None,
    events=None,
    facts=None,
    manifest_hash="test",
    client=None,
    require_clean=True,
    semaphore=None,
    pacer=None,
    cache_only=False,
):
    config = copy.deepcopy(config)
    if mode not in {"historical", "endogenous"} or treatment not in TREATMENTS:
        raise ValueError("unsupported experiment mode/treatment")
    if provider not in {"mock", *LIVE_PROVIDERS}:
        raise ValueError("unsupported provider")
    return_method, finite_spec = return_settings(config["behavior"])
    adaptation = rule_adaptation(config["behavior"], treatment, return_method)
    hybrid = treatment == "hybrid_style_llm"
    rule_orders = treatment == "rule_abm" or hybrid
    if hybrid:
        if finite_spec is None or mode != "endogenous" or cache_only:
            raise ValueError(
                "hybrid_style_llm requires finite endogenous mode without cache replay"
            )
        validate_client(client)
    source_record = (
        {
            "order_source": "rule",
            "style_source": "model_interface",
            "review_provenance": client.review_provenance,
            "review_protocol": client.hybrid_review_protocol,
        }
        if hybrid
        else {}
    )
    adaptation_record = (
        {"rule_style_adaptation": adaptation}
        if "rule_style_adaptation" in config["behavior"]
        else {}
    )
    prompt_version = PROMPT_VERSION if finite_spec is None else FINITE_PROMPT_VERSION
    mean_field_visibility = config["behavior"].get("mean_field_visibility", "visible")
    if not isinstance(mean_field_visibility, str) or mean_field_visibility not in (
        "visible",
        "hidden",
    ):
        raise ValueError("behavior.mean_field_visibility must be visible or hidden")
    # This records the client interface, not evidence of a real model dispatch.
    prompt_mean_field_supplied = (
        None if treatment == "rule_abm" else mean_field_visibility == "visible"
    )
    if provider in LIVE_PROVIDERS and treatment != "rule_abm" and client is None and pacer is None:
        pacer = RequestPacer(4.0)
    config["run"] = {
        "mode": mode,
        "treatment": treatment,
        "provider": provider,
        "mean_field_visibility": mean_field_visibility,
        "prompt_mean_field_supplied": prompt_mean_field_supplied,
        **adaptation_record,
        **source_record,
    }
    if finite_spec is not None:
        config["run"]["strategy_return_method"] = return_method
    condition = event_condition(config)
    output = output or output_path(config, mode, treatment, provider)
    if hybrid and output.exists() and any(output.iterdir()):
        raise RuntimeError("hybrid runs require a new empty output directory; no resume or replay")
    if bars is None:
        bars, events, facts, manifest_hash = load_dataset(config)
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    commit, dirty = git_state()
    if provider in LIVE_PROVIDERS and treatment != "rule_abm" and require_clean and dirty:
        raise RuntimeError("live experiments require a clean worktree; commit validated code first")
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("status") == "complete":
            if (
                old["config_hash"] != config_hash
                or old["data_manifest_hash"] != manifest_hash
                or old.get("source_hash") != source_hash()
            ):
                raise RuntimeError("output directory belongs to different inputs")
            return json.loads((output / "metrics.json").read_text())
        with (output / "run_attempts.jsonl").open("a") as handle:
            handle.write(json.dumps(old) + "\n")
    output.mkdir(parents=True, exist_ok=True)
    dependency_names = [
        "httpx",
        "pydantic",
        "pandas",
        "numpy",
        "scipy",
        "pyarrow",
        "pandas-market-calendars",
    ]
    metadata = {
        "status": "running",
        "config_hash": config_hash,
        "data_manifest_hash": manifest_hash,
        "git_commit": commit,
        "source_hash": source_hash(),
        "dirty_worktree": dirty,
        "python": platform.python_version(),
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in dependency_names
        },
        "started_at": datetime.now(UTC).isoformat(),
        "model": "rule"
        if treatment == "rule_abm"
        else (llm_config(config, provider)["model"] if provider in LIVE_PROVIDERS else provider),
        "prompt_version": prompt_version,
        "mode": mode,
        "treatment": treatment,
        "seed": config["experiment"]["seed"],
        "shock": config["fundamental_shock"]["magnitude"],
        "event_condition": condition,
        "include_secondary": condition == "e1_e2",
        "symbol": config["data"]["primary_symbol"],
        "llm_parameters": config["llm"],
        "n_agents": config["population"]["n_agents"],
        "request_rate_limit_rps": pacer.requests_per_second if pacer else None,
        "cache_only_replay": cache_only,
        "mean_field_visibility": mean_field_visibility,
        "prompt_mean_field_supplied": prompt_mean_field_supplied,
        "population_alignment_interpretation": "alignment_not_proof_of_population_information_use",
        **adaptation_record,
        **source_record,
    }
    if finite_spec is not None:
        metadata.update(strategy_return_method=return_method, finite_rule_pair=finite_spec)
    write_json(metadata_path, metadata)
    write_json(output / "config_snapshot.json", config)
    (output / "observations.jsonl").write_text("")
    if hybrid:
        (output / "hybrid_review_batches.jsonl").write_text("")
    spec = config["data"]
    primary = (
        bars.loc[bars.symbol == spec["primary_symbol"]]
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    initial_price = float(
        primary.loc[primary.session_date < spec["experiment_start"], "close"].iloc[-1]
    )
    agents = population(config, treatment, str(initial_price))
    finite_pairs = {}
    if finite_spec is not None:
        for agent in agents:
            pair = FiniteRulePair(agent, initial_price, finite_spec, 0)
            finite_pairs[agent.persona.agent_id] = pair
            agent.finite_review = pair.summary(agent.portfolio)
        (output / "finite_rule_replay.jsonl").write_text("")
    environment = (
        HistoricalReplayEnvironment(primary, spec["experiment_start"])
        if mode == "historical"
        else EndogenousEventEnvironment(
            primary, spec["experiment_start"], spec["primary_symbol"], agents
        )
    )
    calendar = spec.get("calendar", "XNYS")
    schedule = sessions(spec["experiment_start"], spec["experiment_end"], calendar)
    previous_cutoff = sessions(
        environment.history.session_date.iloc[-1],
        environment.history.session_date.iloc[-1],
        calendar,
    ).market_open.iloc[-1]
    if condition == "no_event":
        events = []
    elif condition == "e1_only":
        events = [event for event in events if event.event_id == config["events"]["primary"]]
    rng = random.Random(config["experiment"]["seed"] + 991)
    policy_rngs = [
        random.Random(config["experiment"]["seed"] * 1000 + i) for i in range(len(agents))
    ]
    majority_threshold = config["behavior"].get("majority_threshold", 0.1)
    mean_field = compute_mean_field(
        agents,
        as_of_session=environment.history.session_date.iloc[-1],
        majority_threshold=majority_threshold,
    )
    fundamental = initial_price
    shock_applied = False
    owned = client is None
    if client is None:
        load_dotenv(ROOT / ".env")
        client = (
            OpenAICompatibleClient(
                llm_config(config, provider),
                None if cache_only else os.getenv(llm_config(config, provider)["api_key_env"]),
                output,
                ROOT / llm_config(config, provider)["cache_dir"],
                semaphore,
                pacer,
                cache_only,
            )
            if provider in LIVE_PROVIDERS and treatment != "rule_abm"
            else MockLLMClient()
        )
    if isinstance(client, OpenAICompatibleClient) and not cache_only:
        await client.verify_model()
    tables = ([], [], [], [], [], [], [], [])
    order_rows, trade_rows, market_rows, agent_rows = tables[:4]
    mean_rows, switch_rows, event_rows, book_rows = tables[4:]

    def prepare_decision(agent, observation, kind):
        system, user = prompts(
            agent,
            observation,
            "vanilla_llm" if hybrid else treatment,
            kind,
            config["experiment"]["seed"],
            mean_field_visibility=mean_field_visibility,
        )
        identity = {
            "agent_id": agent.persona.agent_id,
            "session_date": observation["session_date"],
            "treatment": treatment,
            "kind": kind,
            "mode": mode,
            "seed": config["experiment"]["seed"],
            "shock": config["fundamental_shock"]["magnitude"],
            "event_condition": condition,
            "decision_batch": observation.get("decision_batch"),
        }
        # Preserve existing visible request identities and their cache entries.
        if mean_field_visibility == "hidden":
            identity["mean_field_visibility"] = "hidden"
        if finite_spec is not None:
            identity.update(strategy_return_method=return_method, prompt_version=prompt_version)
        if hybrid:
            identity.update(order_source="rule", style_source="model_interface")
        return system, user, identity

    async def decide(agent, observation, kind):
        system, user, identity = prepare_decision(agent, observation, kind)
        return await client.decide(
            system, user, OrderDecision if kind == "order" else StyleDecision, identity
        )

    def save_tables():
        named = dict(
            zip(
                (
                    "orders",
                    "trades",
                    "market",
                    "agents",
                    "mean_field",
                    "style_switches",
                    "events_seen",
                    "order_book_snapshots",
                ),
                tables,
                strict=True,
            )
        )
        for name, rows in named.items():
            frame = pd.DataFrame(rows)
            frame.to_parquet(output / f"{name}.parquet", index=False)
            frame.to_csv(output / f"{name}.csv", index=False)

    try:
        for day, (session, timing) in enumerate(schedule.iterrows()):
            failure_phase = "trading_and_daily_accounting"
            cutoff = timing.market_open
            if not shock_applied and any(
                e.event_id == config["events"]["primary"] and e.first_seen_at <= cutoff
                for e in events
            ):
                fundamental = initial_price * (1 + config["fundamental_shock"]["magnitude"])
                shock_applied = True
            context = {}
            if mode == "historical":
                for symbol in spec["context_symbols"]:
                    completed = bars.loc[
                        (bars.symbol == symbol) & (bars.session_date < session)
                    ].sort_values("session_date")
                    if len(completed) >= 2:
                        context[symbol] = (
                            float(completed.close.iloc[-1]) / float(completed.close.iloc[-2]) - 1
                        )
            observation = build_observation(
                environment.history,
                session,
                cutoff,
                previous_cutoff,
                events,
                fundamental,
                mean_field,
                context,
            )
            observation.update(
                symbol=spec["primary_symbol"], **available_fundamentals(facts, session)
            )
            price_band = config["behavior"].get("limit_price_band", [0.95, 1.05])
            price_floor = observation["last_close"] * price_band[0]
            price_ceiling = observation["last_close"] * price_band[1]
            observation["permitted_price_band"] = {
                "minimum": round(price_floor, 6),
                "maximum": round(price_ceiling, 6),
                "reference": "session_open_last_trade",
            }
            observation["order_book"] = (
                environment.book_context()
                if mode == "endogenous"
                else {
                    "best_bid": None,
                    "best_ask": None,
                    "spread": None,
                    "spread_bps": None,
                    "last_trade": observation["last_close"],
                    "bid_depth": 0,
                    "ask_depth": 0,
                    "scope": "unavailable_in_historical_replay",
                }
            )
            with (output / "observations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
            previous_cutoff = cutoff
            # Execute the two hypothetical rules before observing today's close or
            # mutating the actual account. These accounts never enter the actual book.
            for pair in finite_pairs.values():
                pair.open_session(observation)
            for event in observation["visible_events"]:
                event_rows.append(
                    {
                        "session_date": session,
                        "event_id": event["event_id"],
                        "published_at": event["first_seen_at"],
                        "visible_at": cutoff.isoformat(),
                        "delay_hours": (
                            cutoff - pd.Timestamp(event["first_seen_at"])
                        ).total_seconds()
                        / 3600,
                    }
                )
            before = [float(a.portfolio.total_equity) for a in agents]
            arrival = list(range(len(agents)))
            rng.shuffle(arrival)
            batch_size = config["behavior"].get("order_decision_batch_size", len(agents))
            microbatch = mode == "endogenous" and not rule_orders and batch_size < len(agents)
            decision_contexts = [observation] * len(agents)
            if microbatch:
                environment.start_session(session)
                results = [None] * len(agents)
                decisions = [None] * len(agents)
                validations = [None] * len(agents)
                for batch_number, start in enumerate(range(0, len(arrival), batch_size)):
                    indices = arrival[start : start + batch_size]
                    batch_observation = {
                        **observation,
                        "order_book": environment.book_context(),
                        "decision_batch": batch_number,
                    }
                    book_rows.append(
                        {
                            "session_date": session,
                            "decision_batch": batch_number,
                            "agent_ids": [agents[index].persona.agent_id for index in indices],
                            **batch_observation["order_book"],
                        }
                    )
                    batch_results = await asyncio.gather(
                        *(decide(agents[index], batch_observation, "order") for index in indices)
                    )
                    for index, result in zip(indices, batch_results, strict=True):
                        results[index] = result
                        decision_contexts[index] = batch_observation
                        valid, violation = enforce_portfolio(
                            result[0], agents[index].portfolio, price_floor, price_ceiling
                        )
                        decisions[index] = valid
                        validations[index] = violation or result[1]["status"]
                        environment.submit(agents[index], valid)
                close, volume, execution, fills = environment.finish_session(agents)
            elif rule_orders:
                results = [
                    (rule_order(a, observation, policy_rngs[i]), {"status": "rule"})
                    for i, a in enumerate(agents)
                ]
                decisions, validations = [], []
                for agent, (decision, meta) in zip(agents, results, strict=True):
                    valid, violation = enforce_portfolio(
                        decision, agent.portfolio, price_floor, price_ceiling
                    )
                    decisions.append(valid)
                    validations.append(violation or meta["status"])
                close, volume, execution, fills = environment.settle(
                    session, agents, decisions, arrival
                )
            else:
                results = await asyncio.gather(*(decide(a, observation, "order") for a in agents))
                decisions, validations = [], []
                for agent, (decision, meta) in zip(agents, results, strict=True):
                    valid, violation = enforce_portfolio(
                        decision, agent.portfolio, price_floor, price_ceiling
                    )
                    decisions.append(valid)
                    validations.append(violation or meta["status"])
                close, volume, execution, fills = environment.settle(
                    session, agents, decisions, arrival
                )
            trade_rows.extend(fills)
            execution = {row["agent_id"]: row for row in execution}
            last = observation["last_close"]
            for i, (agent, decision) in enumerate(zip(agents, decisions, strict=True)):
                fill = execution[agent.persona.agent_id]
                book = decision_contexts[i]["order_book"]
                order_rows.append(
                    {
                        "session_date": session,
                        "agent_id": agent.persona.agent_id,
                        "style": agent.current_style,
                        "mean_field_visibility": mean_field_visibility,
                        "prompt_mean_field_supplied": None
                        if hybrid
                        else prompt_mean_field_supplied,
                        **decision.model_dump(),
                        **fill,
                        "validation_status": validations[i],
                        "fallback": bool(results[i][1].get("fallback")),
                        "requested_action": results[i][0].action,
                        "requested_quantity": results[i][0].quantity,
                        "decision_batch": decision_contexts[i].get("decision_batch"),
                        "seen_best_bid": book["best_bid"],
                        "seen_best_ask": book["best_ask"],
                        "seen_spread_bps": book["spread_bps"],
                        "bid_distance_to_ask_bps": (
                            (book["best_ask"] - decision.limit_price) / last * 10_000
                            if decision.action == "buy" and book["best_ask"] is not None
                            else None
                        ),
                        "ask_distance_to_bid_bps": (
                            (decision.limit_price - book["best_bid"]) / last * 10_000
                            if decision.action == "sell" and book["best_bid"] is not None
                            else None
                        ),
                        "prior_majority_action": mean_field["majority_action"],
                        "prior_majority_strength": mean_field["majority_strength"],
                        "aggressiveness": (
                            (decision.limit_price / last - 1)
                            * (1 if decision.action == "buy" else -1)
                        )
                        if decision.limit_price
                        else 0.0,
                    }
                )
                if finite_spec is None:
                    alternative = (
                        "technical" if agent.current_style == "fundamental" else "fundamental"
                    )
                    exposure = 1.0 if signal(alternative, observation) > 0 else 0.0
                    agent.shadow_growth *= 1 + exposure * (close / last - 1)
                else:
                    pair = finite_pairs[agent.persona.agent_id]
                    rows = pair.close_session(close)
                    agent.finite_review = pair.summary(agent.portfolio)
                    with (output / "finite_rule_replay.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        for row in rows:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                equity = float(agent.portfolio.total_equity)
                agent.peak_equity = max(agent.peak_equity, equity)
                agent.memory.append(
                    {
                        "session": session,
                        "style": agent.current_style,
                        "action": decision.action,
                        "filled_quantity": fill["filled_quantity"],
                        "pnl_change": round(equity - before[i], 4),
                        "market_return": round(close / last - 1, 6),
                        "events": [e["event_id"] for e in observation["visible_events"]],
                    }
                )
                agent.memory = agent.memory[-config["behavior"]["memory_window"] :]
                agent_rows.append(
                    {
                        "session_date": session,
                        "agent_id": agent.persona.agent_id,
                        "current_style": agent.current_style,
                        **agent.context()["portfolio"],
                        **return_columns(agent),
                    }
                )
            # Trading and valuation are complete even if the following review fails.
            # Preserve the pre-review population snapshot until all choices commit.
            mean_field = compute_mean_field(
                agents, decisions, session, majority_threshold=majority_threshold
            )
            mean_rows.append({"session_date": session, **mean_field})
            market_rows.append(
                {
                    "session_date": session,
                    "open": float(environment.history.open.iloc[-1]),
                    "high": float(environment.history.high.iloc[-1]),
                    "low": float(environment.history.low.iloc[-1]),
                    "previous_close": last,
                    "close": close,
                    "volume": volume,
                    "simulated_volume": sum(t["quantity"] for t in fills),
                    "trade_count": len(fills),
                    "fundamental_value": fundamental,
                    "order_imbalance": mean_field["buy_sell_imbalance"],
                }
            )
            if (day + 1) % config["behavior"]["strategy_switch_interval"] == 0:
                interval = config["behavior"]["strategy_switch_interval"]
                closing_book = (
                    {
                        **environment.book_context(),
                        "scope": "post_close_after_day_order_expiry",
                    }
                    if mode == "endogenous"
                    else {**observation["order_book"], "last_trade": close}
                )
                end_observation = {
                    **observation,
                    **features(environment.history),
                    "decision_at_utc": timing.market_close.isoformat(),
                    "decision_at_et": timing.market_close.tz_convert(
                        "America/New_York"
                    ).isoformat(),
                    "history_last_session": session,
                    "phase": "post_close_style_review",
                    "order_book": closing_book,
                    "review_block": {
                        "start_session": schedule.index[day + 1 - interval],
                        "end_session": session,
                        "completed_sessions": interval,
                    },
                }
                # The opening order-price constraint is no longer a current review fact.
                end_observation.pop("permitted_price_band")
                if hybrid:
                    failure_phase = "style_review_dispatch"
                    prepared = [prepare_decision(a, end_observation, "style") for a in agents]
                    style_results = await review_batch(
                        client,
                        prepared,
                        [a.current_style for a in agents],
                        output / "hybrid_review_batches.jsonl",
                    )
                    # Validate every reset before mutating any real agent or ledger.
                    failure_phase = "prepare_review_resets"
                    next_pairs = {}
                    next_summaries = {}
                    for agent, (decision, _) in zip(agents, style_results, strict=True):
                        clone = copy.deepcopy(agent)
                        clone.current_style = decision.target_style
                        next_pairs[agent.persona.agent_id] = FiniteRulePair(
                            clone,
                            close,
                            finite_spec,
                            finite_pairs[agent.persona.agent_id].block_index + 1,
                        )
                        next_summaries[agent.persona.agent_id] = next_pairs[
                            agent.persona.agent_id
                        ].summary(agent.portfolio)
                else:
                    style_results = (
                        [
                            (
                                StyleDecision(
                                    decision="stay",
                                    target_style=a.current_style,
                                    rationale=(
                                        "fixed rule policy: review and reset without switching"
                                    ),
                                )
                                if adaptation == "fixed"
                                else rule_style(a),
                                {"status": "rule"},
                            )
                            for a in agents
                        ]
                        if treatment == "rule_abm"
                        else await asyncio.gather(
                            *(decide(a, end_observation, "style") for a in agents)
                        )
                    )
                if treatment == "rule_abm":
                    decision_source = "fixed_rule" if adaptation == "fixed" else "rule"
                elif isinstance(client, MockLLMClient):
                    decision_source = "MockLLMClient"
                elif isinstance(client, OpenAICompatibleClient):
                    decision_source = "OpenAICompatibleClient"
                else:
                    decision_source = "injected_client"
                failure_phase = "prepare_review_rows"
                pending_switch_rows = []
                for agent, (decision, meta) in zip(agents, style_results, strict=True):
                    request_metadata = (
                        {} if decision_source in {"rule", "fixed_rule", "MockLLMClient"} else meta
                    )
                    coherent = (decision.decision == "stay") == (
                        decision.target_style == agent.current_style
                    )
                    valid_choice = (
                        coherent
                        and meta["status"] in {"ok", "mock", "rule"}
                        and not meta.get("fallback", False)
                    )
                    pending_switch_rows.append(
                        {
                            "session_date": session,
                            "agent_id": agent.persona.agent_id,
                            "from_style": agent.current_style,
                            "phase": end_observation["phase"],
                            "decision_at_utc": end_observation["decision_at_utc"],
                            "prompt_version": prompt_version,
                            "decision_source": decision_source,
                            "mean_field_visibility": mean_field_visibility,
                            "prompt_mean_field_supplied": prompt_mean_field_supplied,
                            **adaptation_record,
                            **source_record,
                            "request_hash": request_metadata.get("request_hash"),
                            "cache_hit": request_metadata.get("cache_hit"),
                            "repair_count": request_metadata.get("repair_count"),
                            "requested_decision": None
                            if meta.get("fallback", False)
                            else decision.decision,
                            "requested_target_style": None
                            if meta.get("fallback", False)
                            else decision.target_style,
                            "decision": decision.decision if valid_choice else "stay",
                            "target_style": decision.target_style
                            if valid_choice
                            else agent.current_style,
                            "choice_valid": valid_choice,
                            "fallback": bool(meta.get("fallback", False)),
                            "rationale": decision.rationale,
                            "validation_status": meta["status"]
                            if coherent
                            else "inconsistent_style",
                            **return_columns(agent),
                        }
                    )
                failure_phase = "apply_completed_review"
                for agent, row in zip(agents, pending_switch_rows, strict=True):
                    valid_choice = row["choice_valid"]
                    if valid_choice:
                        agent.current_style = row["target_style"]
                    agent.block_start_equity = float(agent.portfolio.total_equity)
                    agent.shadow_growth = 1.0
                    if finite_spec is not None:
                        pair = (
                            next_pairs[agent.persona.agent_id]
                            if hybrid
                            else FiniteRulePair(
                                agent,
                                close,
                                finite_spec,
                                finite_pairs[agent.persona.agent_id].block_index + 1,
                            )
                        )
                        finite_pairs[agent.persona.agent_id] = pair
                        agent.finite_review = (
                            next_summaries[agent.persona.agent_id]
                            if hybrid
                            else pair.summary(agent.portfolio)
                        )
                switch_rows.extend(pending_switch_rows)
                if hybrid:
                    append_event(
                        output / "hybrid_review_batches.jsonl",
                        {
                            "phase": "applied",
                            "batch": session,
                            "count": len(agents),
                        },
                    )
            mean_field = compute_mean_field(
                agents, decisions, session, majority_threshold=majority_threshold
            )
            mean_rows[-1] = {"session_date": session, **mean_field}
            failure_phase = "save_daily_progress"
            write_json(
                output / "progress.json",
                {
                    "completed_sessions": day + 1,
                    "total_sessions": len(schedule),
                    "session": session,
                    "close": close,
                },
            )
            print(
                f"{mode}/{treatment}/seed{config['experiment']['seed']}/"
                f"shock{config['fundamental_shock']['magnitude']}: "
                f"{day + 1}/{len(schedule)} close={close:.4f} trades={len(fills)}",
                flush=True,
            )
    except (Exception, asyncio.CancelledError) as exc:
        metadata.update(
            status="interrupted",
            finished_at=datetime.now(UTC).isoformat(),
            error_type=type(exc).__name__,
        )
        if hybrid:
            save_tables()
            metadata.update(
                completed_market_sessions=len(market_rows),
                last_completed_market_session=market_rows[-1]["session_date"]
                if market_rows
                else None,
                applied_review_rows=len(switch_rows),
                last_applied_review=switch_rows[-1]["session_date"] if switch_rows else None,
                evidence_status="incomplete_not_eligible_for_result_comparison",
                failure_phase=failure_phase,
                current_styles={a.persona.agent_id: a.current_style for a in agents},
                artifact_hashes={
                    p.name: digest(p.read_bytes()) for p in sorted(output.glob("*.parquet"))
                },
            )
            for name in ("finite_rule_replay.jsonl", "hybrid_review_batches.jsonl"):
                if (output / name).exists():
                    metadata["artifact_hashes"][name] = digest((output / name).read_bytes())
        write_json(metadata_path, metadata)
        raise
    finally:
        if owned and isinstance(client, OpenAICompatibleClient):
            await client.close()
    save_tables()
    metrics = run_metrics(
        pd.DataFrame(market_rows),
        pd.DataFrame(order_rows),
        pd.DataFrame(switch_rows),
        initial_price,
        sum(a.portfolio.position for a in agents)
        if mode == "endogenous"
        else len(agents) * config["population"]["initial_shares"],
    )
    metrics["trades"] = len(trade_rows)
    write_json(output / "metrics.json", metrics)
    metadata.update(
        status="complete",
        finished_at=datetime.now(UTC).isoformat(),
        sessions=len(schedule),
        quality="engineering_mock"
        if provider == "mock"
        else (
            "degraded"
            if max(metrics["fallback_rate"], metrics["invalid_style_rate"]) > 0.05
            else "pilot"
        ),
    )
    metadata["artifact_hashes"] = {
        p.name: digest(p.read_bytes()) for p in sorted(output.glob("*.parquet"))
    }
    if finite_spec is not None:
        metadata["artifact_hashes"]["finite_rule_replay.jsonl"] = digest(
            (output / "finite_rule_replay.jsonl").read_bytes()
        )
    if hybrid:
        metadata["artifact_hashes"]["hybrid_review_batches.jsonl"] = digest(
            (output / "hybrid_review_batches.jsonl").read_bytes()
        )
    write_json(metadata_path, metadata)
    return metrics


def main(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument(
        "--mode", choices=["historical", "endogenous"], default=mode or "historical"
    )
    parser.add_argument("--treatment", choices=TREATMENTS, default="vanilla_llm")
    parser.add_argument("--provider", choices=["mock", *LIVE_PROVIDERS], default="siliconflow")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--shock", type=float)
    parser.add_argument("--symbol")
    parser.add_argument("--event-condition", choices=["no_event", "e1_only", "e1_e2"])
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.agents is not None:
        config["population"]["n_agents"] = args.agents
    if args.shock is not None:
        config["fundamental_shock"]["magnitude"] = args.shock
    if args.symbol is not None:
        config["data"]["primary_symbol"] = args.symbol
    if args.event_condition is not None:
        config["events"]["condition"] = args.event_condition
    print(
        json.dumps(
            asyncio.run(
                run_experiment(
                    config, args.mode, args.treatment, args.provider, cache_only=args.cache_only
                )
            )
        )
    )
