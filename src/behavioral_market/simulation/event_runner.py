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

ROOT = Path(__file__).resolve().parents[3]
TREATMENTS = ("rule_abm", "vanilla_llm", "low_herding", "high_herding")
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
    manifest_path = ROOT / "data/manifests/meta_compute_2026_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["processed_files"]:
        if digest((ROOT / record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError("processed dataset hash mismatch")
    bars = pd.read_parquet(directory / "market_daily.parquet")
    events = [
        MarketEvent.model_validate(row)
        for row in json.loads((directory / "events.json").read_text())
    ]
    sec = directory / "sec_companyfacts.json"
    return (
        bars,
        events,
        (json.loads(sec.read_text()) if sec.exists() else None),
        digest(manifest_path.read_bytes()),
    )


def output_path(config, mode, treatment, provider):
    seed = config["experiment"]["seed"]
    shock = config["fundamental_shock"]["magnitude"]
    secondary = config["events"]["include_secondary"]
    count = config["population"]["n_agents"]
    return (
        ROOT
        / "outputs/meta_compute_2026"
        / provider
        / mode
        / treatment
        / f"n{count}_seed{seed}_shock{shock:g}_e2{int(secondary)}"
    )


def source_hash():
    paths = sorted((ROOT / "src/behavioral_market").rglob("*.py"))
    return digest(b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in paths))


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
    if provider in LIVE_PROVIDERS and treatment != "rule_abm" and client is None and pacer is None:
        pacer = RequestPacer(4.0)
    config["run"] = {"mode": mode, "treatment": treatment, "provider": provider}
    if bars is None:
        bars, events, facts, manifest_hash = load_dataset(config)
    output = output or output_path(config, mode, treatment, provider)
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
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "treatment": treatment,
        "seed": config["experiment"]["seed"],
        "shock": config["fundamental_shock"]["magnitude"],
        "include_secondary": config["events"]["include_secondary"],
        "llm_parameters": config["llm"],
        "n_agents": config["population"]["n_agents"],
        "request_rate_limit_rps": pacer.requests_per_second if pacer else None,
        "cache_only_replay": cache_only,
    }
    write_json(metadata_path, metadata)
    write_json(output / "config_snapshot.json", config)
    (output / "observations.jsonl").write_text("")
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
    environment = (
        HistoricalReplayEnvironment(primary, spec["experiment_start"])
        if mode == "historical"
        else EndogenousEventEnvironment(
            primary, spec["experiment_start"], spec["primary_symbol"], agents
        )
    )
    schedule = sessions(spec["experiment_start"], spec["experiment_end"])
    previous_cutoff = sessions(
        environment.history.session_date.iloc[-1], environment.history.session_date.iloc[-1]
    ).market_open.iloc[-1]
    events = [
        event
        for event in events
        if config["events"]["include_secondary"] or event.event_id != config["events"]["secondary"]
    ]
    rng = random.Random(config["experiment"]["seed"] + 991)
    policy_rngs = [
        random.Random(config["experiment"]["seed"] * 1000 + i) for i in range(len(agents))
    ]
    mean_field = compute_mean_field(agents, as_of_session=environment.history.session_date.iloc[-1])
    fundamental = initial_price
    shock_applied = False
    owned = client is None
    if client is None:
        load_dotenv(ROOT / ".env")
        client = (
            OpenAICompatibleClient(
                llm_config(config, provider),
                None
                if cache_only
                else os.getenv(llm_config(config, provider)["api_key_env"]),
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
    order_rows, trade_rows, market_rows, agent_rows, mean_rows, switch_rows, event_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    async def decide(agent, observation, kind):
        system, user = prompts(agent, observation, treatment, kind, config["experiment"]["seed"])
        identity = {
            "agent_id": agent.persona.agent_id,
            "session_date": observation["session_date"],
            "treatment": treatment,
            "kind": kind,
            "mode": mode,
            "seed": config["experiment"]["seed"],
            "shock": config["fundamental_shock"]["magnitude"],
            "include_secondary": config["events"]["include_secondary"],
        }
        return await client.decide(
            system, user, OrderDecision if kind == "order" else StyleDecision, identity
        )

    try:
        for day, (session, timing) in enumerate(schedule.iterrows()):
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
            with (output / "observations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
            previous_cutoff = cutoff
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
            if treatment == "rule_abm":
                results = [
                    (rule_order(a, observation, policy_rngs[i]), {"status": "rule"})
                    for i, a in enumerate(agents)
                ]
            else:
                results = await asyncio.gather(*(decide(a, observation, "order") for a in agents))
            decisions, validations = [], []
            for agent, (decision, meta) in zip(agents, results, strict=True):
                valid, violation = enforce_portfolio(decision, agent.portfolio)
                decisions.append(valid)
                validations.append(violation or meta["status"])
            arrival = list(range(len(agents)))
            rng.shuffle(arrival)
            close, volume, execution, fills = environment.settle(
                session, agents, decisions, arrival
            )
            trade_rows.extend(fills)
            execution = {row["agent_id"]: row for row in execution}
            last = observation["last_close"]
            for i, (agent, decision) in enumerate(zip(agents, decisions, strict=True)):
                fill = execution[agent.persona.agent_id]
                order_rows.append(
                    {
                        "session_date": session,
                        "agent_id": agent.persona.agent_id,
                        "style": agent.current_style,
                        **decision.model_dump(),
                        **fill,
                        "validation_status": validations[i],
                        "fallback": bool(results[i][1].get("fallback")),
                        "requested_action": results[i][0].action,
                        "requested_quantity": results[i][0].quantity,
                        "aggressiveness": (
                            (decision.limit_price / last - 1)
                            * (1 if decision.action == "buy" else -1)
                        )
                        if decision.limit_price
                        else 0.0,
                    }
                )
                alternative = "technical" if agent.current_style == "fundamental" else "fundamental"
                exposure = 1.0 if signal(alternative, observation) > 0 else 0.0
                agent.shadow_growth *= 1 + exposure * (close / last - 1)
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
                        "actual_block_return": equity / agent.block_start_equity - 1,
                        "counterfactual_block_return": agent.shadow_growth - 1,
                    }
                )
            if (day + 1) % config["behavior"]["strategy_switch_interval"] == 0:
                end_observation = {
                    **observation,
                    **features(environment.history),
                    "decision_at_utc": timing.market_close.isoformat(),
                    "decision_at_et": timing.market_close.tz_convert(
                        "America/New_York"
                    ).isoformat(),
                    "history_last_session": session,
                    "phase": "post_close_style_review",
                }
                style_results = (
                    [(rule_style(a), {"status": "rule"}) for a in agents]
                    if treatment == "rule_abm"
                    else await asyncio.gather(
                        *(decide(a, end_observation, "style") for a in agents)
                    )
                )
                for agent, (decision, meta) in zip(agents, style_results, strict=True):
                    coherent = (decision.decision == "stay") == (
                        decision.target_style == agent.current_style
                    )
                    switch_rows.append(
                        {
                            "session_date": session,
                            "agent_id": agent.persona.agent_id,
                            "from_style": agent.current_style,
                            "decision": decision.decision if coherent else "stay",
                            "target_style": decision.target_style
                            if coherent
                            else agent.current_style,
                            "rationale": decision.rationale,
                            "validation_status": meta["status"]
                            if coherent
                            else "inconsistent_style",
                            "actual_block_return": float(agent.portfolio.total_equity)
                            / agent.block_start_equity
                            - 1,
                            "counterfactual_block_return": agent.shadow_growth - 1,
                        }
                    )
                    if coherent:
                        agent.current_style = decision.target_style
                    agent.block_start_equity = float(agent.portfolio.total_equity)
                    agent.shadow_growth = 1.0
            mean_field = compute_mean_field(agents, decisions, session)
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
        write_json(metadata_path, metadata)
        raise
    finally:
        if owned and isinstance(client, OpenAICompatibleClient):
            await client.close()
    tables = {
        "agents": agent_rows,
        "orders": order_rows,
        "trades": trade_rows,
        "market": market_rows,
        "mean_field": mean_rows,
        "events_seen": event_rows,
        "style_switches": switch_rows,
    }
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        frame.to_parquet(output / f"{name}.parquet", index=False)
        frame.to_csv(output / f"{name}.csv", index=False)
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
        else ("degraded" if metrics["fallback_rate"] > 0.05 else "pilot"),
    )
    metadata["artifact_hashes"] = {
        p.name: digest(p.read_bytes()) for p in sorted(output.glob("*.parquet"))
    }
    write_json(metadata_path, metadata)
    return metrics


def main(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument(
        "--mode", choices=["historical", "endogenous"], default=mode or "historical"
    )
    parser.add_argument("--treatment", choices=TREATMENTS, default="vanilla_llm")
    parser.add_argument(
        "--provider", choices=["mock", *LIVE_PROVIDERS], default="siliconflow"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--shock", type=float)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.agents is not None:
        config["population"]["n_agents"] = args.agents
    if args.shock is not None:
        config["fundamental_shock"]["magnitude"] = args.shock
    print(
        json.dumps(
            asyncio.run(
                run_experiment(
                    config, args.mode, args.treatment, args.provider, cache_only=args.cache_only
                )
            )
        )
    )
