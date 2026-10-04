# ruff: noqa: E402
"""Freeze, prepare and run a bounded development-only human financial forecast study."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.client import OpenAICompatibleClient, RequestPacer
from behavioral_market.simulation.event_runner import llm_config
from experiments.financial_probability import METHODS, Demonstrations, key, question, select_targets
from experiments.market_quote_memory_baselines import predict
from experiments.probability_elicitation import LLMProbabilityClient
from experiments.replication_development_data import FREEZE, checked_freeze, load_development_frames
from experiments.replication_metadata import PLAN
from experiments.replication_quote_memory import extend_memory
from experiments.replication_quote_states import quote_states
from experiments.replication_training_audit import audit_training, normalized_tables
from experiments.replication_training_data import checked_plan

CONFIG = "config/financial_probability_development_20261004.json"
TRAIN = "outputs/market_quote_memory_training_20261004"
OUT = "outputs/financial_probability_development_20261004"
CODE = [
    "experiments/financial_probability.py",
    "experiments/replication_development_data.py",
    "experiments/replication_quote_memory.py",
    "experiments/replication_quote_states.py",
    "experiments/market_quote_memory_baselines.py",
    "experiments/market_quote_baselines.py",
    "experiments/replication_training_data.py",
    "experiments/replication_training_audit.py",
    "experiments/probability_elicitation.py",
    "src/behavioral_market/llm/client.py",
    "src/behavioral_market/simulation/event_runner.py",
    "src/behavioral_market/data/archive.py",
    "experiments/replication_metadata.py",
    "scripts/run_financial_probability.py",
    "scripts/build_market_quote_memory.py",
    "scripts/analyze_financial_probability.py",
    CONFIG,
    "config/memory_market_2026.json",
]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in values))


def training(root):
    folder = root / TRAIN
    manifest = read(folder / "manifest.json")
    for path, expected in manifest["code_hashes"].items():
        assert digest((root / path).read_bytes()) == expected, path
    for file, field in [
        ("states.jsonl", "states_sha256"),
        ("baseline_model.json", "baseline_model_sha256"),
    ]:
        assert digest((folder / file).read_bytes()) == manifest[field], file
    records = rows(folder / "states.jsonl")
    plan = checked_plan(root)
    assert {r["provenance"]["session"] for r in records} <= set(plan["assignments"]["train"])
    return records, read(folder / "baseline_model.json")


def freeze(root):
    training(root)
    paths = [
        *CODE,
        f"{TRAIN}/manifest.json",
        f"{TRAIN}/states.jsonl",
        f"{TRAIN}/baseline_model.json",
    ]
    frozen = {
        "scope": "development only; test remains closed",
        "partition_sha256": digest((root / PLAN).read_bytes()),
        "input_hashes": {p: digest((root / p).read_bytes()) for p in paths},
        "primary_metric": "session-then-person-balanced Brier on all 400 common targets",
        "coverage": (
            "All 3200 responses valid after at most one schema repair; otherwise no primary scores"
        ),
        "contrasts": [
            "real_labels minus no_examples",
            "real_labels minus neutral_labels",
            "real_labels minus shuffled_labels",
            "real_labels minus memory_ridge_logistic",
            "real_labels minus demonstration_mean",
        ],
        "uncertainty": (
            "10000 paired resamples of 16 whole sessions, percentile 95%; "
            "descriptive multiple contrasts"
        ),
        "selection": (
            "25 eligible orders per development session by seeded hash; "
            "never by current side or error"
        ),
        "interpretation": (
            "Active-order side given partial observation and recorded history; "
            "not HOLD or autonomous market validation"
        ),
    }
    path = root / FREEZE
    if path.exists() and read(path) != frozen:
        raise ValueError("existing freeze differs; use a versioned study")
    write_json(path, frozen)
    print(json.dumps({"frozen": FREEZE, "sha256": digest(path.read_bytes())}))


def prepare(root):
    frozen = checked_freeze(root)
    config = read(root / CONFIG)
    out = root / OUT
    if (out / "preparation.json").exists():
        audit_preparation(root)
        print("Existing frozen preparation verified")
        return
    train, model = training(root)
    frames, metadata, plan = load_development_frames(root)
    audit, tables = audit_training(frames, metadata)
    audit["scope"] = "16 development sessions only; test behavior remains closed"
    errors = audit["accounting_and_reference_errors"]
    critical = [
        "cash_period_residuals",
        "asset_period_residuals",
        "dividend_payment_residuals",
        "next_cash_residuals",
        "invalid_initial_endowments",
        "period_cash_flow_not_conserved",
        "period_asset_flow_not_conserved",
        "trade_price_order_disagreements",
        "trade_precedes_order",
    ]
    if any(errors[name] for name in critical):
        write_json(out / "failed_accounting_audit.json", audit)
        raise ValueError("development accounting invalid; do not fabricate observations")
    normalized, _ = normalized_tables(frames)
    records, exclusions = quote_states(
        normalized["orders"],
        tables["linked_trades"],
        tables["ledger"],
        tables["market_periods"],
        metadata["_markets"],
    )
    selected = select_targets(
        records, set(plan["assignments"]["development"]), config["quota"], config["seed"]
    )
    selected = extend_memory(selected, normalized["orders"], tables["ledger"])
    assert len(selected) == config["expected_targets"]
    examples = Demonstrations(train, model, config["seed"])
    bundles = [{"target_id": key(r), "bundle": examples.bundle(r["state"])} for r in selected]
    baseline = predict(model, selected)
    baseline["demonstration_mean"] = np.array(
        [np.mean([d["real_buy"] for d in b["bundle"]["demonstrations"]]) for b in bundles]
    )
    baseline["constant_half"] = np.full(len(selected), 0.5)
    write_rows(out / "targets.jsonl", selected)
    write_rows(out / "bundles.jsonl", bundles)
    write_json(out / "baselines.json", {name: values.tolist() for name, values in baseline.items()})
    write_json(out / "accounting_audit.json", audit)
    files = ["targets.jsonl", "bundles.jsonl", "baselines.json", "accounting_audit.json"]
    summary = {
        "scope": frozen["scope"],
        "freeze_sha256": digest((root / FREEZE).read_bytes()),
        "files": {f: digest((out / f).read_bytes()) for f in files},
        "eligible": len(records),
        "exclusions": exclusions,
        "targets": len(selected),
        "sessions": len({r["provenance"]["session"] for r in selected}),
        "target_ids": [key(r) for r in selected],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    write_json(out / "preparation.json", summary)
    print(json.dumps({k: summary[k] for k in ("scope", "eligible", "targets", "sessions")}))


def audit_preparation(root):
    checked_freeze(root)
    summary = read(root / OUT / "preparation.json")
    assert summary["freeze_sha256"] == digest((root / FREEZE).read_bytes())
    for path, expected in summary["files"].items():
        assert digest((root / OUT / path).read_bytes()) == expected, path
    return summary


async def run(root, smoke=False):
    checked_freeze(root)
    config = read(root / CONFIG)
    if config["methods"] != list(METHODS) or config["menu_orders"] != ["buy_first", "sell_first"]:
        raise ValueError("configuration/task menu mismatch")
    if not 1 <= config["request_concurrency"] <= 32 or not 0 < config["requests_per_second"] <= 4:
        raise ValueError("invalid request capacity")
    if smoke:
        train, model = training(root)
        examples = Demonstrations(train, model, config["seed"])
        # Deterministic train-only probes, separate from development results.
        selected = sorted(train, key=key)[:2]
        bundles = [{"target_id": key(r), "bundle": examples.bundle(r["state"])} for r in selected]
        methods = ("no_examples", "real_labels")
    else:
        audit_preparation(root)
        bundles = rows(root / OUT / "bundles.jsonl")
        methods = METHODS
    stage = "smoke" if smoke else "main"
    out = root / OUT / stage
    provider = llm_config(read(root / config["provider_config"]), config["provider"])
    provider.update(config["provider_overrides"])
    load_dotenv(root / ".env")
    transport = OpenAICompatibleClient(
        provider,
        os.getenv(provider["api_key_env"], ""),
        out / "transport",
        root / ".cache/financial_probability_development_20261004" / stage,
        semaphore=asyncio.Semaphore(config["request_concurrency"]),
        pacer=RequestPacer(config["requests_per_second"]),
        cache_namespace="financial-probability-development-v1",
    )
    client = LLMProbabilityClient(transport)
    metadata = {
        "status": "running",
        "stage": stage,
        "provider_config": provider,
        "freeze_sha256": digest((root / FREEZE).read_bytes()),
        "preparation_sha256": digest((root / OUT / "preparation.json").read_bytes())
        if not smoke
        else None,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "expected_responses": len(bundles) * len(methods) * 2,
    }
    destination = out / "responses.jsonl"
    if (out / "metadata.json").exists():
        old = read(out / "metadata.json")
        for field in (
            "provider_config",
            "freeze_sha256",
            "preparation_sha256",
            "expected_responses",
        ):
            if old[field] != metadata[field]:
                raise ValueError("run inputs changed")
        if old["status"] == "complete":
            print("Existing completed matrix retained")
            await transport.close()
            return
    write_json(out / "metadata.json", metadata)
    existing = rows(destination) if destination.exists() else []
    by_key = {(r["target_id"], r["method"], r["order"]): r for r in existing}
    if len(by_key) != len(existing):
        raise ValueError("duplicate response identities")
    work = asyncio.Semaphore(config["request_concurrency"])
    completed = len(existing)

    async def one(bundle, method, order):
        nonlocal completed
        identity = {
            "target_id": bundle["target_id"],
            "method": method,
            "order": order,
            "stage": stage,
        }
        if (identity["target_id"], method, order) in by_key:
            return
        async with work:
            state, instructions, criteria, buy_label = question(bundle["bundle"], method, order)
            p, meta = await client.evaluate_choice(state, instructions, criteria, identity)
            row = {
                **identity,
                "probability_buy": p[buy_label] if p is not None else None,
                "metadata": meta,
            }
            with destination.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            completed += 1
            if completed % 40 == 0 or smoke:
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "completed": completed,
                            "expected": metadata["expected_responses"],
                        }
                    ),
                    flush=True,
                )

    tasks = []
    try:
        await transport.verify_model()
        tasks = [
            asyncio.create_task(one(b, m, o))
            for b in bundles
            for m in methods
            for o in ("buy_first", "sell_first")
        ]
        await asyncio.gather(*tasks)
        final = rows(destination)
        valid = sum(r["probability_buy"] is not None for r in final)
        metadata.update(
            status="complete" if valid == metadata["expected_responses"] else "invalid_matrix",
            valid=valid,
        )
        write_json(out / "metadata.json", metadata)
        if metadata["status"] != "complete":
            raise ValueError("incomplete probability coverage; no primary scoring")
    except BaseException as exc:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        metadata.update(status="failed", error_type=type(exc).__name__)
        write_json(out / "metadata.json", metadata)
        raise
    finally:
        await transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["freeze", "prepare", "smoke", "run"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(ROOT)
    elif args.stage == "prepare":
        prepare(ROOT)
    else:
        asyncio.run(run(ROOT, args.stage == "smoke"))


if __name__ == "__main__":
    main()
