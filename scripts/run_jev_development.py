#!/usr/bin/env python3
"""Frozen Jev comparison on the existing, already-observed development set."""

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from behavioral_market.benchmarks.choices13k import frame_items  # noqa: E402
from behavioral_market.data.archive import digest, write_json  # noqa: E402
from experiments import probability_calibration as parent  # noqa: E402
from experiments.jev_client import JevClient  # noqa: E402
from experiments.probability_analysis import mean_bernoulli_js  # noqa: E402
from experiments.risk_statistical_baselines import predict  # noqa: E402

PLAN = ROOT / "config/frozen/jev_development_20260923.json"
OUT = ROOT / "outputs/jev/development_20260923"
LLM = (
    ROOT / "outputs/human_benchmark/probability_calibration/aigc_relay/development/responses.jsonl"
)
MODEL = ROOT / "outputs/risk_statistical_baselines_20260922/model.json"
FILES = [
    "scripts/run_jev_development.py",
    "experiments/jev_client.py",
    "experiments/probability_calibration.py",
    "experiments/probability_analysis.py",
    "experiments/risk_statistical_baselines.py",
    "docs/jev_development_protocol_20260923.md",
]


def inputs():
    config = json.loads((ROOT / "config/probability_calibration.json").read_text())
    plan, base, old, frames = parent.load_plan(config, ROOT)
    targets = {x.item_id: x for x in frame_items(frames["development"])}
    pool = {x.item_id: x for x in frame_items(frames["pool"]) if x.item_id in old["pool_ids"]}
    jobs = parent.jobs_for(plan, config["experiment"]["seed"])
    payloads = []
    for method, item_id, orientation in jobs:
        demos = [] if method == "P0" else old["demonstrations"][str(item_id)]["matched"]
        state, instruction, criteria, mapping = parent.forecast_inputs(
            targets[item_id],
            [pool[i] for i in demos],
            method,
            orientation,
            pool,
            old["label_permutation"],
            base["experiment"]["seed"],
        )
        payloads.append(
            {
                "method": method,
                "item_id": item_id,
                "orientation": orientation,
                "state": state,
                "instructions": instruction,
                "criteria": criteria,
                "mapping": mapping,
            }
        )
    return plan, targets, payloads


def freeze():
    parent_plan, _, payloads = inputs()
    return {
        "scope": "exploratory reused development; no independent test or financial claim",
        "model": "jev-1.13.0",
        "max_rps": 2.0,
        "concurrency": 4,
        "target_ids": parent_plan["target_ids"],
        "calls": len(payloads),
        "parent_plan_hash": digest((ROOT / parent.PLAN).read_bytes()),
        "payloads_hash": digest(json.dumps(payloads, sort_keys=True).encode()),
        "source_hashes": {p: digest((ROOT / p).read_bytes()) for p in FILES},
        "llm_response_hash": digest(LLM.read_bytes()),
        "statistical_model_hash": digest(MODEL.read_bytes()),
        "bootstrap_seed": 20260923,
        "bootstrap_samples": 5000,
    }


def checked_plan(live=False):
    plan = json.loads(PLAN.read_text())
    if plan != freeze():
        raise RuntimeError("frozen Jev inputs or code changed")
    if live:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
            raise RuntimeError("commit protocol/code/plan before calls")
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{PLAN.relative_to(ROOT)}"], cwd=ROOT
        )
        if committed != PLAN.read_bytes():
            raise RuntimeError("plan not committed")
    return plan


def records(stage, phash, complete=False):
    plan, _, payloads = inputs()
    expected = {
        (x["method"], x["item_id"], x["orientation"])
        for x in payloads
        if stage != "smoke" or x["item_id"] in plan["target_ids"][:2]
    }
    rows = parent.read_records(OUT / stage / "responses.jsonl")
    parent.check_matrix(rows, list(expected), phash, stage, complete)
    return rows


async def run(stage):
    plan = checked_plan(live=True)
    phash = digest(PLAN.read_bytes())
    if stage == "development":
        if not (OUT / "smoke/completion.json").exists():
            raise RuntimeError("smoke required")
        records("smoke", phash, True)
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = records(stage, phash)
        done = {(r["method"], r["item_id"], r["orientation"]) for r in rows}
        _, _, payloads = inputs()
        jobs = [p for p in payloads if stage != "smoke" or p["item_id"] in plan["target_ids"][:2]]
        queue = asyncio.Queue()
        for p in jobs:
            if (p["method"], p["item_id"], p["orientation"]) not in done:
                queue.put_nowait(p)
        load_dotenv(ROOT / ".env")
        client = JevClient(
            os.getenv("TYPESAFE_API_KEY") or os.getenv("TYPESAFE_AI_API_KEY"),
            out,
            ROOT / ".cache/jev/development_20260923",
            max_rps=plan["max_rps"],
            trust_env=True,
        )
        manifest = out / "manifest.json"
        if not manifest.exists():
            write_json(
                manifest,
                {
                    "plan_hash": phash,
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "started_at": datetime.now(UTC).isoformat(),
                    "environment_proxy_enabled": True,
                },
            )
        elif json.loads(manifest.read_text())["plan_hash"] != phash:
            raise RuntimeError("manifest differs")
        count = len(done)

        async def worker():
            nonlocal count
            while not queue.empty():
                p = queue.get_nowait()
                identity = {k: p[k] for k in ("method", "item_id", "orientation")}
                identity.update(plan_hash=phash, stage=stage)
                answer = await client.evaluate_choice(
                    p["state"], p["instructions"], p["criteria"], identity
                )
                shown_b = next(k for k, v in p["mapping"].items() if v == "B")
                row = {
                    **identity,
                    **answer,
                    "status": "ok",
                    "original_b_probability": answer["probabilities"][shown_b],
                }
                with (out / "responses.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                if count % 100 == 0 or count == len(jobs):
                    print(
                        json.dumps({"stage": stage, "completed": count, "planned": len(jobs)}),
                        flush=True,
                    )

        tasks = [asyncio.create_task(worker()) for _ in range(plan["concurrency"])]
        try:
            await asyncio.gather(*tasks)
            records(stage, phash, True)
            checked_plan()
            write_json(
                out / "completion.json",
                {
                    "plan_hash": phash,
                    "responses": count,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()


def evaluate():
    plan = checked_plan()
    phash = digest(PLAN.read_bytes())
    rows = records("development", phash, True)
    _, targets, _ = inputs()
    ids = sorted(plan["target_ids"])
    y = np.array([targets[i].human_b_rate for i in ids])
    llm = parent.read_records(LLM)
    parent.check_matrix(
        llm,
        parent.jobs_for({"target_ids": ids}, 20260923),
        plan["parent_plan_hash"],
        "development",
        True,
    )
    predictions = {}
    effects = {}
    for prefix, data in [("Jev", rows), ("LLM", llm)]:
        _, ps, es, _ = parent.paired_predictions(data, ids)
        for method in parent.METHODS:
            predictions[f"{prefix}_{method}"] = np.array([ps[method][i] for i in ids])
            effects[f"{prefix}_{method}"] = float(np.mean(np.abs([es[method][i] for i in ids])))
    predictions["constant_half"] = np.full(len(ids), 0.5)

    def metrics(p, human):
        return {**parent.probability_metrics(p, human), "mean_js": mean_bernoulli_js(p, human)}

    rng = np.random.default_rng(plan["bootstrap_seed"])
    samples = rng.integers(0, len(ids), (plan["bootstrap_samples"], len(ids)))
    contrasts = {}
    pairs = [(f"Jev_{m}", f"LLM_{m}") for m in parent.METHODS]
    pairs += [("Jev_P_true", f"Jev_{m}") for m in ("P0", "P_neutral", "P_shuffled")]
    for a, b in pairs:
        delta = np.abs(predictions[a] - y) - np.abs(predictions[b] - y)
        contrasts[f"{a}_minus_{b}"] = {
            "delta_mae": float(delta.mean()),
            "ci95_item_bootstrap": np.quantile(
                delta[samples].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    risk_ids = [i for i in ids if not targets[i].ambiguity]
    mask = np.array([not targets[i].ambiguity for i in ids])
    stats = predict(json.loads(MODEL.read_text()), [targets[i] for i in risk_ids])
    risk_metrics = {k: metrics(p[mask], y[mask]) for k, p in predictions.items()}
    risk_metrics.update({k: metrics(p, y[mask]) for k, p in stats.items()})
    usage = {
        "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows),
        "responses_missing_usage": sum(not r.get("usage") for r in rows),
        "cache_hits": sum(r["cache_hit"] for r in rows),
    }
    times = [r["latency_seconds"] for r in rows if not r["cache_hit"]]
    report = {
        "scope": plan["scope"],
        "plan_hash": phash,
        "items": len(ids),
        "responses": len(rows),
        "metrics": {k: metrics(p, y) for k, p in predictions.items()},
        "contrasts": contrasts,
        "risk_only_items": len(risk_ids),
        "risk_only_metrics": risk_metrics,
        "mean_absolute_orientation_effect": effects,
        "usage_main_successful_responses": usage,
        "latency_seconds": {
            "median": float(np.median(times)),
            "p95": float(np.quantile(times, 0.95)),
        },
        "response_hash": digest((OUT / "development/responses.jsonl").read_bytes()),
        "limitations": (
            "Reused development, no participant bootstrap; historical LLM latency is not "
            "a controlled speed trial. No financial or market inference."
        ),
    }
    write_json(OUT / "development/evaluation/summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "smoke", "development", "evaluate"]
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        proposed = freeze()
        if PLAN.exists() and json.loads(PLAN.read_text()) != proposed:
            raise RuntimeError("existing frozen plan differs")
        write_json(PLAN, proposed)
        print(json.dumps({"calls": proposed["calls"], "plan": str(PLAN.relative_to(ROOT))}))
    elif args.stage == "evaluate":
        evaluate()
    else:
        asyncio.run(run(args.stage))
