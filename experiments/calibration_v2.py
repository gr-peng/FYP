"""Visible-information retrieval and a frozen, resumable development ablation.

This experiment deliberately reuses development data. It never reads the previous
final-test data, never rewrites a previous result, and makes no new held-out claim.
"""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from behavioral_market.benchmarks.choices13k import frame_items
from behavioral_market.benchmarks.evaluation import evaluate_stage, load_responses
from behavioral_market.benchmarks.prompts import build_prompt, format_option
from behavioral_market.benchmarks.retrieval import OUTPUT_TEMPLATE, SYSTEM_TEMPLATE
from behavioral_market.benchmarks.runner import config_hash, llm_config, make_client
from behavioral_market.benchmarks.sampling import orientations, proportional_stratified_sample
from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation.event_runner import git_state, source_hash

METHODS = ("B0", "C_true", "C_shuffled", "C_neutral", "C_random")
PLAN_PATH = "config/frozen/choices13k_calibration_v2.json"


def stable(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def visible_option(outcomes, hidden=False):
    # Never filter ambiguous outcomes by their hidden probability, even if zero.
    if hidden:
        return {"hidden": True, "payoffs": sorted({float(x) for _, x in outcomes})}
    merged = defaultdict(float)
    for probability, payoff in outcomes:
        merged[float(payoff)] += float(probability)
    return {
        "hidden": False,
        "outcomes": [[round(p, 10), x] for x, p in sorted(merged.items()) if p > 0],
    }


def visible_fingerprint(item):
    options = [visible_option(item.option_a), visible_option(item.option_b, item.ambiguity)]
    # Neither row IDs, Corr (not displayed), labels, nor hidden B probabilities enter.
    return stable(sorted(options, key=lambda value: json.dumps(value, sort_keys=True)))


def observable_features(item):
    a = visible_option(item.option_a)
    b = visible_option(item.option_b, item.ambiguity)
    values = [x for _, x in a["outcomes"]]
    values += b["payoffs"] if item.ambiguity else [x for _, x in b["outcomes"]]
    scale = max(abs(x) for x in values)
    probabilities = [p for p, _ in a["outcomes"]]
    if item.ambiguity:
        gap = None
    else:
        probabilities += [p for p, _ in b["outcomes"]]
        ev_a = sum(p * x for p, x in a["outcomes"])
        ev_b = sum(p * x for p, x in b["outcomes"])
        gap = abs(ev_a - ev_b)
    return {
        "ambiguity": bool(item.ambiguity),
        "has_loss": any(x < 0 for x in values),
        "ev_gap": gap,
        "payoff_scale": scale,
        "outcome_count": len(values),
        "probabilities": sorted(probabilities),
    }


def fit_thresholds(pool):
    gaps = [observable_features(item)["ev_gap"] for item in pool if not item.ambiguity]
    if not gaps:
        raise ValueError("pool needs at least one non-ambiguous item")
    return np.quantile(gaps, [1 / 3, 2 / 3]).tolist()


def retrieve(pool, target, thresholds, seed, strategy="matched", k=4):
    target_key = visible_fingerprint(target)
    available = [item for item in pool if visible_fingerprint(item) != target_key]
    if len(available) < k:
        raise ValueError("insufficient nonoverlapping demonstrations")
    target_features = observable_features(target)

    def gap_bin(features):
        gap = features["ev_gap"]
        return None if gap is None else int(np.searchsorted(thresholds, gap, side="right"))

    def score(item):
        tie = stable(seed, target_key, visible_fingerprint(item), strategy)
        if strategy == "random":
            return (tie,)
        if strategy != "matched":
            raise ValueError("unknown retrieval strategy")
        features = observable_features(item)
        # Ambiguity is matched first. Hidden EV is never used for either operand.
        return (
            features["ambiguity"] != target_features["ambiguity"],
            features["has_loss"] != target_features["has_loss"],
            gap_bin(features) != gap_bin(target_features),
            abs(features["outcome_count"] - target_features["outcome_count"]),
            features["probabilities"] != target_features["probabilities"],
            abs(math.log1p(features["payoff_scale"]) - math.log1p(target_features["payoff_scale"])),
            tie,
        )

    return sorted(available, key=score)[:k]


def label_permutation(pool, seed):
    # A seeded cyclic derangement preserves the pool's empirical label multiset.
    ordered = sorted(pool, key=lambda item: stable(seed, visible_fingerprint(item), "labels"))
    if len(ordered) < 2:
        raise ValueError("label permutation needs at least two items")
    offset = 1 + int(stable(seed, "offset"), 16) % (len(ordered) - 1)
    return {
        str(item.item_id): ordered[(i + offset) % len(ordered)].item_id
        for i, item in enumerate(ordered)
    }


def display_item(item, orientation):
    entries = [("A", item.option_a, True), ("B", item.option_b, not item.ambiguity)]
    if orientation == "B_first":
        entries.reverse()
    elif orientation != "A_first":
        raise ValueError("invalid orientation")
    blocks, mapping = [], {}
    for shown, (original, outcomes, visible) in zip(("A", "B"), entries, strict=True):
        mapping[shown] = original
        blocks.append(f"Option {shown}:\n" + "\n".join(format_option(outcomes, visible)))
    return "\n\n".join(blocks), mapping


def build_prompt_v2(target, demos, method, orientation, pool_by_id, permutation, seed):
    if method not in METHODS:
        raise ValueError("unknown method")
    if method == "B0":
        return build_prompt(target, "Choose the option you prefer.", orientation)
    examples = []
    for index, item in enumerate(demos, 1):
        order = "A_first" if int(stable(seed, item.item_id, "orientation"), 16) % 2 else "B_first"
        gamble, mapping = display_item(item, order)
        inverse = {original: shown for shown, original in mapping.items()}
        rate = item.human_b_rate
        if method == "C_shuffled":
            rate = pool_by_id[permutation[str(item.item_id)]].human_b_rate
        elif method == "C_neutral":
            rate = 0.5
        n_b = math.floor(rate * 10 + 0.5)
        choices = ["B"] * n_b + ["A"] * (10 - n_b)
        np.random.default_rng(int(stable(seed, item.item_id, "records")[:16], 16)).shuffle(choices)
        records = "\n".join(
            f"Example choice record {j}: {inverse[choice]}" for j, choice in enumerate(choices, 1)
        )
        examples.append(f"Earlier problem {index}:\n{gamble}\n\n{records}")
    gamble, mapping = display_item(target, orientation)
    user = (
        "Review these example choice records from earlier decision problems. "
        "Use them as behavioral examples, then make your own choice for the final problem.\n\n"
        + "\n\n".join(examples)
        + f"\n\nNow make your own choice for this new problem.\n{gamble}\n\n{OUTPUT_TEMPLATE}"
    )
    return SYSTEM_TEMPLATE, user, mapping


def experiment_source_hash(root):
    paths = ["experiments/calibration_v2.py", "scripts/run_calibration_v2.py"]
    return stable(source_hash(), *[(p, digest((root / p).read_bytes())) for p in paths])


def read_sources(config, root):
    frames, hashes = {}, {}
    for name, spec in config["sources"].items():
        path = root / spec["path"]
        actual = digest(path.read_bytes())
        if actual != spec["sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
        hashes[name] = actual
        frames[name] = pd.read_parquet(path)
    return frames, hashes


def prepare(config, root):
    frames, hashes = read_sources(config, root)
    pool_all = frame_items(frames["pool"])
    unique_pool = {}
    for item in sorted(pool_all, key=lambda item: item.item_id):
        unique_pool.setdefault(visible_fingerprint(item), item)
    pool = list(unique_pool.values())
    eligible = []
    seen = set(unique_pool)
    for item in sorted(frame_items(frames["development"]), key=lambda item: item.item_id):
        key = visible_fingerprint(item)
        if key not in seen:
            eligible.append(item.item_id)
            seen.add(key)
    frame = frames["development"][frames["development"].item_id.isin(eligible)].copy()
    features = {item.item_id: observable_features(item) for item in frame_items(frame)}
    frame["stratum"] = frame.item_id.map(
        lambda i: f"loss={int(features[i]['has_loss'])}|amb={int(features[i]['ambiguity'])}"
    )
    selected = proportional_stratified_sample(frame, config["items"], config["experiment"]["seed"])
    thresholds = fit_thresholds(pool)
    demos = {}
    for target in frame_items(selected):
        demos[str(target.item_id)] = {
            strategy: [
                item.item_id
                for item in retrieve(
                    pool, target, thresholds, config["experiment"]["seed"], strategy
                )
            ]
            for strategy in ("matched", "random")
        }
    plan = {
        "schema": "calibration-v2-development-1",
        "config_hash": config_hash(config),
        "source_hash": experiment_source_hash(root),
        "input_hashes": hashes,
        "pool_ids": [item.item_id for item in pool],
        "target_ids": selected.item_id.astype(int).tolist(),
        "ev_thresholds_pool_only": thresholds,
        "demonstrations": demos,
        "label_permutation": label_permutation(pool, config["experiment"]["seed"]),
        "methods": list(METHODS),
        "calls": config["items"] * len(METHODS) * config["sampling"]["repeats_per_item"],
        "population": "reused_development_not_untouched",
    }
    path = root / PLAN_PATH
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("existing frozen plan differs; create a separately named experiment")
    write_json(path, plan)
    return plan


def load_plan(config, root, live=False):
    plan = json.loads((root / PLAN_PATH).read_text())
    frames, hashes = read_sources(config, root)
    if (
        plan["config_hash"] != config_hash(config)
        or plan["source_hash"] != experiment_source_hash(root)
        or plan["input_hashes"] != hashes
    ):
        raise RuntimeError("frozen plan/config/source mismatch")
    if live:
        _, dirty = git_state(root)
        if dirty:
            raise RuntimeError("live stage requires a clean committed worktree")
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", PLAN_PATH],
            cwd=root,
            check=True,
            capture_output=True,
        )
    return plan, frames


def jobs_for(plan, repeats, seed, smoke=False):
    targets = plan["target_ids"][:2] if smoke else plan["target_ids"]
    jobs = [
        (method, item, repeat, orientation)
        for item in targets
        for repeat, orientation in enumerate(orientations(item, repeats, seed))
        for method in METHODS
    ]
    return sorted(jobs, key=lambda job: stable(seed, "dispatch", job))


def check_matrix(frame, jobs, complete=False):
    expected = {(m, i, r): o for m, i, r, o in jobs}
    actual = {}
    for row in frame.itertuples():
        key = (row.prompt_id, int(row.item_id), int(row.repeat_id))
        if key in actual or key not in expected or row.display_order != expected[key]:
            raise RuntimeError("response identities or counterbalance do not match frozen plan")
        actual[key] = row.display_order
    if complete and actual != expected:
        raise RuntimeError(f"incomplete matrix: {len(actual)}/{len(expected)}")
    return set(actual)


async def run(config, root, provider, stage="development", cache_only=False):
    plan, frames = load_plan(config, root, live=provider != "mock")
    smoke = stage == "smoke"
    if stage not in {"smoke", "development"}:
        raise ValueError("invalid run stage")
    if provider != "mock" and stage == "development":
        smoke_dir = root / config["experiment"]["output_dir"] / provider / "smoke"
        if not (smoke_dir / "completion.json").exists():
            raise RuntimeError("complete the frozen smoke stage before live development")
        smoke_frame = load_responses(smoke_dir / "responses.jsonl")
        check_matrix(smoke_frame, jobs_for(plan, 2, config["experiment"]["seed"], True), True)
        if not (smoke_frame.status == "ok").all():
            raise RuntimeError("smoke stage contains invalid responses")
    repeats = 2 if smoke else config["sampling"]["repeats_per_item"]
    jobs = jobs_for(plan, repeats, config["experiment"]["seed"], smoke)
    out = root / config["experiment"]["output_dir"] / provider / stage
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another process owns this experiment stage") from None
    client = None
    try:
        frozen = {
            "plan_hash": digest((root / PLAN_PATH).read_bytes()),
            "provider": provider,
            "stage": stage,
            "source_hash": plan["source_hash"],
        }
        manifest = out / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if any(old.get(k) != v for k, v in frozen.items()):
                raise RuntimeError("run belongs to different frozen inputs")
        else:
            commit, dirty = git_state(root)
            write_json(
                manifest,
                {
                    **frozen,
                    "git_commit": commit,
                    "dirty": dirty,
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
        response_path = out / "responses.jsonl"
        old = load_responses(response_path) if response_path.exists() else pd.DataFrame()
        completed = check_matrix(old, jobs) if len(old) else set()
        targets = {item.item_id: item for item in frame_items(frames["development"])}
        pool = {item.item_id: item for item in frame_items(frames["pool"])}
        client = make_client(config, provider, root, out, cache_only, config["llm"]["max_rps"])
        await client.verify_model()
        queue = asyncio.Queue()
        for job in jobs:
            if job[:3] not in completed:
                queue.put_nowait(job)
        count = len(completed)
        print(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "completed": count,
                    "planned": len(jobs),
                    "remaining": queue.qsize(),
                }
            ),
            flush=True,
        )

        async def worker():
            nonlocal count
            while not queue.empty():
                method, item_id, repeat_id, orientation = queue.get_nowait()
                ids = (
                    []
                    if method == "B0"
                    else plan["demonstrations"][str(item_id)][
                        "random" if method == "C_random" else "matched"
                    ]
                )
                system, user, display = build_prompt_v2(
                    targets[item_id],
                    [pool[i] for i in ids],
                    method,
                    orientation,
                    pool,
                    plan["label_permutation"],
                    config["experiment"]["seed"],
                )
                identity = {
                    "run_id": config["experiment"]["name"],
                    "split": stage,
                    "prompt_id": method,
                    "item_id": item_id,
                    "repeat_id": repeat_id,
                    "display_order": orientation,
                    "demonstration_item_ids": ids,
                    "plan_hash": frozen["plan_hash"],
                    "model": "mock"
                    if provider == "mock"
                    else llm_config(config, provider)["model"],
                }
                started = time.perf_counter()
                choice, raw, meta = await client.choose(system, user, identity)
                usage = meta.get("usage", {})
                record = {
                    **identity,
                    "validated_choice": choice,
                    "raw_response": raw,
                    "original_b_choice": display.get(choice) == "B" if choice else None,
                    "first_display_selected": choice == "A" if choice else None,
                    "latency_seconds": time.perf_counter() - started,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    **meta,
                }
                with response_path.open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                count += 1
                if count % 100 == 0 or count == len(jobs):
                    print(json.dumps({"completed": count, "planned": len(jobs)}), flush=True)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(config["llm"]["max_concurrency"])]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        check_matrix(load_responses(response_path), jobs, complete=True)
        write_json(
            out / "completion.json",
            {"responses": count, "completed_at": datetime.now(UTC).isoformat(), **frozen},
        )
    finally:
        if client is not None:
            await client.close()
        lock.close()


def paired_interval(delta, seed, count=5000):
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "items": len(values),
    }


def evaluate(config, root, provider="aigc_relay"):
    plan, frames = load_plan(config, root)
    out = root / config["experiment"]["output_dir"] / provider / "development"
    responses = load_responses(out / "responses.jsonl")
    repeats = config["sampling"]["repeats_per_item"]
    check_matrix(responses, jobs_for(plan, repeats, config["experiment"]["seed"]), True)
    items = frames["development"][frames["development"].item_id.isin(plan["target_ids"])].copy()
    items = items.drop(columns=["ev_gap_bin", "stratum", "split"], errors="ignore")
    rates, metrics = evaluate_stage(responses, items, repeats, out / "evaluation", config)
    by_method = rates.pivot(index="item_id", columns="prompt_id", values="llm_b_rate")
    labels = items.set_index("item_id").human_b_rate.reindex(by_method.index)
    pool = frames["pool"].set_index("item_id").loc[plan["pool_ids"]]
    predictions = {
        "constant_half": np.full(len(labels), 0.5),
        "pool_mean": np.full(len(labels), pool.human_b_rate.mean()),
        "pool_median": np.full(len(labels), pool.human_b_rate.median()),
        "retrieval_mean": np.array(
            [
                pool.loc[plan["demonstrations"][str(i)]["matched"]].human_b_rate.mean()
                for i in by_method.index
            ]
        ),
    }
    baselines = {}
    for name, probabilities in predictions.items():
        expected = np.zeros(len(labels))
        for k in range(repeats + 1):
            mass = math.comb(repeats, k) * probabilities**k * (1 - probabilities) ** (repeats - k)
            expected += mass * np.abs(k / repeats - labels.to_numpy())
        baselines[name] = {
            "mae_probability": float(np.abs(probabilities - labels.to_numpy()).mean()),
            "mae_expected_sampled_10": float(expected.mean()),
            "rmse_probability": float(np.sqrt(np.mean((probabilities - labels.to_numpy()) ** 2))),
        }
    errors = by_method.sub(labels, axis=0).abs()
    contrasts = {
        f"C_true_minus_{other}": paired_interval(
            (errors.C_true - errors[other]).dropna(), config["experiment"]["seed"] + 1
        )
        for other in ("B0", "C_shuffled", "C_neutral", "C_random")
    }
    valid = all(
        v["invalid_rate"] <= config["evaluation"]["invalid_rate_gate"] for v in metrics.values()
    )
    eligible = ("C_true", "C_random")
    selected = min(eligible, key=lambda name: (metrics[name]["mae"], eligible.index(name)))
    summary = {
        "population": "reused development; exploratory, not independent confirmation",
        "metrics": metrics,
        "non_llm_baselines": baselines,
        "paired_item_contrasts": contrasts,
        "invalid_gate_passed": valid,
        "selected_for_external_candidate": selected if valid else None,
        "true_labels_supported_against_both_controls": valid
        and all(
            contrasts[f"C_true_minus_{other}"]["ci95"][1] < 0
            for other in ("C_shuffled", "C_neutral")
        ),
        "source_hash": plan["source_hash"],
        "plan_hash": digest((root / PLAN_PATH).read_bytes()),
        "calls": len(responses),
    }
    write_json(out / "evaluation/summary.json", summary)
    text = [
        "# Calibration V2 development results",
        "",
        "Reused development data: exploratory mechanism evidence only.",
        "",
        "| Method | MAE | RMSE | Pearson | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        text.append(
            f"| {name} | {value['mae']:.5f} | {value['rmse']:.5f} | "
            f"{value['pearson']} | {value['invalid_rate']:.3%} |"
        )
    text += ["", "```json", json.dumps(summary, indent=2), "```", ""]
    (out / "evaluation/report.md").write_text("\n".join(text))
    return summary
