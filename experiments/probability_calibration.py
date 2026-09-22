"""Frozen, resumable human-choice probability experiment on reused development."""

import asyncio
import copy
import fcntl
import json
import math
import os
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from behavioral_market.benchmarks.choices13k import frame_items
from behavioral_market.benchmarks.evaluation import bernoulli_js
from behavioral_market.benchmarks.runner import config_hash, make_client
from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation.event_runner import git_state
from experiments import calibration_v2 as parent
from experiments.probability_elicitation import LLMProbabilityClient, validate_probabilities

METHODS = ("P0", "P_true", "P_neutral", "P_shuffled")
PLAN = "config/frozen/probability_calibration_20260922.json"
OWN_FILES = [
    "experiments/probability_calibration.py",
    "experiments/probability_elicitation.py",
    "scripts/run_probability_calibration.py",
    "docs/probability_calibration_protocol.md",
]
CRITERIA = {"A": "Choose the displayed Option A.", "B": "Choose the displayed Option B."}


def forecast_inputs(target, demos, method, orientation, pool, permutation, example_seed):
    if method not in METHODS:
        raise ValueError("unknown probability method")
    examples = []
    if method != "P0":
        for item in demos:
            order = (
                "A_first"
                if int(parent.stable(example_seed, item.item_id, "orientation"), 16) % 2
                else "B_first"
            )
            gamble, mapping = parent.display_item(item, order)
            rate = item.human_b_rate
            if method == "P_shuffled":
                rate = pool[permutation[str(item.item_id)]].human_b_rate
            elif method == "P_neutral":
                rate = 0.5
            n_b = math.floor(rate * 10 + 0.5)
            examples.append(
                {
                    "decision_problem": gamble,
                    "rounded_example_counts": {
                        displayed: n_b if original == "B" else 10 - n_b
                        for displayed, original in mapping.items()
                    },
                }
            )
    gamble, mapping = parent.display_item(target, orientation)
    state = {"earlier_examples": examples, "current_decision_problem": gamble}
    instructions = (
        "Predict the choice probabilities of a randomly selected adult participant in a "
        "behavioral decision experiment, choosing between the current options without outcome "
        "feedback. Use only the displayed information. Earlier example counts, when supplied, "
        "are aggregate examples rounded to a ten-choice scale; they are not individual histories. "
        "Estimate probabilities for the participant's choice of displayed A and displayed B."
    )
    return state, instructions, dict(CRITERIA), mapping


def source_hash(root):
    return parent.stable(
        parent.experiment_source_hash(root),
        *[(p, digest((root / p).read_bytes())) for p in OWN_FILES],
    )


def parent_inputs(config, root):
    base = json.loads((root / config["parent_config"]).read_text())
    plan, frames = parent.load_plan(base, root)
    return base, plan, frames


def prepare(config, root):
    _, old, _ = parent_inputs(config, root)
    plan = {
        "schema": "probability-calibration-1",
        "config_hash": config_hash(config),
        "source_hash": source_hash(root),
        "parent_plan_hash": digest((root / parent.PLAN_PATH).read_bytes()),
        "target_ids": old["target_ids"],
        "methods": list(METHODS),
        "calls": len(old["target_ids"]) * len(METHODS) * 2,
        "population": "reused development, not an independent test",
    }
    path = root / PLAN
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("frozen probability plan differs; use a separate experiment")
    write_json(path, plan)
    return {"calls": plan["calls"], "targets": len(plan["target_ids"]), "smoke_calls": 16}


def load_plan(config, root, live=False):
    plan = json.loads((root / PLAN).read_text())
    base, old, frames = parent_inputs(config, root)
    if (
        plan["config_hash"] != config_hash(config)
        or plan["source_hash"] != source_hash(root)
        or plan["parent_plan_hash"] != digest((root / parent.PLAN_PATH).read_bytes())
        or plan["target_ids"] != old["target_ids"]
        or plan["methods"] != list(METHODS)
    ):
        raise RuntimeError("probability frozen inputs mismatch")
    if live:
        _, dirty = git_state(root)
        if dirty:
            raise RuntimeError("live probability stage requires a clean committed worktree")
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", PLAN], cwd=root, check=True, capture_output=True
        )
    return plan, base, old, frames


def jobs_for(plan, seed, smoke=False):
    ids = plan["target_ids"][:2] if smoke else plan["target_ids"]
    jobs = [
        (method, item_id, orientation)
        for method in METHODS
        for item_id in ids
        for orientation in ("A_first", "B_first")
    ]
    return sorted(jobs, key=lambda x: parent.stable(seed, "probability-dispatch", x))


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_matrix(records, jobs, plan_hash, stage, complete=False):
    expected, actual = set(jobs), set()
    for row in records:
        key = (row["method"], row["item_id"], row["orientation"])
        if (
            key not in expected
            or key in actual
            or row["plan_hash"] != plan_hash
            or row["stage"] != stage
        ):
            raise RuntimeError("probability response identities mismatch")
        if row["status"] == "ok":
            probabilities, _ = validate_probabilities(
                json.dumps({"probabilities": row["probabilities"]}), CRITERIA
            )
            shown_b = "B" if row["orientation"] == "A_first" else "A"
            if (
                not math.isfinite(row["original_b_probability"])
                or abs(row["original_b_probability"] - probabilities[shown_b]) > 1e-10
            ):
                raise RuntimeError("probability orientation mapping mismatch")
        elif row["original_b_probability"] is not None or row["probabilities"] is not None:
            raise RuntimeError("invalid response must not contain a fallback probability")
        actual.add(key)
    if complete and actual != expected:
        raise RuntimeError(f"incomplete probability matrix {len(actual)}/{len(expected)}")
    return actual


class MockTransport:
    async def completion(self, messages, identity):
        # Deterministic, target-label-free, orientation-consistent engineering fixture.
        p_b = (int(parent.stable(identity["item_id"], identity["method"])[:8], 16) % 81 + 10) / 100
        p_display_b = p_b if identity["orientation"] == "A_first" else 1 - p_b
        raw = json.dumps({"probabilities": {"A": 1 - p_display_b, "B": p_display_b}})
        return {
            "choices": [{"message": {"content": raw}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }, {"cache_hit": False, "mock": True}


async def run(config, root, provider, stage, cache_only=False):
    if stage not in {"smoke", "development"}:
        raise ValueError("unknown probability run stage")
    plan, base, old, frames = load_plan(config, root, live=provider != "mock")
    phash = digest((root / PLAN).read_bytes())
    output_root = root / config["experiment"]["output_dir"] / provider
    if stage == "development":
        smoke = output_root / "smoke"
        if not (smoke / "completion.json").exists():
            raise RuntimeError("probability smoke must complete before development")
        smoke_rows = read_records(smoke / "responses.jsonl")
        check_matrix(
            smoke_rows, jobs_for(plan, config["experiment"]["seed"], True), phash, "smoke", True
        )
        if any(x["status"] != "ok" for x in smoke_rows):
            raise RuntimeError("probability smoke has invalid responses")
    output = output_root / stage
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another process owns this probability stage") from None
    wrapper = None
    try:
        identity_base = {
            "plan_hash": phash,
            "provider": provider,
            "stage": stage,
            "run_id": config["experiment"]["name"],
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if any(prior.get(k) != v for k, v in identity_base.items()):
                raise RuntimeError("output path contains another probability experiment")
        else:
            commit, dirty = git_state(root)
            write_json(
                manifest_path,
                {
                    **identity_base,
                    "source_hash": plan["source_hash"],
                    "git_commit": commit,
                    "dirty": dirty,
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
        path = output / "responses.jsonl"
        jobs = jobs_for(plan, config["experiment"]["seed"], stage == "smoke")
        done = check_matrix(read_records(path), jobs, phash, stage)
        runtime = copy.deepcopy(base)
        runtime["llm"].update(config["llm_overrides"])
        runtime["llm"]["providers"]["aigc_relay"]["cache_dir"] = (
            ".cache/probability_calibration/aigc_relay"
        )
        if provider == "mock":
            client = LLMProbabilityClient(MockTransport())
        else:
            wrapper = make_client(
                runtime, provider, root, output, cache_only, config["llm_overrides"]["max_rps"]
            )
            wrapper.client.cache_namespace = phash
            await wrapper.verify_model()
            client = LLMProbabilityClient(wrapper.client)
        targets = {x.item_id: x for x in frame_items(frames["development"])}
        pool = {x.item_id: x for x in frame_items(frames["pool"]) if x.item_id in old["pool_ids"]}
        queue = asyncio.Queue()
        for job in jobs:
            if job not in done:
                queue.put_nowait(job)
        count = len(done)
        print(
            json.dumps({"pid": os.getpid(), "completed": count, "planned": len(jobs)}), flush=True
        )

        async def worker():
            nonlocal count
            while not queue.empty():
                method, item_id, orientation = queue.get_nowait()
                demo_ids = [] if method == "P0" else old["demonstrations"][str(item_id)]["matched"]
                state, instructions, criteria, mapping = forecast_inputs(
                    targets[item_id],
                    [pool[i] for i in demo_ids],
                    method,
                    orientation,
                    pool,
                    old["label_permutation"],
                    base["experiment"]["seed"],
                )
                identity = {
                    **identity_base,
                    "method": method,
                    "item_id": item_id,
                    "orientation": orientation,
                    "demonstration_ids": demo_ids,
                }
                probabilities, meta = await client.evaluate_choice(
                    state, instructions, criteria, identity
                )
                shown_b = next(k for k, v in mapping.items() if v == "B")
                record = {
                    **identity,
                    **meta,
                    "probabilities": probabilities,
                    "original_b_probability": probabilities[shown_b] if probabilities else None,
                }
                with path.open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                count += 1
                if count % 100 == 0 or count == len(jobs):
                    print(json.dumps({"completed": count, "planned": len(jobs)}), flush=True)
                queue.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(config["llm_overrides"]["max_concurrency"])
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker_task in workers:
                if not worker_task.done():
                    worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        check_matrix(read_records(path), jobs, phash, stage, True)
        write_json(
            output / "completion.json",
            {**identity_base, "responses": count, "completed_at": datetime.now(UTC).isoformat()},
        )
    finally:
        if wrapper is not None:
            await wrapper.close()
        lock.close()


def probability_metrics(probabilities, human):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(human, dtype=float)
    if not len(p) or len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError("invalid probability metric inputs")
    informative = np.ptp(p) > 1e-12 and np.ptp(y) > 1e-12
    expected = sum(
        math.comb(10, k) * p**k * (1 - p) ** (10 - k) * np.abs(k / 10 - y) for k in range(11)
    )
    return {
        "items": len(y),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mean_js": float(np.mean(bernoulli_js(p, y))),
        "pearson": float(np.corrcoef(p, y)[0, 1]) if informative else None,
        "spearman": float(spearmanr(p, y).statistic) if informative else None,
        "mae_expected_sampled_10": float(np.mean(expected)),
    }


def paired_predictions(records, ids):
    predictions, effects, valid_pairs = {}, {}, {}
    by_key = {(r["method"], r["item_id"], r["orientation"]): r for r in records}
    for method in METHODS:
        predictions[method], effects[method] = {}, {}
        for item_id in ids:
            pair = [by_key[(method, item_id, o)] for o in ("A_first", "B_first")]
            if all(r["status"] == "ok" for r in pair):
                first, second = [r["original_b_probability"] for r in pair]
                predictions[method][item_id] = (first + second) / 2
                effects[method][item_id] = second - first
        valid_pairs[method] = len(predictions[method])
    common = sorted(set.intersection(*(set(p) for p in predictions.values())))
    if not common:
        raise RuntimeError("no common valid probability pairs")
    return common, predictions, effects, valid_pairs


def evaluate(config, root, provider):
    plan, _, old, frames = load_plan(config, root)
    out = root / config["experiment"]["output_dir"] / provider / "development"
    rows = read_records(out / "responses.jsonl")
    phash = digest((root / PLAN).read_bytes())
    check_matrix(rows, jobs_for(plan, config["experiment"]["seed"]), phash, "development", True)
    ids, p_by_method, effects, valid_pairs = paired_predictions(rows, plan["target_ids"])
    items = frames["development"].set_index("item_id").loc[ids]
    human = items.human_b_rate.to_numpy()
    predictions = {m: np.array([p_by_method[m][i] for i in ids]) for m in METHODS}
    metrics = {}
    for method, values in predictions.items():
        effect = np.array([effects[method][i] for i in ids])
        records = [r for r in rows if r["method"] == method]
        metrics[method] = {
            **probability_metrics(values, human),
            "valid_item_pairs": valid_pairs[method],
            "invalid_request_rate": sum(r["status"] != "ok" for r in records) / len(records),
            "b_probability_when_first_minus_when_second": float(effect.mean()),
            "mean_absolute_order_effect": float(np.abs(effect).mean()),
            "displayed_first_probability": float((0.5 + effect / 2).mean()),
        }
    pool = frames["pool"].set_index("item_id").loc[old["pool_ids"]]
    controls = {
        "constant_half": np.full(len(ids), 0.5),
        "pool_mean": np.full(len(ids), pool.human_b_rate.mean()),
        "pool_median": np.full(len(ids), pool.human_b_rate.median()),
        "retrieval_mean": np.array(
            [pool.loc[old["demonstrations"][str(i)]["matched"]].human_b_rate.mean() for i in ids]
        ),
    }
    contrasts = {
        f"P_true_minus_{other}": parent.paired_interval(
            np.abs(predictions["P_true"] - human) - np.abs(predictions[other] - human),
            config["experiment"]["seed"] + 1,
            config["evaluation"]["bootstrap_samples"],
        )
        for other in ("P0", "P_neutral", "P_shuffled")
    }
    shrinkage = []
    for method, probabilities in predictions.items():
        for alpha in config["evaluation"]["shrinkage_alphas"]:
            shrinkage.append(
                {
                    "method": method,
                    "alpha": alpha,
                    **probability_metrics(0.5 + alpha * (probabilities - 0.5), human),
                }
            )
    chosen = min(shrinkage, key=lambda x: (x["mae"], x["alpha"], METHODS.index(x["method"])))
    gate = len(ids) / len(plan["target_ids"]) >= config["evaluation"][
        "minimum_common_coverage"
    ] and all(
        v["invalid_request_rate"] <= config["evaluation"]["invalid_rate_gate"]
        for v in metrics.values()
    )
    subgroups = []
    for field in ("ambiguity", "has_loss"):
        for value in (False, True):
            mask = items[field].astype(bool).to_numpy() == value
            if not mask.any():
                continue
            for method, probabilities in {**predictions, **controls}.items():
                subgroups.append(
                    {
                        "field": field,
                        "value": value,
                        "method": method,
                        **probability_metrics(probabilities[mask], human[mask]),
                    }
                )
    preferences = {}
    for label, column in (("higher_ev", "ev"), ("riskier", "var")):
        a, b = items[f"{column}_a"].to_numpy(), items[f"{column}_b"].to_numpy()
        mask = ~items.ambiguity.astype(bool).to_numpy() & ~np.isclose(a, b)
        preferences[label] = {"items": int(mask.sum())}
        for method, probabilities in {**predictions, **controls, "human": human}.items():
            rate = np.where(b > a, probabilities, 1 - probabilities)[mask]
            preferences[label][method] = float(rate.mean()) if len(rate) else None
    calls = [c for r in rows for c in r["calls"]]
    report = {
        "population": "reused development; exploratory",
        "provider": provider,
        "plan_hash": phash,
        "source_hash": plan["source_hash"],
        "requests": len(rows),
        "common_items": len(ids),
        "omitted_item_ids": sorted(set(plan["target_ids"]) - set(ids)),
        "metrics": metrics,
        "controls": {k: probability_metrics(v, human) for k, v in controls.items()},
        "contrasts": contrasts,
        "shrinkage": shrinkage,
        "invalid_gate_passed": gate,
        "correct_label_mechanism_supported": gate
        and all(contrasts[f"P_true_minus_{m}"]["ci95"][1] < 0 for m in ("P_neutral", "P_shuffled")),
        "selected_external_candidate": (
            {
                "method": chosen["method"] if chosen["alpha"] else None,
                "alpha": chosen["alpha"],
                "mae": chosen["mae"],
                "is_constant": chosen["alpha"] == 0,
            }
            if gate
            else None
        ),
        "preferences_observable_only": preferences,
        "usage_successful_calls_including_repairs": {
            "calls": len(calls),
            "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
            "input_tokens": sum(c["usage"].get("prompt_tokens", 0) or 0 for c in calls),
            "output_tokens": sum(c["usage"].get("completion_tokens", 0) or 0 for c in calls),
        },
    }
    destination = out / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "summary.json", report)
    pd.DataFrame(subgroups).to_csv(destination / "subgroups.csv", index=False)
    pd.DataFrame({"item_id": ids, "human_b_rate": human, **predictions, **controls}).to_csv(
        destination / "predictions.csv", index=False
    )
    return report
