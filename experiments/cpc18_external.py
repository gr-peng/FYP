"""Frozen external forecasts with gated CPC18 human-label opening."""

import asyncio
import copy
import fcntl
import json
import os
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from behavioral_market.benchmarks.choices13k import frame_items
from behavioral_market.benchmarks.runner import config_hash, make_client
from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation.event_runner import git_state
from experiments import calibration_v2 as retrieval
from experiments import probability_calibration as forecast
from experiments.cpc18_data import (
    bootstrap_contrasts,
    participant_arrays,
    read_outcomes,
    read_targets,
)
from experiments.probability_analysis import mean_bernoulli_js
from experiments.probability_elicitation import LLMProbabilityClient
from experiments.risk_statistical_baselines import features, predict

PLAN = "config/frozen/cpc18_external_20260922.json"
MODEL = "config/frozen/cpc18_statistical_model_20260922.json"
OWN_FILES = [
    "experiments/cpc18_external.py",
    "experiments/cpc18_data.py",
    "experiments/risk_statistical_baselines.py",
    "experiments/probability_analysis.py",
    "scripts/run_cpc18_external.py",
    "docs/cpc18_external_protocol.md",
]


def source_hash(root):
    return retrieval.stable(
        forecast.source_hash(root), *[(p, digest((root / p).read_bytes())) for p in OWN_FILES]
    )


def inherited(config, root):
    pc = json.loads((root / config["parent_config"]).read_text())
    probability_plan, base, prior, frames = forecast.load_plan(pc, root)
    pool = [x for x in frame_items(frames["pool"]) if x.item_id in prior["pool_ids"]]
    return pc, probability_plan, base, prior, pool


def prepare(config, root):
    pc, probability_plan, base, prior, pool = inherited(config, root)
    folder = root / config["input_folder"]
    screen = json.loads((folder / "feature_overlap_screen.json").read_text())
    ids = screen["unique_risk_only_eligible_ids"]
    if len(ids) != config["expected_items"]:
        raise ValueError("unexpected external target population")
    workbook = folder / "cpc18_210.xlsx"
    if digest(workbook.read_bytes()) != screen["source_cpc18_sha256"]:
        raise ValueError("CPC18 workbook differs from feature screen")
    full_choices = root / "data/processed/human_benchmarks/choices13k/all_items.parquet"
    if digest(full_choices.read_bytes()) != screen["source_choices13k_sha256"]:
        raise ValueError("complete Choices13k source differs from overlap screen")
    targets = read_targets(workbook, ids)
    audit = {x["cpc_game_id"]: x for x in screen["rows"]}
    for item in targets:
        entry = audit[item.item_id]
        if entry["choices13k_matching_rows"] or entry[
            "visible_fingerprint"
        ] != retrieval.visible_fingerprint(item):
            raise ValueError("external structural eligibility mismatch")
    selection = json.loads((root / config["selected_summary"]).read_text())
    if (
        selection["plan_hash"] != digest((root / forecast.PLAN).read_bytes())
        or selection["source_hash"] != probability_plan["source_hash"]
        or not selection["invalid_gate_passed"]
        or selection["selected_external_candidate"]["method"] != "P_true"
        or selection["selected_external_candidate"]["alpha"] != 1
    ):
        raise ValueError("development selection differs from external protocol")
    model = json.loads((root / config["statistical_model"]).read_text())
    result = json.loads((root / config["statistical_results"]).read_text())
    risk_pool = [x for x in pool if not x.ambiguity]
    if (
        result["source_sha256"]
        != digest((root / "experiments/risk_statistical_baselines.py").read_bytes())
        or model["train_ids"] != [x.item_id for x in risk_pool]
        or model["ridge"] != 0.01
        or model["neighbors"] != 16
    ):
        raise ValueError("statistical checkpoint does not match frozen training definition")
    if not np.allclose(model["train_y"], [x.human_b_rate for x in risk_pool], atol=1e-12, rtol=0):
        raise ValueError("statistical training labels mismatch")
    if not np.allclose(
        model["train_z"],
        np.array([features(x) for x in risk_pool]) / model["scale"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("statistical training features mismatch")
    frozen_model = root / MODEL
    if frozen_model.exists() and json.loads(frozen_model.read_text()) != model:
        raise ValueError("refusing to replace frozen statistical checkpoint")
    write_json(frozen_model, model)
    controls = predict(model, targets)
    rates = np.array([x.human_b_rate for x in pool])
    controls.update(
        constant_half=np.full(len(ids), 0.5),
        pool_mean=np.full(len(ids), rates.mean()),
        pool_median=np.full(len(ids), np.median(rates)),
    )
    input_paths = [
        str(workbook.relative_to(root)),
        str((folder / "all_raw.csv").relative_to(root)),
        str((folder / "feature_overlap_screen.json").relative_to(root)),
        str((folder / "raw_metadata_screen.json").relative_to(root)),
        str((folder / "raw_manifest.json").relative_to(root)),
        str(full_choices.relative_to(root)),
        config["selected_summary"],
        config["statistical_model"],
        config["statistical_results"],
        MODEL,
        config["parent_config"],
        forecast.PLAN,
        retrieval.PLAN_PATH,
    ]
    raw_manifest = json.loads((folder / "raw_manifest.json").read_text())
    raw_hash = next(x["sha256"] for x in raw_manifest["files"] if x["path"].endswith("all_raw.csv"))
    if digest((folder / "all_raw.csv").read_bytes()) != raw_hash:
        raise ValueError("raw archive checksum mismatch")
    plan = {
        "schema": "cpc18-external-1",
        "config_hash": config_hash(config),
        "source_hash": source_hash(root),
        "input_hashes": {p: digest((root / p).read_bytes()) for p in input_paths},
        "target_ids": [x.item_id for x in targets],
        "methods": list(forecast.METHODS),
        "calls": len(ids) * len(forecast.METHODS) * 2,
        "selected_method": "P_true",
        "selected_alpha": 1.0,
        "statistical_model": MODEL,
        "training_items": len(risk_pool),
        "example_seed": base["experiment"]["seed"],
        "demonstrations": {
            str(x.item_id): [
                d.item_id
                for d in retrieval.retrieve(
                    pool, x, prior["ev_thresholds_pool_only"], base["experiment"]["seed"]
                )
            ]
            for x in targets
        },
        "label_permutation": prior["label_permutation"],
        "llm_overrides": pc["llm_overrides"],
        "control_predictions": {k: v.tolist() for k, v in controls.items()},
        "external_human_labels_read": False,
    }
    path = root / PLAN
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("external plan differs; never overwrite a frozen study")
    write_json(path, plan)
    return {
        "external_items": len(ids),
        "main_forecasts": plan["calls"],
        "smoke_forecasts": 16,
        "statistical_training_items": len(risk_pool),
        "human_labels_read": False,
    }


def load_plan(config, root, clean=False):
    plan = json.loads((root / PLAN).read_text())
    if plan["config_hash"] != config_hash(config) or plan["source_hash"] != source_hash(root):
        raise RuntimeError("external config/source freeze mismatch")
    for path, checksum in plan["input_hashes"].items():
        if digest((root / path).read_bytes()) != checksum:
            raise RuntimeError(f"external input mismatch: {path}")
    if clean:
        if git_state(root)[1]:
            raise RuntimeError("external live/evaluation stage requires committed clean worktree")
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", PLAN, MODEL],
            cwd=root,
            check=True,
            capture_output=True,
        )
    return plan


async def run(config, root, provider, stage, cache_only=False):
    if stage not in {"smoke", "test"}:
        raise ValueError("unknown external stage")
    plan = load_plan(config, root, clean=provider != "mock")
    _, _, base, _, pool_list = inherited(config, root)
    phash = digest((root / PLAN).read_bytes())
    output_root = root / config["experiment"]["output_dir"] / provider
    if stage == "test":
        smoke = output_root / "smoke"
        if not (smoke / "completion.json").exists():
            raise RuntimeError("external smoke has not completed")
        smoke_rows = forecast.read_records(smoke / "responses.jsonl")
        forecast.check_matrix(
            smoke_rows,
            forecast.jobs_for(plan, config["experiment"]["seed"], True),
            phash,
            "smoke",
            True,
        )
        if any(x["status"] != "ok" for x in smoke_rows):
            raise RuntimeError("invalid external smoke forecast")
    output = output_root / stage
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another process owns this external stage") from None
    wrapper = None
    try:
        fixed = {
            "run_id": config["experiment"]["name"],
            "provider": provider,
            "stage": stage,
            "plan_hash": phash,
        }
        manifest = output / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if any(old.get(k) != v for k, v in fixed.items()):
                raise RuntimeError("external output identity mismatch")
        else:
            write_json(
                manifest,
                {
                    **fixed,
                    "source_hash": plan["source_hash"],
                    "git_commit": git_state(root)[0],
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
        path = output / "responses.jsonl"
        jobs = forecast.jobs_for(plan, config["experiment"]["seed"], stage == "smoke")
        done = forecast.check_matrix(forecast.read_records(path), jobs, phash, stage)
        targets = {
            x.item_id: x
            for x in read_targets(
                root / config["input_folder"] / "cpc18_210.xlsx", plan["target_ids"]
            )
        }
        pool = {x.item_id: x for x in pool_list}
        if provider == "mock":
            client = LLMProbabilityClient(forecast.MockTransport())
        else:
            runtime = copy.deepcopy(base)
            runtime["llm"].update(plan["llm_overrides"])
            runtime["llm"]["providers"][provider]["cache_dir"] = ".cache/cpc18_external/aigc_relay"
            wrapper = make_client(
                runtime, provider, root, output, cache_only, plan["llm_overrides"]["max_rps"]
            )
            wrapper.client.cache_namespace = phash
            await wrapper.verify_model()
            client = LLMProbabilityClient(wrapper.client)
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
                demo_ids = [] if method == "P0" else plan["demonstrations"][str(item_id)]
                state, instruction, criteria, mapping = forecast.forecast_inputs(
                    targets[item_id],
                    [pool[i] for i in demo_ids],
                    method,
                    orientation,
                    pool,
                    plan["label_permutation"],
                    plan["example_seed"],
                )
                identity = {
                    **fixed,
                    "method": method,
                    "item_id": item_id,
                    "orientation": orientation,
                    "demonstration_ids": demo_ids,
                }
                probabilities, meta = await client.evaluate_choice(
                    state, instruction, criteria, identity
                )
                shown_b = next(k for k, v in mapping.items() if v == "B")
                row = {
                    **identity,
                    **meta,
                    "probabilities": probabilities,
                    "original_b_probability": probabilities[shown_b] if probabilities else None,
                }
                with path.open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                if count % 100 == 0 or count == len(jobs):
                    print(json.dumps({"completed": count, "planned": len(jobs)}), flush=True)
                queue.task_done()

        tasks = [
            asyncio.create_task(worker()) for _ in range(plan["llm_overrides"]["max_concurrency"])
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        forecast.check_matrix(forecast.read_records(path), jobs, phash, stage, True)
        write_json(
            output / "completion.json",
            {**fixed, "responses": count, "completed_at": datetime.now(UTC).isoformat()},
        )
    finally:
        if wrapper is not None:
            await wrapper.close()
        lock.close()


def mock_outcomes(ids):
    return pd.DataFrame(
        [
            {
                "SubjID": person,
                "Set": person // 10,
                "Location": "mock",
                "Condition": "mock",
                "GameID": game,
                "Trial": trial,
                "Feedback": 0,
                "B": (person + trial + game) % 2,
            }
            for person in range(20)
            for game in ids
            for trial in range(1, 6)
        ]
    )


def authorize_label_opening(output, plan, config, root, rows, ids):
    invalid = {
        m: sum(r["status"] != "ok" for r in rows if r["method"] == m)
        / sum(r["method"] == m for r in rows)
        for m in forecast.METHODS
    }
    if len(ids) / len(plan["target_ids"]) < config["evaluation"]["minimum_common_coverage"] or any(
        x > config["evaluation"]["invalid_rate_gate"] for x in invalid.values()
    ):
        raise RuntimeError("forecast validity gate failed: human outcomes remain closed")
    fixed = {
        "plan_hash": digest((root / PLAN).read_bytes()),
        "source_hash": plan["source_hash"],
        "responses_sha256": digest((output / "responses.jsonl").read_bytes()),
        "raw_outcomes_sha256": plan["input_hashes"][
            str((root / config["input_folder"] / "all_raw.csv").relative_to(root))
        ],
        "common_item_ids": ids,
    }
    path = output / "label_opening.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if any(prior.get(k) != v for k, v in fixed.items()):
            raise RuntimeError("cannot change predictions/source after label opening")
    else:
        write_json(
            path,
            {**fixed, "opened_at": datetime.now(UTC).isoformat(), "git_commit": git_state(root)[0]},
        )
    return invalid


def evaluate(config, root, provider):
    plan = load_plan(config, root, clean=provider != "mock")
    output = root / config["experiment"]["output_dir"] / provider / "test"
    rows = forecast.read_records(output / "responses.jsonl")
    phash = digest((root / PLAN).read_bytes())
    forecast.check_matrix(
        rows, forecast.jobs_for(plan, config["experiment"]["seed"]), phash, "test", True
    )
    ids, raw_predictions, effects, valid_pairs = forecast.paired_predictions(
        rows, plan["target_ids"]
    )
    if provider == "mock":
        human_rows = mock_outcomes(ids)
        invalid = dict.fromkeys(forecast.METHODS, 0.0)
    else:
        invalid = authorize_label_opening(output, plan, config, root, rows, ids)
        human_rows = read_outcomes(root / config["input_folder"] / "all_raw.csv", ids)
    numerator, denominator, strata = participant_arrays(human_rows, ids)
    human = numerator.sum(axis=0) / denominator.sum(axis=0)
    predictions = {m: np.array([raw_predictions[m][i] for i in ids]) for m in forecast.METHODS}
    indices = [plan["target_ids"].index(i) for i in ids]
    predictions.update({m: np.array(p)[indices] for m, p in plan["control_predictions"].items()})
    metrics = {
        m: {**forecast.probability_metrics(p, human), "mean_js": mean_bernoulli_js(p, human)}
        for m, p in predictions.items()
    }
    for method in forecast.METHODS:
        order_effect = np.array([effects[method][i] for i in ids])
        metrics[method].update(
            invalid_request_rate=invalid[method],
            valid_pairs=valid_pairs[method],
            signed_order_effect=float(order_effect.mean()),
            mean_absolute_order_effect=float(np.abs(order_effect).mean()),
        )
    comparisons = [
        ("P_true", m)
        for m in (
            "P0",
            "P_neutral",
            "P_shuffled",
            "constant_half",
            "symmetric_fractional_logistic",
            "aligned_knn_16",
        )
    ]
    comparisons.append(("symmetric_fractional_logistic", "constant_half"))
    intervals = bootstrap_contrasts(
        numerator,
        denominator,
        strata,
        predictions,
        comparisons,
        config["evaluation"]["bootstrap_samples"],
        config["experiment"]["seed"] + 1,
    )
    items = {
        x.item_id: x for x in read_targets(root / config["input_folder"] / "cpc18_210.xlsx", ids)
    }
    preferences = {}
    for label, feature in (("higher_ev", "ev"), ("riskier", "var")):
        a = np.array([getattr(items[i], f"{feature}_a") for i in ids])
        b = np.array([getattr(items[i], f"{feature}_b") for i in ids])
        mask = ~np.isclose(a, b)
        preferences[label] = {
            "items": int(mask.sum()),
            **{
                m: float(np.where(b > a, p, 1 - p)[mask].mean())
                for m, p in {**predictions, "human": human}.items()
            },
        }
    calls = [call for row in rows for call in row["calls"]]
    report = {
        "provider": provider,
        "external_human_labels_read": provider != "mock",
        "plan_hash": phash,
        "source_hash": plan["source_hash"],
        "requests": len(rows),
        "common_items": len(ids),
        "omitted_ids": sorted(set(plan["target_ids"]) - set(ids)),
        "participants": numerator.shape[0],
        "human_trial_rows": len(human_rows),
        "resampling_strata": len(strata),
        "selected_candidate": {"method": plan["selected_method"], "alpha": plan["selected_alpha"]},
        "metrics": metrics,
        "contrasts": intervals,
        "preferences": preferences,
        "correct_label_mechanism_supported": all(
            intervals[f"P_true_minus_{m}"]["ci95"]["joint"][1] < 0
            for m in ("P_neutral", "P_shuffled")
        ),
        "usage_successful_calls_including_repairs": {
            "calls": len(calls),
            "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
            "input_tokens": sum(c["usage"].get("prompt_tokens", 0) or 0 for c in calls),
            "output_tokens": sum(c["usage"].get("completion_tokens", 0) or 0 for c in calls),
        },
    }
    destination = output / "evaluation"
    write_json(destination / "summary.json", report)
    pd.DataFrame(
        {
            "item_id": ids,
            "human_b_rate": human,
            "participants": (denominator > 0).sum(axis=0),
            **predictions,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    return report
