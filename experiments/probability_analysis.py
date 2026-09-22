"""Reproducible secondary-metric erratum; preserve the pre-call frozen experiment.

The first evaluator passed arrays to a scalar Bernoulli-JS helper, reporting a
sum under the mean_js name. Raw predictions and all selection/MAE results remain
unchanged. Corrected output is written separately, never over the original report.
"""

import copy
import json

import numpy as np
import pandas as pd

from behavioral_market.benchmarks.evaluation import bernoulli_js
from behavioral_market.data.archive import digest, write_json
from experiments import probability_calibration as experiment


def mean_bernoulli_js(probabilities, human):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(human, dtype=float)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("JS requires matching nonempty one-dimensional arrays")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or ((p < 0).any() or (p > 1).any() or (y < 0).any() or (y > 1).any())
    ):
        raise ValueError("invalid Bernoulli probability")
    return float(np.mean([bernoulli_js(a, b) for a, b in zip(p, y, strict=True)]))


def correct(config, root, provider="aigc_relay"):
    plan, _, parent, frames = experiment.load_plan(config, root)
    output = root / config["experiment"]["output_dir"] / provider / "development"
    path = output / "evaluation/summary.json"
    original = json.loads(path.read_text())
    phash = digest((root / experiment.PLAN).read_bytes())
    rows = experiment.read_records(output / "responses.jsonl")
    experiment.check_matrix(
        rows, experiment.jobs_for(plan, config["experiment"]["seed"]), phash, "development", True
    )
    ids, raw_predictions, _, _ = experiment.paired_predictions(rows, plan["target_ids"])
    items = frames["development"].set_index("item_id").loc[ids]
    y = items.human_b_rate.to_numpy()
    predictions = {m: np.array([raw_predictions[m][i] for i in ids]) for m in experiment.METHODS}
    pool = frames["pool"].set_index("item_id").loc[parent["pool_ids"]]
    controls = {
        "constant_half": np.full(len(ids), 0.5),
        "pool_mean": np.full(len(ids), pool.human_b_rate.mean()),
        "pool_median": np.full(len(ids), pool.human_b_rate.median()),
        "retrieval_mean": np.array(
            [pool.loc[parent["demonstrations"][str(i)]["matched"]].human_b_rate.mean() for i in ids]
        ),
    }
    if original["plan_hash"] != phash or original["common_items"] != len(ids):
        raise RuntimeError("original analysis does not match completed frozen experiment")
    revised = copy.deepcopy(original)

    def patch_metric(record, p, human):
        expected = experiment.probability_metrics(p, human)
        for metric in ("mae", "rmse", "mean_js", "mae_expected_sampled_10"):
            if not np.isclose(record[metric], expected[metric], atol=1e-12, rtol=1e-12):
                raise RuntimeError("original metric differs from frozen evaluator")
        record["mean_js"] = mean_bernoulli_js(p, human)

    for name, p in predictions.items():
        patch_metric(revised["metrics"][name], p, y)
    for name, p in controls.items():
        patch_metric(revised["controls"][name], p, y)
    for row in revised["shrinkage"]:
        patch_metric(row, 0.5 + row["alpha"] * (predictions[row["method"]] - 0.5), y)
    subgroups = pd.read_csv(output / "evaluation/subgroups.csv")
    all_predictions = {**predictions, **controls}
    for index, row in subgroups.iterrows():
        mask = items[row["field"]].astype(bool).to_numpy() == bool(row["value"])
        subgroups.at[index, "mean_js"] = mean_bernoulli_js(
            all_predictions[row["method"]][mask], y[mask]
        )
    revised["evaluation_erratum"] = {
        "problem": "Arrays supplied to scalar Bernoulli-JS helper; mean_js was a sum.",
        "correction": "Average per-problem JS in bits; other metrics and predictions unchanged.",
        "original_summary_sha256": digest(path.read_bytes()),
        "responses_sha256": digest((output / "responses.jsonl").read_bytes()),
        "correction_source_sha256": digest(
            (root / "experiments/probability_analysis.py").read_bytes()
        ),
        "selection_and_primary_results_changed": False,
    }
    destination = output / "evaluation_corrected"
    write_json(destination / "summary.json", revised)
    subgroups.to_csv(destination / "subgroups.csv", index=False)
    return {
        "correction": revised["evaluation_erratum"],
        "mean_js": {k: v["mean_js"] for k, v in revised["metrics"].items()},
    }
