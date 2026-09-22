#!/usr/bin/env python3
"""Fixed exploratory controls fitted on the existing pool; external labels stay closed."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.choices13k import frame_items  # noqa: E402
from behavioral_market.data.archive import digest, write_json  # noqa: E402
from experiments.calibration_v2 import load_plan  # noqa: E402
from experiments.probability_analysis import mean_bernoulli_js  # noqa: E402
from experiments.probability_calibration import probability_metrics  # noqa: E402
from experiments.risk_statistical_baselines import fit, predict  # noqa: E402


def main():
    config = json.loads((ROOT / "config/choices13k_calibration_v2.json").read_text())
    plan, frames = load_plan(config, ROOT)
    pool = [item for item in frame_items(frames["pool"]) if item.item_id in plan["pool_ids"]]
    targets = [
        item
        for item in frame_items(frames["development"])
        if item.item_id in plan["target_ids"] and not item.ambiguity
    ]
    model = fit(pool, ridge=0.01, neighbors=16)
    predictions = predict(model, targets)
    human = [item.human_b_rate for item in targets]
    metrics = {
        name: {**probability_metrics(p, human), "mean_js": mean_bernoulli_js(p, human)}
        for name, p in predictions.items()
    }
    output = ROOT / "outputs/risk_statistical_baselines_20260922"
    report = {
        "scope": "exploratory reused development; risk-only; external labels unopened",
        "target_ids": [x.item_id for x in targets],
        "training_items": len(model["train_ids"]),
        "methods": metrics,
        "features": model["features"],
        "source_sha256": digest((ROOT / "experiments/risk_statistical_baselines.py").read_bytes()),
        "hyperparameters": {"ridge": 0.01, "neighbors": 16},
        "model_selection": "two fixed controls; no hyperparameter search",
    }
    write_json(output / "model.json", model)
    write_json(output / "results.json", report)
    forecast_path = ROOT / (
        "outputs/human_benchmark/probability_calibration/aigc_relay/development/"
        "evaluation/predictions.csv"
    )
    if forecast_path.exists():
        llm = (
            pd.read_csv(forecast_path)
            .set_index("item_id")
            .loc[[item.item_id for item in targets], "P_true"]
            .to_numpy()
        )
        prior = predictions["symmetric_fractional_logistic"]
        grid = [
            {
                "llm_weight": weight,
                "mae": float(np.mean(np.abs((1 - weight) * prior + weight * llm - human))),
            }
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        mixture = {
            "scope": "post hoc reused-development mixture diagnostic, not an independent test",
            "risk_items": len(targets),
            "formula": "(1 - w) * fitted_statistical_probability + w * P_true_probability",
            "forecast_sha256": digest(forecast_path.read_bytes()),
            "statistical_model_sha256": digest((output / "model.json").read_bytes()),
            "grid": grid,
            "selected": min(grid, key=lambda x: (x["mae"], x["llm_weight"])),
        }
        write_json(output / "mixture_diagnostic.json", mixture)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
