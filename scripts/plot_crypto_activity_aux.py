#!/usr/bin/env python3
"""Plot all fixed auxiliary development comparisons from verified receipts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.crypto_activity_aux_pipeline import read_evidence, sha  # noqa: E402


def plot(result_path, review_path, output_prefix):
    result_path, review_path, prefix = map(Path, (result_path, review_path, output_prefix))
    result, review = read_evidence(result_path), read_evidence(review_path)
    if (
        result["status"] != "diagnostic_computed_no_refit"
        or review["verdict"] != "fixed_development_diagnostic_numerically_verified"
        or review["development_result_sha256"] != sha(result_path)
        or review["model_artifact_sha256"] != result["model_artifact_sha256"]
        or review["protocol_sha256"] != result["protocol_sha256"]
    ):
        raise ValueError("Complete fixed result and matching independent review required")
    scenarios = result["scenarios"]
    if [s["additional_delay_ms"] for s in scenarios] != [0, 1000, 5000]:
        raise ValueError("Retain every predeclared delay")
    paths = [prefix.with_suffix(suffix) for suffix in (".png", ".pdf", ".json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Never overwrite previously published evidence")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    colors = ("#2a6f97", "#c27b20", "#5b8c5a")
    derived = []
    for scenario, color in zip(scenarios, colors, strict=True):
        days = scenario["scores"]["by_day"]
        if isinstance(days, dict):
            days = [{"day": day, **report} for day, report in days.items()]
        if [day["day"] for day in days] != [f"2024-10-{i}" for i in range(21, 28)]:
            raise ValueError("Retain all seven fixed development dates in order")
        label = f'{scenario["additional_delay_ms"] / 1000:g}s extra delay'
        comparisons = {}
        for axis, a, b in ((axes[0], "H0", "N0"), (axes[1], "H1", "H0")):
            values = [d["models"][a]["mean_log_probability"]
                      - d["models"][b]["mean_log_probability"] for d in days]
            comparisons[f"{a}-{b}"] = values
            axis.plot(range(7), values, marker="o", color=color, label=label, linewidth=1.5)
        derived.append({"delay_ms": scenario["additional_delay_ms"],
                        "days": [d["day"] for d in days], "daily_gains": comparisons})
    for axis, title in zip(axes[:2], (
        "Distribution specification: H0 minus N0", "Book increment: H1 minus H0",
    ), strict=True):
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(7), [f"Oct {i}" for i in range(21, 28)], rotation=45)
        axis.set_ylabel("Mean log-score gain (nats / observation)")
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    overall = scenarios[0]["scores"]["overall"]
    families = ("N0", "N1", "H0", "H1", "Hnoise")
    predicted = [overall["models"][family]["predicted_zero_fraction"] for family in families]
    axes[2].bar(families, predicted, color=("#aaa", "#888", "#2a6f97", "#5b8c5a", "#c27b20"))
    axes[2].axhline(overall["observed_zero_fraction"], color="#aa2c38", linestyle="--",
                    label="Observed zero-print fraction")
    axes[2].set_title("Zero probability: fixed 0s diagnostic", fontsize=10)
    axes[2].set_ylabel("Fraction of one-second observations")
    axes[2].set_ylim(0, 0.28)
    axes[2].legend(fontsize=8)
    fig.suptitle("Frozen auxiliary development: Oct 21-27, 2024\n"
                 "Count diagnostic; no autonomous-market or causal validation", fontsize=12)
    fig.savefig(paths[0], dpi=180)
    fig.savefig(paths[1])
    plt.close(fig)
    evidence = {
        "result_sha256": sha(result_path), "independent_review_sha256": sha(review_path),
        "plotter_sha256": sha(Path(__file__)), "daily_gains": derived,
        "zero_probability": {"observed": overall["observed_zero_fraction"],
                             **dict(zip(families, predicted, strict=True))},
        "figure_sha256": {p.suffix: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]},
        "interpretation": "Reporting only. No fit, winner selection or significance threshold.",
    }
    paths[2].write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return [str(path) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.result, args.review, args.output_prefix)))


if __name__ == "__main__":
    main()
