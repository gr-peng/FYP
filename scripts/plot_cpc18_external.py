#!/usr/bin/env python3
"""Plot the completed, frozen CPC18 result without refitting or new model calls."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "outputs/human_benchmark/cpc18_external/aigc_relay/test/evaluation/summary.json"
    result = json.loads(source.read_text())
    assert result["external_human_labels_read"] and result["common_items"] == 175
    labels = {
        "symmetric_fractional_logistic": "Logistic (487 training items)",
        "aligned_knn_16": "Aligned kNN (k=16)",
        "P_true": "LLM + true human examples",
        "P_shuffled": "LLM + shuffled labels",
        "P_neutral": "LLM + neutral labels",
        "P0": "Plain LLM",
        "constant_half": "Constant 0.5",
    }
    colors = ["#167B72", "#167B72", "#315EA8", "#859ABD", "#859ABD", "#859ABD", "#91949B"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), gridspec_kw={"width_ratios": [1, 1.08]})
    ax = axes[0]
    values = [result["metrics"][k]["mae"] for k in labels]
    ax.barh(list(labels.values()), values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.24)
    ax.set_xlabel("External MAE (lower is better)")
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_title("175 CPC18 risk-choice problems", loc="left", fontweight="bold")
    ax = axes[1]
    peers = ["P0", "P_neutral", "P_shuffled", "symmetric_fractional_logistic", "aligned_knn_16"]
    for i, method in enumerate(peers):
        contrast = result["contrasts"][f"P_true_minus_{method}"]
        value = contrast["estimate"]
        low, high = contrast["ci95"]["joint"]
        ax.errorbar(
            value,
            i,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            color="#315EA8",
            capsize=4,
            markersize=5,
        )
    ax.axvline(0, color="#65676B", linewidth=1)
    ax.set_yticks(
        range(len(peers)),
        [
            "vs plain LLM",
            "vs neutral labels",
            "vs shuffled labels",
            "vs logistic",
            "vs aligned kNN",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.045, 0.09)
    ax.set_xlabel("MAE difference: true-example LLM minus comparator")
    ax.set_title("Joint bootstrap 95% intervals", loc="left", fontweight="bold")
    fig.suptitle(
        "CPC18: LLM calibration helps; statistical models predict better",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.02,
        "Frozen before external labels • 686 participants • 5,000 joint resamples\n"
        "Correct-label mechanism gate failed; this is not a financial-market test.",
        fontsize=9,
        color="#50545B",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93], w_pad=2.8)
    dest = ROOT / "docs/figures"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "cpc18_external_20260922.png", dpi=180)
    fig.savefig(dest / "cpc18_external_20260922.pdf")
    plt.close(fig)
    print(dest / "cpc18_external_20260922.png")


if __name__ == "__main__":
    main()
