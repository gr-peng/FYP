"""Plot descriptive train/development profiles; this is not a simulation score."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    for extension in ("png", "pdf", "json"):
        if args.output_prefix.with_suffix(f".{extension}").exists():
            raise ValueError("Refusing to replace an existing figure or provenance record")
    rows, sources = [], []
    for path in args.profiles:
        body = path.read_bytes()
        report = json.loads(body)
        if report["split"] not in {"train", "development"}:
            raise ValueError("Only train/development profiles can be plotted")
        if any(row["split"] != report["split"] for row in report["days"]):
            raise ValueError("Inconsistent split identity")
        rows.extend(report["days"])
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(body).hexdigest()})
    if len({row["day"] for row in rows}) != len(rows):
        raise ValueError("Duplicate profile day")
    rows.sort(key=lambda row: row["day"])
    x = np.arange(len(rows))
    labels = [row["day"][5:] + ("\ntrain" if row["split"] == "train" else "\ndev") for row in rows]
    colors = ["#246680" if row["split"] == "train" else "#c06a34" for row in rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")

    def intervals(ax, values, offset=0, color="#246680", label=None):
        median = np.array([v["p50"] for v in values])
        low, high = np.array([v["p05"] for v in values]), np.array([v["p95"] for v in values])
        ax.errorbar(
            x + offset,
            median,
            yerr=[median - low, high - median],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )

    intervals(axes[0, 0], [row["spread_bps"] for row in rows])
    axes[0, 0].set(title="Observed spread: median and 5–95% range", ylabel="Basis points")
    for side, offset, color in [("bid", -0.12, "#246680"), ("ask", 0.12, "#c06a34")]:
        intervals(
            axes[0, 1],
            [row["depth_levels"]["5"][f"{side}_depth_btc"] for row in rows],
            offset,
            color,
            side,
        )
    axes[0, 1].set(title="Visible top-5-level depth: median and 5–95% range", ylabel="BTC")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(
        x,
        [row["trades_within_joint_coverage"]["trade_count"]["mean"] for row in rows],
        color=colors,
    )
    axes[1, 0].set(title="Mean trade count in covered seconds", ylabel="Trades / second")
    axes[1, 1].bar(
        x,
        [row["return_horizon_seconds"]["60"]["log_return_bps"]["std"] for row in rows],
        color=colors,
    )
    axes[1, 1].set(
        title="Non-overlapping minute midpoint-return variability",
        ylabel="Standard deviation (bps)",
    )
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "Bybit BTCUSDT perpetual · observed training / development data\n"
        "Retained full seconds only; descriptive quantiles are not confidence intervals",
        fontsize=13,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = args.output_prefix.with_suffix(f".{extension}")
        if output.exists():
            raise ValueError(f"Refusing to replace existing figure: {output}")
        fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {
        "sources": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Observed data only. No model fidelity, execution validity "
        "or LLM improvement score.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
