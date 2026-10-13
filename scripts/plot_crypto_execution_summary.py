"""Plot authenticated training price-proxy errors on identical delay samples."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELAYS = (0, 10, 50, 100, 250)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--day", choices=["2024-11-01", "2024-11-02", "2024-11-03"], required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output_prefix.with_suffix(f".{ext}").exists() for ext in ("png", "pdf", "json")):
        raise FileExistsError("Choose a new figure prefix; do not replace evidence")
    manifest_hash = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "complete" or manifest.get("outputs_published") is not True:
        raise ValueError("Plot only a complete published summary")
    outputs = manifest.get("outputs", [])
    if len(outputs) != 1:
        raise ValueError("Expected one authenticated summary output")
    path = Path(outputs[0]["path"])
    summary_hash = sha(path)
    if summary_hash != outputs[0]["sha256"]:
        raise ValueError("Summary bytes differ from the successful manifest")
    summary = json.loads(path.read_text())
    if summary.get("split") != "train" or summary.get("delays_ms") != list(DELAYS):
        raise ValueError("Only the fixed training delay scenarios can be plotted")
    day = summary["days"][args.day]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    for column, (group, name) in enumerate(
        [
            ("shared_eligible", "Common eligible prints"),
            ("shared_cross_l1", "Common cross-L1 prints"),
        ]
    ):
        counts = {day["delays"][str(delay)]["groups"][group]["n"] for delay in DELAYS}
        if len(counts) != 1 or next(iter(counts)) <= 0:
            raise ValueError("Delay comparison requires a nonempty identical sample")
        n = next(iter(counts))
        for row, (metric, label) in enumerate([("mean", "Mean absolute error"), ("p99", "P99")]):
            ax = axes[row, column]
            for baseline, color, caption in (
                ("sweep", "#246680", "Visible sweep"),
                ("l1", "#c06a34", "Opposite L1"),
            ):
                values = [
                    day["delays"][str(delay)]["groups"][group]["errors_bps"][f"{baseline}_abs_bps"][
                        metric
                    ]
                    for delay in DELAYS
                ]
                ax.plot(DELAYS, values, marker="o", markersize=4, color=color, label=caption)
            ax.set(title=f"{name} (n={n:,}) · {label}", ylabel="Absolute price error (bps)")
            ax.set_xticks(DELAYS)
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
            if row == 1:
                ax.set_xlabel("Assumed extra information delay (ms)")
    window = "00:00–01:00 UTC" if summary["mode"] == "smoke" else "full UTC day"
    fig.suptitle(
        f"Bybit BTCUSDT perpetual · {args.day} {window} · training diagnostic\n"
        "Price proxies conditioned on actual print side/size; identical samples across delays\n"
        "Empirical P99 uses all retained values; no parent-order or market-realism validation",
        fontsize=12,
    )
    if sha(args.manifest) != manifest_hash or sha(path) != summary_hash:
        raise ValueError("Input evidence changed while plotting")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_prefix.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)
    metadata = {
        "manifest": {"path": str(args.manifest.resolve()), "sha256": manifest_hash},
        "summary": {"path": str(path.resolve()), "sha256": summary_hash},
        "script_sha256": sha(__file__),
        "day": args.day,
        "interpretation": "One training window/day; conditional print-price diagnostic only",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
