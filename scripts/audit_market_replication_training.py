#!/usr/bin/env python3
"""Audit only frozen training sessions, preserving all conflicts and provenance."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.replication_metadata import PLAN, sha  # noqa: E402
from experiments.replication_training_audit import audit_training  # noqa: E402
from experiments.replication_training_data import (  # noqa: E402
    load_training_frames,
    load_training_macro_reference,
)

CODE = (
    "experiments/replication_training_data.py",
    "experiments/replication_training_audit.py",
    "scripts/audit_market_replication_training.py",
)


def main():
    hashes = {p: sha((ROOT / p).read_bytes()) for p in CODE}
    frames, metadata, _ = load_training_frames(ROOT)
    report, tables = audit_training(frames, metadata)
    reference = load_training_macro_reference(ROOT).rename(
        columns={
            "slab": "session",
            "mlab": "market",
            "rep": "repeat",
            "ab": "average_bias",
        }
    )
    comparison = tables["market_outcomes"].merge(
        reference,
        on=["session", "market", "repeat"],
        suffixes=("_ours", "_author"),
        how="outer",
        indicator=True,
        validate="1:1",
    )
    if not comparison._merge.eq("both").all():
        raise ValueError("training macro-reference coverage differs")
    differences = {}
    for metric in ["rd", "rad", "rdmax", "average_bias"]:
        difference = (comparison[f"{metric}_ours"] - comparison[f"{metric}_author"]).abs()
        differences[metric] = {
            "max_absolute_difference": float(difference.max()),
            "matched_nonmissing": int(difference.notna().sum()),
            "above_1e_6": int(difference.gt(1e-6).sum()),
            "author_storage_dtype": str(comparison[f"{metric}_author"].dtype),
            "equal_after_author_precision_conversion": int(
                (
                    comparison[f"{metric}_ours"].astype(comparison[f"{metric}_author"].dtype)
                    == comparison[f"{metric}_author"]
                ).sum()
            ),
        }
    report["author_processed_macro_comparison"] = differences
    if hashes != {p: sha((ROOT / p).read_bytes()) for p in CODE}:
        raise RuntimeError("audit code changed during execution; rerun before recording results")
    report["plan_sha256"] = sha((ROOT / PLAN).read_bytes())
    report["code_hashes"] = hashes
    output = ROOT / "outputs/market_replication_training_20260922"
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
