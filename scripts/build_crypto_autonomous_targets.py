#!/usr/bin/env python3
"""Build the fixed three-day TRAIN targets, initializations, and metric scales."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.crypto_autonomous_target_pipeline import build_train_targets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_train_targets(ROOT, args.output, split=args.split,
                                 prepare_only=args.prepare_only)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "outputs_published": result["outputs_published"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
