#!/usr/bin/env python3
"""Extract the fixed train-only effective flow; no model fit or generation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.crypto_effective_flow_reader import run_extraction  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_extraction(ROOT, args.output, split=args.split, mode=args.mode,
                            prepare_only=args.prepare_only)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "outputs_published": result["outputs_published"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
