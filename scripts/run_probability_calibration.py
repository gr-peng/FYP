#!/usr/bin/env python3
"""Frozen probability prediction and label-control development experiment."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.probability_calibration import evaluate, prepare, run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "smoke", "development", "evaluate"]
    )
    parser.add_argument("--provider", default="aigc_relay", choices=["aigc_relay", "mock"])
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/probability_calibration.json").read_text())
    if args.stage == "prepare":
        print(json.dumps(prepare(config, ROOT)))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate(config, ROOT, args.provider), indent=2))
    else:
        asyncio.run(run(config, ROOT, args.provider, args.stage, args.cache_only))
