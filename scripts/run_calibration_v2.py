#!/usr/bin/env python3
"""Explicit-stage entry point for the visible-information calibration ablation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.calibration_v2 import evaluate, prepare, run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "smoke", "development", "evaluate"]
    )
    parser.add_argument("--provider", default="aigc_relay", choices=["aigc_relay", "mock"])
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/choices13k_calibration_v2.json").read_text())
    if args.stage == "prepare":
        plan = prepare(config, ROOT)
        print(json.dumps({"calls": plan["calls"], "items": len(plan["target_ids"])}))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate(config, ROOT, args.provider), indent=2))
    else:
        asyncio.run(run(config, ROOT, args.provider, args.stage, args.cache_only))
