#!/usr/bin/env python3
"""Prepare, forecast, then explicitly open and evaluate independent CPC18 labels."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.cpc18_external import evaluate, prepare, run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "test", "evaluate"], required=True)
    parser.add_argument("--provider", choices=["aigc_relay", "mock"], default="aigc_relay")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/cpc18_external.json").read_text())
    if args.stage == "prepare":
        print(json.dumps(prepare(config, ROOT)))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate(config, ROOT, args.provider), indent=2))
    else:
        asyncio.run(run(config, ROOT, args.provider, args.stage, args.cache_only))
