#!/usr/bin/env python3
"""Run one explicit stage of the Choices13k data-driven calibration experiment."""

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.data_driven import run_data_driven_stage  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["plan", "prepare", "development", "freeze", "final_test", "evaluate"],
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/choices13k_data_driven.json"
    )
    parser.add_argument("--provider", choices=["siliconflow", "aigc_relay", "mock"])
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-rps", type=float)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Engineering only; live scientific stages keep the clean-tree gate.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.provider == "mock":
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] += "_mock"

    async def execute():
        value = await run_data_driven_stage(
            config,
            ROOT,
            args.stage,
            args.provider,
            args.cache_only,
            args.max_rps,
            args.allow_dirty,
        )
        if isinstance(value, tuple):
            value = {"responses": len(value[0]), "items": len(value[1])}
        print(json.dumps({"stage": args.stage, "result": value}, default=str), flush=True)

    asyncio.run(execute())


if __name__ == "__main__":
    main()
