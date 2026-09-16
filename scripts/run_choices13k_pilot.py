#!/usr/bin/env python3
"""Run an explicit, resumable stage of the preregistered Choices13k pilot."""

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.runner import run_stage  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["plan", "prepare", "calibration", "freeze", "heldout", "evaluate", "all"],
        help="No stage is implicit; plan is offline and all runs the gated sequence.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/choices13k_pilot.json")
    parser.add_argument("--provider", choices=["siliconflow", "aigc_relay", "mock"])
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-rps", type=float)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Engineering only; live scientific runs should keep the default clean-tree gate.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.provider == "mock":
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] += "_mock"
    stages = (
        ["prepare", "calibration", "freeze", "heldout", "evaluate"]
        if args.stage == "all"
        else [args.stage]
    )

    async def execute():
        results = {}
        for stage in stages:
            value = await run_stage(
                config,
                ROOT,
                stage,
                args.provider,
                args.cache_only,
                args.max_rps,
                args.allow_dirty,
            )
            if isinstance(value, tuple):
                value = {"responses": len(value[0]), "items": len(value[1])}
            results[stage] = value
            print(json.dumps({"stage": stage, "result": value}, default=str), flush=True)
        return results

    asyncio.run(execute())


if __name__ == "__main__":
    main()
