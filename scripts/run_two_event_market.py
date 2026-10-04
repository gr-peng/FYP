"""Run explicit rule/mock/live diagnostic matrices, never silently call a paid provider."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.two_event_market import run_suite  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", required=True, choices=["rule", "mock", "aigc_relay", "siliconflow"]
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/two_event_market_pilot.json")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    asyncio.run(run_suite(config, args.provider, cache_only=args.cache_only))


if __name__ == "__main__":
    main()
