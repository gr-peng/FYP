#!/usr/bin/env python3
"""Evaluate a completed frozen Choices13k held-out run; makes no API requests."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.runner import final_evaluation  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/choices13k_pilot.json")
    args = parser.parse_args()
    print(json.dumps(final_evaluation(json.loads(args.config.read_text()), ROOT), indent=2))
