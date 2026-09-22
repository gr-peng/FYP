#!/usr/bin/env python3
"""Preserve frozen reports and write a separately versioned JS aggregation erratum."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.probability_analysis import correct  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["aigc_relay", "mock"], default="aigc_relay")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/probability_calibration.json").read_text())
    print(json.dumps(correct(config, ROOT, args.provider), indent=2))
