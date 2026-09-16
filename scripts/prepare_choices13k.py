#!/usr/bin/env python3
"""Clone, hash, parse, and freeze the Choices13k pilot split without calling an LLM."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.prepare import prepare  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/choices13k_pilot.json")
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.config.read_text()), ROOT), indent=2))
