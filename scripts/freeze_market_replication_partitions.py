#!/usr/bin/env python3
"""Freeze metadata-only session partitions before human market behavior access."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.replication_metadata import freeze  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(freeze(ROOT), indent=2))
