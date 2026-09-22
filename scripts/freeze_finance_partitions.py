#!/usr/bin/env python3
"""Freeze session partitions without loading financial decisions or outcomes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finance_data import build_split_plan  # noqa: E402


def main():
    plan = build_split_plan(ROOT)
    path = ROOT / "config/frozen/financial_data_partitions_20260922.json"
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("refusing to overwrite a different frozen partition plan")
    path.write_text(encoded)
    print(
        json.dumps(
            {
                name: {split: len(ids) for split, ids in spec["assignments"].items()}
                for name, spec in plan["datasets"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
