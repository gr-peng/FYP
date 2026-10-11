# ruff: noqa: E402
"""Describe audited train/development features without opening the event test."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behavioral_market.data.archive import write_json
from experiments.crypto_profile import profile_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/crypto_microstructure_20261011.json"
    )
    parser.add_argument("--split", choices=["train", "development"], required=True)
    parser.add_argument("--days", nargs="+", help="Optional subset of the selected frozen split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Refuse overwrites: each report preserves its evidence and implementation identity.
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    report = profile_study(ROOT, args.config, args.split, args.days)
    report["runner_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    json.dumps(report, allow_nan=False)  # Refuse non-standard NaN/Infinity in machine output.
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": args.split,
                "days": [day["day"] for day in report["days"]],
            }
        )
    )


if __name__ == "__main__":
    main()
