# ruff: noqa: E402
"""Summarize one authenticated train-only execution run into a new directory."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.crypto_execution_summary import summarize_execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/crypto_execution_diagnostic_20261011.json"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="A complete published execution manifest"
    )
    parser.add_argument("--split", choices=["train"], required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="A new immutable summary directory"
    )
    args = parser.parse_args(argv)
    state = summarize_execution(ROOT, args.config, args.manifest, args.output, split=args.split)
    print(
        json.dumps(
            {
                "status": state["status"],
                "run_id": state["run_id"],
                "outputs_published": state["outputs_published"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
