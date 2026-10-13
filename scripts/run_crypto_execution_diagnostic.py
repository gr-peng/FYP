# ruff: noqa: E402
"""Run the frozen train-only execution diagnostic, or prepare its input receipt."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.crypto_execution_diagnostic import run_diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/crypto_execution_diagnostic_20261011.json"
    )
    parser.add_argument("--split", choices=["train"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="A new immutable run directory")
    args = parser.parse_args(argv)
    state = run_diagnostic(
        ROOT,
        args.config,
        args.output,
        split=args.split,
        mode=args.mode,
        prepare_only=args.prepare_only,
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "run_id": state["run_id"],
                "output": str(args.output),
                "outputs_published": state["outputs_published"],
            }
        )
    )


if __name__ == "__main__":
    main()
