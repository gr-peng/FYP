# ruff: noqa: E402
"""Separate train-only fit, reviewed development export, and fixed evaluation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import crypto_activity_aux_pipeline as pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "config/crypto_activity_aux_20261011.json"
    )
    commands = parser.add_subparsers(dest="phase", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--output", type=Path, required=True)
    for name in ("export-development", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--models", type=Path, required=True)
        sub.add_argument("--review", type=Path, required=True)
        sub.add_argument("--access-receipt", type=Path, required=True)
        if name == "evaluate":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.phase == "fit":
        result = pipeline.fit(ROOT, args.protocol, args.output)
    elif args.phase == "export-development":
        result = pipeline.export_development(
            ROOT, args.protocol, args.models, args.review, args.access_receipt, args.workers
        )
    else:
        result = pipeline.evaluate(
            ROOT, args.protocol, args.models, args.review, args.access_receipt, args.output
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
