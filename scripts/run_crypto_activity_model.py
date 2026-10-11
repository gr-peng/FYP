# ruff: noqa: E402
"""Run the frozen train/development NB2 diagnostic; never access the event test."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behavioral_market.data.archive import write_json
from experiments.crypto_activity_model import run_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "config/crypto_activity_probe_20261011.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new evidence path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = run_probe(ROOT, args.protocol)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_hash:
        raise ValueError("Runner implementation changed during fitting")
    report["source_sha256"]["scripts/run_crypto_activity_model.py"] = runner_hash
    json.dumps(report, allow_nan=False)
    temporary = args.output.with_name(f".{args.output.name}.{uuid4().hex}.tmp")
    try:
        write_json(temporary, report)
        # Atomic publication that cannot replace evidence created while fitting.
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "scenarios_ms": [s["additional_delay_ms"] for s in report["scenarios"]],
            }
        )
    )


if __name__ == "__main__":
    main()
