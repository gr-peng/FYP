"""Explicit, resumable stages for the two-event research; no implicit paid calls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.two_event_data import evaluate, prepare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["plan", "prepare", "evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/two_event_study.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "plan":
        print(json.dumps(config, ensure_ascii=False, indent=2))
    elif args.stage == "prepare":
        result = prepare(config)
        print(f"Verified {len(result['processed_files'])} artifacts; no model calls.")
    else:
        result = evaluate(config)
        print(f"Evaluated {len(result['scenarios'])} scenarios; no model calls.")
        print(ROOT / "outputs" / config["study_id"] / "empirical/REPORT.md")


if __name__ == "__main__":
    main()
