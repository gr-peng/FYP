#!/usr/bin/env python3
"""Build partial-observation quote targets from frozen training sessions only."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.replication_metadata import PLAN, sha  # noqa: E402
from experiments.replication_quote_states import quote_states  # noqa: E402
from experiments.replication_training_audit import audit_training, normalized_tables  # noqa: E402
from experiments.replication_training_data import load_training_frames  # noqa: E402

CODE = [
    "experiments/replication_quote_states.py",
    "experiments/replication_training_data.py",
    "experiments/replication_training_audit.py",
    "scripts/build_market_quote_training.py",
]


def main():
    hashes = {p: sha((ROOT / p).read_bytes()) for p in CODE}
    frames, metadata, plan = load_training_frames(ROOT)
    audit, tables = audit_training(frames, metadata)
    normalized, _ = normalized_tables(frames)
    records, exclusions = quote_states(
        normalized["orders"],
        tables["linked_trades"],
        tables["ledger"],
        tables["market_periods"],
        metadata["_markets"],
    )
    if hashes != {p: sha((ROOT / p).read_bytes()) for p in CODE}:
        raise RuntimeError("state-construction code changed during execution")
    output = ROOT / "outputs/market_quote_training_20260922"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "states.jsonl"
    with path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "scope": "Training only; conditional side of an active order, partial observation",
        "plan_sha256": sha((ROOT / PLAN).read_bytes()),
        "source_sha256": plan["source_sha256"],
        "code_hashes": hashes,
        "states_sha256": sha(path.read_bytes()),
        "source_participant_submissions": audit["orders"]["participant_submissions"],
        "retained": len(records),
        "excluded": exclusions,
        "sessions": len({r["provenance"]["session"] for r in records}),
        "participants": len({r["provenance"]["participant"] for r in records}),
        "retained_order_types": dict(Counter(r["target"]["kind"] for r in records)),
        "event_weighted_buy_rate_diagnostic": sum(r["target"]["buy"] for r in records)
        / len(records),
        "target_boundaries": [
            "No HOLD target or unconditional trading probability.",
            "All input history is strictly before the target's recorded second.",
            "Current book and available funds remain unknown; total balances are reconstructed.",
            "Quoted quantity, quoted price and order kind are targets, not predictors.",
            "Replacing a same-second order with an arbitrary UUID order is forbidden.",
            "Eligibility narrows the event population; it is not evidence of a complete agent.",
        ],
    }
    if len(records) + sum(exclusions.values()) != summary["source_participant_submissions"]:
        raise ValueError("quote eligibility accounting does not cover all submissions")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
