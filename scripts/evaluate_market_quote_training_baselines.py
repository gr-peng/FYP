#!/usr/bin/env python3
"""Exploratory five-fold session CV inside TRAIN only; development/test stay closed."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.market_quote_baselines import fit, metrics, predict  # noqa: E402
from experiments.replication_metadata import PLAN, sha  # noqa: E402
from experiments.replication_training_data import checked_plan  # noqa: E402


def main():
    plan = checked_plan(ROOT)
    folder = ROOT / "outputs/market_quote_training_20260922"
    source = json.loads((folder / "summary.json").read_text())
    data = (folder / "states.jsonl").read_bytes()
    if source["states_sha256"] != sha(data) or source["plan_sha256"] != sha(
        (ROOT / PLAN).read_bytes()
    ):
        raise ValueError("training state source mismatch")
    for path, checksum in source["code_hashes"].items():
        if sha((ROOT / path).read_bytes()) != checksum:
            raise ValueError("state-construction source changed")
    paths = [
        "experiments/market_quote_baselines.py",
        "scripts/evaluate_market_quote_training_baselines.py",
    ]
    hashes = {p: sha((ROOT / p).read_bytes()) for p in paths}
    records = [json.loads(line) for line in data.splitlines()]
    sessions = {r["provenance"]["session"] for r in records}
    if not sessions <= set(plan["assignments"]["train"]):
        raise ValueError("non-training financial records")
    folds = {}
    for study in (1, 2):
        ordered = sorted(
            [s for s in sessions if plan["session_studies"][s] == study],
            key=lambda s: sha(f"20260927:training-cv:{s}".encode()),
        )
        folds.update({s: i % 5 for i, s in enumerate(ordered)})
    output, fold_metrics = {}, {}
    for fold in range(5):
        indices = [i for i, r in enumerate(records) if folds[r["provenance"]["session"]] == fold]
        train = [r for r in records if folds[r["provenance"]["session"]] != fold]
        validation = [records[i] for i in indices]
        model = fit(train)
        predictions = predict(model, validation)
        for method, values in predictions.items():
            output.setdefault(method, np.full(len(records), np.nan))[indices] = values
        fold_metrics[str(fold)] = metrics(validation, predictions)
        print(
            json.dumps({"completed_training_fold": fold, "validation_records": len(indices)}),
            flush=True,
        )
    report = {
        "scope": "Exploratory session CV inside TRAIN; not independent development/test",
        "fixed_ridge": 0.01,
        "folds": folds,
        "code_hashes": hashes,
        "state_summary_sha256": sha((folder / "summary.json").read_bytes()),
        "out_of_fold": metrics(records, output),
        "by_fold": fold_metrics,
        "per_session": {},
    }
    for session in sorted(sessions):
        ids = [i for i, r in enumerate(records) if r["provenance"]["session"] == session]
        report["per_session"][session] = metrics(
            [records[i] for i in ids], {m: p[ids] for m, p in output.items()}
        )
    model = fit(records)
    if hashes != {p: sha((ROOT / p).read_bytes()) for p in paths}:
        raise RuntimeError("baseline code changed during training")
    (folder / "baseline_cv.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (folder / "baseline_model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps(report["out_of_fold"], indent=2))


if __name__ == "__main__":
    main()
