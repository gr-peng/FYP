#!/usr/bin/env python3
"""Fit on frozen train sessions and evaluate fixed baselines on development only."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finance_data import load_development_rows, sha256  # noqa: E402
from experiments.scopic_baselines import fit, metrics, predict  # noqa: E402
from experiments.scopic_states import pre_ranking_decisions, sell_opportunities  # noqa: E402

CODE = [
    "experiments/finance_data.py",
    "experiments/scopic_states.py",
    "experiments/scopic_baselines.py",
    "scripts/run_scopic_baselines.py",
    "docs/scopic_baseline_protocol.md",
    "config/frozen/financial_data_partitions_20260922.json",
]


def main(stage):
    hashes = {path: sha256(ROOT / path) for path in CODE}
    frozen = ROOT / "config/frozen/scopic_baseline_20260922.json"
    if stage == "freeze":
        encoded = json.dumps({"input_hashes": hashes}, indent=2, sort_keys=True) + "\n"
        if frozen.exists() and frozen.read_text() != encoded:
            raise ValueError("refusing to rewrite frozen baseline hashes")
        frozen.write_text(encoded)
        print("Frozen code, protocol and financial partitions; no development behavior read.")
        return
    if json.loads(frozen.read_text())["input_hashes"] != hashes:
        raise ValueError("baseline inputs differ from frozen definition")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the frozen baseline before development evaluation")
    plan = json.loads((ROOT / CODE[-1]).read_text())

    def opportunities(split):
        rows = load_development_rows(ROOT, plan, "scopic", split)
        return sell_opportunities(pre_ranking_decisions(rows))

    train = opportunities("train")
    model = fit(train)
    dev = opportunities("development")
    overall = metrics(dev, predict(model, dev))
    sessions = {}
    for sid in sorted({x["provenance"]["session"] for x in dev}):
        subset = [x for x in dev if x["provenance"]["session"] == sid]
        sessions[sid] = metrics(subset, predict(model, subset))
    output = {
        "scope": "development only; test remains closed",
        "frozen_input_hashes": hashes,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model": model,
        "development": overall,
        "by_session": sessions,
        "uncertainty": "Two development sessions; descriptive metrics only, no IID CI.",
    }
    folder = ROOT / "outputs/scopic_baselines_20260922"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "evaluate"], required=True)
    main(parser.parse_args().stage)
