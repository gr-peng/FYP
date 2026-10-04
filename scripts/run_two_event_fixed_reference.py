# ruff: noqa: E402
"""Run a rule-only reference check; real prices are exogenous by construction."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation.event_runner import git_state
from experiments.two_event_data import verify_manifest
from experiments.two_event_fixed_reference import run_reference


def main():
    config = json.loads((ROOT / "config/two_event_month_extension_v6.json").read_text())
    study = json.loads((ROOT / config["data_config"]).read_text())
    data = ROOT / "data/processed" / study["study_id"]
    manifest = json.loads((data / "manifest.json").read_text())
    verify_manifest(manifest, ROOT)
    out = ROOT / "outputs" / study["study_id"] / "fixed_history_rule_reference_v6"
    rows = []
    for spec in config["scenarios"]:
        bars = pd.read_parquet(data / f"{spec['scenario_id']}_daily.parquet")
        paired = []
        for condition in ("news", "no_news"):
            result = run_reference(config, spec, bars, 42, condition)
            write_json(out / f"{spec['symbol']}_{condition}.json", result)
            paired.append(result)
            rows.append({k: v for k, v in result.items() if k not in ("decisions", "daily")})
        assert paired[0]["daily"] == paired[1]["daily"]
        assert [r["decision"] for r in paired[0]["decisions"]] == [
            r["decision"] for r in paired[1]["decisions"]
        ]
    commit, dirty = git_state()
    write_json(
        out / "summary.json",
        {
            "runs": rows,
            "git_commit": commit,
            "dirty_worktree": dirty,
            "data_manifest_sha256": digest((data / "manifest.json").read_bytes()),
            "interpretation": "Price path equals the input by construction; this verifies the "
            "adapter, not price realism. OHLC-touch fills assume external liquidity and ignore "
            "queue/spread/impact.",
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
