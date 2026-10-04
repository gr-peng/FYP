# ruff: noqa: E402
"""Reproduce past-only financial memory states and freeze train-only baseline fits."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from experiments.market_quote_memory_baselines import fit
from experiments.replication_quote_memory import extend_memory
from experiments.replication_training_audit import (
    account_ledger,
    attach_trade_actors,
    normalized_tables,
)
from experiments.replication_training_data import checked_plan, load_training_frames

original = ROOT / "outputs/market_quote_training_20260922"
source = json.loads((original / "summary.json").read_text())
raw = (original / "states.jsonl").read_bytes()
assert digest(raw) == source["states_sha256"]
for path, expected in source["code_hashes"].items():
    assert digest((ROOT / path).read_bytes()) == expected, path
plan = checked_plan(ROOT)
base = [json.loads(line) for line in raw.splitlines()]
assert {r["provenance"]["session"] for r in base} <= set(plan["assignments"]["train"])
frames, _, _ = load_training_frames(ROOT)
tables, _ = normalized_tables(frames)
linked = attach_trade_actors(tables["orders"], tables["trades"])
ledger = account_ledger(tables["rounds"], linked)
records = extend_memory(base, tables["orders"], ledger)
data = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records).encode()
previous = ROOT / "outputs/market_quote_memory_draft_20261004"
old = [json.loads(line) for line in (previous / "states.jsonl").read_text().splitlines()]
assert old == records, "formal reconstruction differs from exploratory draft"
model = fit(records)
old_model = json.loads((previous / "baseline_model.json").read_text())
for key in ("beta", "mean", "scale", "train_constant"):
    assert np.allclose(model[key], old_model[key], rtol=1e-10, atol=1e-12), key
out = ROOT / "outputs/market_quote_memory_training_20261004"
out.mkdir(exist_ok=True, parents=True)
(out / "states.jsonl").write_bytes(data)
write_json(out / "baseline_model.json", model)
manifest = {
    "scope": "Train-only reproducible reconstruction; development/test remain closed",
    "input_states_sha256": digest(raw),
    "states_sha256": digest(data),
    "baseline_model_sha256": digest((out / "baseline_model.json").read_bytes()),
    "partition_sha256": digest(
        (ROOT / "config/frozen/market_replication_partitions_20260922.json").read_bytes()
    ),
    "source_sha256": plan["source_sha256"],
    "records": len(records),
    "sessions": len({r["provenance"]["session"] for r in records}),
    "code_hashes": {
        p: digest((ROOT / p).read_bytes())
        for p in [
            "experiments/replication_quote_memory.py",
            "experiments/market_quote_memory_baselines.py",
            "experiments/market_quote_baselines.py",
            "experiments/replication_training_data.py",
            "experiments/replication_training_audit.py",
            "experiments/replication_quote_states.py",
        ]
    },
    "builder_sha256": digest(Path(__file__).read_bytes()),
    "exploratory_state_values_reproduced": True,
    "exploratory_fit_reproduced": True,
}
write_json(out / "manifest.json", manifest)
(out / "builder.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(manifest, indent=2))
