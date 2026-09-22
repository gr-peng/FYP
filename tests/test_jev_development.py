import asyncio
import json

import pytest

from scripts import run_jev_development as study


def test_resumption_and_original_option_mapping(tmp_path, monkeypatch):
    plan = {"target_ids": [1, 2], "max_rps": 2, "concurrency": 4}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(study, "PLAN", path)
    monkeypatch.setattr(study, "OUT", tmp_path / "output")
    monkeypatch.setattr(study, "checked_plan", lambda live=False: plan)
    payloads = []
    for method in study.parent.METHODS:
        for item_id in [1, 2]:
            for orientation in ["A_first", "B_first"]:
                payloads.append(
                    {
                        "method": method,
                        "item_id": item_id,
                        "orientation": orientation,
                        "state": {},
                        "instructions": "predict",
                        "criteria": {"A": "A", "B": "B"},
                        "mapping": {"A": "A", "B": "B"}
                        if orientation == "A_first"
                        else {"A": "B", "B": "A"},
                    }
                )
    monkeypatch.setattr(study, "inputs", lambda: (plan, {}, payloads))
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def evaluate_choice(self, state, instructions, criteria, identity):
            calls.append(identity)
            return {"probabilities": {"A": 0.2, "B": 0.8}, "cache_hit": False}

        async def close(self):
            pass

    monkeypatch.setattr(study, "JevClient", FakeClient)
    asyncio.run(study.run("smoke"))
    asyncio.run(study.run("smoke"))
    assert len(calls) == 16
    rows = study.records("smoke", study.digest(path.read_bytes()), True)
    assert all(
        r["original_b_probability"] == (0.8 if r["orientation"] == "A_first" else 0.2) for r in rows
    )
    records = study.OUT / "smoke/responses.jsonl"
    with records.open("a") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(RuntimeError, match="identities"):
        study.records("smoke", study.digest(path.read_bytes()), True)


def test_main_requires_completed_smoke(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    monkeypatch.setattr(study, "PLAN", path)
    monkeypatch.setattr(study, "OUT", tmp_path / "output")
    monkeypatch.setattr(study, "checked_plan", lambda live=False: {})
    with pytest.raises(RuntimeError, match="smoke required"):
        asyncio.run(study.run("development"))
