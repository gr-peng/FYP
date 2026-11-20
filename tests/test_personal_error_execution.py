"""Executable synthetic forecasts; no actual E forecasts or original human rows."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from scripts import decompose_personal_forecast_error as tool


def fixture(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    packet.mkdir()
    records = [
        {
            "session": f"fictional{s}",
            "participant": f"s{s}p{i % 3}",
            "order_id": f"o{i}",
            "fold": s,
            "observed_buy": int(i % 3 == 0),
            "probabilities": {
                m: (0.1 + 0.1 * (i % 5) + 0.05 * j) for j, m in enumerate(tool.model.METHODS)
            },
        }
        for s in range(3)
        for i in range(6 + s)
    ]
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    rows = [
        {
            "provenance": {k: r[k] for k in ("session", "participant", "order_id")},
            "target": {"buy": r["observed_buy"]},
        }
        for r in records
    ]
    probabilities = {
        k: np.array([r["probabilities"][k] for r in records]) for k in tool.model.METHODS
    }
    sessions = sorted({r["session"] for r in records})
    analysis = {"overall": tool.metrics(rows, probabilities), "per_session": {}}
    for session in sessions:
        ids = [i for i, r in enumerate(records) if r["session"] == session]
        analysis["per_session"][session] = tool.metrics(
            [rows[i] for i in ids], {k: p[ids] for k, p in probabilities.items()}
        )
    tool.save(packet / "old_summary.json", analysis)
    tool.save(
        packet / "old_folds.json",
        {
            "session_to_fold": {s: i for i, s in enumerate(sessions)},
            "components": [[s] for s in sessions],
            "recorded_participants": 9,
            "cross_session_recorded_participant_codes": 0,
        },
    )
    plan = {"prediction": {"sha256": tool.sha(path.read_bytes()), "bytes": path.stat().st_size}}
    tool.save(packet / "plan.json", {"scope": "synthetic_fixture_not_actual_approval"})
    tool.save(
        packet / "independent_preflight.json",
        {
            "accepted_for_execution": True,
            "mustfix": [],
            "plan_sha256": tool.sha((packet / "plan.json").read_bytes()),
        },
    )
    monkeypatch.setattr(tool, "SOURCE", tmp_path)
    monkeypatch.setattr(tool, "PREDICTION", path.name)
    monkeypatch.setattr(tool, "EXPECTED", {"records": 21, "sessions": 3, "recorded_codes": 9})
    monkeypatch.setattr(tool, "checked_packet", lambda _p: plan)
    return packet, path, plan


def test_complete_synthetic_run_reproduces_original_metrics_and_components(tmp_path, monkeypatch):
    packet, _path, _plan = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    result = tool.execute(packet, out)
    assert result["population"]["records"] == 21
    assert result["population"]["session_person_units"] == 9
    assert len(tool.read(out / "person_components.json")) == 9
    assert tool.read(out / "admission.json")["original_metric_numbers"] == 4 * 23
    assert tool.read(out / "status.json")["status"] == "complete_pending_independent_audit"
    assert set(result["overall"]) == set(tool.model.METHODS)
    for v in result["overall"].values():
        assert v["total_error"] == pytest.approx(v["personal_mean_bias"] + v["within_person_error"])
        assert v["within_person_error"] == pytest.approx(
            v["prediction_variance"] + v["label_variance"] - 2 * v["prediction_label_covariance"]
        )
    assert len(result["contrasts"]) == 2
    assert all(c["draws"] == 10000 for c in result["contrasts"].values())
    assert "没有重新训练" in (out / "REPORT.txt").read_text()


@pytest.mark.parametrize(
    "failure",
    ["approval", "hash", "fold", "count", "score", "components", "person", "admission", "late"],
)
def test_bad_source_or_persistence_stops_without_complete_report(tmp_path, monkeypatch, failure):
    packet, path, plan = fixture(tmp_path, monkeypatch)
    called = []
    original_decompose, original_save = tool.model.decompose, tool.save

    def tracked(records):
        called.append(True)
        value = original_decompose(records)
        if failure == "late":
            path.write_text(path.read_text() + " ")
        return value

    monkeypatch.setattr(tool.model, "decompose", tracked)
    if failure == "approval":
        tool.save(packet / "independent_preflight.json", {"accepted_for_execution": False})
        monkeypatch.setattr(tool, "verified_bytes", lambda *_a: pytest.fail("unapproved read"))
    elif failure == "hash":
        # Same length, invalid JSON: hash rejection must precede JSON parsing.
        path.write_bytes(b"!" * plan["prediction"]["bytes"])
    elif failure == "count":
        monkeypatch.setattr(tool, "EXPECTED", {"records": 22, "sessions": 3, "recorded_codes": 9})
    elif failure == "score":
        value = tool.read(packet / "old_summary.json")
        value["overall"]["methods"][tool.model.METHODS[0]]["brier_session_person_balanced"] += 0.01
        tool.save(packet / "old_summary.json", value)
    elif failure in {"fold", "components", "person"}:
        value = tool.read(packet / "old_folds.json")
        if failure == "fold":
            value["session_to_fold"]["fictional0"] = 1
        elif failure == "components":
            value["components"] = [["fictional0", "fictional1"], ["fictional2"]]
        else:
            value["cross_session_recorded_participant_codes"] = 1
        tool.save(packet / "old_folds.json", value)
    elif failure == "admission":

        def fail_save(destination, value):
            if destination.name == "admission.json":
                raise OSError("synthetic durable admission failure")
            return original_save(destination, value)

        monkeypatch.setattr(tool, "save", fail_save)
    out = tmp_path / "out"
    with pytest.raises((ValueError, OSError)) as exc:
        tool.execute(packet, out)
    if failure == "hash":
        assert "hash mismatch before decoding" in str(exc.value)
    assert bool(called) == (failure == "late")
    assert not (out / "summary.json").exists()
    assert not (out / "REPORT.txt").exists()
    if failure != "approval":
        assert tool.read(out / "status.json")["status"] == "incomplete"


def test_verified_bytes_rejects_symlink_fifo_size_and_digest(tmp_path):
    path = tmp_path / "ordinary"
    path.write_bytes(b"test")
    expected = {"bytes": 4, "sha256": tool.sha(b"test")}
    assert tool.verified_bytes(path, expected) == b"test"
    with pytest.raises(ValueError, match="type/size"):
        tool.verified_bytes(path, dict(expected, bytes=3))
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.verified_bytes(path, dict(expected, sha256="0" * 64))
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        tool.verified_bytes(link, expected)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="type/size"):
        tool.verified_bytes(fifo, expected)


def test_prepare_never_reads_real_forecasts_and_checks_copied_metadata(tmp_path, monkeypatch):
    forbidden = tool.SOURCE / tool.PREDICTION
    original_text, original_bytes, original_open = Path.read_text, Path.read_bytes, os.open

    def text_guard(path, *a, **k):
        assert path != forbidden
        return original_text(path, *a, **k)

    def bytes_guard(path, *a, **k):
        assert path != forbidden
        return original_bytes(path, *a, **k)

    def open_guard(path, *a, **k):
        assert Path(path) != forbidden
        return original_open(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", text_guard)
    monkeypatch.setattr(Path, "read_bytes", bytes_guard)
    monkeypatch.setattr(os, "open", open_guard)
    design = tool.ROOT.parent / "iterations/personal_error_decomposition_20261107/design_v2.json"
    plan = tool.prepare(tmp_path / "packet", design)
    assert plan["preparation_read_prediction_bytes"] is False
    assert tool.checked_packet(tmp_path / "packet") == plan
    copied = tmp_path / "packet/old_folds.json"
    copied.write_text(copied.read_text() + " ")
    with pytest.raises(ValueError, match="frozen input"):
        tool.checked_packet(tmp_path / "packet")
