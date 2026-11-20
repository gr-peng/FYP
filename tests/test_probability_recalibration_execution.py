"""Synthetic executable packets; no actual derived prediction rows are decoded."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from scripts import validate_probability_recalibration as tool


def fixture(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    packet.mkdir()
    table = [
        {
            "target_id": f"fictional{s}:o{i}",
            "provenance": {
                "session": f"fictional{s}",
                "order_id": f"o{i}",
                "participant": f"s{s}p{i % 5}",
            },
            "observed_buy": int(i % 3 == 0),
            "predictions": {k: 0.1 + 0.2 * (i % 5) for k in tool.admission.METHODS},
        }
        for s in range(16)
        for i in range(25)
    ]
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(table))
    rows = [{"provenance": r["provenance"], "target": {"buy": r["observed_buy"]}} for r in table]
    probabilities = {
        k: np.array([r["predictions"][k] for r in table]) for k in tool.admission.METHODS
    }
    sessions = sorted({r["provenance"]["session"] for r in table})
    analysis = {"overall": tool.metrics(rows, probabilities), "per_session": {}}
    for session in sessions:
        ids = [i for i, r in enumerate(rows) if r["provenance"]["session"] == session]
        analysis["per_session"][session] = tool.metrics(
            [rows[i] for i in ids], {k: p[ids] for k, p in probabilities.items()}
        )
    tool.save(packet / "accounting.json", {"scope": "synthetic"})
    tool.save(
        packet / "preparation.json",
        {
            "target_ids": [r["target_id"] for r in table],
            "sessions": 16,
            "files": {"accounting_audit.json": tool.sha((packet / "accounting.json").read_bytes())},
        },
    )
    source = packet / "sources/scripts/analyze_financial_probability.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source receipt, not historical code\n")
    prep_sha = tool.sha((packet / "preparation.json").read_bytes())
    analysis.update(
        preparation_sha256=prep_sha, analysis_source_sha256=tool.sha(source.read_bytes())
    )
    tool.save(packet / "analysis.json", analysis)
    tool.save(
        packet / "completion.json",
        {
            "preparation_sha256": prep_sha,
            "status": "complete",
            "valid": 3200,
            "expected_responses": 3200,
            "provider_config": {"model": "DeepSeek-V4-Flash"},
        },
    )
    tool.save(
        packet / "partitions.json",
        {
            "assignments": {
                "train": ["fictional_training"],
                "development": sessions,
                "test": ["fictional_final"],
            },
        },
    )
    plan = {"table_sha256": tool.sha(path.read_bytes()), "table_bytes": path.stat().st_size}
    tool.save(packet / "plan.json", {"evidence_kind": "synthetic_fixture_not_actual_approval"})
    tool.save(
        packet / "independent_preflight.json",
        {
            "accepted_for_execution": True,
            "mustfix": [],
            "plan_sha256": tool.sha((packet / "plan.json").read_bytes()),
        },
    )
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "TABLE", path.name)
    monkeypatch.setattr(tool, "checked_packet", lambda _p: plan)
    return packet, path, plan


def test_complete_synthetic_executable_keeps_all_predictions_and_models(tmp_path, monkeypatch):
    packet, _path, _plan = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    summary = tool.execute(packet, out)
    assert summary["overall"]["records"] == 400
    assert set(summary["overall"]["methods"]) == {
        "raw",
        "intercept",
        "monotone",
        "training_constant",
    }
    models = tool.read(out / "models.json")
    assert len(models) == 16 and all(len(r["models"]) == 2 for r in models)
    predictions = tool.read(out / "predictions.json")
    assert len({r["target_id"] for r in predictions}) == 400
    assert all(r["predictions"]["raw"] == r["probability_buy"] for r in predictions)
    receipt = tool.read(out / "admission.json")
    assert receipt["checked_metric_numbers"] == 17 * 29
    assert receipt["components_sha256_before_fit"] == tool.sha(
        (out / "components.json").read_bytes()
    )
    assert tool.read(out / "status.json")["status"] == "complete_pending_independent_audit"
    for reference in ("raw", "training_constant"):
        difference = (
            summary["overall"]["methods"]["monotone"]["brier_session_person_balanced"]
            - summary["overall"]["methods"][reference]["brier_session_person_balanced"]
        )
        assert summary["contrasts"]["monotone-minus-" + reference]["difference"] == pytest.approx(
            difference
        )


@pytest.mark.parametrize(
    "failure", ["approval", "version", "components", "admission", "fit", "source_change"]
)
def test_failure_never_emits_complete_scores_or_success_report(tmp_path, monkeypatch, failure):
    packet, path, _plan = fixture(tmp_path, monkeypatch)
    fits = []
    original_fit, original_save = tool.calibration.cross_validate, tool.save

    def tracked_fit(rows):
        fits.append(True)
        if failure == "fit":
            raise ValueError("synthetic failed optimizer")
        result = original_fit(rows)
        if failure == "source_change":
            path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(tool.calibration, "cross_validate", tracked_fit)
    if failure == "approval":
        tool.save(packet / "independent_preflight.json", {"accepted_for_execution": False})
        monkeypatch.setattr(tool, "bounded_read", lambda *_a: pytest.fail("unaccepted file read"))
    elif failure == "version":
        completion = tool.read(packet / "completion.json")
        completion["provider_config"]["model"] = "different_model"
        tool.save(packet / "completion.json", completion)
        monkeypatch.setattr(
            tool, "bounded_read", lambda *_a: pytest.fail("unaccepted model source")
        )
    elif failure in {"components", "admission"}:

        def fail_save(destination, value):
            if destination.name == failure + ".json":
                raise OSError("synthetic persistence failure")
            return original_save(destination, value)

        monkeypatch.setattr(tool, "save", fail_save)
    out = tmp_path / "out"
    with pytest.raises((ValueError, OSError)):
        tool.execute(packet, out)
    assert bool(fits) == (failure in {"fit", "source_change"})
    assert not (out / "summary.json").exists()
    assert not (out / "REPORT.txt").exists()
    if failure != "approval":
        assert tool.read(out / "status.json")["status"] == "incomplete"


def test_bounded_read_rejects_size_symlinks_and_fifo(tmp_path):
    path = tmp_path / "ordinary"
    path.write_bytes(b"test")
    assert tool.bounded_read(path, 4) == b"test"
    with pytest.raises(ValueError, match="type or size"):
        tool.bounded_read(path, 3)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        tool.bounded_read(link, 4)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="type or size"):
        tool.bounded_read(fifo, 4)


def test_prepare_and_packet_check_never_open_probability_table(tmp_path, monkeypatch):
    forbidden = tool.ROOT / tool.TABLE
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
    design = tool.ROOT.parent / "iterations/probability_recalibration_20261107/design_v1.json"
    plan = tool.prepare(tmp_path / "packet", design)
    assert tool.checked_packet(tmp_path / "packet") == plan
    copied = tmp_path / "packet/design.json"
    copied.write_text(copied.read_text() + " ")
    with pytest.raises(ValueError, match="frozen input"):
        tool.checked_packet(tmp_path / "packet")
