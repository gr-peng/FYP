"""Evidence readers cannot silently expand into raw data or hide missing inputs."""

import json

import pytest

from experiments.research_progress import (
    read_sources,
    validate_report_interpretation,
    verify_sources_unchanged,
)


def test_missing_summary_stops_reconciliation(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_sources(tmp_path, {"required": "missing.json"})


def test_changed_summary_is_not_published_as_a_stable_snapshot(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"completed": 3}))
    values, identities = read_sources(tmp_path, {"study": "summary.json"})
    assert values["study"]["completed"] == 3
    path.write_text(json.dumps({"completed": 9}))
    with pytest.raises(ValueError, match="changed"):
        verify_sources_unchanged(identities)


def test_summary_read_does_not_follow_embedded_raw_paths(tmp_path):
    raw = tmp_path / "do_not_read.bin"
    raw.write_bytes(b"This is deliberately not valid JSON")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"raw_path": str(raw), "outputs": [str(raw)]}))
    values, identities = read_sources(tmp_path, {"study": "summary.json"})
    assert set(identities) == {"study"}
    assert values["study"]["raw_path"] == str(raw)
    verify_sources_unchanged(identities)


def test_allowlist_cannot_escape_workspace_through_symlink(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError, match="leaves workspace"):
        read_sources(workspace, {"study": "linked.json"})


def test_new_positive_financial_result_cannot_silently_reuse_old_negative_prose():
    findings = {
        "persona": {"directional_support": "higher_error"},
        "risk_examples": {"directional_support": "lower_error"},
        "external_risk": {"directional_support": "lower_error"},
        "external_shuffled_control": {"directional_support": "not_established"},
        "financial": {"directional_support": "lower_error"},
    }
    with pytest.raises(ValueError, match="narrative: financial"):
        validate_report_interpretation(findings)
