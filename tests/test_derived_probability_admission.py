"""All source tables and archived scores below are synthetic fixtures."""

import copy
import json

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access
from test_probability_recalibration import records

from experiments import derived_probability_admission as admission
from experiments import probability_recalibration as calibration


def packet():
    minimal = records()
    sessions = sorted({r["session"] for r in minimal})
    table = [
        {
            "target_id": r["target_id"],
            "provenance": {k: r[k] for k in ("session", "participant", "order_id")},
            "observed_buy": r["observed_buy"],
            "predictions": {name: r["probability_buy"] for name in admission.METHODS},
        }
        for r in minimal
    ]
    prep = {"target_ids": [r["target_id"] for r in minimal], "sessions": len(sessions)}
    scoring = calibration.scoring_records(minimal)
    predictions = {
        name: np.array([r["probability_buy"] for r in minimal]) for name in admission.METHODS
    }
    analysis = {"overall": admission.metrics(scoring, predictions), "per_session": {}}
    for session in sessions:
        ids = [i for i, r in enumerate(minimal) if r["session"] == session]
        analysis["per_session"][session] = admission.metrics(
            [scoring[i] for i in ids], {k: p[ids] for k, p in predictions.items()}
        )
    return table, prep, analysis, sessions


def admit(parts):
    table, prep, analysis, sessions = parts
    raw = json.dumps(table).encode()
    return admission.admit(raw, admission.sha(raw), prep, analysis, sessions, count=36, quota=6)


def test_exact_saved_receipts_allow_only_minimal_primary_projection():
    parts = packet()
    original = copy.deepcopy(parts)
    actual, receipt = admit(parts)
    assert parts == original
    assert actual == records()
    assert receipt["checked_metric_numbers"] == 7 * (9 * 3 + 2)
    assert receipt["consistency_only"] is True
    assert receipt["original_responses_labels_and_menu_averaging_reauthenticated"] is False
    assert all(set(r) == calibration.FIELDS for r in actual)


@pytest.mark.parametrize(
    "bad",
    ["duplicate", "order", "label", "participant", "nonprimary", "method", "score", "session"],
)
def test_inconsistent_table_or_receipts_stop_before_calibration(bad):
    table, prep, analysis, sessions = packet()
    if bad == "duplicate":
        table[-1] = copy.deepcopy(table[0])
    elif bad == "order":
        table.reverse()
    elif bad == "label":
        table[0]["observed_buy"] = 1 - table[0]["observed_buy"]
    elif bad == "participant":
        table[0]["provenance"]["participant"] = ""
    elif bad == "nonprimary":
        table[0]["predictions"]["shuffled_labels"] = 0.99
    elif bad == "method":
        del table[0]["predictions"]["real_labels"]
    elif bad == "score":
        analysis["overall"]["methods"]["no_examples"]["brier_session_person_balanced"] += 1e-8
    else:
        sessions[-1] = "unapproved"
    with pytest.raises((ValueError, KeyError)):
        admit((table, prep, analysis, sessions))


def test_fixed_absolute_tolerance_has_no_relative_relaxation():
    assert admission.compare_metrics({"a": 1000.0}, {"a": 1000.0 + 5e-11}) == 1
    with pytest.raises(ValueError, match="metric differs"):
        admission.compare_metrics({"a": 1000.0}, {"a": 1000.0 + 1e-8})


def test_bad_hash_prevents_any_json_decoding(monkeypatch):
    monkeypatch.setattr(
        admission, "strict_json", lambda _v: pytest.fail("decoded unverified bytes")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        admission.admit(b"not JSON", "wrong", {}, {}, [])
