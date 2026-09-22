import numpy as np
import pandas as pd
import pytest

from experiments.cpc18_data import (
    FEATURE_COLUMNS,
    bootstrap_contrasts,
    participant_arrays,
    select_outcomes,
    targets_from_frame,
)
from experiments.cpc18_external import authorize_label_opening, mock_outcomes


def feature_frame():
    row = dict.fromkeys(FEATURE_COLUMNS, np.nan)
    row.update(GameID=1, Amb=0, ap1=1.0, av1=10.0, bp1=0.5, bv1=30.0, bp2=0.5, bv2=-5.0)
    return pd.DataFrame([row])


def test_feature_boundary_excludes_human_rates_and_ambiguous_cases():
    frame = feature_frame()
    item = targets_from_frame(frame, [1])[0]
    assert not hasattr(item, "human_b_rate")
    assert item.ev_a == 10
    assert item.ev_b == 12.5
    with pytest.raises(ValueError, match="allowlist"):
        targets_from_frame(frame.assign(**{"B.1": 0.99}), [1])
    with pytest.raises(ValueError, match="non-ambiguous"):
        targets_from_frame(frame.assign(Amb=1), [1])


def test_no_feedback_selection_precedes_outcome_conversion():
    frame = mock_outcomes([1])
    later = frame.copy()
    later["Trial"] += 5
    later["Feedback"] = 1
    later["B"] = "unreadable_later_outcome"
    excluded = frame.copy()
    excluded["GameID"] = 99
    excluded["B"] = "unreadable_excluded_outcome"
    selected = select_outcomes(pd.concat([frame, later, excluded]), [1])
    assert len(selected) == len(frame)
    assert selected.B.isin([0, 1]).all()


def test_participant_cluster_keeps_repeated_choices_together():
    rows = [
        {
            "SubjID": person,
            "GameID": 1,
            "Trial": trial,
            "Set": 1,
            "Location": "site",
            "Condition": "order",
            "B": person,
        }
        for person in (0, 1)
        for trial in range(1, 6)
    ]
    numerator, denominator, strata = participant_arrays(pd.DataFrame(rows), [1])
    assert numerator.shape == (2, 1)
    np.testing.assert_array_equal(denominator, [[5], [5]])
    result = bootstrap_contrasts(
        numerator, denominator, strata, {"zero": [0.0], "one": [1.0]}, [("zero", "one")], 1000, 41
    )["zero_minus_one"]
    assert result["estimate"] == 0
    assert result["ci95"]["problem_only"] == [0, 0]
    assert result["ci95"]["participant_only"] == [-1, 1]
    assert result["ci95"]["joint"] == [-1, 1]
    with pytest.raises(ValueError, match="partition"):
        bootstrap_contrasts(
            numerator,
            denominator,
            [[0, 0]],
            {"zero": [0.0], "one": [1.0]},
            [("zero", "one")],
            10,
            41,
        )


def test_invalid_forecast_gate_prevents_any_label_opening_record(tmp_path):
    rows = [
        {"method": method, "status": "invalid" if method == "P_true" else "ok"}
        for method in ("P0", "P_true", "P_neutral", "P_shuffled")
    ]
    config = {"evaluation": {"minimum_common_coverage": 0.99, "invalid_rate_gate": 0.01}}
    with pytest.raises(RuntimeError, match="outcomes remain closed"):
        authorize_label_opening(tmp_path, {"target_ids": [1]}, config, tmp_path, rows, [1])
    assert not (tmp_path / "label_opening.json").exists()
