import copy

import numpy as np
import pytest
from test_choices13k_data_driven import make_item

from experiments.calibration_v2 import label_permutation
from experiments.probability_calibration import (
    METHODS,
    check_matrix,
    forecast_inputs,
    jobs_for,
    paired_predictions,
    probability_metrics,
)


def test_forecast_is_blind_to_target_labels_and_hidden_probabilities():
    pool = [make_item(i, i / 10) for i in range(1, 8)]
    lookup = {x.item_id: x for x in pool}
    permutation = label_permutation(pool, 9)
    target = make_item(100, ambiguity=True)
    changed = target.model_copy(
        update={
            "human_b_rate": 0.98123,
            "n_humans": 731,
            "ev_b": 1e9,
            "var_b": 1e8,
            "option_b": [(1.0, 120.0), (0.0, 1.0)],
        }
    )
    # make_item's B support depends on ID, so use the same support with altered hidden mass.
    changed = changed.model_copy(
        update={"option_b": [(1.0, target.option_b[0][1]), (0.0, target.option_b[1][1])]}
    )
    for method in METHODS:
        assert forecast_inputs(target, pool[:4], method, "A_first", lookup, permutation, 9) == (
            forecast_inputs(changed, pool[:4], method, "A_first", lookup, permutation, 9)
        )
    flipped_pool = [x.model_copy(update={"human_b_rate": 1 - x.human_b_rate}) for x in pool]
    assert forecast_inputs(target, pool[:4], "P_neutral", "A_first", lookup, permutation, 9) == (
        forecast_inputs(
            target,
            flipped_pool[:4],
            "P_neutral",
            "A_first",
            {x.item_id: x for x in flipped_pool},
            permutation,
            9,
        )
    )


def matrix_fixture():
    plan = {"target_ids": [10, 20]}
    jobs = jobs_for(plan, 1)
    rows = []
    for method, item, orientation in jobs:
        shown_b = 0.2 if orientation == "A_first" else 0.8
        rows.append(
            {
                "method": method,
                "item_id": item,
                "orientation": orientation,
                "plan_hash": "frozen",
                "stage": "development",
                "status": "ok",
                "probabilities": {"A": 1 - shown_b, "B": shown_b},
                "original_b_probability": 0.2,
            }
        )
    return jobs, rows


def test_complete_counterbalance_and_original_choice_mapping():
    jobs, rows = matrix_fixture()
    assert len(jobs) == 16
    assert len(check_matrix(rows, jobs, "frozen", "development", True)) == 16
    ids, predictions, effects, count = paired_predictions(rows, [10, 20])
    assert ids == [10, 20]
    assert all(v == 2 for v in count.values())
    for method in METHODS:
        assert predictions[method][10] == 0.2
        assert effects[method][10] == 0
    with pytest.raises(RuntimeError, match="identities"):
        check_matrix(rows + [rows[0]], jobs, "frozen", "development")
    changed = copy.deepcopy(rows)
    changed[0]["original_b_probability"] = 0.9
    with pytest.raises(RuntimeError, match="mapping"):
        check_matrix(changed, jobs, "frozen", "development")
    with pytest.raises(RuntimeError, match="incomplete"):
        check_matrix(rows[:-1], jobs, "frozen", "development", True)


def test_missing_orientation_is_not_replaced_by_single_prediction():
    jobs, rows = matrix_fixture()
    affected = rows[0]["item_id"]
    rows[0].update(status="invalid", probabilities=None, original_b_probability=None)
    check_matrix(rows, jobs, "frozen", "development", True)
    ids, _, _, _ = paired_predictions(rows, [10, 20])
    assert ids == [i for i in [10, 20] if i != affected]
    rows[0]["original_b_probability"] = 0.5
    with pytest.raises(RuntimeError, match="fallback"):
        check_matrix(rows, jobs, "frozen", "development")


def test_sampled_error_and_constant_discrimination():
    result = probability_metrics([0, 1], [0.3, 0.3])
    assert np.isclose(result["mae_expected_sampled_10"], 0.5)
    assert np.isclose(result["mae"], 0.5)
    result = probability_metrics([0.5, 0.5], [0.5, 0.5])
    assert result["mae"] == 0
    assert result["pearson"] is None
    assert np.isclose(result["mae_expected_sampled_10"], 0.123046875)
