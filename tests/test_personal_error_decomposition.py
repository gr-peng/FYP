"""Exact synthetic checks for descriptive forecast-error accounting."""

import copy
import math

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import personal_error_decomposition as model


def rows_for(session, person, labels, probabilities, prefix="o"):
    return [
        {
            "session": session,
            "participant": person,
            "order_id": f"{prefix}{i}",
            "fold": 0,
            "observed_buy": y,
            "probabilities": {m: p for m in model.METHODS},
        }
        for i, (y, p) in enumerate(zip(labels, probabilities, strict=True))
    ]


def test_noisy_forecast_shrinkage_is_not_event_matching():
    noisy = model.person_terms([0, 0, 1, 1], [0, 1, 0, 1])
    constant = model.person_terms([0, 0, 1, 1], [0.5] * 4)
    assert noisy["personal_mean_bias"] == constant["personal_mean_bias"] == 0
    assert noisy["prediction_label_covariance"] == constant["prediction_label_covariance"] == 0
    assert noisy["within_person_error"] == 0.5
    assert constant["within_person_error"] == 0.25
    assert noisy["prediction_variance"] - constant["prediction_variance"] == 0.25


def test_matching_can_improve_even_when_prediction_variance_increases():
    constant = model.person_terms([0, 1], [0.5, 0.5])
    perfect = model.person_terms([0, 1], [0, 1])
    assert perfect["total_error"] == 0
    assert perfect["prediction_variance"] - constant["prediction_variance"] == 0.25
    assert perfect["prediction_label_covariance"] - constant["prediction_label_covariance"] == 0.25
    assert perfect["within_person_error"] - constant["within_person_error"] == -0.25


def test_singleton_and_constant_labels_have_different_centered_errors():
    singleton = model.person_terms([1], [0.2])
    constant = model.person_terms([1, 1], [0, 1])
    assert singleton["total_error"] == singleton["personal_mean_bias"] == pytest.approx(0.64)
    assert singleton["within_person_error"] == 0
    assert constant["label_variance"] == 0
    assert constant["within_person_error"] == constant["prediction_variance"] == 0.25


def test_equal_total_error_can_hide_opposite_bias_and_centered_changes():
    first = model.person_terms([0, 0], [0.5, 0.5])
    second = model.person_terms([0, 0], [0, math.sqrt(0.5)])
    assert first["total_error"] == pytest.approx(second["total_error"])
    assert second["personal_mean_bias"] - first["personal_mean_bias"] == pytest.approx(-0.125)
    assert second["within_person_error"] - first["within_person_error"] == pytest.approx(0.125)


def test_unequal_people_sessions_and_events_match_hand_weights():
    rows = (
        rows_for("a", "x", [1], [0], "x")
        + rows_for("a", "y", [0, 0, 0], [0, 0, 0], "y")
        + rows_for("b", "z", [0, 1], [0.5, 0.5], "z")
    )
    original = copy.deepcopy(rows)
    result = model.decompose(rows)
    assert rows == original
    for value in result["overall"].values():
        assert value["total_error"] == 0.375
        assert value["personal_mean_bias"] == 0.25
        assert value["within_person_error"] == 0.125
    population = result["population"]
    assert population["single_target"] == {"people": 1, "balanced_weight": 0.25}
    assert population["constant_labels_including_single_target"] == {
        "people": 2,
        "balanced_weight": 0.5,
    }
    assert population["constant_labels_multiple_targets"] == {"people": 1, "balanced_weight": 0.25}


def test_repeating_events_within_a_person_does_not_change_person_weight():
    rows = rows_for("a", "x", [0, 1], [0.2, 0.6]) + rows_for("b", "z", [1], [0.1])
    original = model.decompose(rows)["overall"]
    extra = copy.deepcopy(rows[:2])
    for r in extra:
        r["order_id"] += "repeat"
    assert model.decompose(rows + extra)["overall"] == original


def test_bootstrap_multiplicity_and_all_algebra_use_identical_draws(monkeypatch):
    class FixedDraws:
        def integers(self, high, size):
            assert high == 3 and size == (1, 3)
            return np.array([[0, 0, 1]])

    monkeypatch.setattr(model.np.random, "default_rng", lambda _seed: FixedDraws())
    rows = []
    for session, change in {"a": 0.2, "b": 0.2, "c": 0, "d": 0, "e": -0.1}.items():
        extra = rows_for(session, session, [0, 0], [0.5, 0.5])
        for row in extra:
            row["probabilities"][model.METHODS[1]] = math.sqrt(0.25 + change)
            row["probabilities"][model.METHODS[2]] = math.sqrt(0.25 + change)
        rows += extra
    result = model.decompose(rows)
    comparisons = model.contrasts(result, [["a"], ["b", "c", "d"], ["e"]], draws=1)
    first = comparisons[model.METHODS[1] + "-minus-" + model.METHODS[0]]
    assert first["percentile_95"]["total_error"] == pytest.approx([0.12, 0.12])
    assert first["percentile_95"]["personal_mean_bias"] == pytest.approx([0.12, 0.12])
    assert first["percentile_95"]["within_person_error"] == [0, 0]
    assert first["max_bootstrap_reconstruction_error"] <= model.TOLERANCE


@pytest.mark.parametrize(
    "bad", ["duplicate", "missing_series", "nonfinite", "label", "identity", "fold"]
)
def test_invalid_common_rows_fail_before_decomposition(monkeypatch, bad):
    rows = rows_for("a", "x", [0, 1], [0.2, 0.6])
    if bad == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif bad == "missing_series":
        del rows[0]["probabilities"][model.METHODS[0]]
    elif bad == "nonfinite":
        rows[0]["probabilities"][model.METHODS[0]] = float("nan")
    elif bad == "label":
        rows[0]["observed_buy"] = True
    elif bad == "identity":
        rows[0]["participant"] = " "
    else:
        rows[0]["fold"] = 5
    monkeypatch.setattr(model, "person_terms", lambda *_a: pytest.fail("invalid rows analyzed"))
    with pytest.raises(ValueError):
        model.decompose(rows)


def test_bootstrap_rejects_missing_or_overlapping_components():
    result = model.decompose(rows_for("a", "x", [0, 1], [0.2, 0.6]))
    for components in ([], [["a"], ["a"]], [["wrong"]]):
        with pytest.raises(ValueError):
            model.contrasts(result, components)
