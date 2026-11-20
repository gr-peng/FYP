"""Synthetic only; this file cannot admit the historical prediction table."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import probability_recalibration as model


def records():
    return [
        {
            "target_id": f"s{s}:o{i}", "session": f"s{s}", "participant": f"s{s}p{i % 2}",
            "order_id": f"o{i}", "observed_buy": int(i % 3 == 0), "probability_buy": 0.1 + i / 10,
        }
        for s in range(6) for i in range(6)
    ]


@pytest.mark.parametrize("mode", model.MODES)
def test_gradient_matches_independent_weighted_finite_difference(mode):
    x = np.array([-2.1, 0.2, 2.5, -0.8])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    theta = np.array([0.3, 0.8]) if mode == "monotone" else np.array([0.3])
    loss, grad = model.objective(theta, x, y, w, mode)

    def independent(v):
        b = v[1] if mode == "monotone" else 1
        p = 1 / (1 + np.exp(-(v[0] + b * x)))
        loss = -sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        return loss + (0.005 * (b - 1) ** 2 if mode == "monotone" else 0)

    eps = 1e-5
    numerical = [
        (independent(theta + np.eye(len(theta))[i] * eps)
         - independent(theta - np.eye(len(theta))[i] * eps)) / (2 * eps)
        for i in range(len(theta))
    ]
    assert loss == pytest.approx(independent(theta))
    assert grad == pytest.approx(numerical, abs=1e-8)


@pytest.mark.parametrize("mode", model.MODES)
def test_constant_forecast_returns_training_rate_without_spurious_slope(mode):
    rows = records()
    for r in rows:
        r["probability_buy"] = 0.8
    fitted = model.fit(rows, mode)
    assert model.predict(fitted, [0.8])[0] == pytest.approx(1 / 3, abs=1e-7)
    assert fitted["slope"] == pytest.approx(1, abs=1e-5)


def test_monotone_correction_stops_at_zero_instead_of_reversing_forecasts():
    rows = records()
    for i, row in enumerate(rows):
        row["observed_buy"] = i % 2
        row["probability_buy"] = 0.01 if i % 2 else 0.99
    fitted = model.fit(rows, "monotone")
    assert fitted["slope"] == pytest.approx(0, abs=1e-10)
    assert model.predict(fitted, [0.01, 0.99]) == pytest.approx([0.5, 0.5], abs=1e-7)


def test_common_coverage_preserves_raw_zero_one_and_input_rows():
    rows = records()
    rows[0]["probability_buy"], rows[1]["probability_buy"] = 0.0, 1.0
    original = copy.deepcopy(rows)
    result = model.cross_validate(rows)
    assert rows == original
    assert len(result["models"]) == 6
    assert result["predictions"]["raw"][:2].tolist() == [0.0, 1.0]
    for p in result["predictions"].values():
        assert p.shape == (36,) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


def test_held_component_labels_do_not_change_its_fitted_correction_or_constant():
    rows = records()
    first = model.cross_validate(rows)
    held = first["models"][0]["held_sessions"]
    for row in rows:
        if row["session"] in held:
            row["observed_buy"] = 1 - row["observed_buy"]
    second = model.cross_validate(rows)
    assert first["models"][0] == second["models"][0]


def test_in_sample_gain_can_disappear_and_reverse_out_of_component():
    rows = records()
    for row in rows:
        row["probability_buy"] = 0.8
        row["observed_buy"] = int(row["session"] in {"s3", "s4", "s5"})
    full = model.fit(rows, "monotone")
    result = model.cross_validate(rows)
    y = np.array([r["observed_buy"] for r in rows])
    raw = result["predictions"]["raw"]
    in_sample = model.predict(full, raw)
    assert np.mean((raw - y) ** 2) == pytest.approx(0.34)
    assert np.mean((in_sample - y) ** 2) == pytest.approx(0.25, abs=1e-7)
    assert np.mean((result["predictions"]["monotone"] - y) ** 2) == pytest.approx(0.36, abs=1e-7)


def test_fit_balances_sessions_then_people_not_individual_event_counts():
    rows = records()[:6]
    for i, row in enumerate(rows):
        row["session"] = "a" if i < 4 else "b"
        row["participant"] = "a1" if i == 0 else "a2" if i < 4 else "b1"
        row["target_id"] = row["session"] + ":" + row["order_id"]
        row["observed_buy"] = int(i == 0 or i >= 4)
        row["probability_buy"] = 0.2
    fitted = model.fit(rows, "intercept")
    assert fitted["train_constant"] == pytest.approx(0.75)
    assert model.predict(fitted, [0.2])[0] == pytest.approx(0.75, abs=1e-7)


def test_transitive_shared_people_stay_together():
    rows = records()
    # Linking two sessions leaves five components, still eligible.
    for row in rows:
        if row["session"] == "s1":
            row["participant"] = row["participant"].replace("s1", "s0")
    result = model.cross_validate(rows)
    assert ["s0", "s1"] in result["components"]
    for group in result["models"]:
        for fitted in group["models"].values():
            assert set(fitted["training_sessions"]).isdisjoint(group["held_sessions"])


def test_all_complements_checked_before_any_fit(monkeypatch):
    rows = records()
    for row in rows:
        row["observed_buy"] = int(row["session"] == "s5")
    monkeypatch.setattr(model, "fit", lambda *_args: pytest.fail("fit before admission completed"))
    with pytest.raises(ValueError, match="constant-label"):
        model.cross_validate(rows)


def test_failed_optimizer_is_not_replaced_by_raw_or_constant(monkeypatch):
    monkeypatch.setattr(model, "minimize", lambda *_a, **_k: SimpleNamespace(success=False, x=[0]))
    with pytest.raises(ValueError, match="fit failed"):
        model.cross_validate(records())


@pytest.mark.parametrize("bad", ["duplicate", "identity", "probability", "label", "extra"])
def test_invalid_common_records_are_rejected(bad):
    rows = records()
    if bad == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif bad == "identity":
        rows[0]["target_id"] = "wrong"
    elif bad == "probability":
        rows[0]["probability_buy"] = float("nan")
    elif bad == "label":
        rows[0]["observed_buy"] = True
    else:
        rows[0]["unapproved"] = "field"
    with pytest.raises(ValueError):
        model.cross_validate(rows)
