import copy
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import nbinom

from experiments import crypto_activity_aux as aux
from experiments import crypto_activity_model as v1

DAY = "2024-10-14"
START = int(datetime.fromisoformat(DAY).replace(tzinfo=UTC).timestamp() * 1000)


def make_joint(n=40):
    buckets = START + np.arange(n) * 1000
    counts = np.arange(n) % 7
    return pd.DataFrame(
        {
            "bucket_ms": buckets,
            "available_at_ms": buckets + 1005,
            "trade_available_at_ms": np.where(counts > 0, buckets + 1000, np.nan),
            "run_id": np.ones(n, dtype=int),
            "trade_count": counts,
            "trade_quantity_btc": counts / 10,
            "spread_bps": np.full(n, 0.02),
            "imbalance_l1": np.full(n, 0.2),
            "visible_bid_levels": np.full(n, 500),
            "visible_ask_levels": np.full(n, 500),
            "bid_depth_5_units": np.full(n, 1000),
            "ask_depth_5_units": np.full(n, 2000),
        }
    )


def samples(joint, delay=0):
    return aux.build_activity_samples(joint, split="train", day=DAY, delay_ms=delay)


@pytest.mark.parametrize("delay,second", [(0, 2), (1000, 3), (5000, 7)])
def test_delayed_target_is_complete_and_activity_is_not_refreshed(delay, second):
    frame, _ = samples(make_joint(), delay)
    first = frame.iloc[0]
    assert first.target_bucket_ms == START + second * 1000
    assert first.activity_bucket_ms == START
    assert (frame.target_bucket_ms >= frame.decision_at_ms).all()
    assert (frame.delayed_activity_available_at_ms <= frame.decision_at_ms).all()
    assert frame.target_bucket_ms.is_unique


@pytest.mark.parametrize("delay", aux.DELAYS_MS)
def test_alignment_matches_frozen_v1_with_whole_day_shift(delay):
    joint = make_joint(60)
    joint.loc[:4, "available_at_ms"] = START + 5500
    joint.loc[8:10, "visible_bid_levels"] = 4
    joint.loc[25:, "run_id"] = 2
    joint = joint.drop(index=[16, 17]).reset_index(drop=True)
    actual, coverage = samples(joint, delay)
    shift = 1730419200000 - START
    shifted = joint.copy()
    for name in ("bucket_ms", "available_at_ms", "trade_available_at_ms"):
        shifted[name] += shift
    expected, expected_coverage = v1.build_activity_samples(
        shifted, split="train", day="2024-11-01", delay_ms=delay
    )
    expected["day"] = DAY
    for name in expected.columns:
        if name.endswith("_ms"):
            expected[name] -= shift
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    assert coverage == expected_coverage
    assert coverage["broken_path_seconds"] > 0
    assert coverage["depth_rejected_seconds"] == 3


def test_boundary_asof_and_target_values_cannot_change_features():
    joint = make_joint()
    joint["available_at_ms"] = joint.bucket_ms + 1000
    frame, _ = samples(joint)
    assert frame.iloc[0].target_bucket_ms == START + 1000
    changed = joint.copy()
    changed.loc[1, "trade_count"] = 1000
    other, _ = samples(changed)
    assert other.iloc[0].target_count == 1000
    names = list(aux.BASELINE_FEATURES + aux.BOOK_FEATURES + aux.NOISE_FEATURES)
    pd.testing.assert_series_equal(frame.iloc[0][names], other.iloc[0][names])


@pytest.mark.parametrize(
    "split,day",
    [("event_test", "2024-11-05"), ("train", "2024-11-01"), ("train", "2024-10-21")],
)
def test_auxiliary_split_gate_runs_before_frame_access(split, day):
    with pytest.raises(ValueError, match="closed"):
        aux.build_activity_samples(None, split=split, day=day, delay_ms=0)


def test_mislabeled_day_and_fractional_publication_are_rejected():
    joint = make_joint()
    joint.loc[0, "bucket_ms"] -= 1000
    with pytest.raises(ValueError, match="UTC day"):
        samples(joint)
    joint = make_joint()
    joint.loc[1, "trade_available_at_ms"] += 0.5
    with pytest.raises(ValueError, match="Trade availability"):
        samples(joint)


def test_noise_serialization_and_reordering_are_exact_and_target_free():
    buckets = np.array([START, START + 1000, START])
    result = aux.noise_features(buckets)
    expected = []
    for index in range(4):
        digest = hashlib.sha256(
            f"crypto-activity-aux-noise-v1|20261011|{START}|{index}".encode("ascii")
        ).digest()
        u1 = ((int.from_bytes(digest[:8], "big") >> 12) + 0.5) / 2**52
        u2 = ((int.from_bytes(digest[8:16], "big") >> 12) + 0.5) / 2**52
        expected.append(np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))
    np.testing.assert_allclose(result[0], expected, rtol=1e-15)
    np.testing.assert_array_equal(result[0], result[2])
    np.testing.assert_array_equal(result[[1, 0]], aux.noise_features(buckets[[1, 0]]))
    np.testing.assert_array_equal(result[:1], aux.noise_features(buckets[:1]))
    assert aux.NOISE_SPEC["seed"] == 20261011


def finite_difference(function, parameters):
    result = []
    for index in range(len(parameters)):
        step = np.zeros(len(parameters))
        step[index] = 1e-5
        result.append((function(parameters + step) - function(parameters - step)) / 2e-5)
    return np.array(result)


def test_truncated_likelihood_gradient_and_full_n_regularization():
    design = np.array([[1, -1.0], [1, 0.2], [1, 0.7], [1, 1.1]])
    y = np.array([1, 2, 4, 12])
    parameters = np.array([0.8, 0.3, np.log(0.7)])
    total, penalty = 11, 0.01
    value, gradient = aux.truncated_nb2_objective(parameters, design, y, total, penalty)
    mean, alpha = np.exp(design @ parameters[:-1]), np.exp(parameters[-1])
    distribution = nbinom(1 / alpha, 1 / (1 + alpha * mean))
    expected = -np.sum(distribution.logpmf(y) - distribution.logsf(0)) / total
    expected += penalty * parameters[1] ** 2 / 2
    assert value == pytest.approx(expected, rel=1e-12)
    numerical = finite_difference(
        lambda theta: aux.truncated_nb2_objective(theta, design, y, total, penalty)[0],
        parameters,
    )
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
    zero_penalty = aux.truncated_nb2_objective(parameters, design, y, total, 0)
    half_weight = aux.truncated_nb2_objective(parameters, design, y, 2 * total, 0)
    assert half_weight[0] == pytest.approx(zero_penalty[0] / 2)
    np.testing.assert_allclose(half_weight[1], zero_penalty[1] / 2)
    np.testing.assert_allclose(gradient - zero_penalty[1], [0, penalty * parameters[1], 0])


def test_logistic_gradient_and_unpenalized_intercept():
    design = np.column_stack([np.ones(5), np.arange(5) - 2])
    positive = np.array([0, 1, 1, 0, 1])
    beta = np.array([0.9, 0.2])
    value, gradient = aux.logistic_objective(beta, design, positive, 0.01)
    assert np.isfinite(value)
    numerical = finite_difference(
        lambda b: aux.logistic_objective(b, design, positive, 0.01)[0], beta
    )
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)
    _, unpenalized = aux.logistic_objective(beta, design, positive, 0)
    np.testing.assert_allclose(gradient - unpenalized, [0, 0.002], atol=1e-15)


def test_hurdle_pmf_has_correct_zero_mass_positive_likelihood_and_mean():
    y = np.arange(1000)
    mu, alpha, probability = 3.0, 0.7, 0.8
    eta, gate = np.log(mu), np.log(probability / (1 - probability))
    actual = np.exp(aux.hurdle_log_probability(y, eta, alpha, gate))
    distribution = nbinom(1 / alpha, 1 / (1 + alpha * mu))
    expected = probability * distribution.pmf(y) / distribution.sf(0)
    expected[0] = 1 - probability
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
    assert actual.sum() == pytest.approx(1)
    assert np.sum(y * actual) == pytest.approx(probability * mu / distribution.sf(0))


@pytest.mark.parametrize("alpha", [0.0001, 0.7, 1000.0])
def test_medians_and_count_bins_match_reference_cdf_and_survival(alpha):
    mu = np.array([0.01, 0.7, 3.0, 60.0, 1000.0])
    eta = np.log(mu)
    distribution = nbinom(1 / alpha, 1 / (1 + alpha * mu))
    np.testing.assert_array_equal(aux.distribution_median(eta, alpha), distribution.ppf(0.5))
    gate = np.array([-4.0, 0.0, 0.1, 1.2, 5.0])
    probability = expit(gate)
    quantiles = distribution.cdf(0) + distribution.sf(0) * (1 - 0.5 / probability)
    expected_median = np.zeros(len(mu))
    active = probability > 0.5
    expected_median[active] = distribution.ppf(np.maximum(quantiles, 0))[active]
    np.testing.assert_array_equal(aux.distribution_median(eta, alpha, gate), expected_median)
    bins = aux.count_bin_probabilities(eta, alpha, gate)
    np.testing.assert_allclose(bins.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(bins[:, 0], 1 - probability, atol=1e-15)
    np.testing.assert_allclose(
        bins[:, -1], probability * distribution.sf(50) / distribution.sf(0), rtol=1e-8
    )


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(7349)
    joint = make_joint(900)
    values = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * 4), len(joint))
    while (values == 0).any():
        mask = values == 0
        values[mask] = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * 4), mask.sum())
    values[rng.uniform(size=len(joint)) < 0.2] = 0
    joint["trade_count"] = values
    joint["trade_quantity_btc"] = values / 10
    joint["trade_available_at_ms"] = joint.bucket_ms + 1000
    joint["imbalance_l1"] = rng.uniform(-1, 1, len(joint))
    joint["bid_depth_5_units"] = rng.uniform(500, 3000, len(joint))
    joint["ask_depth_5_units"] = rng.uniform(500, 3000, len(joint))
    frame, _ = samples(joint)
    models = aux.fit_models(frame, copy.deepcopy(aux.NUMERICAL_SPEC))
    return frame, models


def test_fit_is_complete_json_serializable_and_normalizes_full_train_only(fitted):
    frame, models = fitted
    aux.validate_models(json.loads(json.dumps(models, allow_nan=False)))
    assert set(models) == set(aux.MODEL_FEATURES)
    for model_id, model in models.items():
        values = frame[model["feature_names"]].to_numpy()
        np.testing.assert_allclose(model["training_mean"], values.mean(axis=0))
        expected_scale = values.std(axis=0)
        expected_scale[np.ptp(values, axis=0) == 0] = 1
        np.testing.assert_allclose(model["training_scale"], expected_scale)
        assert model["diagnostics"]["training_samples"] == len(frame)
        if model_id.startswith("H"):
            prediction = aux._predictions(frame, model)
            assert prediction["zero"].mean() == pytest.approx(
                (frame.target_count == 0).mean(), abs=1e-5
            )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.pop("Hnoise"),
        lambda m: m["H0"].update(gate_coefficients=None),
        lambda m: m["N0"].update(feature_names=["wrong"]),
        lambda m: m["N1"].update(alpha=float("nan")),
        lambda m: m["N0"]["training_scale"].__setitem__(0, 0),
        lambda m: m["H1"]["diagnostics"].update(training_samples=100),
        lambda m: m["H0"]["diagnostics"]["count_optimizer"].update(success=False),
        lambda m: m["H0"].pop("training_mean"),
    ],
)
def test_model_validation_rejects_incomplete_or_unaccepted_artifacts(fitted, mutation):
    models = copy.deepcopy(fitted[1])
    mutation(models)
    with pytest.raises(ValueError):
        aux.validate_models(models)


def test_score_never_optimizes_or_updates_models_and_reports_components(fitted, monkeypatch):
    frame, models = fitted
    frozen = json.dumps(models, sort_keys=True, allow_nan=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Scoring must not call an optimizer")

    monkeypatch.setattr(aux, "minimize", forbidden)
    development = frame.copy()
    development["split"], development["day"] = "development", "2024-10-21"
    development["target_bucket_ms"] += 7 * 86_400_000
    development["imbalance_l1"] *= 0.9
    score = aux.score_models(development, models)
    json.dumps(score, allow_nan=False)
    assert json.dumps(models, sort_keys=True, allow_nan=False) == frozen
    assert sum(s["samples"] for s in score["by_day"]) == len(frame)
    assert sum(s["samples"] for s in score["by_utc_block"]) == len(frame)
    for model_id, summary in score["overall"]["models"].items():
        assert summary["mean_log_probability"] == pytest.approx(
            summary["mean_binary_log_probability"]
            + summary["mean_positive_log_probability_contribution"]
        )
        prediction = aux._predictions(development, models[model_id])
        expected = prediction["mean"].sum() / np.exp(prediction["log_positive_probability"]).sum()
        assert summary["predicted_positive_mean_mixture"] == pytest.approx(expected)
        assert summary["observed_to_predicted_positive_mean_ratio"] == pytest.approx(
            summary["observed_positive_mean"] / expected
        )
        assert sum(b["observed_fraction"] for b in summary["count_bins"].values()) == (
            pytest.approx(1)
        )
    assert len(score["overall"]["paired_comparisons"]) == 4


def test_extreme_hurdle_probability_uses_stable_pooled_positive_mean(fitted):
    frame, models = fitted
    model = copy.deepcopy(models["H0"])
    model["count_coefficients"] = [700.0] + [0.0] * len(aux.BASELINE_FEATURES)
    model["gate_coefficients"] = [-1000.0] + [0.0] * len(aux.BASELINE_FEATURES)
    one = frame.iloc[:1].copy()
    one["target_count"] = 1
    prediction = aux._predictions(one, model)
    assert np.exp(prediction["log_positive_probability"])[0] == 0
    summary = aux._summary(one.target_count.to_numpy(), prediction, np.array([0]))
    assert summary["predicted_positive_mean_mixture"] == pytest.approx(np.exp(700))
    assert np.isfinite(summary["observed_to_predicted_positive_mean_ratio"])


def test_fit_rejects_development_and_changed_numerics_before_optimizer(fitted, monkeypatch):
    frame = fitted[0].copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Disallowed fit reached an optimizer")

    monkeypatch.setattr(aux, "minimize", forbidden)
    frame["split"] = "development"
    with pytest.raises(ValueError, match="train only"):
        aux.fit_models(frame, aux.NUMERICAL_SPEC)
    with pytest.raises(ValueError, match="specification"):
        aux.fit_models(fitted[0], {**aux.NUMERICAL_SPEC, "l2_penalty": 0.02})


def test_optimizer_failure_is_hard_failure_without_fallback(fitted, monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(success=False, message="synthetic rejected optimizer")

    monkeypatch.setattr(aux, "minimize", failed)
    with pytest.raises(aux.FitError, match="synthetic rejected"):
        aux.fit_models(fitted[0], aux.NUMERICAL_SPEC)
    assert len(calls) == 1


def test_single_class_or_too_few_positive_training_rows_fail(fitted):
    for target in (0, 1):
        frame = fitted[0].copy()
        frame["target_count"] = target
        with pytest.raises(aux.FitError, match="both zero/nonzero"):
            aux.fit_models(frame, aux.NUMERICAL_SPEC)


def test_reused_numerical_source_is_hash_bound(monkeypatch):
    monkeypatch.setattr(aux, "NB2_HELPER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source hash changed"):
        aux.assert_helper_source()
