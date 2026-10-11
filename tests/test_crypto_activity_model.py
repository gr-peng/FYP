import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from experiments import crypto_activity_model as activity
from scripts import run_crypto_activity_model as runner

DAY = "2024-11-01"
START = 1730419200000
PROTOCOL = Path(__file__).resolve().parents[1] / "config/crypto_activity_probe_20261011.json"


def make_joint(seconds=None, *, day=DAY, runs=None):
    seconds = list(range(30) if seconds is None else seconds)
    start = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    buckets = start + np.array(seconds, dtype="int64") * 1000
    n = len(seconds)
    counts = np.arange(n) % 7
    return pd.DataFrame(
        {
            "bucket_ms": buckets,
            "available_at_ms": buckets + 1005,
            "trade_available_at_ms": np.where(counts > 0, buckets + 1000, np.nan),
            "run_id": [1] * n if runs is None else runs,
            "trade_count": counts,
            "trade_quantity_btc": counts / 10,
            "spread_bps": [0.02] * n,
            "imbalance_l1": [0.2] * n,
            "visible_bid_levels": [500] * n,
            "visible_ask_levels": [500] * n,
            "bid_depth_5_units": [1000] * n,
            "ask_depth_5_units": [2000] * n,
        }
    )


def samples(joint, delay=0):
    return activity.build_activity_samples(joint, split="train", day=DAY, delay_ms=delay)


@pytest.mark.parametrize("delay,target_second", [(0, 2), (1000, 3), (5000, 7)])
def test_target_is_first_full_second_after_delayed_availability(delay, target_second):
    frame, _ = samples(make_joint(), delay)
    first = frame.iloc[0]
    assert first.signal_bucket_ms == START
    assert first.target_bucket_ms == START + target_second * 1000
    assert first.target_end_ms == first.target_bucket_ms + 1000
    assert first.activity_bucket_ms == START  # No refreshing activity during extra latency.
    assert (frame.target_bucket_ms >= frame.decision_at_ms).all()
    assert (frame.delayed_activity_available_at_ms <= frame.decision_at_ms).all()
    assert frame.target_bucket_ms.is_unique


def test_exact_boundary_decision_may_target_second_beginning_there():
    joint = make_joint()
    joint["available_at_ms"] = joint.bucket_ms + 1000
    frame, _ = samples(joint)
    assert frame.iloc[0].target_bucket_ms == START + 1000
    assert frame.iloc[0].target_count == joint.iloc[1].trade_count
    assert frame.iloc[0].log1p_last_trade_count == 0  # Target count cannot enter features.


def test_activity_asof_can_be_newer_than_delayed_book_but_never_newer_than_availability():
    joint = make_joint()
    joint.loc[:4, "available_at_ms"] = START + 5500
    joint.loc[1:4, "visible_bid_levels"] = 4  # These cannot supply candidate book signals.
    frame, counts = samples(joint)
    first = frame[frame.signal_bucket_ms == START].iloc[0]
    assert first.activity_bucket_ms == START + 4000
    assert first.activity_available_at_ms == START + 5000
    assert first.target_bucket_ms == START + 6000
    assert first.log1p_last_trade_count == pytest.approx(np.log1p(4))
    assert counts["depth_rejected_seconds"] == 4


def test_duplicate_targets_keep_latest_decision_then_latest_signal_without_target_selection():
    joint = make_joint()
    joint.loc[:4, "available_at_ms"] = START + 5500
    frame, counts = samples(joint)
    first = frame[frame.target_bucket_ms == START + 6000]
    assert len(first) == 1 and first.iloc[0].signal_bucket_ms == START + 4000
    assert counts["duplicate_target_signals_removed"] == 4
    changed = joint.copy()
    changed.loc[6, "trade_count"] = 9999
    other, _ = samples(changed)
    first, other_first = first.iloc[0], other[other.target_bucket_ms == START + 6000].iloc[0]
    assert other_first.target_count == 9999
    for name in (*activity.BASELINE_FEATURES, *activity.BOOK_FEATURES, "signal_bucket_ms"):
        assert first[name] == other_first[name]


def test_delay_path_cannot_cross_missing_joint_second_or_book_segment():
    joint = make_joint([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11])
    frame, counts = samples(joint, 5000)
    assert not frame.signal_bucket_ms.isin([START, START + 1000, START + 2000]).any()
    assert counts["broken_path_seconds"] == 3
    joint = make_joint(runs=[1] * 5 + [2] * 25)
    frame, counts = samples(joint)
    assert START + 3000 not in frame.signal_bucket_ms.to_numpy()
    assert counts["broken_path_seconds"] == 2


def test_depth_eligibility_is_common_and_rejections_are_accounted_for():
    joint = make_joint()
    joint.loc[0, "visible_bid_levels"] = 4
    joint.loc[1, "visible_ask_levels"] = 3
    frame, counts = samples(joint)
    assert not frame.signal_bucket_ms.isin([START, START + 1000]).any()
    assert counts["depth_rejected_seconds"] == 2
    assert counts["depth_rejected_fraction"] == pytest.approx(2 / len(joint))
    assert sum(
        counts[key]
        for key in (
            "depth_rejected_seconds",
            "missing_target_seconds",
            "broken_path_seconds",
            "duplicate_target_signals_removed",
            "samples",
        )
    ) == len(joint)


def test_mislabeled_day_and_cross_midnight_buckets_are_rejected():
    with pytest.raises(ValueError, match="UTC day"):
        samples(make_joint(day="2024-11-02"))
    with pytest.raises(ValueError, match="UTC day"):
        samples(make_joint([86398, 86399, 86400]))


def test_fractional_trade_availability_is_not_silently_truncated():
    joint = make_joint()
    joint.loc[1, "trade_available_at_ms"] += 0.5
    with pytest.raises(ValueError, match="Trade availability"):
        samples(joint)


@pytest.mark.parametrize(
    "split,day", [("event_test", "2024-11-05"), ("train", "2024-11-05"), ("all", DAY)]
)
def test_event_or_wrong_split_rejected_before_frame_access(split, day):
    with pytest.raises(ValueError, match="closed"):
        activity.build_activity_samples(None, split=split, day=day, delay_ms=0)


def test_nb2_log_score_matches_reference_distribution_including_zero_and_tail_counts():
    y, mu, alpha = np.array([0, 3, 100]), np.array([0.5, 2.0, 30.0]), 0.7
    actual = activity.nb2_log_probability(y, np.log(mu), alpha)
    expected = nbinom.logpmf(y, 1 / alpha, (1 / alpha) / (1 / alpha + mu))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.exp(activity.nb2_log_probability(np.arange(1000), np.log(3.0), 1.5)).sum() == (
        pytest.approx(1.0)
    )


def test_nb2_joint_coefficient_and_dispersion_gradient_matches_finite_difference():
    design = np.array([[1, -1, 0.5], [1, 0, -1], [1, 1, 0], [1, 2, 1]], dtype=float)
    y = np.array([0, 2, 10, 100])
    parameters = np.array([0.4, 0.3, -0.2, np.log(0.7)])
    _, gradient = activity.nb2_objective(parameters, design, y, 0.01)
    numerical = []
    for i in range(len(parameters)):
        offset = np.eye(len(parameters))[i] * 1e-5
        high = activity.nb2_objective(parameters + offset, design, y, 0.01)[0]
        low = activity.nb2_objective(parameters - offset, design, y, 0.01)[0]
        numerical.append((high - low) / 2e-5)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-7)


def synthetic_regression(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mean = np.exp(1.2 + 0.6 * x)
    alpha = 0.5
    y = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mean))
    return pd.DataFrame(
        {
            "state": x,
            "constant": np.full(n, 0.1),
            "target_count": y,
            "split": "train",
            "day": DAY,
            "target_bucket_ms": START + np.arange(n) * 1000,
        }
    )


def test_nb2_recovers_overdispersed_synthetic_relation_with_train_only_standardization():
    training = synthetic_regression()
    model = activity.fit_nb2(
        training, ["state", "constant"], activity.read_protocol(PROTOCOL)["model"]
    )
    assert model.alpha == pytest.approx(0.5, abs=0.12)
    assert model.coefficients[1] / model.scale[0] == pytest.approx(0.6, abs=0.07)
    assert model.coefficients[0] - model.coefficients[1] * model.mean[0] / model.scale[0] == (
        pytest.approx(1.2, abs=0.1)
    )
    assert model.diagnostics["constant_training_features"] == ["constant"]
    np.testing.assert_allclose(model.mean, training[["state", "constant"]].mean())
    np.testing.assert_allclose(model.scale, [training.state.std(ddof=0), 1])
    original = copy.deepcopy(model.to_dict())
    model.log_mean(pd.DataFrame({"state": [500.0], "constant": [100.0]}))
    assert model.to_dict() == original  # Development features never refit scaling.
    with pytest.raises(ValueError, match="training rows only"):
        activity.fit_nb2(
            training.assign(split="development"),
            ["state"],
            activity.read_protocol(PROTOCOL)["model"],
        )


def test_proper_score_detects_known_synthetic_incremental_signal_and_common_calibration():
    training, development = synthetic_regression(), synthetic_regression(seed=19, n=700)
    settings = activity.read_protocol(PROTOCOL)["model"]
    baseline = activity.fit_nb2(training, ["constant"], settings)
    book = activity.fit_nb2(training, ["constant", "state"], settings)
    report = activity.evaluate_pair(development, baseline, book, block_hours=3)
    assert report["overall"]["book_minus_baseline_log_score"] > 0.05
    assert report["overall"]["samples"] == 700
    assert report["by_day"][0]["samples"] == 700
    assert report["by_utc_block"][0]["samples"] == 700
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("success", [False, True])
def test_failed_or_false_success_optimizer_never_silently_falls_back(monkeypatch, success):
    calls = []

    def failed(fun, initial, **kwargs):
        calls.append(1)
        return SimpleNamespace(success=success, x=np.zeros_like(initial), message="forced failure")

    monkeypatch.setattr(activity, "minimize", failed)
    with pytest.raises(activity.FitError, match="optimizer"):
        activity.fit_nb2(
            synthetic_regression(n=100), ["state"], activity.read_protocol(PROTOCOL)["model"]
        )
    assert len(calls) == 1


def test_all_zero_training_counts_fail_explicitly():
    with pytest.raises(activity.FitError, match="all-zero"):
        activity.fit_nb2(
            synthetic_regression(n=100).assign(target_count=0),
            ["state"],
            activity.read_protocol(PROTOCOL)["model"],
        )


def test_protocol_rejects_test_access_or_posthoc_feature_and_regularization_changes(tmp_path):
    original = activity.read_protocol(PROTOCOL)
    changes = [
        lambda p: p["splits"].update(event_test=["2024-11-05"]),
        lambda p: p["additional_book_features"].append("future_quantity"),
        lambda p: p["model"].update(l2_penalty=0.005),
    ]
    path = tmp_path / "modified.json"
    for change in changes:
        protocol = copy.deepcopy(original)
        change(protocol)
        path.write_text(json.dumps(protocol))
        with pytest.raises(ValueError):
            activity.read_protocol(path)


def test_run_uses_authorized_reader_and_only_train_rows_for_both_fits(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"splits": {**activity.FIXED_SPLITS, "event_test": ["2024-11-05"]}})
    )
    protocol = activity.read_protocol(PROTOCOL)
    protocol.update(
        source_config="source.json",
        source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    protocol_path = tmp_path / "probe.json"
    protocol_path.write_text(json.dumps(protocol))
    reads, fits, prepares, joins = [], [], [], []

    def read(root, config, split, day, kind):
        reads.append((split, day, kind))
        return pd.DataFrame({"day": [day]}), {
            "audit_run_id": "current-successful-audit",
            "input_sha256": "raw-sha",
            "features_path": f"{day}-{kind}.parquet",
            "features_sha256": "feature-sha",
            "processing_source_sha256": {"auditor.py": "auditor-sha"},
        }

    def prepare(raw, audit, day):
        prepares.append(day)
        return make_joint(day=day)

    def join(book, trades, audit):
        joins.append(1)
        return book, {"joint_covered_seconds": len(book)}

    def fit(training, names, settings):
        fits.append(training.copy())
        assert set(training.split) == {"train"}
        return activity.NB2Fit(
            tuple(names),
            np.zeros(len(names)),
            np.ones(len(names)),
            np.r_[np.log(2.0), np.zeros(len(names))],
            0.5,
            {"success": True, "synthetic_mock": True},
        )

    monkeypatch.setattr(activity.crypto_features, "read_feature_day", read)
    monkeypatch.setattr(activity.crypto_profile, "prepare_book", prepare)
    monkeypatch.setattr(activity.crypto_profile, "covered_trade_seconds", join)
    monkeypatch.setattr(activity, "fit_nb2", fit)
    report = activity.run_probe(tmp_path, protocol_path)
    assert len(reads) == 8 and len(prepares) == len(joins) == 4
    assert not any(split == "event_test" for split, _, _ in reads)
    assert len(fits) == 6
    for first, second in zip(fits[::2], fits[1::2], strict=True):
        pd.testing.assert_frame_equal(first, second)  # Same fitting rows for both models.
    assert [s["additional_delay_ms"] for s in report["scenarios"]] == [0, 1000, 5000]
    assert report["sources"][0]["audit_run_id"] == "current-successful-audit"
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    assert report["protocol_sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    json.dumps(report, allow_nan=False)


def test_cli_refuses_overwrite_before_loading_or_fitting(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original evidence")

    def forbidden(*args):
        pytest.fail("No data read or fit should run for an existing output")

    monkeypatch.setattr(runner, "run_probe", forbidden)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--protocol", str(tmp_path / "missing"), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text() == "original evidence"


def test_cli_cannot_overwrite_evidence_created_while_fitting(tmp_path, monkeypatch):
    output = tmp_path / "report.json"

    def concurrent_evidence(*args):
        output.write_text("another completed run")
        return {"source_sha256": {}, "status": "diagnostic_computed", "scenarios": []}

    monkeypatch.setattr(runner, "run_probe", concurrent_evidence)
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(output)])
    assert output.read_text() == "another completed run"
    assert list(tmp_path.iterdir()) == [output]
