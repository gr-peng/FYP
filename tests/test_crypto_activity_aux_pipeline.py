import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import crypto_activity_aux_pipeline as pipeline


def test_evidence_publication_is_immutable_and_detects_tampering(tmp_path):
    path = tmp_path / "model.json"
    pipeline.publish(path, {"example": [1, 2, 3]})
    before = path.read_bytes()
    assert pipeline.read_evidence(path) == {"example": [1, 2, 3]}
    with pytest.raises(FileExistsError):
        pipeline.publish(path, {"example": [3]})
    assert path.read_bytes() == before
    edited = json.loads(before)
    edited["example"] = [100]
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="digest"):
        pipeline.read_evidence(path)


def test_publication_rejects_nan_and_caller_digest(tmp_path):
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        pipeline.publish(path, {"x": float("nan")})
    with pytest.raises(ValueError, match="Caller"):
        pipeline.publish(path, {"content_sha256": "invented"})
    assert not path.exists()


@pytest.mark.parametrize(
    "phase,split,day",
    [
        ("fit", "development", "2024-10-21"),
        ("fit", "event_test", "2024-11-05"),
        ("fit", "train", "2024-11-01"),
        ("evaluate", "train", "2024-10-14"),
        ("evaluate", "development", "2024-11-04"),
        ("unknown", "train", "2024-10-14"),
    ],
)
def test_phase_gate_rejects_before_any_reader(monkeypatch, phase, split, day):
    def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized reader reached")

    monkeypatch.setattr(pipeline.crypto_features, "read_feature_day", forbidden)
    with pytest.raises(ValueError, match="phase"):
        pipeline.phase_feature_day(
            Path("missing"), Path("missing"), phase=phase, split=split, day=day, kind="orderbook"
        )


def test_phase_gate_delegates_only_the_allowed_day(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline.crypto_features, "read_feature_day", lambda *args: calls.append(args) or "ok"
    )
    assert (
        pipeline.phase_feature_day(
            Path("a"), Path("b"), phase="fit", split="train", day="2024-10-14", kind="trades"
        )
        == "ok"
    )
    assert calls == [(Path("a"), Path("b"), "train", "2024-10-14", "trades")]


def test_changed_numeric_spec_rejected_before_source_open(tmp_path):
    protocol = copy.deepcopy(pipeline.expected_protocol())
    protocol["model"]["l2_penalty"] = 0.1
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Specification"):
        pipeline.read_protocol(tmp_path, path)


def test_missing_calendar_day_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="calendar"):
        pipeline.scenario_samples({}, "train", 0)


def test_source_or_runtime_change_rejected(monkeypatch):
    frozen = {
        "protocol_sha256": "p",
        "source_sha256": {"file": "f"},
        "source_config_sha256": "c",
        "runtime": {"python": "x"},
    }
    monkeypatch.setattr(pipeline, "identity", lambda *args: frozen)
    pipeline.assert_identity(None, None, frozen)
    altered = copy.deepcopy(frozen)
    altered["source_sha256"]["file"] = "new"
    with pytest.raises(ValueError, match="changed"):
        pipeline.assert_identity(None, None, altered)


def test_review_rejects_wrong_model_hash_before_access(monkeypatch, tmp_path):
    model = tmp_path / "model.json"
    model.write_text("frozen model bytes")
    artifact = {
        "protocol_sha256": "p",
        "source_sha256": {"source": "s"},
        "frozen_at": "2026-10-11T00:00:00+00:00",
    }
    monkeypatch.setattr(pipeline, "verify_models", lambda *args: artifact)
    review = {
        "kind": "independent_auxiliary_activity_prefit_and_freeze_review",
        "verdict": "pass_for_development_diagnostic",
        "model_artifact_sha256": "wrong",
        "protocol_sha256": "p",
        "source_sha256": {"source": "s"},
        "aux_development_market_accessed": False,
        "reviewer": "independent",
        "checks": ["numerics"],
        "reviewed_at": "2026-10-11T00:01:00+00:00",
    }
    path = tmp_path / "review.json"
    pipeline.publish(path, review)
    with pytest.raises(ValueError, match="review"):
        pipeline.verify_review(tmp_path, None, model, path)


def test_incomplete_or_reordered_scenario_artifact_fails(monkeypatch, tmp_path):
    artifact = {
        "kind": "frozen_auxiliary_activity_models",
        "status": "all_15_fitted_train_only",
        "protocol": {"fixed": True},
        "scenarios": [{"additional_delay_ms": 0}, {"additional_delay_ms": 1000}],
    }
    monkeypatch.setattr(pipeline, "read_protocol", lambda *args: ({"fixed": True}, {}))
    monkeypatch.setattr(pipeline, "assert_identity", lambda *args: None)
    path = tmp_path / "model.json"
    pipeline.publish(path, artifact)
    with pytest.raises(ValueError, match="three delays"):
        pipeline.verify_models(tmp_path, None, path)


def test_review_cannot_precede_fitted_artifact(monkeypatch, tmp_path):
    model = tmp_path / "model.json"
    model.write_text("frozen model bytes")
    artifact = {
        "protocol_sha256": "p",
        "source_sha256": {"source": "s"},
        "started_at": "2026-10-11T00:00:00+00:00",
        "frozen_at": "2026-10-11T00:02:00+00:00",
    }
    monkeypatch.setattr(pipeline, "verify_models", lambda *args: artifact)
    path = tmp_path / "review.json"
    pipeline.publish(
        path,
        {
            "kind": "independent_auxiliary_activity_prefit_and_freeze_review",
            "verdict": "pass_for_development_diagnostic",
            "model_artifact_sha256": pipeline.sha(model),
            "protocol_sha256": "p",
            "source_sha256": {"source": "s"},
            "aux_development_market_accessed": False,
            "reviewer": "independent",
            "checks": ["a"],
            "reviewed_at": "2026-10-11T00:01:00+00:00",
        },
    )
    with pytest.raises(ValueError, match="predates"):
        pipeline.verify_review(tmp_path, None, model, path)


@pytest.mark.parametrize(
    "timestamps",
    [
        ("2026-01-01T00:00:00", "2026-01-01T00:00:01+00:00"),
        ("2026-01-01T00:00:00+08:00", "2026-01-01T00:00:01+00:00"),
        ("2026-01-01T00:00:01+00:00", "2026-01-01T00:00:00+00:00"),
    ],
)
def test_chronology_rejects_naive_non_utc_or_reversed_dates(timestamps):
    with pytest.raises(ValueError):
        pipeline.check_chronology(*timestamps)


def test_reader_race_cannot_bind_old_features_to_new_failed_receipt(monkeypatch, tmp_path):
    day = pipeline.core.FIXED_SPLITS["train"][0]
    path = (
        tmp_path / "outputs/crypto_microstructure_aux_20261011/integrity" / f"{day}_orderbook.json"
    )
    path.parent.mkdir(parents=True)
    old = {"status": "complete", "audit_run_id": "old"}
    path.write_text(json.dumps(old))

    def raced_reader(*args, **kwargs):
        path.write_text(json.dumps({"status": "failed", "audit_run_id": "new"}))
        return None, old

    monkeypatch.setattr(pipeline, "phase_feature_day", raced_reader)
    with pytest.raises(ValueError, match="changed during feature read"):
        pipeline.read_joint_days(tmp_path, phase="fit")


def test_export_receipt_chain_rejects_later_reaudit(tmp_path):
    base = tmp_path / "outputs/crypto_microstructure_aux_20261011/integrity"
    base.mkdir(parents=True)
    results = []
    for day in pipeline.core.FIXED_SPLITS["development"]:
        for kind in ("orderbook", "trades"):
            item = {
                "day": day,
                "kind": kind,
                "status": "complete",
                "features_published": True,
                "split": "development",
                "audit_run_id": "original",
                "config_sha256": pipeline.SOURCE_CONFIG_SHA256,
            }
            (base / f"{day}_{kind}.json").write_text(json.dumps(item))
            results.append(item)
    batch = {"status": "complete", "audit_run_id": "original", "results": results}
    assert len(pipeline.exported_receipts(tmp_path, batch)) == 14
    changed = {**results[0], "audit_run_id": "later"}
    (base / f"{changed['day']}_{changed['kind']}.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="does not belong"):
        pipeline.exported_receipts(tmp_path, batch)


def synthetic_joint(day):
    rng = np.random.default_rng(971)
    n = 160
    start = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    buckets = start + np.arange(n, dtype="int64") * 1000
    counts = rng.negative_binomial(1.2, 0.25, n)
    return pd.DataFrame(
        {
            "bucket_ms": buckets,
            "available_at_ms": buckets + 1010,
            "trade_available_at_ms": buckets + 1000,
            "run_id": np.ones(n, dtype=int),
            "trade_count": counts,
            "trade_quantity_btc": counts * 0.02,
            "spread_bps": 0.014 + rng.random(n) * 0.001,
            "imbalance_l1": rng.uniform(-0.5, 0.5, n),
            "visible_bid_levels": np.full(n, 500),
            "visible_ask_levels": np.full(n, 500),
            "bid_depth_5_units": rng.integers(1000, 3000, n),
            "ask_depth_5_units": rng.integers(1000, 3000, n),
        }
    )


@pytest.fixture(scope="module")
def frozen_synthetic_models():
    frames = {day: synthetic_joint(day) for day in pipeline.core.FIXED_SPLITS["train"]}
    samples, _ = pipeline.scenario_samples(frames, "train", 0)
    return pipeline.core.fit_models(samples, pipeline.NUMERIC_SPEC)


def test_evaluate_real_scoring_never_calls_fitting_or_optimizer(
    monkeypatch, tmp_path, frozen_synthetic_models
):
    artifact = {
        "scenarios": [
            {"additional_delay_ms": delay, "models": frozen_synthetic_models}
            for delay in (0, 1000, 5000)
        ]
    }
    model, review, access = [tmp_path / name for name in ("model", "review", "access")]
    for path in (model, review, access):
        path.write_text("immutable fixture")
    monkeypatch.setattr(pipeline, "verify_review", lambda *args: (artifact, {}))
    monkeypatch.setattr(pipeline, "verify_access", lambda *args: {})
    monkeypatch.setattr(pipeline, "committed_identity", lambda *args: "fixture-commit")
    monkeypatch.setattr(pipeline, "assert_identity", lambda *args: None)
    monkeypatch.setattr(pipeline, "verify_provenance_unchanged", lambda *args: None)
    monkeypatch.setattr(pipeline, "identity", lambda *args: {"fixture_identity": True})
    phases = []

    def reader(root, *, phase):
        phases.append(phase)
        assert phase == "evaluate"
        return (
            {day: synthetic_joint(day) for day in pipeline.core.FIXED_SPLITS["development"]},
            [],
            [],
        )

    monkeypatch.setattr(pipeline, "read_joint_days", reader)

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation reached fitting or optimization")

    monkeypatch.setattr(pipeline.core, "fit_models", forbidden)
    monkeypatch.setattr(pipeline.core, "minimize", forbidden)
    monkeypatch.setattr(pipeline.core.v1, "fit_nb2", forbidden)
    monkeypatch.setattr(pipeline.core.v1, "minimize", forbidden)
    output = tmp_path / "scores.json"
    pipeline.evaluate(tmp_path, None, model, review, access, output)
    report = pipeline.read_evidence(output)
    assert report["status"] == "diagnostic_computed_no_refit"
    assert phases == ["evaluate"]
    assert len(report["scenarios"]) == 3
    for scenario in report["scenarios"]:
        assert set(scenario["scores"]["overall"]["models"]) == set(pipeline.core.MODEL_FEATURES)
        assert len(scenario["scores"]["by_day"]) == 7


def test_fit_orchestration_loads_train_only(monkeypatch, tmp_path, frozen_synthetic_models):
    protocol = pipeline.expected_protocol()
    monkeypatch.setattr(pipeline, "read_protocol", lambda *args: (protocol, {}))
    monkeypatch.setattr(pipeline, "committed_identity", lambda *args: "fixture-commit")
    monkeypatch.setattr(pipeline, "identity", lambda *args: {"fixture_identity": True})
    monkeypatch.setattr(pipeline, "assert_identity", lambda *args: None)
    monkeypatch.setattr(pipeline, "verify_provenance_unchanged", lambda *args: None)
    phases, fitted_splits = [], []

    def reader(root, *, phase):
        phases.append(phase)
        assert phase == "fit"
        return ({day: synthetic_joint(day) for day in pipeline.core.FIXED_SPLITS["train"]}, [], [])

    def fit_only_train(samples, spec):
        fitted_splits.append(set(samples.split))
        assert sorted(set(samples.day)) == pipeline.core.FIXED_SPLITS["train"]
        return frozen_synthetic_models

    monkeypatch.setattr(pipeline, "read_joint_days", reader)
    monkeypatch.setattr(pipeline.core, "fit_models", fit_only_train)
    output = tmp_path / "models.json"
    pipeline.fit(tmp_path, None, output)
    assert phases == ["fit"] and fitted_splits == [{"train"}] * 3
    assert pipeline.read_evidence(output)["status"] == "all_15_fitted_train_only"
