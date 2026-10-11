import hashlib
import json

import pandas as pd
import pytest

from experiments.crypto_features import read_feature_day
from scripts import audit_crypto_microstructure as runner
from scripts.audit_crypto_microstructure import audit_task


def fixture(tmp_path):
    config = {
        "study_id": "fixture",
        "venue": "bybit",
        "market": "linear_usdt_perpetual",
        "symbol": "BTCUSDT",
        "splits": {"train": ["2024-11-01"]},
        "urls": {"orderbook": "https://example.com/{date}.zip"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    features = tmp_path / "book.parquet"
    pd.DataFrame({"bucket_ms": [1000], "available_at_ms": [2050]}).to_parquet(features)
    raw = tmp_path / "data/raw/fixture/orderbook/2024-11-01.zip"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fixture raw bytes")
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    meta = {
        "sha256": raw_hash,
        "bytes": raw.stat().st_size,
        "config_sha256": config_hash,
        "url": "https://example.com/2024-11-01.zip",
        "split": "train",
        "day": "2024-11-01",
        "kind": "orderbook",
        **{key: config[key] for key in ["venue", "market", "symbol"]},
    }
    raw.with_name(raw.name + ".metadata.json").write_text(json.dumps(meta))
    report = tmp_path / "outputs/fixture/integrity/2024-11-01_orderbook.json"
    report.parent.mkdir(parents=True)
    state = {
        "status": "complete",
        "audit_run_id": "fixture-success",
        "features_published": True,
        "config_sha256": config_hash,
        "split": "train",
        "day": "2024-11-01",
        "kind": "orderbook",
        "input_sha256": raw_hash,
        "zip_crc_verified": True,
        "features_path": "book.parquet",
        "feature_rows": 1,
        "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
    }
    report.write_text(json.dumps(state))
    return path, report, state, features


def test_failed_audit_blocks_old_file_even_when_still_present(tmp_path):
    config, report, state, features = fixture(tmp_path)
    assert len(read_feature_day(tmp_path, config, "train", "2024-11-01", "orderbook")[0]) == 1
    state["features_published"] = False
    report.write_text(json.dumps(state))
    assert features.exists()
    with pytest.raises(ValueError, match="Current audit"):
        read_feature_day(tmp_path, config, "train", "2024-11-01", "orderbook")


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "wrong-source"),
        ("config_sha256", "wrong-config"),
        ("features_sha256", "wrong-features"),
        ("split", "development"),
    ],
)
def test_manifest_identity_drift_is_rejected(tmp_path, field, value):
    config, report, state, _ = fixture(tmp_path)
    state[field] = value
    report.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        read_feature_day(tmp_path, config, "train", "2024-11-01", "orderbook")


def test_event_test_gate_precedes_any_file_read(tmp_path):
    with pytest.raises(ValueError, match="closed"):
        read_feature_day(
            tmp_path, tmp_path / "nonexistent", "event_test", "2024-11-05", "orderbook"
        )
    with pytest.raises(ValueError, match="frozen"):
        audit_task({}, "unused", "event_test", "2024-11-05", "orderbook", True)


@pytest.mark.parametrize("failure", ["bytes", "metadata_missing", "metadata_identity"])
def test_preflight_failure_revokes_old_successful_receipt(tmp_path, monkeypatch, failure):
    config_path, report, state, features = fixture(tmp_path)
    raw = tmp_path / "data/raw/fixture/orderbook/2024-11-01.zip"
    metadata = raw.with_name(raw.name + ".metadata.json")
    if failure == "bytes":
        raw.write_bytes(b"corrupted raw")
    elif failure == "metadata_missing":
        metadata.unlink()
    else:
        meta = json.loads(metadata.read_text())
        meta["day"] = "2024-11-02"
        metadata.write_text(json.dumps(meta))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit_task(
            json.loads(config_path.read_text()),
            state["config_sha256"],
            "train",
            "2024-11-01",
            "orderbook",
            True,
        )
    failed = json.loads(report.read_text())
    assert failed["status"] == "failed" and not failed["features_published"]
    assert failed["audit_run_id"] != state["audit_run_id"]
    assert features.exists()
    with pytest.raises(ValueError, match="Current audit"):
        read_feature_day(tmp_path, config_path, "train", "2024-11-01", "orderbook")


def test_raw_bytes_changed_without_rerun_are_rejected(tmp_path):
    config, _, _, _ = fixture(tmp_path)
    raw = tmp_path / "data/raw/fixture/orderbook/2024-11-01.zip"
    raw.write_bytes(b"unnoticed mutation")
    with pytest.raises(ValueError, match="Actual raw bytes"):
        read_feature_day(tmp_path, config, "train", "2024-11-01", "orderbook")


def test_running_receipt_blocks_reads_after_interrupted_audit(tmp_path):
    config, report, state, _ = fixture(tmp_path)
    state["status"] = "running"
    report.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Current audit"):
        read_feature_day(tmp_path, config, "train", "2024-11-01", "orderbook")
