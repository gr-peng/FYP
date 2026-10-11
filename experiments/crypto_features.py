"""Consume only features published by the current successful per-file audit.

File existence alone is insufficient: failed reprocessing may leave a previous
valid parquet in place. The audit manifest binds input, split and output hashes.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


def read_feature_day(root: Path, config_path: Path, split: str, day: str, kind: str):
    if split not in {"train", "development"}:
        raise ValueError("Event-test feature access is closed pending evaluation freeze")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if day not in config["splits"][split] or kind not in {"orderbook", "trades"}:
        raise ValueError("Day or feature kind is outside the frozen selection")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    report = root / "outputs" / config["study_id"] / "integrity" / f"{day}_{kind}.json"
    audit = json.loads(report.read_text())
    if (
        audit.get("status") != "complete"
        or not audit.get("audit_run_id")
        or not audit.get("features_published")
        or audit["config_sha256"] != config_hash
        or audit["split"] != split
        or audit["day"] != day
        or audit["kind"] != kind
    ):
        raise ValueError("Current audit does not authorize these features")
    if kind == "trades" and audit["quality"] != "valid":
        raise ValueError("Invalid trade tape cannot supply model features")
    if kind == "orderbook" and not audit["zip_crc_verified"]:
        raise ValueError("Orderbook archive CRC not verified")
    url = config["urls"][kind].format(date=day)
    raw = root / "data/raw" / config["study_id"] / kind / url.rsplit("/", 1)[-1]
    meta = json.loads(raw.with_name(raw.name + ".metadata.json").read_text())
    if meta["sha256"] != audit["input_sha256"] or meta["config_sha256"] != config_hash:
        raise ValueError("Feature input no longer matches the archived source identity")
    with raw.open("rb") as source:
        raw_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if raw_hash != audit["input_sha256"] or raw.stat().st_size != meta["bytes"]:
        raise ValueError("Actual raw bytes no longer match the successful audit")
    path = root / audit["features_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != audit["features_sha256"]:
        raise ValueError("Published feature hash differs")
    frame = pd.read_parquet(path)
    if len(frame) != audit["feature_rows"]:
        raise ValueError("Published row count differs")
    return frame, audit
