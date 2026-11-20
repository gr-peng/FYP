"""Guarded TRAIN source bridge for labels, past-only starts, and fixed scales.

The three artifacts have separate interfaces. The generator receives copied
initializations and an empirical model, never target labels or a raw reader.
This builder does not generate paths, choose models, or access held-out dates.
"""

import gzip
import hashlib
import json
import platform
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from behavioral_market.data.archive import write_json
from experiments import crypto_effective_flow_reader as reader
from experiments.crypto_autonomous_metrics import fit_metrics
from experiments.crypto_autonomous_targets import collect_train_day
from experiments.crypto_effective_flow import DAY_US, TRAIN_DAYS
from experiments.crypto_execution_diagnostic import (
    AUDIT_SOURCES,
    audited_inputs,
    day_start_us,
    frozen_identity,
    sha256,
)
from experiments.crypto_flow_template_pipeline import (
    CONFIG_PATH,
    FROZEN_PATHS,
    expected_protocol,
)

TARGET_SOURCES = tuple(dict.fromkeys((
    *FROZEN_PATHS,
    "experiments/crypto_autonomous_targets.py",
    "experiments/crypto_autonomous_target_pipeline.py",
    "scripts/build_crypto_autonomous_targets.py",
)))


def _json_ready(value):
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _publish(output, filename, kind, payload, rows):
    path = output / filename
    write_json(path, _json_ready(payload))
    return {"kind": kind, "path": str(path.resolve()), "sha256": sha256(path), "rows": rows}


def _bind_receipt(path, expected):
    body = path.read_bytes()
    # All frozen producers here use write_json's exact encoding. Bind the
    # bytes whose expected state was checked, not a separate later hash read.
    expected_body = (json.dumps(expected, ensure_ascii=False, indent=2,
                                default=str, allow_nan=False) + "\n").encode()
    if body != expected_body:
        raise ValueError("On-disk preflight receipt differs from verified state")
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


def _preflight_references(output, preflight):
    prepared = dict(preflight, status="running", completed_at=None)
    return [_bind_receipt(output / "source_preflight" / filename, expected)
            for filename, expected in (("manifest.json", preflight),
                                       ("prepared_receipt.json", prepared))]


def _verify_artifacts(references):
    for record in references:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Target/preflight artifact changed before publication or parsing")


def _read_day(inputs, day):
    with zipfile.ZipFile(inputs["orderbook"]["path"]) as archive:
        names = archive.namelist()
        if names != [f"{day}_BTCUSDT_ob500.data"]:
            raise ValueError("Unexpected audited LOB archive member")
        with archive.open(names[0]) as book_handle, gzip.open(
            inputs["trades"]["path"], "rt", encoding="utf-8", newline="",
        ) as trade_handle:
            return collect_train_day(
                map(json.loads, book_handle),
                reader._read_prints(trade_handle, day, day_start_us(day) + DAY_US), day,
            )


def _check_day(day, result):
    start = day_start_us(day)
    if (result.day != day or result.span_us != (start, start + DAY_US)
            or tuple(b.start_ms for b in result.minute_labels)
            != tuple(start // 1000 + i * 60_000 for i in range(1440))
            or tuple(b.start_us for b in result.initializations)
            != tuple(start + h * 3_600_000_000 + 600_000_000 for h in range(24))
            or tuple(b.start_us for b in result.price_blocks)
            != tuple(start + i * 300_000_000 for i in range(288))):
        raise ValueError("Target collector lost or changed the full fixed calendar")


def _day_artifacts(output, result):
    day = result.day
    common = {"day": day, "span_us": result.span_us, "split": "train"}
    records = [
        _publish(output, f"{day}_minute_labels.json", "minute_labels", {
            **common, "labels": [asdict(label) for label in result.minute_labels],
        }, 1440),
        _publish(output, f"{day}_initializations.json", "past_only_initializations", {
            **common, "initializations": [asdict(init) for init in result.initializations],
        }, 24),
        _publish(output, f"{day}_price_blocks.json", "price_only_candidates", {
            **common, "blocks": [asdict(block) for block in result.price_blocks],
        }, 288),
    ]
    counts = {
        "minute_candidates": 1440,
        "complete_minutes": sum(b.complete for b in result.minute_labels),
        "minute_issues": dict(Counter(i for b in result.minute_labels for i in b.issues)),
        "initialization_candidates": 24,
        "complete_initializations": sum(not b.issues for b in result.initializations),
        "initialization_issues": dict(Counter(i for b in result.initializations for i in b.issues)),
        "price_block_candidates": 288,
        "complete_price_blocks": sum(not b.issues and all(v is not None for v in b.returns_bps)
                                     for b in result.price_blocks),
        "observed_prints": result.observed_prints,
        "observed_quantity_units": result.observed_quantity_units,
    }
    return records, counts


def _fence(root, identity, preflight, plan_bytes):
    if (root / CONFIG_PATH).read_bytes() != plan_bytes:
        raise ValueError("Parsed target protocol changed")
    paths = [root / path for path in TARGET_SOURCES]
    if frozen_identity(root, paths)["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Frozen target source changed")
    source_bytes = (root / reader.SOURCE_PATH).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != reader.SOURCE_SHA:
        raise ValueError("Original TRAIN source configuration changed")
    source = json.loads(source_bytes)
    audit_sources = {p: identity["source_sha256"][p] for p in AUDIT_SOURCES}
    for day in TRAIN_DAYS:
        current = audited_inputs(root, source, reader.SOURCE_SHA, day, audit_sources)
        if current != preflight["inputs"][day]:
            raise ValueError("Current target input/audit evidence changed")
    # A change during a later input's hashing must not bypass the code fence.
    if ((root / CONFIG_PATH).read_bytes() != plan_bytes
            or frozen_identity(root, paths)["source_sha256"] != identity["source_sha256"]):
        raise ValueError("Frozen target source changed during input verification")


def build_train_targets(root, output, *, split="train", prepare_only=False):
    if split != "train":
        raise ValueError("Development/event/unknown target access is closed")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    state = {
        "kind": "crypto_autonomous_train_targets", "status": "running",
        "run_id": uuid4().hex, "outputs_published": False, "outputs": [],
        "split": "train", "days": list(TRAIN_DAYS), "day_counts": {},
        "started_at": datetime.now(UTC).isoformat(), "completed_at": None,
        "runtime": {"python": platform.python_version()},
        "interpretation": "TRAIN labels/scales and past-only contexts; no generation or held-out",
    }
    write_json(manifest, state)
    try:
        plan_bytes = (root / CONFIG_PATH).read_bytes()
        plan = json.loads(plan_bytes)
        if plan != expected_protocol():
            raise ValueError("Target protocol differs from the fixed prototype")
        identity = frozen_identity(root, [root / p for p in TARGET_SOURCES])
        plan_sha = hashlib.sha256(plan_bytes).hexdigest()
        if identity["source_sha256"][CONFIG_PATH] != plan_sha:
            raise ValueError("Parsed target protocol differs from frozen bytes")
        preflight = reader.run_extraction(root, output / "source_preflight", split="train",
                                         mode="train", prepare_only=True)
        if (preflight["status"] != "prepared_no_market_rows_read"
                or preflight["days"] != list(TRAIN_DAYS)
                or preflight["outputs_published"] is not False or preflight["outputs"]):
            raise ValueError("Expected a complete all-TRAIN input preflight without parsing")
        for path, digest in preflight["source_sha256"].items():
            if identity["source_sha256"].get(path) != digest:
                raise ValueError("Source preflight and target freeze disagree")
        references = _preflight_references(output, preflight)
        _fence(root, identity, preflight, plan_bytes)
        _verify_artifacts(references)
        state.update(**identity, config=plan, config_sha256=plan_sha,
                     inputs=preflight["inputs"], source_preflight=references)
        prepared_path = output / "prepared_receipt.json"
        write_json(prepared_path, state)
        prepared = _bind_receipt(prepared_path, state)
        state["prepared_receipt_sha256"] = prepared["sha256"]
        all_references = (*references, prepared)
        _verify_artifacts(all_references)
        if prepare_only:
            state["status"] = "prepared_no_market_rows_read"
        else:
            labels = []
            for day in TRAIN_DAYS:
                _verify_artifacts(all_references)
                result = _read_day(preflight["inputs"][day], day)
                _check_day(day, result)
                outputs, counts = _day_artifacts(output, result)
                labels.extend(result.minute_labels)
                state["outputs"].extend(outputs)
                state["day_counts"][day] = counts
                write_json(manifest, state)
            metrics = fit_metrics(tuple(labels), split="train")
            state["outputs"].append(_publish(output, "metrics.json", "frozen_train_metrics",
                                              asdict(metrics), len(labels)))
            _fence(root, identity, preflight, plan_bytes)
            state.update(status="complete", outputs_published=True)
        _verify_artifacts((*all_references, *state["outputs"]))
        state["completed_at"] = datetime.now(UTC).isoformat()
        write_json(manifest, state)
        return state
    except BaseException as error:
        state.update(status="failed", outputs_published=False, error=repr(error),
                     completed_at=datetime.now(UTC).isoformat())
        write_json(manifest, state)
        raise
