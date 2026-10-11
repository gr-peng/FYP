"""Frozen train-only fitting, reviewed export, and optimization-free evaluation.

These gates protect this experiment's scientific workflow; they are not a
security boundary against someone running unrelated file readers manually.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy

from behavioral_market.data.archive import write_json
from experiments import crypto_activity_aux as core
from experiments import crypto_features, crypto_profile

SOURCE_CONFIG = "config/crypto_microstructure_aux_20261011.json"
SOURCE_CONFIG_SHA256 = "ed325ecb674361c72ad55af5b9b2c35728c2f6bce222ede3584e2b26c2aff52d"
SOURCE_PATHS = (
    "experiments/crypto_activity_aux.py",
    "experiments/crypto_activity_aux_pipeline.py",
    "experiments/crypto_activity_model.py",
    "experiments/crypto_features.py",
    "experiments/crypto_profile.py",
    "experiments/bybit_reconstruction.py",
    "experiments/bybit_trades.py",
    "src/behavioral_market/data/bybit_book.py",
    "src/behavioral_market/data/archive.py",
    "scripts/run_crypto_activity_aux.py",
    "scripts/audit_crypto_microstructure.py",
)
NUMERIC_SPEC = {
    "l2_penalty": 0.01,
    "alpha_bounds": [0.0001, 1000.0],
    "optimizer": {
        "method": "L-BFGS-B",
        "maxiter": 500,
        "ftol": 1e-10,
        "gtol": 1e-7,
        "accepted_projected_gradient_inf_norm": 0.001,
    },
}


def now():
    return datetime.now(UTC).isoformat()


def progress(phase, step, **details):
    print(
        json.dumps({"at": now(), "phase": phase, "step": step, **details}),
        file=sys.stderr,
        flush=True,
    )


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sample_sha(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def publish(path, payload):
    """Complete immutable JSON, including a digest of its canonical payload."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Evidence already exists: {path}")
    if "content_sha256" in payload:
        raise ValueError("Caller must not supply the content digest")
    value = {**payload, "content_sha256": content_sha(payload)}
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_json(temporary, value)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def read_evidence(path):
    value = json.loads(Path(path).read_text())
    expected = value.pop("content_sha256", None)
    if expected != content_sha(value):
        raise ValueError("Evidence content digest differs")
    return value


def expected_protocol():
    return {
        "protocol_version": 1,
        "study_id": "crypto_activity_aux_20261011",
        "source_config": SOURCE_CONFIG,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "splits": core.FIXED_SPLITS,
        "additional_delay_ms": [0, 1000, 5000],
        "target_seconds": 1,
        "minimum_visible_levels_per_side": 5,
        "model_features": {key: list(value) for key, value in core.MODEL_FEATURES.items()},
        "model": NUMERIC_SPEC,
        "noise": core.NOISE_SPEC,
        "calibration_block_hours": 3,
        "count_bins": ["0", "1", "2", "3-5", "6-10", "11-20", "21-50", "51+"],
        "median": "inf{k integer >=0: F(k)>=0.5}",
        "hurdle_objective": "full_N_mean_negative_log_probability_plus_half_l2_all_slopes",
        "comparisons": ["H0-N0", "N1-N0", "H1-H0", "Hnoise-H0"],
        "selection_policy": "Report every fixed model/delay/day; "
        "no winner selection or realism pass label",
        "aux_development_market_data_unopened_at_design": True,
        "method_development_used_previously_seen_november_results": True,
        "interpretation": "Retrospective method development with parameters fit on October 14-20 "
        "and evaluated on October 21-27. A conditional count diagnostic, not autonomous "
        "market generation, execution validation, prospective research, or a significance test. "
        "One fixed noise seed is not an empirical null distribution. Original November "
        "5-7 event test and final human financial sessions remain sealed.",
    }


def read_protocol(root, path):
    protocol = json.loads(Path(path).read_text())
    if protocol != expected_protocol():
        raise ValueError("Specification differs from this fixed auxiliary experiment")
    source_path = Path(root) / SOURCE_CONFIG
    if sha(source_path) != SOURCE_CONFIG_SHA256:
        raise ValueError("Auxiliary data configuration hash differs")
    source = json.loads(source_path.read_text())
    if source["splits"] != {**core.FIXED_SPLITS, "event_test": []}:
        raise ValueError("Auxiliary date selection differs; event test is closed")
    return protocol, source


def runtime():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def source_hashes(root):
    return {name: sha(Path(root) / name) for name in SOURCE_PATHS}


def committed_identity(root):
    # Freeze every experiment dependency, without blocking unrelated parallel work.
    paths = [*SOURCE_PATHS, SOURCE_CONFIG, "config/crypto_activity_aux_20261011.json"]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=root, text=True
    ).strip()
    if status:
        raise ValueError("Commit reviewed code/configuration before running the experiment")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def identity(root, protocol_path):
    return {
        "protocol_sha256": sha(protocol_path),
        "source_sha256": source_hashes(root),
        "source_config_sha256": sha(Path(root) / SOURCE_CONFIG),
        "runtime": runtime(),
    }


def assert_identity(root, protocol_path, frozen):
    actual = identity(root, protocol_path)
    if any(frozen.get(key) != value for key, value in actual.items()):
        raise ValueError("Frozen protocol, implementation, data configuration or runtime changed")


def phase_feature_day(root, source_config, *, phase, split, day, kind):
    """Reject an unauthorized phase/split/day before reaching any data reader."""
    allowed = {"fit": "train", "evaluate": "development"}
    if phase not in allowed or split != allowed[phase] or day not in core.FIXED_SPLITS[split]:
        raise ValueError("This phase cannot read the requested market split/day")
    return crypto_features.read_feature_day(Path(root), Path(source_config), split, day, kind)


def read_joint_days(root, *, phase):
    if phase not in {"fit", "evaluate"}:
        raise ValueError("Unknown experiment phase")
    split = "train" if phase == "fit" else "development"
    frames, provenance, coverage = {}, [], []
    for day in core.FIXED_SPLITS[split]:
        progress(phase, "read_audited_day", day=day)
        loaded, audits, receipt_hashes = {}, {}, {}
        for kind in ("orderbook", "trades"):
            report = (
                Path(root)
                / "outputs/crypto_microstructure_aux_20261011/integrity"
                / f"{day}_{kind}.json"
            )
            before = report.read_bytes()
            loaded[kind], audits[kind] = phase_feature_day(
                root, Path(root) / SOURCE_CONFIG, phase=phase, split=split, day=day, kind=kind
            )
            if report.read_bytes() != before or json.loads(before) != audits[kind]:
                raise ValueError("Audit receipt changed during feature read")
            receipt_hashes[kind] = hashlib.sha256(before).hexdigest()
        book, trades = loaded["orderbook"], loaded["trades"]
        ba, ta = audits["orderbook"], audits["trades"]
        prepared = crypto_profile.prepare_book(book, ba, day)
        frames[day], counts = crypto_profile.covered_trade_seconds(prepared, trades, ta)
        coverage.append({"split": split, "day": day, **counts})
        for kind, audit in (("orderbook", ba), ("trades", ta)):
            report = (
                Path(root)
                / "outputs/crypto_microstructure_aux_20261011/integrity"
                / f"{day}_{kind}.json"
            )
            provenance.append(
                {
                    "split": split,
                    "day": day,
                    "kind": kind,
                    "audit_receipt_sha256": receipt_hashes[kind],
                    **{
                        key: audit[key]
                        for key in (
                            "audit_run_id",
                            "input_sha256",
                            "features_path",
                            "features_sha256",
                            "processing_source_sha256",
                            "feature_rows",
                        )
                    },
                }
            )
    return frames, provenance, coverage


def verify_provenance_unchanged(root, sources):
    """Check that a concurrent re-audit/replacement did not revoke our inputs."""
    for item in sources:
        report = (
            Path(root)
            / "outputs/crypto_microstructure_aux_20261011/integrity"
            / f"{item['day']}_{item['kind']}.json"
        )
        if sha(report) != item["audit_receipt_sha256"]:
            raise ValueError("Input audit changed during this phase")
        if sha(Path(root) / item["features_path"]) != item["features_sha256"]:
            raise ValueError("Input features changed during this phase")
        source = json.loads((Path(root) / SOURCE_CONFIG).read_text())
        raw_name = source["urls"][item["kind"]].format(date=item["day"]).rsplit("/", 1)[-1]
        raw = Path(root) / "data/raw" / source["study_id"] / item["kind"] / raw_name
        if sha(raw) != item["input_sha256"]:
            raise ValueError("Input raw bytes changed since feature audit")


def scenario_samples(frames, split, delay):
    pieces, coverage = [], []
    if set(frames) != set(core.FIXED_SPLITS[split]):
        raise ValueError("Incomplete fixed calendar; cannot drop or replace a date")
    for day in core.FIXED_SPLITS[split]:
        samples, counts = core.build_activity_samples(
            frames[day], split=split, day=day, delay_ms=delay
        )
        if samples.empty:
            raise ValueError(f"No eligible samples for {split}/{day}/{delay}")
        pieces.append(samples)
        coverage.append({"split": split, "day": day, **counts})
    return pd.concat(pieces, ignore_index=True), coverage


def fit(root, protocol_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    protocol, _ = read_protocol(root, protocol_path)
    start, commit, frozen = now(), committed_identity(root), identity(root, protocol_path)
    frames, sources, coverage = read_joint_days(root, phase="fit")
    scenarios = []
    for delay in protocol["additional_delay_ms"]:
        samples, selected = scenario_samples(frames, "train", delay)
        progress("fit", "fit_five_models", delay_ms=delay, samples=len(samples))
        models = core.fit_models(samples, protocol["model"])
        core.validate_models(models)
        progress("fit", "score_frozen_training_predictions", delay_ms=delay)
        scenarios.append(
            {
                "additional_delay_ms": delay,
                "coverage": selected,
                "sample_sha256": sample_sha(samples),
                "samples": len(samples),
                "models": models,
                "training_scores": core.score_models(samples, models),
            }
        )
    assert_identity(root, protocol_path, frozen)
    verify_provenance_unchanged(root, sources)
    artifact = {
        "schema_version": 1,
        "kind": "frozen_auxiliary_activity_models",
        "status": "all_15_fitted_train_only",
        "started_at": start,
        "frozen_at": now(),
        "git_commit": commit,
        "git_identity_scope": "Listed sources and fixed configurations committed; "
        "unrelated parallel work may be uncommitted",
        "protocol": protocol,
        **frozen,
        "sources": sources,
        "joint_coverage": coverage,
        "scenarios": scenarios,
    }
    publish(output, artifact)
    return {"output": str(output), "sha256": sha(output), "status": artifact["status"]}


def verify_models(root, protocol_path, model_path):
    protocol, _ = read_protocol(root, protocol_path)
    artifact = read_evidence(model_path)
    if (
        artifact.get("kind") != "frozen_auxiliary_activity_models"
        or artifact.get("status") != "all_15_fitted_train_only"
        or artifact.get("protocol") != protocol
    ):
        raise ValueError("Missing complete train-only model artifact")
    assert_identity(root, protocol_path, artifact)
    scenarios = artifact.get("scenarios", [])
    if [row.get("additional_delay_ms") for row in scenarios] != [0, 1000, 5000]:
        raise ValueError("Frozen artifact must contain all three delays in fixed order")
    expected = {
        (day, kind) for day in core.FIXED_SPLITS["train"] for kind in ("orderbook", "trades")
    }
    sources = artifact.get("sources", [])
    if (
        len(sources) != len(expected)
        or any(item.get("split") != "train" for item in sources)
        or {(item["day"], item["kind"]) for item in sources} != expected
    ):
        raise ValueError("Frozen models lack complete training-only provenance")
    for row in scenarios:
        core.validate_models(row["models"])
        if not row.get("sample_sha256") or row.get("samples", 0) <= 0:
            raise ValueError("Frozen models lack training sample identity")
        for model in row["models"].values():
            if (
                model["diagnostics"]["training_days"] != core.FIXED_SPLITS["train"]
                or model["diagnostics"]["training_samples"] != row["samples"]
            ):
                raise ValueError(
                    "Frozen model population differs from the complete training sample"
                )
    verify_provenance_unchanged(root, sources)
    return artifact


def verify_review(root, protocol_path, model_path, review_path):
    artifact = verify_models(root, protocol_path, model_path)
    review = read_evidence(review_path)
    if (
        review.get("kind") != "independent_auxiliary_activity_prefit_and_freeze_review"
        or review.get("verdict") != "pass_for_development_diagnostic"
        or review.get("model_artifact_sha256") != sha(model_path)
        or review.get("protocol_sha256") != artifact["protocol_sha256"]
        or review.get("source_sha256") != artifact["source_sha256"]
        or review.get("aux_development_market_accessed") is not False
        or not review.get("reviewer")
        or not review.get("checks")
    ):
        raise ValueError("Independent review does not authorize this frozen artifact")
    check_chronology(artifact["started_at"], artifact["frozen_at"], review["reviewed_at"], now())
    return artifact, review


def check_chronology(*timestamps):
    parsed = [datetime.fromisoformat(value) for value in timestamps]
    if any(value.utcoffset() is None or value.utcoffset().total_seconds() != 0 for value in parsed):
        raise ValueError("Evidence timestamps must use explicit UTC")
    if any(left > right for left, right in zip(parsed, parsed[1:], strict=False)):
        raise ValueError("Evidence chronology invalid: later phase predates earlier phase")


def exported_receipts(root, batch):
    expected = {
        (day, kind) for day in core.FIXED_SPLITS["development"] for kind in ("orderbook", "trades")
    }
    results = batch.get("results", [])
    if (
        batch.get("status") != "complete"
        or not batch.get("audit_run_id")
        or len(results) != len(expected)
        or {(item.get("day"), item.get("kind")) for item in results} != expected
    ):
        raise ValueError("Development export batch is incomplete")
    receipts = {}
    for item in results:
        path = (
            Path(root)
            / "outputs/crypto_microstructure_aux_20261011/integrity"
            / f"{item['day']}_{item['kind']}.json"
        )
        body = path.read_bytes()
        if (
            json.loads(body) != item
            or item.get("status") != "complete"
            or item.get("features_published") is not True
            or item.get("split") != "development"
            or item.get("audit_run_id") != batch["audit_run_id"]
            or item.get("config_sha256") != SOURCE_CONFIG_SHA256
        ):
            raise ValueError("Current development receipt does not belong to the completed export")
        receipts[str(path.relative_to(root))] = hashlib.sha256(body).hexdigest()
    return receipts


def export_development(root, protocol_path, model_path, review_path, access_path, workers=3):
    """Record first access, preserve integrity-only receipts, then export features."""
    artifact, _ = verify_review(root, protocol_path, model_path, review_path)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1-4 export workers")
    committed_identity(root)
    base = Path(root) / "outputs/crypto_microstructure_aux_20261011"
    processed = Path(root) / "data/processed/crypto_microstructure_aux_20261011/development"
    if processed.exists() and any(processed.iterdir()):
        raise ValueError(
            "Development features already exist; first-access export cannot be repeated"
        )
    receipts = [base / "audit_development_all.json"] + [
        base / "integrity" / f"{day}_{kind}.json"
        for day in core.FIXED_SPLITS["development"]
        for kind in ("orderbook", "trades")
    ]
    for path in receipts[1:]:
        report = json.loads(path.read_text())
        if report.get("status") != "complete" or report.get("features_published") is not False:
            raise ValueError("Expected completed integrity-only development receipts")
    backup = base / "before_activity_aux_development_export"
    if backup.exists():
        raise FileExistsError("Earlier export attempt exists; preserve and diagnose it")
    publish(
        access_path,
        {
            "schema_version": 1,
            "kind": "auxiliary_development_first_access_authorization",
            "authorized_at": now(),
            "model_artifact_sha256": sha(model_path),
            "review_sha256": sha(review_path),
            "protocol_sha256": artifact["protocol_sha256"],
            "source_sha256": artifact["source_sha256"],
            "days": core.FIXED_SPLITS["development"],
            "purpose": "Generate features only after complete fitted model freeze "
            "and independent review",
            "integrity_only_receipts": {
                str(path.relative_to(root)): sha(path) for path in receipts
            },
        },
    )
    backup.mkdir()
    for path in receipts:
        shutil.copy2(path, backup / path.name)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(root) / "scripts/audit_crypto_microstructure.py"),
            "--config",
            str(Path(root) / SOURCE_CONFIG),
            "--split",
            "development",
            "--kind",
            "all",
            "--export-features",
            "--workers",
            str(workers),
        ],
        cwd=root,
        check=False,
    )
    assert_identity(root, protocol_path, artifact)
    completion = Path(access_path).with_name(Path(access_path).stem + ".completion.json")
    batch = json.loads((base / "audit_development_all.json").read_text())
    publish(
        completion,
        {
            "kind": "auxiliary_development_export_completion",
            "completed_at": now(),
            "access_receipt_sha256": sha(access_path),
            "exit_code": result.returncode,
            "audit_batch_sha256": sha(base / "audit_development_all.json"),
            "per_file_receipts_sha256": exported_receipts(root, batch)
            if result.returncode == 0
            else {},
        },
    )
    if result.returncode != 0:
        raise RuntimeError("Development export failed; retain failure and do not score")
    return {"access_receipt": str(access_path), "completion": str(completion), "status": "exported"}


def verify_access(root, artifact, model_path, review_path, access_path):
    access = read_evidence(access_path)
    if (
        access.get("kind") != "auxiliary_development_first_access_authorization"
        or access.get("model_artifact_sha256") != sha(model_path)
        or access.get("review_sha256") != sha(review_path)
        or access.get("protocol_sha256") != artifact["protocol_sha256"]
        or access.get("source_sha256") != artifact["source_sha256"]
        or access.get("days") != core.FIXED_SPLITS["development"]
    ):
        raise ValueError("Development access receipt differs from frozen identities")
    review = read_evidence(review_path)
    completion_path = Path(access_path).with_name(Path(access_path).stem + ".completion.json")
    completion = read_evidence(completion_path)
    batch_path = (
        Path(root) / "outputs/crypto_microstructure_aux_20261011/audit_development_all.json"
    )
    batch = json.loads(batch_path.read_text())
    if (
        completion.get("kind") != "auxiliary_development_export_completion"
        or completion.get("exit_code") != 0
        or completion.get("access_receipt_sha256") != sha(access_path)
        or completion.get("audit_batch_sha256") != sha(batch_path)
        or batch.get("status") != "complete"
        or len(batch.get("results", [])) != 14
    ):
        raise ValueError("Development export lacks a current successful completion receipt")
    check_chronology(
        review["reviewed_at"], access["authorized_at"], completion["completed_at"], now()
    )
    if completion.get("per_file_receipts_sha256") != exported_receipts(root, batch):
        raise ValueError("Development per-file export receipts changed")
    return access


def evaluate(root, protocol_path, model_path, review_path, access_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    artifact, _ = verify_review(root, protocol_path, model_path, review_path)
    verify_access(root, artifact, model_path, review_path, access_path)
    commit, start = committed_identity(root), now()
    frozen_inputs = {
        "model_artifact_sha256": sha(model_path),
        "review_sha256": sha(review_path),
        "access_receipt_sha256": sha(access_path),
    }
    frames, sources, coverage = read_joint_days(root, phase="evaluate")
    scenarios = []
    for model_scenario in artifact["scenarios"]:
        delay = model_scenario["additional_delay_ms"]
        samples, selected = scenario_samples(frames, "development", delay)
        progress("evaluate", "score_five_frozen_models", delay_ms=delay, samples=len(samples))
        scenarios.append(
            {
                "additional_delay_ms": delay,
                "coverage": selected,
                "sample_sha256": sample_sha(samples),
                "samples": len(samples),
                "scores": core.score_models(samples, model_scenario["models"]),
            }
        )
    assert_identity(root, protocol_path, artifact)
    verify_provenance_unchanged(root, sources)
    verify_access(root, artifact, model_path, review_path, access_path)
    if frozen_inputs != {
        "model_artifact_sha256": sha(model_path),
        "review_sha256": sha(review_path),
        "access_receipt_sha256": sha(access_path),
    }:
        raise ValueError("Frozen model/review/access evidence changed during evaluation")
    publish(
        output,
        {
            "schema_version": 1,
            "kind": "auxiliary_activity_development_scores",
            "status": "diagnostic_computed_no_refit",
            "market_read_started_at": start,
            "completed_at": now(),
            "git_commit": commit,
            **frozen_inputs,
            **identity(root, protocol_path),
            "sources": sources,
            "joint_coverage": coverage,
            "scenarios": scenarios,
        },
    )
    return {"output": str(output), "sha256": sha(output), "status": "diagnostic_computed_no_refit"}
