"""Exact per-day summaries of authenticated, train-only execution outputs.

This consumer never opens raw archives. Quantiles are order statistics of the
retained float64 bps residuals, not quantiles of the exact rational prices.
"""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from behavioral_market.data.archive import write_json
from experiments.crypto_execution_diagnostic import (
    AUDIT_SOURCES,
    DELAYS,
    ERRORS,
    EXECUTION_SOURCES,
    GROUPS,
    ROW_SCHEMA,
    day_start_us,
    validate_plan,
)

CONFIG_NAME = "config/crypto_execution_diagnostic_20261011.json"
SOURCE_CONFIG_NAME = "config/crypto_microstructure_20261011.json"
SUMMARY_SOURCES = (
    "experiments/crypto_execution_summary.py",
    "scripts/summarize_crypto_execution.py",
    "src/behavioral_market/data/archive.py",
)
REASONS = ("eligible", "no_state", "invalid_state", "outside_day_state", "stale", "partial_depth")
SIDES = ("Buy", "Sell")
AGES = ("[0,25)ms", "[25,100)ms", "[100,250)ms", "[250,1000)ms", "[1000,+inf)ms", "unknown")
SIZES = ("<=1", "(1,5]", ">5", "unknown")
STRATA = {"by_side": SIDES, "by_age_bin": AGES, "by_size_bin": SIZES}
QUANTILES = {"median": 0.5, "p90": 0.9, "p99": 0.99}
READ_COLUMNS = (
    "day",
    "trade_id",
    "trade_row",
    "timestamp_us",
    "side",
    "quantity_units",
    "observed_price_units",
    "delay_ms",
    "reason",
    "age_bin",
    "size_bin",
    "cross_l1",
    "shared_eligible",
    "shared_cross_l1",
    "age_us",
    "covered_quantity_units",
    "unresolved_quantity_units",
    "l1_quantity_units",
    "sweep_price_denominator",
    *ERRORS,
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty recorded path is required")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _committed_sources(root, commit, hashes, expected):
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ValueError("A full producer Git commit is required")
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError("Recorded source/config provenance differs from the fixed file set")
    for name in expected:
        if not _digest(hashes[name]):
            raise ValueError("Invalid recorded source hash")
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if hashlib.sha256(blob).hexdigest() != hashes[name] or sha256(root / name) != hashes[name]:
            raise ValueError(f"Committed or current source differs: {name}")


def _summary_identity(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    hashes = {name: sha256(root / name) for name in SUMMARY_SOURCES}
    _committed_sources(root, commit, hashes, SUMMARY_SOURCES)
    return {"git_commit": commit, "source_sha256": hashes}


def _validate_input_provenance(root, state, source, source_hash, days):
    evidence = {}
    if set(state.get("inputs", {})) != set(days):
        raise ValueError("Input provenance days differ from the scoring days")
    for day in days:
        inputs = state["inputs"][day]
        if not isinstance(inputs, dict) or set(inputs) != {"orderbook", "trades"}:
            raise ValueError("Both audited raw-source receipts are required")
        for kind, receipt in inputs.items():
            url = source["urls"][kind].format(date=day)
            raw = root / "data/raw" / source["study_id"] / kind / url.rsplit("/", 1)[-1]
            expected = {
                "path": raw,
                "metadata_path": raw.with_name(raw.name + ".metadata.json"),
                "audit_path": root
                / "outputs"
                / source["study_id"]
                / "integrity"
                / f"{day}_{kind}.json",
            }
            if not isinstance(receipt, dict) or any(
                _path(root, receipt.get(key)) != path.resolve() for key, path in expected.items()
            ):
                raise ValueError("Recorded raw-source identity differs from the frozen day")
            if (
                any(
                    not _digest(receipt.get(key))
                    for key in ("sha256", "metadata_sha256", "audit_sha256")
                )
                or type(receipt.get("bytes")) is not int
                or receipt["bytes"] <= 0
                or not isinstance(receipt.get("audit_run_id"), str)
                or not receipt["audit_run_id"]
            ):
                raise ValueError("Incomplete audited raw-source provenance")
            identity = {
                "url": url,
                "split": "train",
                "day": day,
                "kind": kind,
                "config_sha256": source_hash,
                **{key: source[key] for key in ("venue", "market", "symbol")},
            }
            parsed = {}
            for path_key, hash_key in (
                ("metadata_path", "metadata_sha256"),
                ("audit_path", "audit_sha256"),
            ):
                path = expected[path_key]
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != receipt[hash_key]:
                    raise ValueError(
                        "Current metadata/audit receipt differs from producer provenance"
                    )
                evidence[path] = receipt[hash_key]
                parsed[path_key] = json.loads(content)
            meta, audit = parsed["metadata_path"], parsed["audit_path"]
            if (
                any(meta.get(key) != value for key, value in identity.items())
                or meta.get("sha256") != receipt["sha256"]
                or meta.get("bytes") != receipt["bytes"]
                or audit.get("status") != "complete"
                or audit.get("audit_run_id") != receipt["audit_run_id"]
                or any(
                    audit.get(key) != identity[key]
                    for key in ("split", "day", "kind", "config_sha256")
                )
                or audit.get("input_sha256") != receipt["sha256"]
                or audit.get("processing_source_sha256")
                != {name: state.get("source_sha256", {}).get(name) for name in AUDIT_SOURCES}
            ):
                raise ValueError("Current source identity/audit authorization differs")
            if kind == "orderbook" and (
                audit.get("zip_crc_verified") is not True
                or audit.get("structural_pass") is not True
                or audit.get("errors") != {}
            ):
                raise ValueError("Current LOB integrity evidence is not valid")
            if kind == "trades" and (
                audit.get("quality") != "valid"
                or audit.get("records_scan_complete") is not True
                or audit.get("gzip_integrity")
                != {"status": "passed", "read_to_eof": True, "crc_verified": True}
            ):
                raise ValueError("Current trade integrity evidence is not valid")
    if state.get("source_config_sha256") != source_hash:
        raise ValueError("Source configuration identity differs")
    return evidence


@dataclass
class VerifiedRun:
    manifest_path: Path
    manifest_bytes: bytes
    prepared_path: Path
    prepared_bytes: bytes
    state: dict
    config_path: Path
    config_bytes: bytes
    source_path: Path
    source_bytes: bytes
    outputs: dict
    paths: dict
    evidence: dict


def verify_run(root, config_path, manifest_path, *, split="train"):
    """Authenticate the full fixed run before reading any output values."""
    if split != "train":
        raise ValueError("Development and event-test summary consumption is closed")
    root, config_path, manifest_path = (
        Path(root).resolve(),
        Path(config_path).resolve(),
        Path(manifest_path).resolve(),
    )
    if config_path != root / CONFIG_NAME or manifest_path.name != "manifest.json":
        raise ValueError("The fixed execution configuration and a run manifest are required")
    config_bytes = config_path.read_bytes()
    plan = json.loads(config_bytes)
    if plan.get("source_config") != SOURCE_CONFIG_NAME:
        raise ValueError("Only the frozen original source configuration is allowed")
    source_path = root / SOURCE_CONFIG_NAME
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != plan.get("source_config_sha256"):
        raise ValueError("Source configuration hash differs")
    source = json.loads(source_bytes)
    manifest_bytes = manifest_path.read_bytes()
    state = json.loads(manifest_bytes)
    if (
        state.get("status") != "complete"
        or state.get("outputs_published") is not True
        or state.get("stage") != "outputs_verified"
        or state.get("split") != "train"
        or not isinstance(state.get("run_id"), str)
        or not state["run_id"]
        or state.get("config") != plan
        or state.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest()
        or state.get("quantiles_computed") is not False
    ):
        raise ValueError("A matching, complete, published execution manifest is required")
    days = validate_plan(plan, source, state.get("mode"))
    windows = {
        day: [
            day_start_us(day),
            day_start_us(day) + (3_600_000_000 if state["mode"] == "smoke" else 86_400_000_000),
        ]
        for day in days
    }
    if state.get("days") != days or state.get("windows_us") != windows:
        raise ValueError("Run days/windows differ from the fixed train-only protocol")
    try:
        started, completed = (
            datetime.fromisoformat(state[key]) for key in ("started_at", "completed_at")
        )
        if (
            started.utcoffset() != timedelta(0)
            or completed.utcoffset() != timedelta(0)
            or completed < started
            or completed > datetime.now(UTC)
        ):
            raise ValueError("Invalid execution timestamps")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "UTC execution start/completion timestamps in the past are required"
        ) from exc
    if not isinstance(state.get("runtime"), dict) or any(
        not state["runtime"].get(key)
        for key in ("python", "python_implementation", "pyarrow", "zlib_runtime", "platform")
    ):
        raise ValueError("Producer runtime provenance is incomplete")
    evidence = _validate_input_provenance(root, state, source, source_hash, days)
    expected_sources = (CONFIG_NAME, SOURCE_CONFIG_NAME, *EXECUTION_SOURCES)
    _committed_sources(root, state.get("git_commit"), state.get("source_sha256"), expected_sources)
    for name, content in ((CONFIG_NAME, config_bytes), (SOURCE_CONFIG_NAME, source_bytes)):
        if state["source_sha256"][name] != hashlib.sha256(content).hexdigest():
            raise ValueError("Configuration bytes differ from producer provenance")
    outputs, paths = {}, {}
    records = state.get("outputs")
    if not isinstance(records, list) or len(records) != 3 * len(days):
        raise ValueError("Exactly three bound output files per scoring day are required")
    directory = manifest_path.parent
    prepared_path = directory / "prepared_receipt.json"
    prepared_bytes = prepared_path.read_bytes()
    prepared = json.loads(prepared_bytes)
    matching_keys = (
        "run_id",
        "split",
        "mode",
        "started_at",
        "runtime",
        "git_commit",
        "source_sha256",
        "config_sha256",
        "source_config_sha256",
        "config",
        "days",
        "inputs",
        "windows_us",
        "interpretation",
        "quantiles_computed",
    )
    if (
        prepared.get("status") != "running"
        or prepared.get("outputs_published") is not False
        or prepared.get("stage") != "inputs_verified_before_market_parsing"
        or prepared.get("outputs") != []
        or prepared.get("completed_at") is not None
        or any(prepared.get(key) != state.get(key) for key in matching_keys)
    ):
        raise ValueError("A matching pre-market prepared receipt is required")
    expected_paths = {
        directory / f"{day}_{suffix}": (day, kind)
        for day in days
        for kind, suffix in (
            ("prints", "prints.parquet"),
            ("blocks", "blocks.parquet"),
            ("summary", "summary.json"),
        )
    }
    for receipt in records:
        if not isinstance(receipt, dict):
            raise ValueError("Malformed output receipt")
        path = _path(root, receipt.get("path"))
        if path not in expected_paths or path in outputs or not _digest(receipt.get("sha256")):
            raise ValueError("Unexpected, duplicated, or unbound execution output")
        day, kind = expected_paths[path]
        rows = receipt.get("rows")
        if (kind == "summary" and rows is not None) or (
            kind != "summary" and (type(rows) is not int or rows < 0)
        ):
            raise ValueError("Output row-count receipt is invalid")
        if kind == "prints" and rows % len(DELAYS):
            raise ValueError("Print rows must contain all five delays")
        if kind == "blocks" and rows != (windows[day][1] - windows[day][0]) // 300_000_000 * len(
            DELAYS
        ):
            raise ValueError("Block rows differ from the frozen day window")
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Actual execution output hash differs from its receipt")
        outputs[path] = receipt
        paths[(day, kind)] = path
    if set(outputs) != set(expected_paths):
        raise ValueError("Missing execution output")
    # Only metadata is inspected here; all file hashes have already been checked.
    for day in days:
        for kind in ("prints", "blocks"):
            path = paths[(day, kind)]
            file = pq.ParquetFile(path)
            if file.metadata.num_rows != outputs[path]["rows"]:
                raise ValueError("Parquet metadata rows differ from the output receipt")
            if kind == "prints" and not file.schema_arrow.equals(ROW_SCHEMA, check_metadata=False):
                raise ValueError("Print schema differs from the frozen execution schema")
        daily = json.loads(paths[(day, "summary")].read_bytes())
        if (
            daily.get("day") != day
            or daily.get("window_us") != windows[day]
            or daily.get("print_delay_rows") != outputs[paths[(day, "prints")]]["rows"]
            or daily.get("quantiles_computed") is not False
            or not isinstance(daily.get("delays"), dict)
            or set(daily["delays"]) != {str(delay) for delay in DELAYS}
            or any(
                daily["delays"][str(d)].get("candidate_count")
                != daily["print_delay_rows"] // len(DELAYS)
                for d in DELAYS
            )
        ):
            raise ValueError("Daily output identity or five-delay denominators differ")
    verified = VerifiedRun(
        manifest_path,
        manifest_bytes,
        prepared_path,
        prepared_bytes,
        state,
        config_path,
        config_bytes,
        source_path,
        source_bytes,
        outputs,
        paths,
        evidence,
    )
    _recheck(verified)
    return verified


def _recheck(run):
    if (
        run.manifest_path.read_bytes() != run.manifest_bytes
        or run.prepared_path.read_bytes() != run.prepared_bytes
        or run.config_path.read_bytes() != run.config_bytes
        or run.source_path.read_bytes() != run.source_bytes
        or any(sha256(path) != receipt["sha256"] for path, receipt in run.outputs.items())
        or any(sha256(path) != digest for path, digest in run.evidence.items())
        or any(
            sha256(run.config_path.parent.parent / name) != digest
            for name, digest in run.state["source_sha256"].items()
        )
    ):
        raise ValueError("Bound execution inputs changed during summary processing")


def exact_statistics(values):
    """No sampling/sketch: inverted-CDF empirical quantiles of float64 values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Residuals must be finite one-dimensional float64 values")
    quantiles = (
        np.quantile(values, list(QUANTILES.values()), method="inverted_cdf")
        if len(values)
        else [None] * len(QUANTILES)
    )
    return {
        "mean": float(np.mean(values)) if len(values) else None,
        **{
            name: float(value) if value is not None else None
            for name, value in zip(QUANTILES, quantiles, strict=True)
        },
    }


@dataclass
class Counts:
    candidates: int = 0
    reasons: Counter = field(default_factory=Counter)
    known_coverage: int = 0
    covered: int = 0
    unresolved: int = 0

    def add(self, values, mask):
        self.candidates += int(mask.sum())
        self.reasons.update(values["reason"][mask].tolist())
        covered, unresolved = values["covered_quantity_units"], values["unresolved_quantity_units"]
        coverage_known = ~values["covered_quantity_units__null"]
        known = mask & coverage_known
        if (
            np.any(
                values["covered_quantity_units__null"] != values["unresolved_quantity_units__null"]
            )
            or np.any(covered[known] < 0)
            or np.any(unresolved[known] < 0)
        ):
            raise ValueError("Invalid visible-quantity coverage")
        self.known_coverage += int(known.sum())
        self.covered += sum(int(x) for x in covered[known])
        self.unresolved += sum(int(x) for x in unresolved[known])


@dataclass
class DelayBuffer:
    counts: Counts = field(default_factory=Counts)
    strata: dict = field(
        default_factory=lambda: {
            key: {label: Counts() for label in labels} for key, labels in STRATA.items()
        }
    )
    chunks: list = field(default_factory=list)

    def add(self, values):
        self.counts.add(values, np.ones(len(values["reason"]), dtype=bool))
        for key, column in (
            ("by_side", "side"),
            ("by_age_bin", "age_bin"),
            ("by_size_bin", "size_bin"),
        ):
            for label, counts in self.strata[key].items():
                counts.add(values, values[column] == label)
        eligible = values["reason"] == "eligible"
        self.chunks.append(
            {
                name: value[eligible]
                for name, value in values.items()
                if name
                in (
                    *ERRORS,
                    "side",
                    "age_bin",
                    "size_bin",
                    "cross_l1",
                    "shared_eligible",
                    "shared_cross_l1",
                )
            }
        )

    def report(self):
        fields = (
            *ERRORS,
            "side",
            "age_bin",
            "size_bin",
            "cross_l1",
            "shared_eligible",
            "shared_cross_l1",
        )
        values = {
            name: np.concatenate([chunk[name] for chunk in self.chunks])
            if self.chunks
            else np.array([])
            for name in fields
        }
        report = _group_report(self.counts, values, np.ones(len(values["side"]), dtype=bool))
        for key, column in (
            ("by_side", "side"),
            ("by_age_bin", "age_bin"),
            ("by_size_bin", "size_bin"),
        ):
            report[key] = {
                label: _group_report(counts, values, values[column] == label)
                for label, counts in self.strata[key].items()
            }
        return report


def _group_report(counts, values, selected):
    eligible_n = counts.reasons["eligible"]
    if int(selected.sum()) != eligible_n:
        raise ValueError("Eligible sample denominator differs from residual sample")
    report = {
        "candidate_count": counts.candidates,
        "eligible_count": eligible_n,
        "exclusion_count": counts.candidates - eligible_n,
        "eligible_fraction": eligible_n / counts.candidates if counts.candidates else None,
        "reasons": {name: counts.reasons[name] for name in REASONS},
        "known_visible_coverage_count": counts.known_coverage,
        "covered_quantity_units_sum": counts.covered,
        "unresolved_quantity_units_sum": counts.unresolved,
        "groups": {},
    }
    for group in GROUPS:
        mask = (
            selected
            if group == "eligible"
            else selected & values[group if group != "cross_l1" else "cross_l1"].astype(bool)
        )
        n = int(mask.sum())
        errors = {name: exact_statistics(values[name][mask]) for name in ERRORS}
        report["groups"][group] = {
            "n": n,
            "fraction_of_candidates": n / counts.candidates if counts.candidates else None,
            "fraction_of_eligible": n / eligible_n if eligible_n else None,
            "errors_bps": errors,
            "sweep_minus_l1_mae_bps": float(
                np.mean(values["sweep_abs_bps"][mask] - values["l1_abs_bps"][mask])
            )
            if n
            else None,
            "sweep_minus_mid_mae_bps": float(
                np.mean(values["sweep_abs_bps"][mask] - values["mid_abs_bps"][mask])
            )
            if n
            else None,
        }
    return report


def _array(table, name):
    column = table[name].combine_chunks()
    if pa.types.is_integer(column.type):
        return column.fill_null(-1).to_numpy(zero_copy_only=False)
    return column.to_numpy(zero_copy_only=False)


def _validate_batch(table, day, window, previous_row, previous_time, seen_ids):
    values = {name: _array(table, name) for name in READ_COLUMNS}
    values.update(
        {
            name + "__null": table[name].combine_chunks().is_null().to_numpy(zero_copy_only=False)
            for name in READ_COLUMNS
            if pa.types.is_integer(table[name].type)
        }
    )
    width = len(DELAYS)
    matrices = {name: array.reshape(-1, width) for name, array in values.items()}
    if not np.array_equal(matrices["delay_ms"], np.tile(DELAYS, (len(table) // width, 1))):
        raise ValueError("Each print must retain the ordered five-delay set")
    for name in (
        "day",
        "trade_id",
        "trade_row",
        "timestamp_us",
        "side",
        "quantity_units",
        "observed_price_units",
    ):
        matrix = matrices[name]
        if (
            any(x is None for x in matrix[:, 0])
            or (name + "__null" in values and values[name + "__null"].any())
            or not np.all(matrix == matrix[:, :1])
        ):
            raise ValueError("Five-delay rows disagree on the paired print identity")
    source_rows, times = matrices["trade_row"][:, 0], matrices["timestamp_us"][:, 0]
    if (
        np.any(source_rows <= previous_row)
        or np.any(np.diff(source_rows) <= 0)
        or np.any(times < previous_time)
        or np.any(np.diff(times) < 0)
        or np.any(times < window[0])
        or np.any(times >= window[1])
        or not np.all(values["day"] == day)
        or not np.isin(values["side"], SIDES).all()
        or np.any(values["quantity_units"] <= 0)
        or np.any(values["observed_price_units"] <= 0)
        or any(not isinstance(x, str) or not x.strip() for x in matrices["trade_id"][:, 0])
        or not np.isin(values["reason"], REASONS).all()
        or not np.isin(values["age_bin"], AGES).all()
        or not np.isin(values["size_bin"], SIZES).all()
    ):
        raise ValueError("Invalid print identity, order, window, or fixed stratum")
    ids = set(matrices["trade_id"][:, 0])
    if len(ids) != len(source_rows) or seen_ids.intersection(ids):
        raise ValueError("Duplicate print IDs are not independent paired observations")
    seen_ids.update(ids)
    eligible = values["reason"] == "eligible"
    for name in ("cross_l1", "shared_eligible", "shared_cross_l1"):
        if any(type(x) not in (bool, np.bool_) for x in values[name]):
            raise ValueError("Group flags must be nonnull booleans")
    shared = (matrices["reason"] == "eligible").all(axis=1)
    cross = matrices["cross_l1"].astype(bool)
    if not np.array_equal(
        matrices["shared_eligible"], np.repeat(shared[:, None], width, axis=1)
    ) or not np.array_equal(
        matrices["shared_cross_l1"], np.repeat((shared & cross.all(axis=1))[:, None], width, axis=1)
    ):
        raise ValueError("Stored shared groups differ from the five-delay intersection")
    age = values["age_us"]
    known_age = ~values["age_us__null"]
    classified = np.isin(values["reason"], ("eligible", "stale", "partial_depth"))
    expected_age = np.array(AGES, dtype=object)[
        np.searchsorted(
            [25_000, 100_000, 250_000, 1_000_000], np.where(known_age, age, 0), side="right"
        )
    ]
    expected_age[~classified] = "unknown"
    if (
        np.any(age[known_age] < 0)
        or np.any(classified & ~known_age)
        or not np.array_equal(values["age_bin"], expected_age)
    ):
        raise ValueError("Age stratum differs from its fixed boundaries")
    l1 = values["l1_quantity_units"]
    known_l1 = ~values["l1_quantity_units__null"]
    # Comparisons against int64 overflow are avoided by promoting the threshold.
    expected_size = np.full(len(table), "unknown", dtype=object)
    expected_size[known_l1] = np.where(
        values["quantity_units"][known_l1] <= l1[known_l1],
        "<=1",
        np.where(
            values["quantity_units"][known_l1] <= 5 * l1[known_l1].astype(object), "(1,5]", ">5"
        ),
    )
    if (
        np.any(l1[known_l1] <= 0)
        or not np.array_equal(known_l1, classified)
        or not np.array_equal(values["size_bin"], expected_size)
    ):
        raise ValueError("Size stratum differs from the fixed L1 quantity ratio")
    if (
        np.any(~known_age[eligible])
        or np.any(age[eligible] >= 1_000_000)
        or np.any(~known_l1[eligible])
        or np.any(values["covered_quantity_units"][eligible] != values["quantity_units"][eligible])
        or np.any(values["unresolved_quantity_units"][eligible] != 0)
        or np.any(values["sweep_price_denominator"][eligible] != values["quantity_units"][eligible])
        or not np.array_equal(
            values["cross_l1"][eligible], values["quantity_units"][eligible] > l1[eligible]
        )
    ):
        raise ValueError("Eligible prices are not fully covered and paired")
    for name in ERRORS:
        residual = values[name]
        if (
            not np.isfinite(residual[eligible]).all()
            or not np.isnan(residual[~eligible]).all()
            or (name != "sweep_signed_bps" and np.any(residual[eligible] < 0))
        ):
            raise ValueError("Residual availability differs from the common eligible sample")
    if not np.array_equal(
        values["sweep_abs_bps"][eligible], np.abs(values["sweep_signed_bps"][eligible])
    ):
        raise ValueError("Signed and absolute sweep residuals disagree")
    return values, int(source_rows[-1]), int(times[-1])


def summarize_day(path, day, window):
    """Stream validation; retain eligible residuals for exact per-day quantiles."""
    buffers = {delay: DelayBuffer() for delay in DELAYS}
    seen_ids = set()
    pending = None
    previous_row, previous_time, rows = 0, -1, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_535, columns=READ_COLUMNS):
        table = pa.Table.from_batches([batch])
        if pending is not None:
            table = pa.concat_tables([pending, table])
        full = len(table) // len(DELAYS) * len(DELAYS)
        pending = table.slice(full) if full < len(table) else None
        if not full:
            continue
        values, previous_row, previous_time = _validate_batch(
            table.slice(0, full), day, window, previous_row, previous_time, seen_ids
        )
        rows += full
        for index, delay in enumerate(DELAYS):
            buffers[delay].add(
                {name: value[index :: len(DELAYS)] for name, value in values.items()}
            )
    if pending is not None:
        raise ValueError("Incomplete five-delay print at the end of Parquet")
    return {
        "day": day,
        "window_us": window,
        "whole_utc_day": window == [day_start_us(day), day_start_us(day) + 86_400_000_000],
        "print_delay_rows": rows,
        "delays": {str(delay): buffer.report() for delay, buffer in buffers.items()},
    }


def _equal_day_metric(reports):
    available = {day: report for day, report in reports.items() if report["n"] > 0}
    differences = {day: report["sweep_minus_l1_mae_bps"] for day, report in available.items()}
    return {
        "days_with_samples": list(available),
        "days_without_samples": [day for day in reports if day not in available],
        "day_count": len(available),
        "mean_daily_mae_bps": {
            name: float(
                np.mean([report["errors_bps"][name]["mean"] for report in available.values()])
            )
            if available
            else None
            for name in ERRORS[:3]
        },
        "daily_sweep_minus_l1_mae_bps": differences,
        "equal_day_sweep_minus_l1_mae_bps": float(np.mean(list(differences.values())))
        if available
        else None,
        "equal_day_sweep_minus_mid_mae_bps": float(
            np.mean([r["sweep_minus_mid_mae_bps"] for r in available.values()])
        )
        if available
        else None,
        "days_sweep_lower_l1_mae": sum(value < 0 for value in differences.values()),
        "days_sweep_higher_l1_mae": sum(value > 0 for value in differences.values()),
    }


def equal_day_summary(days):
    if not all(day["whole_utc_day"] for day in days.values()):
        return {
            "available": False,
            "reason": "The fixed smoke window is one hour and cannot supply whole-day comparisons",
        }
    delays = {}
    for delay in DELAYS:
        reports = {day: result["delays"][str(delay)] for day, result in days.items()}
        result = {
            "groups": {
                group: _equal_day_metric(
                    {day: report["groups"][group] for day, report in reports.items()}
                )
                for group in GROUPS
            }
        }
        for key, labels in STRATA.items():
            result[key] = {
                label: {
                    "groups": {
                        group: _equal_day_metric(
                            {
                                day: report[key][label]["groups"][group]
                                for day, report in reports.items()
                            }
                        )
                        for group in GROUPS
                    }
                }
                for label in labels
            }
        delays[str(delay)] = result
    return {
        "available": True,
        "weighting": (
            "One equal weight per complete UTC day with nonempty paired samples in the "
            "relevant stratum; empty days are listed, never assigned zero error"
        ),
        "delays": delays,
    }


def _publish_exclusive(path, content):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".summary-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication which cannot replace an existing file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summarize_execution(root, config_path, manifest_path, output, *, split="train"):
    """Write a new immutable hash-bound summary; no raw or held-out data access."""
    if split != "train":
        raise ValueError("Development and event-test summary consumption is closed")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "manifest.json"
    state = {
        "status": "running",
        "outputs_published": False,
        "run_id": uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "outputs": [],
    }
    write_json(state_path, state)
    try:
        run = verify_run(root, config_path, manifest_path, split=split)
        identity = _summary_identity(root)
        state.update(
            **identity,
            producer_manifest_path=str(run.manifest_path),
            producer_manifest_sha256=hashlib.sha256(run.manifest_bytes).hexdigest(),
            producer_prepared_receipt_path=str(run.prepared_path),
            producer_prepared_receipt_sha256=hashlib.sha256(run.prepared_bytes).hexdigest(),
            producer_run_id=run.state["run_id"],
            config_sha256=run.state["config_sha256"],
        )
        days = {
            day: summarize_day(run.paths[(day, "prints")], day, run.state["windows_us"][day])
            for day in run.state["days"]
        }
        for day, report in days.items():
            if report["print_delay_rows"] != run.outputs[run.paths[(day, "prints")]]["rows"]:
                raise ValueError("Consumed print count differs from the authenticated receipt")
        summary = {
            "producer_run_id": run.state["run_id"],
            "producer_manifest_sha256": state["producer_manifest_sha256"],
            "producer_prepared_receipt_sha256": state["producer_prepared_receipt_sha256"],
            "producer_git_commit": run.state["git_commit"],
            "summary_identity": identity,
            "config_sha256": run.state["config_sha256"],
            "mode": run.state["mode"],
            "split": "train",
            "delays_ms": list(DELAYS),
            "quantiles_computed": True,
            "quantile_policy": {
                "method": "inverted_cdf",
                "probabilities": QUANTILES,
                "basis": (
                    "All retained float64 bps residuals within each day/delay/paired group and "
                    "fixed stratum; no sketch or sampling; exact rational prices remain in "
                    "producer Parquet"
                ),
                "pooling": (
                    "Per-day quantiles only; cross-day comparisons give equal weight to "
                    "whole-day paired MAE differences"
                ),
            },
            "source_evidence": (
                "Current metadata/audit receipt hashes and authorization are verified. Raw "
                "archive byte hashes are inherited from the authenticated producer; this "
                "summary does not reopen or rehash raw archives."
            ),
            "interpretation": (
                "Conditional print-price diagnostic given observed side and size; unknown "
                "parent-order structure remains. No IID print inference or full-fill/market-"
                "realism conclusion."
            ),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
            },
            "days": days,
            "equal_day": equal_day_summary(days),
        }
        _recheck(run)
        for name, digest in identity["source_sha256"].items():
            if sha256(root / name) != digest:
                raise ValueError("Summary source changed during processing")
        path = output / "summary.json"
        _publish_exclusive(path, summary)
        state.update(
            status="complete",
            outputs_published=True,
            outputs=[{"path": str(path), "sha256": sha256(path)}],
        )
    except Exception as exc:
        state.update(
            status="failed",
            outputs_published=False,
            error={"type": type(exc).__name__, "detail": str(exc)[:400]},
        )
        raise
    finally:
        state["completed_at"] = datetime.now(UTC).isoformat()
        write_json(state_path, state)
    return state
