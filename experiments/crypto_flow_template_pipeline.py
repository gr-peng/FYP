"""Authenticated, train-only empirical templates and an immutable model loader.

The fitter parses only complete, hash-bound reader outputs. Raw archives are
hashed by audited_inputs, never parsed here. The loader opens only the model
artifact and frozen code/configuration; it neither reads raw data nor refits.
Engineering validation and a committed protocol are not market-realism results.
"""

import gzip
import hashlib
import json
import os
import platform
import subprocess
import zlib
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from behavioral_market.data.archive import write_json
from experiments import crypto_effective_flow_reader as reader
from experiments.crypto_autonomous_metrics import DIMENSIONS
from experiments.crypto_effective_flow import (
    DAY_US,
    DELAYS_MS,
    GRID_US,
    MAX_AGE_US,
    MAX_UNITS,
    MIN_CELL_SIZE,
    TRAIN_DAYS,
    FlowTemplate,
    Intent,
    ObservedPrint,
    RelativeLiquidity,
    StateFeatures,
    TemplatePool,
    TrainingRow,
    fit_template_pool,
)
from experiments.crypto_execution_diagnostic import (
    AUDIT_SOURCES,
    SPLITS,
    audited_inputs,
    day_start_us,
    frozen_identity,
    sha256,
)

CONFIG_PATH = "config/crypto_autonomous_flow_20261013.json"
CANDIDATES_PER_DAY = DAY_US // GRID_US
MODEL_SOURCES = tuple(
    dict.fromkeys(
        (
            "experiments/crypto_flow_template_pipeline.py",
            "scripts/fit_crypto_flow_templates.py",
            "experiments/crypto_autonomous_flow.py",
            "experiments/crypto_autonomous_metrics.py",
            "experiments/__init__.py",
            *reader.READER_SOURCES,
            "src/behavioral_market/__init__.py",
            "src/behavioral_market/data/__init__.py",
            "src/behavioral_market/market/__init__.py",
            "src/behavioral_market/market/perpetual_exchange.py",
            "src/behavioral_market/market/perpetual.py",
            "src/behavioral_market/market/matching_engine.py",
            "src/behavioral_market/market/order_book.py",
            "src/behavioral_market/market/order.py",
            "src/behavioral_market/market/money.py",
            "src/behavioral_market/market/trade.py",
            # Importing the market package also imports these spot components.
            "src/behavioral_market/market/exchange.py",
            "src/behavioral_market/market/portfolio.py",
        )
    )
)
FROZEN_PATHS = (CONFIG_PATH, reader.CONFIG_PATH, reader.SOURCE_PATH, *MODEL_SOURCES)
SAMPLER = "one parent getrandbits(128); local Random(draw).randrange; whole-row empirical draw"
SPREAD_THRESHOLD = "exact common-cohort zero-delay train median; even midpoint"
FEATURE_NAMES = ("imbalance_l1", "spread_bps", "trade_pressure")
REFERENCE_FIELDS = (
    "observed_at_us",
    "valid",
    "segment_id",
    "segment_start_us",
    "message_index",
    "engine_us",
    "available_us",
    "publication_prefix_ms",
    "invalid_count",
    "bid_l1",
    "bid_outer_price_units",
    "bid_visible_levels",
    "ask_l1",
    "ask_outer_price_units",
    "ask_visible_levels",
)
RECORD_FIELDS = (
    "day",
    "grid_us",
    "eligible",
    "exclusion_reasons",
    "start_prefix",
    "end_prefix",
    "signal_prefixes",
    "features_by_delay",
    "observed_prints",
    "print_alignment",
    "censored_endpoints",
    "effective_levels",
    "template",
)
OUTPUT_FIELDS = (
    "path",
    "sha256",
    "rows",
    "eligible_rows",
    "exclusions",
    "observed_prints",
    "observed_quantity_units",
    "print_alignment",
)


def expected_protocol():
    """Fixed before any prototype fit/generation; no selected winner or pass bar."""
    return {
        "study_id": "crypto_autonomous_flow_20261013",
        "source_config": reader.SOURCE_PATH,
        "source_config_sha256": reader.SOURCE_SHA,
        "extraction_config": reader.CONFIG_PATH,
        "allowed_splits": ["train"],
        "train_days": list(TRAIN_DAYS),
        "candidate_rows_per_day": CANDIDATES_PER_DAY,
        "common_delay_cohort": True,
        "methods": ["M0", "M1"],
        "orderings": {"S1": ["remove", "add", "aggress"], "S2": ["aggress", "remove", "add"]},
        "delays_ms": list(DELAYS_MS),
        "seeds": list(range(20)),
        "grid_us": GRID_US,
        "horizon_us": 300_000_000,
        "model": {
            "state_dimensions": list(FEATURE_NAMES),
            "cells_per_delay": 18,
            "imbalance_edges": [[-1, 3], [1, 3]],
            "pressure_edges": [[-1, 3], [1, 3]],
            "spread_threshold": SPREAD_THRESHOLD,
            "fallback_min_rows": MIN_CELL_SIZE,
            "sampler": SAMPLER,
            "net_event_order": "bid descending then ask ascending",
            "intent_order": "original print source-row order",
        },
        "train_windows": {
            "days": list(TRAIN_DAYS),
            "hours_utc": list(range(24)),
            "start_minute": 10,
            "end_minute": 15,
            "total_candidates": 72,
        },
        "first_smoke": {"day": TRAIN_DAYS[0], "start_second_utc": 600, "end_second_utc": 900},
        "generator": {
            "initial_view_max_depth_per_side": 500,
            "missing_initial_levels": "do not pad or replenish",
            "price_storage_scale": 100,
            "quantity_storage_scale": 1000,
            "accounts": ["bid-maker", "ask-maker", "buy-taker", "sell-taker"],
            "cash_per_account": "G=sum(initial two-sided price_units*quantity_units)/100000",
            "max_leverage": "1",
            "maker_fee_rate": "0",
            "taker_fee_rate": "0",
            "min_open_notional": "0",
            "topups_allowed": False,
            "DAY": "reserve all remaining gross exposure/adverse entry/positive maker fees",
            "IOC": (
                "atomic full visible fill plan and affected accounts/reservations; "
                "cancel remainder without reserving unfilled nominal"
            ),
            "mark": "own two-sided midpoint half-up integer unit at step begin and end",
            "decimal_precision": 80,
            "print_timestamp_offset_us": 1000,
            "print_merge": "same intent and same execution price only",
            "empty_side_policy": "retain failed seed and undefined tail; no carry/reset/refill",
        },
        "metrics": {
            "dimensions": list(DIMENSIONS),
            "train_minute_candidates": 4320,
            "scale_ddof": 0,
            "log1p_indices": [1, 2, 3, 4, 5],
            "ensemble_size": 20,
            "energy_score": "fair U-statistic",
            "return_bps": "10000*log(mid_end/mid_start); also for P",
            "minute_means": "600 left 100ms grid points equally timed",
            "depth": "visible BTC within 5bps of midpoint in supplied top500; coverage separately",
            "interval_quantiles": [[1, 20], [19, 20]],
            "quantile_method": "inverted_cdf",
            "incomplete_policy": "retain denominators; no drop or zero imputation",
            "targets_builder": (
                "validate failed source grids and reset crossings; retain candidates"
            ),
            "market_pass_threshold": None,
        },
        "access": {
            "development": "closed",
            "event_test": "closed",
            "LLM": "closed",
            "future_price_or_q1_for_generation": False,
        },
        "freeze_required_sources": list(FROZEN_PATHS),
        "interpretation": (
            "Effective aggregate flow with finite experimental collateral; "
            "no historical queue/MMR/liquidation/ADL/funding assertion"
        ),
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _same_json(first, second):
    # Dict/list equality treats bools and integral floats as equal to integers.
    return _canonical(first) == _canonical(second)


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _mapping(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"Unexpected or missing {name} fields")
    return value


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid exact integer {name}")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(value):
    def reject_constant(token):
        raise ValueError(f"Nonfinite JSON number: {token}")

    return json.loads(value, object_pairs_hook=_pairs, parse_constant=reject_constant)


def _fraction(value):
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(v) is not int for v in value)
        or value[1] <= 0
    ):
        raise ValueError("Feature requires an exact numerator/positive-denominator rational")
    result = Fraction(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError("Rational must be in canonical reduced form")
    return result


def _features(value):
    _mapping(value, FEATURE_NAMES, "feature")
    return StateFeatures(*(_fraction(value[name]) for name in FEATURE_NAMES))


def _feature_json(value):
    return {
        name: [getattr(value, name).numerator, getattr(value, name).denominator]
        for name in FEATURE_NAMES
    }


def _path(root, value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("A nonempty bound path is required")
    value = Path(value)
    return (value if value.is_absolute() else root / value).resolve()


def _committed_sources(root, commit, hashes, expected):
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ValueError("A full committed source identity is required")
    _mapping(hashes, expected, "source provenance")
    for name in expected:
        if not _digest(hashes[name]):
            raise ValueError("Invalid committed source hash")
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if _hash_bytes(committed) != hashes[name] or sha256(root / name) != hashes[name]:
            raise ValueError(f"Committed or current source differs: {name}")


def _timestamps(state):
    try:
        start, end = (datetime.fromisoformat(state[k]) for k in ("started_at", "completed_at"))
        if (
            start.utcoffset() != timedelta(0)
            or end.utcoffset() != timedelta(0)
            or end < start
            or end > datetime.now(UTC)
        ):
            raise ValueError("Not past UTC timestamps")
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError("Explicit UTC start/completion in the past are required") from error


def _count_map(value, name):
    if type(value) is not dict or any(type(k) is not str or not k for k in value):
        raise ValueError(f"Invalid {name} count mapping")
    for count in value.values():
        _integer(count, name, minimum=1)
    return value


def _gzip_rows(path):
    """Check all current gzip bytes/CRC and line denominators, without JSON parsing."""
    count = 0
    with gzip.open(path, "rb") as handle:
        for line in handle:
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError("Incomplete or empty JSONL record")
            count += 1
    return count


def _normalized_inputs(root, inputs):
    _mapping(inputs, TRAIN_DAYS, "raw input days")
    result = {}
    for day in TRAIN_DAYS:
        _mapping(inputs[day], ("orderbook", "trades"), "raw input kinds")
        result[day] = {}
        for kind, receipt in inputs[day].items():
            _mapping(
                receipt,
                (
                    "path",
                    "sha256",
                    "bytes",
                    "metadata_path",
                    "metadata_sha256",
                    "audit_path",
                    "audit_sha256",
                    "audit_run_id",
                ),
                "raw receipt",
            )
            copy = dict(receipt)
            for key in ("path", "metadata_path", "audit_path"):
                copy[key] = str(_path(root, copy[key]))
            for key in ("sha256", "metadata_sha256", "audit_sha256"):
                if not _digest(copy[key]):
                    raise ValueError("Incomplete raw/metadata/audit hashes")
            _integer(copy["bytes"], "raw bytes", minimum=1)
            if type(copy["audit_run_id"]) is not str or not copy["audit_run_id"]:
                raise ValueError("Missing raw audit identity")
            result[day][kind] = copy
    return result


@dataclass(frozen=True)
class VerifiedExtraction:
    root: Path
    manifest_path: Path
    manifest_bytes: bytes
    prepared_bytes: bytes
    state: dict
    config_bytes: bytes
    source_bytes: bytes
    source: dict
    output_paths: tuple[Path, ...]
    inputs: dict


def verify_train_extraction(root, manifest_path):
    """Verify a complete master and every source/output hash before candidate values."""
    root = Path(root).resolve()
    manifest_path = _path(root, manifest_path)
    if manifest_path.name != "manifest.json":
        raise ValueError("A run_extraction master manifest is required")
    manifest_bytes = manifest_path.read_bytes()
    state = _json(manifest_bytes)
    windows = {day: [day_start_us(day), day_start_us(day) + DAY_US] for day in TRAIN_DAYS}
    if (
        type(state) is not dict
        or state.get("status") != "complete"
        or state.get("outputs_published") is not True
        or state.get("split") != "train"
        or state.get("mode") != "train"
        or state.get("days") != list(TRAIN_DAYS)
        or not _same_json(state.get("windows_us"), windows)
    ):
        raise ValueError("Only complete three-full-day train extraction masters may fit")
    if type(state.get("run_id")) is not str or not state["run_id"]:
        raise ValueError("Missing extraction run identity")
    _timestamps(state)
    if type(state.get("runtime")) is not dict or any(
        not state["runtime"].get(k) for k in ("python", "python_implementation")
    ):
        raise ValueError("Missing extraction runtime identity")
    config_bytes = (root / reader.CONFIG_PATH).read_bytes()
    source_bytes = (root / reader.SOURCE_PATH).read_bytes()
    if (
        not _same_json(state.get("config"), reader.expected_protocol())
        or not _same_json(_json(config_bytes), reader.expected_protocol())
        or state.get("config_sha256") != _hash_bytes(config_bytes)
        or state.get("source_config_sha256") != reader.SOURCE_SHA
        or _hash_bytes(source_bytes) != reader.SOURCE_SHA
    ):
        raise ValueError("Extraction configuration/hash chain differs")
    source = _json(source_bytes)
    if (
        source.get("splits") != SPLITS
        or source.get("venue") != "bybit"
        or source.get("market") != "linear_usdt_perpetual"
        or source.get("symbol") != "BTCUSDT"
        or source.get("book_depth") != 500
        or source.get("price_storage_scale") != 100
        or source.get("quantity_storage_scale") != 1000
    ):
        raise ValueError("Only the original approved train source is allowed")
    source_paths = (reader.CONFIG_PATH, reader.SOURCE_PATH, *reader.READER_SOURCES)
    _committed_sources(root, state.get("git_commit"), state.get("source_sha256"), source_paths)
    prepared_bytes = (manifest_path.parent / "prepared_receipt.json").read_bytes()
    expected_prepared = dict(
        state, status="running", outputs_published=False, outputs=[], completed_at=None
    )
    if not _same_json(_json(prepared_bytes), expected_prepared):
        raise ValueError("Prepared extraction receipt does not match the complete master")
    records = state.get("outputs")
    if type(records) is not list or len(records) != len(TRAIN_DAYS):
        raise ValueError("Exactly three complete daily outputs are required")
    output_paths = []
    for day, receipt in zip(TRAIN_DAYS, records, strict=True):
        _mapping(receipt, OUTPUT_FIELDS, "daily output")
        path = _path(root, receipt["path"])
        if path != manifest_path.parent / f"{day}_effective_flow.jsonl.gz":
            raise ValueError("Output day/path is outside its fixed extraction run")
        if not _digest(receipt["sha256"]) or receipt["rows"] != CANDIDATES_PER_DAY:
            raise ValueError("Every output must bind all 864000 daily candidate rows")
        _integer(receipt["rows"], "daily rows", minimum=CANDIDATES_PER_DAY)
        _integer(receipt["eligible_rows"], "eligible rows", maximum=CANDIDATES_PER_DAY)
        _integer(receipt["observed_prints"], "observed prints")
        _integer(receipt["observed_quantity_units"], "observed print quantity")
        _count_map(receipt["exclusions"], "exclusions")
        alignment = _count_map(receipt["print_alignment"], "print alignment")
        if (
            set(alignment)
            - {"unusable_endpoints", "common_visible_domain", "outside_common_visible_domain"}
            or sum(alignment.values()) != receipt["observed_prints"]
        ):
            raise ValueError("Print alignment denominators differ")
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Actual daily gzip hash differs")
        output_paths.append(path)
    inputs = _normalized_inputs(root, state.get("inputs"))
    audit_sources = {name: state["source_sha256"][name] for name in AUDIT_SOURCES}
    for day in TRAIN_DAYS:
        current = audited_inputs(root, source, reader.SOURCE_SHA, day, audit_sources)
        if _normalized_inputs(root, {**inputs, day: current})[day] != inputs[day]:
            raise ValueError("Current raw/metadata/audit actual hashes differ from the master")
    # All gzip hashes precede all line counting, and all counts precede JSON parsing.
    for path, receipt in zip(output_paths, records, strict=True):
        if _gzip_rows(path) != receipt["rows"]:
            raise ValueError("Current gzip row count differs from the complete master")
    result = VerifiedExtraction(
        root,
        manifest_path,
        manifest_bytes,
        prepared_bytes,
        state,
        config_bytes,
        source_bytes,
        source,
        tuple(output_paths),
        inputs,
    )
    # Line counting can outlive a mutation of a raw file checked earlier.
    # Revalidate all inputs, then fence the producer/config/output identities
    # again before any candidate JSON is consumed.
    _fence_extraction(result, rehash_raw=True)
    return result


def _fence_extraction(verified, *, rehash_raw):
    root, state = verified.root, verified.state
    if rehash_raw:
        sources = {name: state["source_sha256"][name] for name in AUDIT_SOURCES}
        for day in TRAIN_DAYS:
            current = audited_inputs(root, verified.source, reader.SOURCE_SHA, day, sources)
            if (
                _normalized_inputs(root, {**verified.inputs, day: current})[day]
                != verified.inputs[day]
            ):
                raise ValueError("Raw input changed during consumption")
    if (
        verified.manifest_path.read_bytes() != verified.manifest_bytes
        or (verified.manifest_path.parent / "prepared_receipt.json").read_bytes()
        != verified.prepared_bytes
        or (root / reader.CONFIG_PATH).read_bytes() != verified.config_bytes
        or (root / reader.SOURCE_PATH).read_bytes() != verified.source_bytes
    ):
        raise ValueError("Extraction/configuration receipts changed during consumption")
    _committed_sources(
        root,
        state["git_commit"],
        state["source_sha256"],
        (reader.CONFIG_PATH, reader.SOURCE_PATH, *reader.READER_SOURCES),
    )
    for path, receipt in zip(verified.output_paths, state["outputs"], strict=True):
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Daily gzip changed during consumption")
    for day in TRAIN_DAYS:
        for receipt in verified.inputs[day].values():
            for path_key, hash_key in (
                ("metadata_path", "metadata_sha256"),
                ("audit_path", "audit_sha256"),
            ):
                if sha256(Path(receipt[path_key])) != receipt[hash_key]:
                    raise ValueError("Metadata/audit receipt changed during consumption")


def _reference(value, timestamp):
    _mapping(value, REFERENCE_FIELDS, "source prefix")
    if value["observed_at_us"] != timestamp or type(value["observed_at_us"]) is not int:
        raise ValueError("Source prefix is not at its precise observation clock")
    if type(value["valid"]) is not bool:
        raise ValueError("Source validity must be boolean")
    _integer(value["invalid_count"], "source error counter")
    for name in (
        "segment_id",
        "segment_start_us",
        "message_index",
        "engine_us",
        "available_us",
        "publication_prefix_ms",
    ):
        if value[name] is not None:
            _integer(value[name], name, minimum=1 if name == "message_index" else 0)
    for name in ("segment_start_us", "engine_us", "available_us"):
        if value[name] is not None and value[name] % 1000:
            raise ValueError("All source clocks must retain millisecond precision")
    if value["message_index"] is not None:
        if (
            value["available_us"] is None
            or value["publication_prefix_ms"] is None
            or (value["publication_prefix_ms"] + 1) * 1000 > value["available_us"]
            or value["available_us"] >= timestamp
            or (
                value["engine_us"] is not None and value["engine_us"] + 1000 > value["available_us"]
            )
        ):
            raise ValueError("Consumed source prefix violates available/publication clocks")
    elif (
        any(
            value[name] is not None
            for name in (
                "engine_us",
                "available_us",
                "publication_prefix_ms",
                "segment_id",
                "segment_start_us",
            )
        )
        or value["invalid_count"]
    ):
        raise ValueError("Unknown message prefix cannot carry known source clocks or errors")
    if (value["segment_id"] is None) != (value["segment_start_us"] is None):
        raise ValueError("Snapshot segment identity and origin must be jointly known")
    for side in ("bid", "ask"):
        levels, l1, outer = (
            value[f"{side}_{name}"] for name in ("visible_levels", "l1", "outer_price_units")
        )
        _integer(levels, "visible level count", maximum=500)
        if value["valid"]:
            if type(l1) is not list or len(l1) != 2 or not levels:
                raise ValueError("Known usable source prefix requires both L1 prices/quantities")
            for units in (*l1, outer):
                _integer(units, "L1/outer storage units", minimum=1, maximum=MAX_UNITS)
            if (
                (side == "bid" and outer > l1[0])
                or (side == "ask" and outer < l1[0])
                or (levels == 1 and outer != l1[0])
                or (levels > 1 and outer == l1[0])
            ):
                raise ValueError("Source outer boundary differs from its ordered visible depth")
        elif l1 is not None or outer is not None or levels:
            raise ValueError("Invalid source cannot expose usable prices or depth")
    if value["valid"]:
        if any(
            value[name] is None
            for name in (
                "segment_id",
                "segment_start_us",
                "message_index",
                "engine_us",
                "available_us",
                "publication_prefix_ms",
            )
        ):
            raise ValueError("Unknown consumed source prefix")
        if (
            value["bid_l1"][0] >= value["ask_l1"][0]
            or value["segment_start_us"] > value["engine_us"]
            or value["engine_us"] + 1000 > value["available_us"]
            or (value["publication_prefix_ms"] + 1) * 1000 > value["available_us"]
            or value["available_us"] >= timestamp
        ):
            raise ValueError("Source prefix violates available/engine/publication clocks")
    elif value["message_index"] is None and any(
        value[name] is not None for name in ("engine_us", "available_us", "publication_prefix_ms")
    ):
        raise ValueError("Unknown message prefix cannot carry known source clocks")
    return value


def _prefix_progress(first, second):
    for key in ("message_index", "available_us", "segment_id", "invalid_count"):
        if first[key] is not None and second[key] is not None and second[key] < first[key]:
            raise ValueError("Source prefix/segment/error clock moved backwards")
    if first["message_index"] is not None and first["message_index"] == second["message_index"]:
        if any(first[key] != second[key] for key in REFERENCE_FIELDS if key != "observed_at_us"):
            raise ValueError("Same source prefix changed its sparse book or identity")
    if first["segment_id"] is not None and first["segment_id"] == second["segment_id"]:
        if first["segment_start_us"] != second["segment_start_us"]:
            raise ValueError("Same segment changed snapshot origin")
        if (
            first["publication_prefix_ms"] is not None
            and second["publication_prefix_ms"] is not None
            and second["publication_prefix_ms"] < first["publication_prefix_ms"]
        ):
            raise ValueError("Same segment publication prefix moved backwards")
    if first["valid"] and second["valid"] and second["engine_us"] < first["engine_us"]:
        raise ValueError("Usable endpoint engine clock moved backwards")


def _usable(reference):
    # Mirror the producer's pure state_features validity rule. Day boundaries
    # are label exclusions; even excluded rows retain computable lag features.
    return (
        reference["valid"]
        and reference["observed_at_us"] - reference["engine_us"] < MAX_AGE_US
    )


def _label_reasons(start, end, day_start):
    reasons = []
    for label, reference in (("start", start), ("end", end)):
        if not reference["valid"]:
            reasons.append(f"{label}_invalid_or_absent")
            continue
        if any(
            not day_start <= reference[key] < day_start + DAY_US
            for key in ("segment_start_us", "engine_us", "available_us")
        ):
            reasons.append(f"{label}_outside_day_state")
        if reference["observed_at_us"] - reference["engine_us"] >= MAX_AGE_US:
            reasons.append(f"{label}_stale")
    if start["valid"] and end["valid"]:
        if start["segment_id"] != end["segment_id"]:
            reasons.append("snapshot_or_reset_crossing")
        if start["invalid_count"] != end["invalid_count"]:
            reasons.append("invalid_message_crossing")
    return reasons


def _event_key(event):
    return (0, -event.price_offset_units) if event.side == "bid" else (1, event.price_offset_units)


def _template(value, row_id):
    _mapping(value, ("row_id", "adds", "removes", "intents"), "narrow template")
    if value["row_id"] != row_id:
        raise ValueError("Template row_id must be its exact day:A candidate identity")
    groups = []
    seen = set()
    for name in ("adds", "removes"):
        if type(value[name]) is not list:
            raise ValueError("Template event arrays are required")
        events = []
        for event in value[name]:
            _mapping(event, ("side", "price_offset_units", "quantity_units"), "relative event")
            parsed = RelativeLiquidity(**event)
            key = (parsed.side, parsed.price_offset_units)
            if key in seen:
                raise ValueError("Minimal net template cannot duplicate a side/price residual")
            seen.add(key)
            events.append(parsed)
        if events != sorted(events, key=_event_key):
            raise ValueError("Net liquidity must retain canonical bid/ask price order")
        groups.append(tuple(events))
    if type(value["intents"]) is not list:
        raise ValueError("Intention array is required")
    intents = []
    for intent in value["intents"]:
        _mapping(intent, ("side", "quantity_units"), "narrow intent")
        intents.append(Intent(**intent))
    return FlowTemplate(row_id, *groups, tuple(intents))


def _prints(value, a, b, previous_row, previous_time):
    if type(value) is not list:
        raise ValueError("Complete interval print array is required")
    result, ids = [], set()
    for item in value:
        _mapping(
            item,
            ("timestamp_us", "side", "quantity_units", "price_units", "source_row", "trade_id"),
            "observed print",
        )
        print_ = ObservedPrint(**item)
        if (
            not a <= print_.timestamp_us < b
            or print_.timestamp_us < previous_time
            # The source reader may skip an archive's pre-UTC-day prefix;
            # subsequent selected records must still be contiguous.
            or (previous_row is not None and print_.source_row != previous_row + 1)
            or print_.trade_id in ids
        ):
            raise ValueError("Print half-open allocation/full source-row order differs")
        ids.add(print_.trade_id)
        previous_row, previous_time = print_.source_row, print_.timestamp_us
        result.append(print_)
    return tuple(result), previous_row, previous_time


def _pressure(tape, signal_us):
    buy = sell = 0
    for print_ in tape:
        if signal_us - 1_000_000 <= print_.timestamp_us < signal_us:
            if print_.side == "Buy":
                buy += print_.quantity_units
            else:
                sell += print_.quantity_units
    return Fraction(buy - sell, buy + sell) if buy + sell else Fraction(0)


def _validate_label(record, start, end, prints, reasons, row_id):
    """Check sparse net identities; full unchanged books remain producer evidence."""
    alignments = record["print_alignment"]
    if type(alignments) is not list or len(alignments) != len(prints):
        raise ValueError("Print alignment must preserve every source print")
    for alignment in alignments:
        _mapping(
            alignment,
            ("source_row", "domain", "within_engine_span", "absent_at_both_endpoints"),
            "print alignment",
        )
        _integer(alignment["source_row"], "alignment source row", minimum=1)
        for name in ("within_engine_span", "absent_at_both_endpoints"):
            if alignment[name] is not None and type(alignment[name]) is not bool:
                raise ValueError("Alignment flags must be explicit boolean or null")
    if reasons:
        if (
            record["template"] is not None
            or record["effective_levels"] != []
            or record["censored_endpoints"] != []
        ):
            raise ValueError("Unusable label endpoints cannot provide fitted flow")
        for print_, alignment in zip(prints, alignments, strict=True):
            if alignment != {
                "source_row": print_.source_row,
                "domain": "unusable_endpoints",
                "within_engine_span": None,
                "absent_at_both_endpoints": None,
            }:
                raise ValueError("Unusable label alignment differs")
        return None
    template = _template(record["template"], row_id)
    if template.intents != tuple(Intent(t.side, t.quantity_units) for t in prints):
        raise ValueError("Narrow intent tape differs from complete observed directions/quantities")
    boundaries = {
        "bid": max(start["bid_outer_price_units"], end["bid_outer_price_units"]),
        "ask": min(start["ask_outer_price_units"], end["ask_outer_price_units"]),
    }

    def known(side, price):
        return price >= boundaries[side] if side == "bid" else price <= boundaries[side]

    consumed = Counter()
    for print_ in prints:
        side = "ask" if print_.side == "Buy" else "bid"
        if known(side, print_.price_units):
            consumed[(side, print_.price_units)] += print_.quantity_units
    if type(record["effective_levels"]) is not list:
        raise ValueError("Sparse effective level array is required")
    levels, adds, removes, keys = {}, [], [], []
    for level in record["effective_levels"]:
        _mapping(
            level,
            (
                "side",
                "price_units",
                "start_quantity_units",
                "end_quantity_units",
                "consumed_proxy_units",
                "residual_units",
            ),
            "effective level",
        )
        side, price = level["side"], level["price_units"]
        if side not in {"bid", "ask"}:
            raise ValueError("Unknown effective side")
        _integer(price, "effective price", minimum=1, maximum=MAX_UNITS)
        for key in ("start_quantity_units", "end_quantity_units"):
            _integer(level[key], key, maximum=MAX_UNITS)
        _integer(level["consumed_proxy_units"], "consumed quantity")
        if type(level["residual_units"]) is not int:
            raise ValueError("Residual must be an exact signed integer")
        key = side, price
        residual = (
            level["end_quantity_units"]
            - level["start_quantity_units"]
            + level["consumed_proxy_units"]
        )
        if (
            key in levels
            or not known(side, price)
            or level["residual_units"] != residual
            or level["consumed_proxy_units"] != consumed[key]
            or not (residual or level["consumed_proxy_units"])
        ):
            raise ValueError("Sparse net residual/common-domain/consumption identity differs")
        for endpoint, reference in (("start", start), ("end", end)):
            if price == reference[f"{side}_l1"][0] and (
                level[f"{endpoint}_quantity_units"] != reference[f"{side}_l1"][1]
            ):
                raise ValueError("Sparse endpoint L1 quantity differs")
        levels[key] = level
        keys.append((0, -price) if side == "bid" else (1, price))
        if residual:
            event = RelativeLiquidity(side, price - start[f"{side}_l1"][0], abs(residual))
            (adds if residual > 0 else removes).append(event)
    if keys != sorted(keys) or not set(consumed) <= set(levels):
        raise ValueError("Effective level order or complete consumed domain differs")
    if template.adds != tuple(adds) or template.removes != tuple(removes):
        raise ValueError("Template differs from the authenticated minimal net residuals")
    for print_, alignment in zip(prints, alignments, strict=True):
        side = "ask" if print_.side == "Buy" else "bid"
        in_domain = known(side, print_.price_units)
        absent = (
            (
                levels[(side, print_.price_units)]["start_quantity_units"] == 0
                and levels[(side, print_.price_units)]["end_quantity_units"] == 0
            )
            if in_domain
            else None
        )
        expected = {
            "source_row": print_.source_row,
            "domain": "common_visible_domain" if in_domain else "outside_common_visible_domain",
            "within_engine_span": start["engine_us"] <= print_.timestamp_us < end["engine_us"],
            "absent_at_both_endpoints": absent,
        }
        if alignment != expected:
            raise ValueError("Print common-domain/engine alignment differs")
    censored = record["censored_endpoints"]
    if type(censored) is not list or len(censored) != 4:
        raise ValueError("All censored endpoint denominators are required")
    for item, (side, endpoint) in zip(
        censored, (("bid", "start"), ("bid", "end"), ("ask", "start"), ("ask", "end")), strict=True
    ):
        if type(item) is not list or len(item) != 4 or item[:2] != [side, endpoint]:
            raise ValueError("Censored endpoint identity differs")
        reference = start if endpoint == "start" else end
        _integer(item[2], "censored levels", maximum=reference[f"{side}_visible_levels"])
        _integer(item[3], "censored quantity")
        if (item[2] == 0) != (item[3] == 0):
            raise ValueError("Censored count/quantity identity differs")
    return template


def _parse_records(records, day, *, end_us):
    """Sequential pure parser, also usable with a short synthetic day prefix."""
    start_us = day_start_us(day)
    if day not in TRAIN_DAYS or end_us % GRID_US or not start_us < end_us <= start_us + DAY_US:
        raise ValueError("Only approved aligned training day prefixes may be parsed")
    history, tape, rows = {}, deque(), []
    previous_end, previous_row, previous_time = None, None, -1
    counts = {
        "candidate_rows": 0,
        "eligible_rows": 0,
        "observed_prints": 0,
        "observed_quantity_units": 0,
        "exclusions": Counter(),
        "print_alignment": Counter(),
    }
    for record in records:
        a = start_us + counts["candidate_rows"] * GRID_US
        if a >= end_us:
            raise ValueError("Extra candidate beyond the complete grid")
        _mapping(record, RECORD_FIELDS, "sparse reader record")
        if (
            record["day"] != day
            or type(record["grid_us"]) is not list
            or any(type(t) is not int for t in record["grid_us"])
            or record["grid_us"] != [a, a + GRID_US]
            or type(record["eligible"]) is not bool
        ):
            raise ValueError("Candidate must retain complete day:A grid order")
        start = _reference(record["start_prefix"], a)
        end = _reference(record["end_prefix"], a + GRID_US)
        if previous_end is not None and start != previous_end:
            raise ValueError("Adjacent candidate source prefixes differ")
        _prefix_progress(start, end)
        history[a] = start
        signals, values = record["signal_prefixes"], record["features_by_delay"]
        if (
            type(signals) is not list
            or type(values) is not list
            or len(signals) != 3
            or len(values) != 3
        ):
            raise ValueError("All three lag source prefixes and feature slots are required")
        reasons = _label_reasons(start, end, start_us)
        label_reasons = tuple(reasons)
        features = []
        # The current interval's prints and effective labels are not read until
        # every feature has been checked against the strictly previous tape.
        for delay, signal, value in zip(DELAYS_MS, signals, values, strict=True):
            s = a - delay * 1000
            expected_signal = history.get(s) if s >= start_us else None
            if signal is not None:
                _reference(signal, s)
            if signal != expected_signal:
                raise ValueError(
                    "Lag source prefix is unknown or differs from its precise A-delay history"
                )
            feature = _features(value) if value is not None else None
            if signal is None or s - 1_000_000 < start_us:
                reasons.append(f"delay_{delay}_incomplete_past_context")
                expected_feature = None
            elif (
                signal["segment_id"] != start["segment_id"]
                or signal["invalid_count"] != start["invalid_count"]
            ):
                reasons.append(f"delay_{delay}_segment_or_error_crossing")
                expected_feature = None
            elif not _usable(signal):
                reasons.append(f"delay_{delay}_invalid_or_stale_signal")
                expected_feature = None
            else:
                bid, ask = signal["bid_l1"], signal["ask_l1"]
                expected_feature = StateFeatures(
                    Fraction(bid[1] - ask[1], bid[1] + ask[1]),
                    Fraction(20000 * (ask[0] - bid[0]), ask[0] + bid[0]),
                    _pressure(tape, s),
                )
            if feature != expected_feature:
                raise ValueError(
                    "Feature rational differs from causal L1/spread/past-print pressure"
                )
            features.append(feature)
        if record["exclusion_reasons"] != reasons or record["eligible"] != (not reasons):
            raise ValueError("Eligibility/common-delay exclusion denominators differ")
        prints, previous_row, previous_time = _prints(
            record["observed_prints"], a, a + GRID_US, previous_row, previous_time
        )
        template = _validate_label(record, start, end, prints, label_reasons, f"{day}:{a}")
        if record["eligible"]:
            if template is None or any(feature is None for feature in features):
                raise ValueError("Eligible rows require one whole template and every delay")
            rows.append(TrainingRow(day, template, tuple(features)))
            counts["eligible_rows"] += 1
        counts["candidate_rows"] += 1
        counts["observed_prints"] += len(prints)
        counts["observed_quantity_units"] += sum(t.quantity_units for t in prints)
        counts["exclusions"].update(reasons)
        counts["print_alignment"].update(x["domain"] for x in record["print_alignment"])
        tape.extend(prints)
        while tape and tape[0].timestamp_us < a + GRID_US - 6_000_000:
            tape.popleft()
        history.pop(a - 5_000_000 - GRID_US, None)
        previous_end = end
    if counts["candidate_rows"] != (end_us - start_us) // GRID_US:
        raise ValueError("Missing candidate in the complete sequential grid")
    counts["exclusions"] = dict(counts["exclusions"])
    counts["print_alignment"] = dict(counts["print_alignment"])
    return tuple(rows), counts


def _parse_day(path, day):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return _parse_records(
            (_json(line) for line in handle), day, end_us=day_start_us(day) + DAY_US
        )


def _row_json(row):
    def event(value):
        return {
            "side": value.side,
            "price_offset_units": value.price_offset_units,
            "quantity_units": value.quantity_units,
        }

    return {
        "template": {
            "row_id": row.template.row_id,
            "adds": [event(x) for x in row.template.adds],
            "removes": [event(x) for x in row.template.removes],
            "intents": [
                {"side": x.side, "quantity_units": x.quantity_units} for x in row.template.intents
            ],
        },
        "features_by_delay": [_feature_json(x) for x in row.features_by_delay],
    }


def _narrow_row(value):
    _mapping(value, ("template", "features_by_delay"), "model row")
    if type(value["template"]) is not dict or type(value["template"].get("row_id")) is not str:
        raise ValueError("Stable model row identity is required")
    row_id = value["template"]["row_id"]
    day = row_id.split(":", 1)[0]
    template = _template(value["template"], row_id)
    values = value["features_by_delay"]
    if type(values) is not list or len(values) != len(DELAYS_MS):
        raise ValueError("Model row requires all delay features")
    row = TrainingRow(day, template, tuple(_features(value) for value in values))
    earliest = day_start_us(day) + max(DELAYS_MS) * 1000 + 1_000_000
    if int(row_id.split(":", 1)[1]) < earliest:
        raise ValueError("Common-cohort row predates its complete lag/print context")
    return row


def _parameters(pool, content_hash):
    return {
        "kind": "joint_empirical_template_pool",
        "rows": len(pool.rows),
        "row_identity_sha256": pool.row_identity_sha256,
        "row_content_sha256": content_hash,
        "spread_median": [pool.spread_median.numerator, pool.spread_median.denominator],
        "spread_threshold": SPREAD_THRESHOLD,
        "delays_ms": list(DELAYS_MS),
        "cells_per_delay": 18,
        "fallback_min_rows": MIN_CELL_SIZE,
        "sampler": SAMPLER,
        "imbalance_edges": [[-1, 3], [1, 3]],
        "pressure_edges": [[-1, 3], [1, 3]],
        "indexes": [[list(cell) for cell in delay] for delay in pool.indexes],
    }


def _write_pool(pool, output):
    rows_path, params_path = output / "rows.jsonl.gz", output / "parameters.json"
    content = hashlib.sha256()
    with rows_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in pool.rows:
                line = _canonical(_row_json(row)) + b"\n"
                content.update(line)
                compressed.write(line)
        raw.flush()
        os.fsync(raw.fileno())
    content_hash = content.hexdigest()
    write_json(params_path, _parameters(pool, content_hash))
    return [
        {"path": str(rows_path), "sha256": sha256(rows_path), "rows": len(pool.rows)},
        {"path": str(params_path), "sha256": sha256(params_path), "rows": None},
    ], content_hash


def _model_identity(root, config_bytes):
    identity = frozen_identity(root, [root / name for name in FROZEN_PATHS])
    if identity["source_sha256"][CONFIG_PATH] != _hash_bytes(config_bytes):
        raise ValueError("Model configuration changed during source freeze verification")
    return identity


def _bind_extraction(verified):
    state = verified.state
    return {
        "manifest_path": str(verified.manifest_path),
        "manifest_sha256": _hash_bytes(verified.manifest_bytes),
        "prepared_sha256": _hash_bytes(verified.prepared_bytes),
        "run_id": state["run_id"],
        "split": "train",
        "mode": "train",
        "days": state["days"],
        "windows_us": state["windows_us"],
        "git_commit": state["git_commit"],
        "source_sha256": state["source_sha256"],
        "config_sha256": state["config_sha256"],
        "source_config_sha256": reader.SOURCE_SHA,
        "inputs": verified.inputs,
        "outputs": state["outputs"],
    }


def _counts_match(counts, receipt):
    _mapping(
        counts,
        (
            "candidate_rows",
            "eligible_rows",
            "observed_prints",
            "observed_quantity_units",
            "exclusions",
            "print_alignment",
        ),
        "parsed daily counts",
    )
    for name in ("candidate_rows", "eligible_rows", "observed_prints", "observed_quantity_units"):
        _integer(counts[name], name)
    for name in ("rows", "eligible_rows", "observed_prints", "observed_quantity_units"):
        _integer(receipt[name], name)
    _count_map(counts["exclusions"], "parsed exclusions")
    alignment = _count_map(counts["print_alignment"], "parsed print alignment")
    if (
        counts["eligible_rows"] > counts["candidate_rows"]
        or sum(counts["exclusions"].values()) < counts["candidate_rows"] - counts["eligible_rows"]
        or (counts["eligible_rows"] == counts["candidate_rows"] and counts["exclusions"])
        or sum(alignment.values()) != counts["observed_prints"]
    ):
        raise ValueError("Invalid complete candidate/exclusion denominators")
    expected = {
        "candidate_rows": receipt["rows"],
        **{
            name: receipt[name]
            for name in (
                "eligible_rows",
                "observed_prints",
                "observed_quantity_units",
                "exclusions",
                "print_alignment",
            )
        },
    }
    if counts != expected:
        raise ValueError("Parsed complete candidate/eligible/exclusion counts differ from master")


def fit_flow_templates(root, manifest_path, output, *, split="train", prepare_only=False):
    """Fit one prescribed common-cohort pool, or publish only an authenticated preflight."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "stage": "verifying_inputs",
        "outputs_published": False,
        "outputs": [],
        "run_id": uuid4().hex,
        "split": split,
        "mode": "train",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        },
    }
    manifest = output / "manifest.json"
    write_json(manifest, state)
    try:
        if split != "train":
            raise ValueError("Development/event model fitting is closed")
        config_bytes = (root / CONFIG_PATH).read_bytes()
        plan = _json(config_bytes)
        if not _same_json(plan, expected_protocol()):
            raise ValueError("Model protocol differs from the fixed machine proposal")
        identity = _model_identity(root, config_bytes)
        verified = verify_train_extraction(root, manifest_path)
        binding = _bind_extraction(verified)
        state.update(
            **identity,
            config=plan,
            config_sha256=_hash_bytes(config_bytes),
            source_config_sha256=reader.SOURCE_SHA,
            extraction_config_sha256=_hash_bytes(verified.config_bytes),
            input_extraction=binding,
            provenance_sha256=_hash_bytes(_canonical(binding)),
            stage="inputs_verified_before_candidate_parsing",
        )
        _committed_sources(root, identity["git_commit"], identity["source_sha256"], FROZEN_PATHS)
        _fence_extraction(verified, rehash_raw=False)
        prepared_path = output / "prepared_receipt.json"
        write_json(prepared_path, state)
        prepared_bytes = prepared_path.read_bytes()
        if not _same_json(_json(prepared_bytes), state):
            raise ValueError("Written model prepared receipt differs from verified state")
        state["prepared_receipt_sha256"] = _hash_bytes(prepared_bytes)

        def check_prepared():
            if prepared_path.read_bytes() != prepared_bytes:
                raise ValueError("Model prepared receipt changed before parsing/publication")

        check_prepared()
        if prepare_only:
            state.update(
                status="prepared_no_candidate_values_read",
                completed_at=datetime.now(UTC).isoformat(),
            )
            write_json(manifest, state)
            return state
        cohort, day_counts = [], {}
        for day, path, receipt in zip(
            TRAIN_DAYS, verified.output_paths, verified.state["outputs"], strict=True
        ):
            check_prepared()
            rows, counts = _parse_day(path, day)
            _counts_match(counts, receipt)
            cohort.extend(rows)
            day_counts[day] = counts
        pool = fit_template_pool(tuple(cohort))
        outputs, content_hash = _write_pool(pool, output)
        _fence_extraction(verified, rehash_raw=True)
        _committed_sources(root, identity["git_commit"], identity["source_sha256"], FROZEN_PATHS)
        if (root / CONFIG_PATH).read_bytes() != config_bytes:
            raise ValueError("Model configuration changed during fitting")
        if any(sha256(Path(receipt["path"])) != receipt["sha256"] for receipt in outputs):
            raise ValueError("Model outputs changed before publication")
        check_prepared()
        state.update(
            status="complete",
            stage="model_outputs_verified",
            outputs_published=True,
            outputs=outputs,
            day_counts=day_counts,
            candidate_rows=len(TRAIN_DAYS) * CANDIDATES_PER_DAY,
            eligible_rows=len(pool.rows),
            row_content_sha256=content_hash,
            row_identity_sha256=pool.row_identity_sha256,
            completed_at=datetime.now(UTC).isoformat(),
        )
        write_json(manifest, state)
        return state
    except BaseException as error:
        state.update(
            status="failed",
            outputs_published=False,
            outputs=[],
            error=repr(error),
            completed_at=datetime.now(UTC).isoformat(),
        )
        write_json(manifest, state)
        raise


@dataclass(frozen=True, slots=True)
class ModelReceiptIdentity:
    run_id: str
    manifest_path: str
    manifest_sha256: str
    compressed_rows_sha256: str
    row_content_sha256: str
    parameters_sha256: str
    provenance_sha256: str
    git_commit: str
    source_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class LoadedTemplateModel:
    pool: TemplatePool
    identity: ModelReceiptIdentity


def load_flow_templates(root, manifest_path):
    """Load saved cells/threshold and validate them; no raw/label readers or refit.

    A subsequent runner must require an independent review receipt bound to this
    manifest identity before real generation. This function does not issue one.
    """
    root = Path(root).resolve()
    manifest_path = _path(root, manifest_path)
    if manifest_path.name != "manifest.json":
        raise ValueError("An immutable model manifest is required")
    manifest_bytes = manifest_path.read_bytes()
    state = _json(manifest_bytes)
    if (
        type(state) is not dict
        or state.get("status") != "complete"
        or state.get("stage") != "model_outputs_verified"
        or state.get("outputs_published") is not True
        or state.get("split") != "train"
        or state.get("mode") != "train"
        or state.get("candidate_rows") != len(TRAIN_DAYS) * CANDIDATES_PER_DAY
    ):
        raise ValueError("Only complete published train models may load")
    _integer(state["candidate_rows"], "model candidate rows")
    if type(state.get("run_id")) is not str or not state["run_id"]:
        raise ValueError("Missing model run identity")
    if type(state.get("runtime")) is not dict or any(
        not state["runtime"].get(k) for k in ("python", "python_implementation")
    ):
        raise ValueError("Missing model runtime identity")
    _timestamps(state)
    config_bytes = (root / CONFIG_PATH).read_bytes()
    if (
        not _same_json(state.get("config"), expected_protocol())
        or not _same_json(_json(config_bytes), expected_protocol())
        or state.get("config_sha256") != _hash_bytes(config_bytes)
        or state.get("source_config_sha256") != reader.SOURCE_SHA
        or state.get("extraction_config_sha256") != sha256(root / reader.CONFIG_PATH)
        or not _same_json(
            _json((root / reader.CONFIG_PATH).read_bytes()), reader.expected_protocol()
        )
        or sha256(root / reader.SOURCE_PATH) != reader.SOURCE_SHA
    ):
        raise ValueError("Loaded model configuration identity differs")
    _committed_sources(root, state.get("git_commit"), state.get("source_sha256"), FROZEN_PATHS)
    binding = state.get("input_extraction")
    _mapping(
        binding,
        (
            "manifest_path",
            "manifest_sha256",
            "prepared_sha256",
            "run_id",
            "split",
            "mode",
            "days",
            "windows_us",
            "git_commit",
            "source_sha256",
            "config_sha256",
            "source_config_sha256",
            "inputs",
            "outputs",
        ),
        "model extraction binding",
    )
    if (
        not _digest(binding["manifest_sha256"])
        or not _digest(binding["prepared_sha256"])
        or type(binding["run_id"]) is not str
        or not binding["run_id"]
    ):
        raise ValueError("Missing bound extraction receipt identity")
    extraction_manifest = _path(root, binding["manifest_path"])
    if extraction_manifest.name != "manifest.json":
        raise ValueError("Bound extraction master path differs")
    if (
        type(binding) is not dict
        or binding.get("split") != "train"
        or binding.get("mode") != "train"
        or binding.get("days") != list(TRAIN_DAYS)
        or not _same_json(
            binding.get("windows_us"),
            {day: [day_start_us(day), day_start_us(day) + DAY_US] for day in TRAIN_DAYS},
        )
        or binding.get("source_config_sha256") != reader.SOURCE_SHA
        or binding.get("config_sha256") != state["extraction_config_sha256"]
        or state.get("provenance_sha256") != _hash_bytes(_canonical(binding))
    ):
        raise ValueError("Loaded model provenance/content binding differs")
    _normalized_inputs(root, binding.get("inputs"))  # Validate identities; do not open raw files.
    _committed_sources(
        root,
        binding.get("git_commit"),
        binding.get("source_sha256"),
        (reader.CONFIG_PATH, reader.SOURCE_PATH, *reader.READER_SOURCES),
    )
    prepared_bytes = (manifest_path.parent / "prepared_receipt.json").read_bytes()
    if state.get("prepared_receipt_sha256") != _hash_bytes(prepared_bytes):
        raise ValueError("Model prepared receipt hash differs")
    prepared = _json(prepared_bytes)
    matching = (
        "run_id",
        "split",
        "mode",
        "started_at",
        "runtime",
        "git_commit",
        "source_sha256",
        "config",
        "config_sha256",
        "source_config_sha256",
        "extraction_config_sha256",
        "input_extraction",
        "provenance_sha256",
    )
    if (
        type(prepared) is not dict
        or prepared.get("status") != "running"
        or prepared.get("stage") != "inputs_verified_before_candidate_parsing"
        or prepared.get("outputs_published") is not False
        or prepared.get("outputs") != []
        or prepared.get("completed_at") is not None
        or any(not _same_json(prepared.get(key), state.get(key)) for key in matching)
    ):
        raise ValueError("Model prepared receipt differs")
    _mapping(state.get("day_counts"), TRAIN_DAYS, "model candidate days")
    receipts = binding.get("outputs")
    if type(receipts) is not list or len(receipts) != len(TRAIN_DAYS):
        raise ValueError("Model extraction daily denominators are missing")
    eligible = 0
    for day, receipt in zip(TRAIN_DAYS, receipts, strict=True):
        _mapping(receipt, OUTPUT_FIELDS, "bound extraction daily output")
        if (
            receipt["rows"] != CANDIDATES_PER_DAY
            or not _digest(receipt["sha256"])
            or _path(root, receipt["path"])
            != extraction_manifest.parent / f"{day}_effective_flow.jsonl.gz"
        ):
            raise ValueError("Model cannot bind a partial extraction")
        _counts_match(state["day_counts"][day], receipt)
        eligible += _integer(receipt["eligible_rows"], "eligible rows", maximum=CANDIDATES_PER_DAY)
    if _integer(state.get("eligible_rows"), "model eligible rows", minimum=1) != eligible:
        raise ValueError("Model whole common-cohort denominator differs")
    outputs = state.get("outputs")
    if type(outputs) is not list or len(outputs) != 2:
        raise ValueError("Both immutable model files are required")
    paths = []
    for receipt, name, rows in zip(
        outputs, ("rows.jsonl.gz", "parameters.json"), (eligible, None), strict=True
    ):
        _mapping(receipt, ("path", "sha256", "rows"), "model output")
        path = _path(root, receipt["path"])
        if (
            path != manifest_path.parent / name
            or receipt["rows"] != rows
            or not _digest(receipt["sha256"])
            or sha256(path) != receipt["sha256"]
        ):
            raise ValueError("Actual model output path/hash/row receipt differs")
        paths.append(path)
    if _gzip_rows(paths[0]) != eligible:
        raise ValueError("Actual model row count differs")
    parameter_bytes = paths[1].read_bytes()
    if _hash_bytes(parameter_bytes) != outputs[1]["sha256"]:
        raise ValueError("Parameters changed before parsing")
    if sha256(paths[0]) != outputs[0]["sha256"]:
        raise ValueError("Model rows changed before parsing")
    if (
        manifest_path.read_bytes() != manifest_bytes
        or (manifest_path.parent / "prepared_receipt.json").read_bytes() != prepared_bytes
    ):
        raise ValueError("Model receipts changed before parsing")
    _committed_sources(root, state["git_commit"], state["source_sha256"], FROZEN_PATHS)
    parameters = _json(parameter_bytes)
    _mapping(
        parameters,
        (
            "kind",
            "rows",
            "row_identity_sha256",
            "row_content_sha256",
            "spread_median",
            "spread_threshold",
            "delays_ms",
            "cells_per_delay",
            "fallback_min_rows",
            "sampler",
            "imbalance_edges",
            "pressure_edges",
            "indexes",
        ),
        "frozen model parameters",
    )
    fixed = {
        "kind": "joint_empirical_template_pool",
        "rows": eligible,
        "spread_threshold": SPREAD_THRESHOLD,
        "delays_ms": list(DELAYS_MS),
        "cells_per_delay": 18,
        "fallback_min_rows": MIN_CELL_SIZE,
        "sampler": SAMPLER,
        "imbalance_edges": [[-1, 3], [1, 3]],
        "pressure_edges": [[-1, 3], [1, 3]],
    }
    if not _same_json({key: parameters[key] for key in fixed}, fixed):
        raise ValueError("Saved model parameters differ from the fixed fit")
    rows, content = [], hashlib.sha256()
    with gzip.open(paths[0], "rb") as handle:
        for line in handle:
            row = _narrow_row(_json(line))
            if line != _canonical(_row_json(row)) + b"\n":
                raise ValueError("Model rows must retain their exact canonical content")
            content.update(line)
            rows.append(row)
    if (
        len(rows) != eligible
        or content.hexdigest() != parameters["row_content_sha256"]
        or content.hexdigest() != state.get("row_content_sha256")
        or parameters["row_identity_sha256"] != state.get("row_identity_sha256")
    ):
        raise ValueError("Whole model row content or identity hashes differ")
    row_days = Counter(row.day for row in rows)
    if any(row_days[day] != state["day_counts"][day]["eligible_rows"] for day in TRAIN_DAYS):
        raise ValueError("Loaded row counts differ from complete daily common cohorts")
    indexes = parameters["indexes"]
    if (
        type(indexes) is not list
        or len(indexes) != 3
        or any(type(delay) is not list or len(delay) != 18 for delay in indexes)
        or any(type(cell) is not list for delay in indexes for cell in delay)
        or any(type(index) is not int for delay in indexes for cell in delay for index in cell)
    ):
        raise ValueError("Frozen cell indexes must be exact integer arrays")
    pool = TemplatePool(
        tuple(rows),
        _fraction(parameters["spread_median"]),
        tuple(tuple(tuple(cell) for cell in delay) for delay in indexes),
        parameters["row_identity_sha256"],
    )
    if (
        manifest_path.read_bytes() != manifest_bytes
        or (manifest_path.parent / "prepared_receipt.json").read_bytes() != prepared_bytes
        or paths[1].read_bytes() != parameter_bytes
        or any(
            sha256(path) != receipt["sha256"] for path, receipt in zip(paths, outputs, strict=True)
        )
    ):
        raise ValueError("Model artifact changed during loading")
    _committed_sources(root, state["git_commit"], state["source_sha256"], FROZEN_PATHS)
    identity = ModelReceiptIdentity(
        state["run_id"],
        str(manifest_path),
        _hash_bytes(manifest_bytes),
        outputs[0]["sha256"],
        content.hexdigest(),
        outputs[1]["sha256"],
        state["provenance_sha256"],
        state["git_commit"],
        tuple(sorted(state["source_sha256"].items())),
    )
    return LoadedTemplateModel(pool, identity)
