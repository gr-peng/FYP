"""Streaming, publication-causal conditional print-price diagnostics.

Public trades are prints, not identified parent orders. Every sweep is read-only.
Full-source integrity comes from a matching, current audit plus actual byte hashes;
windowed readers do not claim to reverify a complete archive's CRC themselves.
"""

import csv
import gzip
import hashlib
import json
import platform
import subprocess
import zipfile
import zlib
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from behavioral_market.data.archive import write_json
from behavioral_market.data.bybit_book import BookIntegrityError, BybitBook, units

DELAYS = (0, 10, 50, 100, 250)
TRAIN_DAYS = ["2024-11-01", "2024-11-02", "2024-11-03"]
SPLITS = {
    "train": TRAIN_DAYS,
    "development": ["2024-11-04"],
    "event_test": ["2024-11-05", "2024-11-06", "2024-11-07"],
}
AUDIT_SOURCES = (
    "scripts/audit_crypto_microstructure.py",
    "experiments/bybit_reconstruction.py",
    "experiments/bybit_trades.py",
    "src/behavioral_market/data/bybit_book.py",
)
EXECUTION_SOURCES = (
    "experiments/crypto_execution_diagnostic.py",
    "scripts/run_crypto_execution_diagnostic.py",
    "src/behavioral_market/data/archive.py",
    *AUDIT_SOURCES,
)
GROUPS = ("eligible", "cross_l1", "shared_eligible", "shared_cross_l1")
ERRORS = ("sweep_abs_bps", "l1_abs_bps", "mid_abs_bps", "sweep_signed_bps")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def day_start_us(day):
    return (date.fromisoformat(day) - date(1970, 1, 1)).days * 86_400_000_000


@dataclass(frozen=True)
class Trade:
    timestamp_us: int
    side: str
    quantity_units: int
    price_units: int
    trade_id: str
    source_row: int


class PublishedBookCursor:
    """One raw-order cursor and one book; pending events never change live state."""

    def __init__(self, messages, day, delay_ms):
        if delay_ms not in DELAYS:
            raise ValueError("Delay is outside the frozen set")
        self.messages = iter(enumerate(messages, 1))
        self.start = day_start_us(day)
        self.end = self.start + 86_400_000_000
        self.delay_ms = delay_ms
        self.book = BybitBook()
        self.pending = None
        self.exhausted = False
        self.prefix_ts = None
        self.available_us = -1
        self.message_index = None
        self.segment = -1
        self.segment_start_us = None
        self.last_time = -1
        self.invalid_messages = 0

    def _peek(self):
        if self.pending is not None or self.exhausted:
            return
        try:
            index, message = next(self.messages)
        except StopIteration:
            self.exhausted = True
            return
        ts, cts = message["ts"], message["cts"]
        if any(type(t) is not int or t <= 0 for t in (ts, cts)):
            raise BookIntegrityError("Unusable raw timestamp; cannot assign publication order")
        prefix = (
            ts
            if message["type"] == "snapshot" or self.prefix_ts is None
            else max(self.prefix_ts, ts)
        )
        available = max(
            (prefix + 1) * 1000 + self.delay_ms * 1000,
            (cts + 1) * 1000,
            self.available_us,  # Preserve the raw-order barrier across snapshots too.
        )
        self.pending = (index, message, prefix, available)

    def advance(self, timestamp_us):
        if timestamp_us < self.last_time:
            raise ValueError("Trade times must be nondecreasing; no silent sorting")
        self.last_time = timestamp_us
        while True:
            self._peek()
            if self.pending is None or self.pending[3] >= timestamp_us:
                break
            index, message, prefix, available = self.pending
            self.pending = None
            self.prefix_ts, self.available_us = prefix, available
            self.message_index = index
            try:
                update = self.book.apply(message)
                if update.kind == "snapshot":
                    self.segment += 1
                    self.segment_start_us = update.engine_ms * 1000
            except BookIntegrityError:
                self.invalid_messages += 1
                # BybitBook invalidates itself; only a later valid snapshot recovers.

    def predict(self, timestamp_us, side, quantity_units):
        """The observed execution price is deliberately absent from this interface."""
        if side not in {"Buy", "Sell"} or type(quantity_units) is not int or quantity_units <= 0:
            raise ValueError("Invalid conditioning side/quantity")
        if type(timestamp_us) is not int or not self.start <= timestamp_us < self.end:
            raise ValueError("Trade outside selected UTC day")
        self.advance(timestamp_us)
        result = {
            "delay_ms": self.delay_ms,
            "reason": "no_state" if self.message_index is None else "invalid_state",
            "segment_id": self.segment if self.segment >= 0 else None,
            "book_message_index": self.message_index,
            "publication_prefix_ms": self.prefix_ts,
            "available_us": self.available_us if self.message_index is not None else None,
            "age_bin": "unknown",
            "size_bin": "unknown",
            "cross_l1": False,
        }
        if not self.book.valid:
            return result
        update = self.book.last_update
        age = timestamp_us - update.engine_ms * 1000
        result.update(
            cts_ms=update.engine_ms,
            update_id=update.update_id,
            sequence=update.sequence,
            age_us=age,
        )
        if not (
            self.start <= update.engine_ms * 1000 < self.end
            and self.start <= self.available_us < self.end
            and self.segment_start_us is not None
            and self.start <= self.segment_start_us < self.end
        ):
            result["reason"] = "outside_day_state"
            return result
        result["age_bin"] = next(
            (
                label
                for edge, label in (
                    (25_000, "[0,25)ms"),
                    (100_000, "[25,100)ms"),
                    (250_000, "[100,250)ms"),
                    (1_000_000, "[250,1000)ms"),
                )
                if age < edge
            ),
            "[1000,+inf)ms",
        )
        bid, _ = next(self.book.levels("bid"))
        ask, _ = next(self.book.levels("ask"))
        opposite = "ask" if side == "Buy" else "bid"
        l1_price, l1_quantity = next(self.book.levels(opposite))
        quantity_text = f"{quantity_units // 1000}.{quantity_units % 1000:03d}"
        sweep = self.book.sweep(side.lower(), quantity_text)
        cost = sum(fill["price_units"] * fill["quantity_units"] for fill in sweep["fills"])
        result.update(
            l1_price_units=l1_price,
            l1_quantity_units=l1_quantity,
            mid_twice_units=bid + ask,
            covered_quantity_units=sweep["filled_units"],
            unresolved_quantity_units=sweep["unfilled_units"],
            sweep_cost_product_units=str(cost),  # Exact arbitrary-size integer.
            sweep_price_denominator=sweep["filled_units"],
            crossed_levels=len(sweep["fills"]),
            cross_l1=quantity_units > l1_quantity,
            size_bin="<=1"
            if quantity_units <= l1_quantity
            else ("(1,5]" if quantity_units <= 5 * l1_quantity else ">5"),
            reason="stale"
            if age >= 1_000_000
            else ("partial_depth" if sweep["unfilled_units"] else "eligible"),
        )
        return result


def diagnostic_rows(trades, messages_by_delay, day):
    """Yield five paired rows per print, retaining neither prints nor past books."""
    if set(messages_by_delay) != set(DELAYS):
        raise ValueError("All five frozen delays are required")
    cursors = {d: PublishedBookCursor(messages_by_delay[d], day, d) for d in DELAYS}
    previous = -1
    for trade in trades:
        if (
            trade.timestamp_us < previous
            or type(trade.price_units) is not int
            or trade.price_units <= 0
            or not trade.trade_id.strip()
        ):
            raise ValueError("Invalid or reversed trade; do not repair the input")
        previous = trade.timestamp_us
        predictions = [
            cursors[d].predict(trade.timestamp_us, trade.side, trade.quantity_units) for d in DELAYS
        ]
        shared = all(row["reason"] == "eligible" for row in predictions)
        shared_cross = shared and all(row["cross_l1"] for row in predictions)
        for row in predictions:
            row.update(
                day=day,
                trade_id=trade.trade_id,
                trade_row=trade.source_row,
                timestamp_us=trade.timestamp_us,
                side=trade.side,
                quantity_units=trade.quantity_units,
                observed_price_units=trade.price_units,
                shared_eligible=shared,
                shared_cross_l1=shared_cross,
            )
            if row["reason"] == "eligible":
                observed = Fraction(trade.price_units)
                sweep = Fraction(int(row["sweep_cost_product_units"]), trade.quantity_units)
                l1 = Fraction(row["l1_price_units"])
                mid = Fraction(row["mid_twice_units"], 2)
                for name, prediction in (("sweep", sweep), ("l1", l1), ("mid", mid)):
                    row[f"{name}_abs_bps"] = float(abs(observed - prediction) / prediction * 10000)
                sign = 1 if trade.side == "Buy" else -1
                row["sweep_signed_bps"] = float(sign * (observed - sweep) / sweep * 10000)
            yield row


_INTEGER_COLUMNS = (
    "trade_row",
    "timestamp_us",
    "quantity_units",
    "observed_price_units",
    "delay_ms",
    "segment_id",
    "book_message_index",
    "publication_prefix_ms",
    "available_us",
    "cts_ms",
    "update_id",
    "sequence",
    "age_us",
    "l1_price_units",
    "l1_quantity_units",
    "mid_twice_units",
    "covered_quantity_units",
    "unresolved_quantity_units",
    "sweep_price_denominator",
    "crossed_levels",
)
ROW_SCHEMA = pa.schema(
    [(name, pa.int64()) for name in _INTEGER_COLUMNS]
    + [
        (name, pa.string())
        for name in (
            "day",
            "trade_id",
            "side",
            "reason",
            "age_bin",
            "size_bin",
            "sweep_cost_product_units",
        )
    ]
    + [(name, pa.bool_()) for name in ("cross_l1", "shared_eligible", "shared_cross_l1")]
    + [(name, pa.float64()) for name in ERRORS],
    metadata={
        b"interpretation": b"Conditional print-price diagnostic, not parent-order fill validation",
        b"exact_prices": b"Price x100; quantity x1000; cost product stored as integer text",
        b"errors": b"Float64 reporting values; recompute exact ratios from retained integers",
        b"quantiles": b"Not computed; all rows retained for subsequent exact analysis",
    },
)


class Accumulator:
    def __init__(self):
        self.count = 0
        self.known_coverage_count = 0
        self.covered_quantity = self.unresolved_quantity = 0
        self.reasons, self.ages, self.sizes = Counter(), Counter(), Counter()
        self.groups = {group: {"n": 0, **dict.fromkeys(ERRORS, 0.0)} for group in GROUPS}

    def add(self, row):
        self.count += 1
        self.reasons[row["reason"]] += 1
        self.ages[row["age_bin"]] += 1
        self.sizes[row["size_bin"]] += 1
        if row.get("covered_quantity_units") is not None:
            self.known_coverage_count += 1
            self.covered_quantity += row["covered_quantity_units"]
            self.unresolved_quantity += row["unresolved_quantity_units"]
        eligible = row["reason"] == "eligible"
        flags = (
            eligible,
            eligible and row["cross_l1"],
            row["shared_eligible"],
            row["shared_cross_l1"],
        )
        for group, include in zip(GROUPS, flags, strict=True):
            if include:
                stats = self.groups[group]
                stats["n"] += 1
                for name in ERRORS:
                    stats[name] += row[name]

    def report(self):
        groups = {}
        for name, stats in self.groups.items():
            n = stats["n"]
            groups[name] = {
                "n": n,
                **{key + "_mean": stats[key] / n if n else None for key in ERRORS},
                "sweep_minus_l1_mae_bps": (
                    (stats["sweep_abs_bps"] - stats["l1_abs_bps"]) / n if n else None
                ),
            }
        return {
            "candidate_count": self.count,
            "reasons": dict(self.reasons),
            "known_visible_coverage_count": self.known_coverage_count,
            "covered_quantity_units_sum": self.covered_quantity,
            "unresolved_quantity_units_sum": self.unresolved_quantity,
            "age_bins": dict(self.ages),
            "size_bins": dict(self.sizes),
            "groups": groups,
        }


def write_day(rows, output, day, start_us, end_us, row_group_size=65536):
    """Bounded row buffers; exact observations retained instead of guessed quantiles."""
    start_of_day = day_start_us(day)
    if (
        start_us % 300_000_000
        or end_us % 300_000_000
        or not start_of_day <= start_us < end_us <= start_of_day + 86_400_000_000
    ):
        raise ValueError("Scoring windows must contain whole UTC five-minute blocks of one day")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    destination = output / f"{day}_prints.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    blocks, daily = {}, {delay: Accumulator() for delay in DELAYS}
    buffer, count = [], 0
    try:
        with pq.ParquetWriter(temporary, ROW_SCHEMA, compression="zstd") as writer:
            for row in rows:
                if not start_us <= row["timestamp_us"] < end_us:
                    raise ValueError("Row escaped the frozen scoring window")
                delay = row["delay_ms"]
                bucket = row["timestamp_us"] // 300_000_000 * 300_000_000
                key = (bucket, delay)
                if key not in blocks:
                    blocks[key] = Accumulator()
                blocks[key].add(row)
                daily[delay].add(row)
                buffer.append(row)
                count += 1
                if len(buffer) >= row_group_size:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=ROW_SCHEMA))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=ROW_SCHEMA))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    block_rows = []
    for bucket in range(start_us, end_us, 300_000_000):
        for delay in DELAYS:
            report = blocks.get((bucket, delay), Accumulator()).report()
            block_rows.append(
                {
                    "bucket_ms": bucket // 1000,
                    "delay_ms": delay,
                    "candidate_count": report["candidate_count"],
                    "report_json": json.dumps(report, allow_nan=False, sort_keys=True),
                }
            )
    block_path = output / f"{day}_blocks.parquet"
    block_schema = pa.schema(
        [
            ("bucket_ms", pa.int64()),
            ("delay_ms", pa.int64()),
            ("candidate_count", pa.int64()),
            ("report_json", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(block_rows, schema=block_schema), block_path)
    summary_path = output / f"{day}_summary.json"
    write_json(
        summary_path,
        {
            "day": day,
            "window_us": [start_us, end_us],
            "delays": {str(delay): accumulator.report() for delay, accumulator in daily.items()},
            "quantiles_computed": False,
            "quantile_basis": "Every observation and exact price ratio retained in print Parquet",
            "aggregation": "Float64 means; no independence or full-fill-validity claim",
            "print_delay_rows": count,
        },
    )
    return [
        {"path": str(path), "sha256": sha256(path), "rows": number}
        for path, number in (
            (destination, count),
            (block_path, len(block_rows)),
            (summary_path, None),
        )
    ]


def _trade_rows(handle, day, start_us, end_us):
    reader = csv.DictReader(handle)
    required = {"timestamp", "symbol", "side", "size", "price", "trdMatchID"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Trade schema differs from audited schema")
    previous = -1
    for index, row in enumerate(reader, 1):
        timestamp = units(row["timestamp"], 6)
        if timestamp < previous or row["symbol"] != "BTCUSDT":
            raise ValueError("Trade identity/order differs from audit")
        previous = timestamp
        if timestamp >= end_us:
            break
        if timestamp < start_us:
            continue
        if not day_start_us(day) <= timestamp < day_start_us(day) + 86_400_000_000:
            raise ValueError("Trade outside selected day")
        yield Trade(
            timestamp,
            row["side"],
            units(row["size"], 3),
            units(row["price"], 2),
            row["trdMatchID"],
            index,
        )


def score_raw_day(paths, output, day, start_us, end_us, row_group_size):
    with ExitStack() as stack:
        messages = {}
        for delay in DELAYS:
            archive = stack.enter_context(zipfile.ZipFile(paths["orderbook"]))
            names = archive.namelist()
            if len(names) != 1 or names[0] != f"{day}_BTCUSDT_ob500.data":
                raise ValueError("Unexpected orderbook ZIP member")
            handle = stack.enter_context(archive.open(names[0]))
            messages[delay] = map(json.loads, handle)
        trades = stack.enter_context(gzip.open(paths["trades"], "rt", encoding="utf-8", newline=""))
        rows = diagnostic_rows(_trade_rows(trades, day, start_us, end_us), messages, day)
        return write_day(rows, output, day, start_us, end_us, row_group_size)


def frozen_identity(root, paths):
    """No uncommitted execution/config changes may enter a real diagnostic run."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    hashes = {}
    for path in paths:
        relative = str(Path(path).resolve().relative_to(root.resolve()))
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        actual = (root / relative).read_bytes()
        if actual != committed:
            raise ValueError(f"Execution input is not frozen in commit {commit}: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return {"git_commit": commit, "source_sha256": hashes}


def validate_plan(plan, source, mode):
    if (
        plan["splits"] != SPLITS
        or source["splits"] != SPLITS
        or plan["runner_allowed_splits"] != ["train"]
        or plan["source_study_id"] != "crypto_microstructure_20261011"
        or source["study_id"] != plan["source_study_id"]
        or source["venue"] != "bybit"
        or source["market"] != "linear_usdt_perpetual"
        or source["symbol"] != "BTCUSDT"
        or source["book_depth"] != 500
        or source["price_storage_scale"] != 100
        or source["quantity_storage_scale"] != 1000
        or plan["delays_ms"] != list(DELAYS)
        or plan["primary_delay_ms"] != 0
        or plan["max_book_age_us"] != 1_000_000
        or plan["block_seconds"] != 300
        or plan["age_bin_edges_us"] != [25_000, 100_000, 250_000, 1_000_000]
        or plan["baselines"] != ["visible_sweep", "opposite_l1", "midpoint"]
        or plan["smoke"]
        != {
            "day": "2024-11-01",
            "start_second_utc": 0,
            "end_second_utc": 3600,
        }
    ):
        raise ValueError("Plan differs from the frozen November train-only protocol")
    if mode not in {"smoke", "train"}:
        raise ValueError("Only smoke or train mode is available")
    return ["2024-11-01"] if mode == "smoke" else TRAIN_DAYS


def audited_inputs(root, source, source_hash, day, current_sources):
    if day not in TRAIN_DAYS:
        raise ValueError("Raw input access is restricted to original November train dates")
    receipts = {}
    for kind in ("orderbook", "trades"):
        url = source["urls"][kind].format(date=day)
        raw = root / "data/raw" / source["study_id"] / kind / url.rsplit("/", 1)[-1]
        meta_path = raw.with_name(raw.name + ".metadata.json")
        audit_path = root / "outputs" / source["study_id"] / "integrity" / f"{day}_{kind}.json"
        # Parse and hash the same bytes. Reading the hash later could accidentally
        # bind a parsed successful audit to the bytes of a newly failed re-audit.
        meta_bytes, audit_bytes = meta_path.read_bytes(), audit_path.read_bytes()
        meta, audit = json.loads(meta_bytes), json.loads(audit_bytes)
        identity = {
            "url": url,
            "split": "train",
            "day": day,
            "kind": kind,
            "config_sha256": source_hash,
            **{key: source[key] for key in ("venue", "market", "symbol")},
        }
        if any(meta.get(key) != value for key, value in identity.items()):
            raise ValueError("Raw metadata identity differs")
        if (
            audit.get("status") != "complete"
            or not audit.get("audit_run_id")
            or any(
                audit.get(key) != identity[key]
                for key in (
                    "split",
                    "day",
                    "kind",
                    "config_sha256",
                )
            )
            or audit.get("processing_source_sha256") != current_sources
        ):
            raise ValueError("Current matching complete audit required")
        if kind == "orderbook":
            if (
                not audit.get("zip_crc_verified")
                or not audit.get("structural_pass")
                or audit["errors"]
            ):
                raise ValueError("First runner requires a structurally valid whole LOB source")
        elif (
            audit.get("quality") != "valid"
            or not audit.get("records_scan_complete")
            or audit.get("gzip_integrity")
            != {
                "status": "passed",
                "read_to_eof": True,
                "crc_verified": True,
            }
        ):
            raise ValueError("Full valid trade tape and GZIP integrity required")
        actual_hash = sha256(raw)
        if (
            raw.stat().st_size != meta["bytes"]
            or actual_hash != meta["sha256"]
            or actual_hash != audit.get("input_sha256")
        ):
            raise ValueError("Actual raw bytes differ from audited source")
        if meta_path.read_bytes() != meta_bytes or audit_path.read_bytes() != audit_bytes:
            raise ValueError("Input receipt changed during preflight")
        receipts[kind] = {
            "path": str(raw),
            "sha256": actual_hash,
            "bytes": meta["bytes"],
            "metadata_path": str(meta_path),
            "metadata_sha256": hashlib.sha256(meta_bytes).hexdigest(),
            "audit_path": str(audit_path),
            "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "audit_run_id": audit["audit_run_id"],
        }
    return receipts


def run_diagnostic(root, config_path, output, *, split="train", mode="smoke", prepare_only=False):
    if split != "train":
        raise ValueError("Development and event-test execution diagnostics are closed")
    root, config_path, output = Path(root), Path(config_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Immutable run directory; never reuse old results.
    manifest_path = output / "manifest.json"
    state = {
        "status": "running",
        "run_id": uuid4().hex,
        "outputs_published": False,
        "split": split,
        "mode": mode,
        "outputs": [],
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pyarrow": pa.__version__,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "platform": platform.platform(),
        },
    }
    write_json(manifest_path, state)
    try:
        plan_bytes = config_path.read_bytes()
        plan_hash = hashlib.sha256(plan_bytes).hexdigest()
        plan = json.loads(plan_bytes)
        if plan["source_config"] != "config/crypto_microstructure_20261011.json":
            raise ValueError("Only the original frozen source configuration is allowed")
        source_path = root / plan["source_config"]
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if source_hash != plan["source_config_sha256"]:
            raise ValueError("Source configuration hash differs")
        source = json.loads(source_bytes)
        days = validate_plan(plan, source, mode)
        identity = frozen_identity(
            root,
            [
                config_path,
                source_path,
                *[root / name for name in EXECUTION_SOURCES],
            ],
        )
        if (
            config_path.read_bytes() != plan_bytes
            or source_path.read_bytes() != source_bytes
            or identity["source_sha256"][str(config_path.resolve().relative_to(root.resolve()))]
            != plan_hash
            or identity["source_sha256"][str(source_path.resolve().relative_to(root.resolve()))]
            != source_hash
        ):
            raise ValueError("Configuration changed during preflight")
        state.update(
            **identity,
            config_sha256=plan_hash,
            source_config_sha256=source_hash,
            config=plan,
            days=days,
            inputs={},
            windows_us={},
            interpretation=plan["interpretation"],
            quantiles_computed=False,
        )
        audit_sources = {name: identity["source_sha256"][name] for name in AUDIT_SOURCES}
        for day in days:
            start = day_start_us(day)
            end = start + (3_600_000_000 if mode == "smoke" else 86_400_000_000)
            state["windows_us"][day] = [start, end]
            state["inputs"][day] = audited_inputs(root, source, source_hash, day, audit_sources)
        state["stage"] = "inputs_verified_before_market_parsing"
        write_json(output / "prepared_receipt.json", state)
        write_json(manifest_path, state)
        if prepare_only:
            state["status"] = "prepared"
        else:
            for day in days:
                paths = {kind: Path(r["path"]) for kind, r in state["inputs"][day].items()}
                state["outputs"].extend(
                    score_raw_day(
                        paths,
                        output,
                        day,
                        *state["windows_us"][day],
                        plan["row_group_size"],
                    )
                )
                write_json(manifest_path, state)
            # Reject changing configuration, code, receipts or raw bytes during processing.
            for name, expected in state["source_sha256"].items():
                if sha256(root / name) != expected:
                    raise ValueError("Frozen source/config changed during run")
            for receipts in state["inputs"].values():
                for receipt in receipts.values():
                    for path_key, hash_key in (
                        ("path", "sha256"),
                        ("metadata_path", "metadata_sha256"),
                        ("audit_path", "audit_sha256"),
                    ):
                        if sha256(receipt[path_key]) != receipt[hash_key]:
                            raise ValueError("Input evidence changed during run")
            state.update(status="complete", stage="outputs_verified", outputs_published=True)
    except Exception as exc:
        state.update(
            status="failed",
            outputs_published=False,
            error={"type": type(exc).__name__, "detail": str(exc)[:400]},
        )
        raise
    finally:
        state["completed_at"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, state)
    return state
