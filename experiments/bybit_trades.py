"""Audit the official Bybit BTCUSDT trade tape without repairing observations.

Prices use 1/100 USDT and quantities 1/1000 BTC. A product unit therefore
represents 1/100000 USDT. Source notional/valuation columns are ignored: the
integer product is recomputed from price and size.

Features contain only seconds with trades and become available at the END of
the second, ``available_at_ms = bucket_ms + 1000``. Equal timestamps retain
source row order. Neither this order nor an L2 book identifies individual
orders' queue priority. No features are published unless the entire CSV and
all gzip trailers pass validation. Duplicate detection stores all distinct
trade IDs in memory; memory use scales with the number of trades in a day.
"""

import csv
import gzip
import io
import os
import tempfile
import zlib
from collections import Counter
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from behavioral_market.data.bybit_book import BookIntegrityError, units

_REQUIRED = ("timestamp", "symbol", "side", "size", "price", "trdMatchID")
_FEATURE_COLUMNS = (
    "bucket_ms",
    "available_at_ms",
    "trade_count",
    "buy_quantity_units",
    "sell_quantity_units",
    "quote_volume_product_units",
    "last_price_units",
    "last_timestamp_us",
)
_MAX_INT64 = 2**63 - 1
_DAY_US = 86_400_000_000


def _publish_features(rows: list[dict], path: Path) -> None:
    """Replace an output atomically only after successful validation."""
    metadata = {
        b"availability": b"Each row is available at bucket_ms + 1000, not bucket start.",
        b"empty_seconds": b"Omitted; no forward fill or synthetic trades.",
        b"price_scale": b"100 units per USDT/BTC",
        b"quantity_scale": b"1000 units per BTC",
        b"quote_volume_product_scale": b"100000 units per USDT",
        b"equal_timestamps": b"Original file order; no sorting or deduplication.",
    }
    schema = pa.schema([(name, pa.int64()) for name in _FEATURE_COLUMNS], metadata=metadata)
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def audit_trades_file(
    path: Path, expected_day: str, features_path: Path | None = None
) -> dict:
    """Validate one UTC day of BTCUSDT trades and optionally publish second bars.

    Data errors are returned as ``quality='invalid'``. Anomaly counts cover
    every scanned CSV record; examples are capped at 20. A fatal CSV/encoding
    error stops record parsing, but the remaining gzip stream is drained to
    validate its trailer. ``records_scan_complete`` distinguishes that case.
    Filesystem errors opening the input or publishing an output raise normally.
    On invalid data an existing destination is left untouched and never claimed
    as published by this call; callers must use ``features_published``.
    """
    path = Path(path)
    day = date.fromisoformat(expected_day)
    if day.isoformat() != expected_day:
        raise ValueError("expected_day must have exact YYYY-MM-DD format")
    start_us = (day - date(1970, 1, 1)).days * _DAY_US
    end_us = start_us + _DAY_US
    if features_path is not None:
        features_path = Path(features_path)
        if features_path.resolve() == path.resolve():
            raise ValueError("Feature destination must differ from source archive")

    errors: Counter = Counter()
    examples = []
    counts = {"data_rows": 0, "valid_rows": 0, "invalid_rows": 0}
    sides = {"Buy": 0, "Sell": 0, "other": 0}
    seen_ids: set[str] = set()
    first_timestamp = last_timestamp = previous_timestamp = None
    rows: list[dict] = []
    header = None
    scan_complete = False
    integrity = {"status": "not_checked", "read_to_eof": False, "crc_verified": False}

    def flag(kind, row_number=None, detail=None):
        errors[kind] += 1
        if len(examples) < 20:
            examples.append({"kind": kind, "row": row_number, "detail": detail})

    def parse_record(record, indices):
        nonlocal first_timestamp, last_timestamp, previous_timestamp
        number = counts["data_rows"]
        before = sum(errors.values())
        if len(record) != len(header):
            flag("invalid_row_width", number, {"columns": len(record)})
            counts["invalid_rows"] += 1
            return
        values = {key: record[index] for key, index in indices.items()}
        timestamp = price = quantity = None
        try:
            timestamp = units(values["timestamp"], 6)
        except BookIntegrityError:
            flag("invalid_timestamp", number)
        if timestamp is not None:
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            if not start_us <= timestamp < end_us:
                flag("wrong_utc_day", number)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                flag("backward_timestamp", number)
            previous_timestamp = timestamp
        if values["symbol"] != "BTCUSDT":
            flag("invalid_symbol", number)
        side = values["side"]
        sides[side if side in sides and side != "other" else "other"] += 1
        if side not in {"Buy", "Sell"}:
            flag("invalid_side", number)
        for name, places in (("price", 2), ("size", 3)):
            label = "price" if name == "price" else "quantity"
            try:
                value = units(values[name], places)
                if value <= 0:
                    flag(f"nonpositive_{label}", number)
                if name == "price":
                    price = value
                else:
                    quantity = value
            except BookIntegrityError:
                flag(f"invalid_{label}", number)
        trade_id = values["trdMatchID"]
        if not trade_id.strip():
            flag("missing_trade_id", number)
        elif trade_id in seen_ids:
            flag("duplicate_trade_id", number)
        else:
            seen_ids.add(trade_id)
        if sum(errors.values()) != before:
            counts["invalid_rows"] += 1
            return
        counts["valid_rows"] += 1
        if features_path is None:
            return
        bucket = timestamp // 1_000_000 * 1000
        if not rows or rows[-1]["bucket_ms"] != bucket:
            rows.append(dict.fromkeys(_FEATURE_COLUMNS, 0))
            rows[-1].update(bucket_ms=bucket, available_at_ms=bucket + 1000)
        aggregate = rows[-1]
        aggregate["trade_count"] += 1
        aggregate[f"{side.lower()}_quantity_units"] += quantity
        aggregate["quote_volume_product_units"] += price * quantity
        aggregate["last_price_units"] = price
        aggregate["last_timestamp_us"] = timestamp

    with path.open("rb") as source:
        if source.read(2) != b"\x1f\x8b":
            flag("invalid_gzip_header")
            integrity["status"] = "failed"
        else:
            source.seek(0)
            with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
                text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
                try:
                    try:
                        reader = csv.reader(text, strict=True)
                        header = next(reader, None)
                        if (
                            header is None
                            or len(header) != len(set(header))
                            or not set(_REQUIRED).issubset(header)
                        ):
                            flag("invalid_csv_header")
                            indices = None
                        else:
                            indices = {name: header.index(name) for name in _REQUIRED}
                        for record in reader:
                            counts["data_rows"] += 1
                            if indices is None:
                                counts["invalid_rows"] += 1
                            else:
                                parse_record(record, indices)
                        scan_complete = True
                    except (csv.Error, UnicodeError) as exc:
                        flag("csv_parse_error", detail=type(exc).__name__)
                        # TextIOWrapper may have buffered bytes. Only compression
                        # integrity, not row coverage, is recoverable after this.
                        while compressed.read(1024 * 1024):
                            pass
                    integrity.update(status="passed", read_to_eof=True, crc_verified=True)
                except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
                    flag("gzip_integrity_error", detail=str(exc))
                    integrity.update(status="failed", error_type=type(exc).__name__)
                finally:
                    text.close()

    if counts["data_rows"] == 0:
        flag("empty_trade_file")
    # All arithmetic above is Python integer arithmetic. Refuse overflow rather
    # than silently downcasting/wrapping when creating the int64 parquet schema.
    if features_path is not None and any(
        not 0 <= value <= _MAX_INT64 for row in rows for value in row.values()
    ):
        flag("feature_int64_overflow")
    valid = not errors and scan_complete and integrity["crc_verified"]
    if valid and features_path is not None:
        _publish_features(rows, features_path)
    published = valid and features_path is not None
    return {
        "source_path": str(path),
        "expected_day": expected_day,
        "symbol": "BTCUSDT",
        "quality": "valid" if valid else "invalid",
        "counts": {**counts, "unique_trade_ids": len(seen_ids), "side_counts": sides},
        "first_timestamp_us": first_timestamp,
        "last_timestamp_us": last_timestamp,
        "records_scan_complete": scan_complete,
        "gzip_integrity": integrity,
        "anomaly_counts": dict(errors),
        "anomaly_examples": examples,
        "anomaly_examples_limit": 20,
        "features_published": published,
        "features_path": str(features_path) if published else None,
        "feature_rows": len(rows) if published else 0,
        "feature_availability": "bucket_ms + 1000; only nonempty trade seconds",
        "scales": {"timestamp": 1_000_000, "price": 100, "quantity": 1000, "quote": 100_000},
    }
