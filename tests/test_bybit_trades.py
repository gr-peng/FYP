"""Small semantic fixtures for exact tape parsing and fail-closed publication."""

import csv
import gzip
import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from experiments.bybit_trades import audit_trades_file

DAY = "2024-11-01"
EPOCH_US = 1_730_419_200_000_000
HEADER = ["timestamp", "symbol", "side", "size", "price", "trdMatchID", "grossValue"]


def row(timestamp="1730419200.123456", side="Buy", size="0.001", price="100.01", tid="a"):
    # Deliberately incorrect grossValue: not a trusted observed state/input.
    return [timestamp, "BTCUSDT", side, size, price, tid, "-999999999"]


def archive(tmp_path, rows, name="trades.csv.gz"):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(HEADER)
    writer.writerows(rows)
    path = tmp_path / name
    path.write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    return path


def test_exact_microseconds_same_time_source_order_and_nonempty_seconds(tmp_path):
    path = archive(
        tmp_path,
        [
            row(size="0.002", price="100.01", tid="a"),
            row(side="Sell", size="0.003", price="100.02", tid="b"),
            row(timestamp="1730419202.000001", price="100.03", tid="c"),
        ],
    )
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "valid"
    assert result["first_timestamp_us"] == EPOCH_US + 123456
    assert result["last_timestamp_us"] == EPOCH_US + 2_000_001
    assert result["gzip_integrity"] == {
        "status": "passed", "read_to_eof": True, "crc_verified": True
    }
    assert result["counts"]["unique_trade_ids"] == 3
    data = pq.read_table(output)
    bars = data.to_pylist()
    assert [bar["bucket_ms"] for bar in bars] == [EPOCH_US // 1000, EPOCH_US // 1000 + 2000]
    assert bars[0]["available_at_ms"] == EPOCH_US // 1000 + 1000
    assert bars[0]["trade_count"] == 2
    assert bars[0]["buy_quantity_units"] == 2
    assert bars[0]["sell_quantity_units"] == 3
    assert bars[0]["quote_volume_product_units"] == 10001 * 2 + 10002 * 3
    assert bars[0]["last_price_units"] == 10002
    assert bars[0]["last_timestamp_us"] == EPOCH_US + 123456
    assert b"bucket_ms + 1000" in data.schema.metadata[b"availability"]


def test_quality_only_returns_no_price_statistics_or_features(tmp_path):
    path = archive(tmp_path, [row()])
    result = audit_trades_file(path, DAY)
    assert result["quality"] == "valid"
    assert result["features_published"] is False
    assert result["features_path"] is None
    assert result["feature_rows"] == 0
    assert not list(tmp_path.glob("*.parquet"))
    assert not any("price" in key or "volume" in key for key in result)


def test_duplicate_id_detected_across_seconds_without_deduplication(tmp_path):
    path = archive(
        tmp_path,
        [row(tid="a"), row(timestamp="1730419201", tid="b"), row(timestamp="1730419202", tid="a")],
    )
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["duplicate_trade_id"] == 1
    assert result["counts"]["data_rows"] == 3
    assert result["counts"]["invalid_rows"] == 1
    assert result["counts"]["unique_trade_ids"] == 2
    assert result["records_scan_complete"]
    assert result["gzip_integrity"]["crc_verified"]
    assert not output.exists()


def test_backwards_timestamp_is_not_sorted_and_existing_output_is_not_published(tmp_path):
    path = archive(tmp_path, [row(timestamp="1730419202", tid="a"), row(tid="b")])
    output = tmp_path / "seconds.parquet"
    output.write_bytes(b"prior artifact")
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["backward_timestamp"] == 1
    assert result["last_timestamp_us"] < result["first_timestamp_us"]
    assert not result["features_published"]
    assert output.read_bytes() == b"prior artifact"


@pytest.mark.parametrize("damage", ["crc", "truncated"])
def test_full_stream_crc_and_trailer_required_before_publication(tmp_path, damage):
    path = archive(tmp_path, [row()])
    contents = bytearray(path.read_bytes())
    if damage == "crc":
        contents[-8] ^= 0x01
    else:
        contents = contents[:-4]
    path.write_bytes(contents)
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["gzip_integrity"]["status"] == "failed"
    assert not result["gzip_integrity"]["crc_verified"]
    assert result["anomaly_counts"]["gzip_integrity_error"] == 1
    assert not output.exists()


@pytest.mark.parametrize(
    ("index", "value", "kind"),
    [
        (0, "1730419200.1234567", "invalid_timestamp"),
        (0, "1730505600", "wrong_utc_day"),
        (1, "ETHUSDT", "invalid_symbol"),
        (2, "BUY", "invalid_side"),
        (3, "0", "nonpositive_quantity"),
        (3, "0.0001", "invalid_quantity"),
        (4, "-1", "nonpositive_price"),
        (4, "100.011", "invalid_price"),
        (5, "", "missing_trade_id"),
    ],
)
def test_invalid_values_are_reported_and_prevent_features(tmp_path, index, value, kind):
    malformed = row()
    malformed[index] = value
    path = archive(tmp_path, [malformed])
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"][kind] == 1
    assert result["gzip_integrity"]["crc_verified"]
    assert not output.exists()


def test_notional_product_remains_exact_above_float_integer_precision(tmp_path):
    path = archive(tmp_path, [row(price="900719925.47", size="1000.001")])
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "valid"
    product = pq.read_table(output).to_pylist()[0]["quote_volume_product_units"]
    assert product == 90_071_992_547 * 1_000_001
    assert product > 2**53


def test_malformed_csv_still_drains_gzip_for_crc_check(tmp_path):
    path = tmp_path / "malformed.csv.gz"
    path.write_bytes(gzip.compress((",".join(HEADER) + '\n"unterminated').encode()))
    result = audit_trades_file(path, DAY)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["csv_parse_error"] == 1
    assert not result["records_scan_complete"]
    assert result["gzip_integrity"]["crc_verified"]


def test_invalid_utf8_still_checks_gzip_integrity(tmp_path):
    path = tmp_path / "invalid_encoding.csv.gz"
    payload = (",".join(HEADER) + "\n" + ",".join(row())).encode() + b"\xff\n"
    path.write_bytes(gzip.compress(payload))
    result = audit_trades_file(path, DAY)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["csv_parse_error"] == 1
    assert not result["records_scan_complete"]
    assert result["gzip_integrity"]["crc_verified"]


def test_all_gzip_members_are_checked_not_only_first(tmp_path):
    path = archive(tmp_path, [row(tid="a")])
    later = (",".join(row(timestamp="1730419202", tid="b")) + "\n").encode()
    member = bytearray(gzip.compress(later))
    member[-8] ^= 1
    path.write_bytes(path.read_bytes() + member)
    result = audit_trades_file(path, DAY, tmp_path / "features.parquet")
    assert result["quality"] == "invalid"
    assert not result["gzip_integrity"]["crc_verified"]
    assert result["anomaly_counts"]["gzip_integrity_error"] == 1
    assert not (tmp_path / "features.parquet").exists()


def test_feature_integer_overflow_fails_without_wrapping(tmp_path):
    path = archive(tmp_path, [row(price="900719925474099.99", size="1000")])
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["feature_int64_overflow"] == 1
    assert result["gzip_integrity"]["crc_verified"]
    assert not output.exists()


@pytest.mark.parametrize("payload", [b"", b"timestamp,timestamp\n1,2\n"])
def test_empty_or_invalid_header_cannot_publish_features(tmp_path, payload):
    path = tmp_path / "header.csv.gz"
    path.write_bytes(gzip.compress(payload))
    output = tmp_path / "seconds.parquet"
    result = audit_trades_file(path, DAY, output)
    assert result["quality"] == "invalid"
    assert result["anomaly_counts"]["invalid_csv_header"] == 1
    assert result["gzip_integrity"]["crc_verified"]
    assert not output.exists()


def test_feature_destination_cannot_replace_input(tmp_path):
    path = archive(tmp_path, [row()])
    with pytest.raises(ValueError, match="differ"):
        audit_trades_file(path, DAY, Path(path))
