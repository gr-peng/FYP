import json
from zipfile import ZIP_STORED, ZipFile

import pandas as pd

from experiments.bybit_reconstruction import audit_book_file

START = 1730419200000


def event(offset, u, kind="delta", quantity="1.000"):
    return {
        "topic": "orderbook.500.BTCUSDT",
        "type": kind,
        "ts": START + offset + 5,
        "cts": START + offset,
        "data": {
            "s": "BTCUSDT",
            "u": u,
            "seq": 1000 + u,
            "b": [["99.00", quantity]],
            "a": [["100.00", "2.000"]],
        },
    }


def archive(tmp_path, rows, name="2024-11-01_BTCUSDT_ob500.data.zip"):
    path = tmp_path / name
    with ZipFile(path, "w", compression=ZIP_STORED) as target:
        target.writestr(name.removesuffix(".zip"), "\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_snapshot_is_not_order_flow_and_features_are_available_after_bucket(tmp_path):
    rows = [
        event(0, 10, "snapshot"),
        event(500, 11, quantity="1.100"),
        event(1500, 12, quantity="1.200"),
        event(2500, 13),
    ]
    features = tmp_path / "features.parquet"
    result = audit_book_file(archive(tmp_path, rows), "2024-11-01", features)
    frame = pd.read_parquet(features)
    assert result["structural_pass"] and result["zip_crc_verified"]
    assert result["feature_rows"] == 2
    assert frame.iloc[0].positive_visible_depth_change_units == 100  # Not initial 3000.
    assert frame.iloc[0].cts_ms == START + 500
    assert frame.iloc[0].available_at_ms == START + 1505
    assert frame.iloc[0].state_age_at_bucket_end_ms == 500


def test_lost_update_does_not_create_filled_gap_seconds(tmp_path):
    rows = [
        event(0, 10, "snapshot"),
        event(500, 11),
        event(1500, 12),
        event(2500, 14),
        event(3500, 15),
        event(4000, 1, "snapshot"),
        event(4500, 2),
        event(5500, 3),
    ]
    output = tmp_path / "features.parquet"
    result = audit_book_file(archive(tmp_path, rows), "2024-11-01", output)
    assert not result["structural_pass"] and result["zip_crc_verified"]
    assert result["counts"]["invalid_messages"] == 2
    frame = pd.read_parquet(output)
    assert frame.bucket_ms.tolist() == [START, START + 4000]
    assert frame.segment.tolist() == [0, 1]


def test_future_changes_do_not_change_past_observed_features(tmp_path):
    prefix = [event(0, 10, "snapshot"), event(500, 11), event(1500, 12)]
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    path = archive(tmp_path, prefix + [event(2500, 13, quantity="0.200")])
    audit_book_file(path, "2024-11-01", first)
    path = archive(tmp_path, prefix + [event(2500, 13, quantity="500.000")])
    audit_book_file(path, "2024-11-01", second)
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))


def test_crc_failure_never_publishes_features(tmp_path):
    path = archive(tmp_path, [event(0, 10, "snapshot"), event(1500, 11)])
    body = bytearray(path.read_bytes())
    position = body.index(b'"99.00"')
    body[position + 1] = ord("8")
    path.write_bytes(body)
    output = tmp_path / "features.parquet"
    result = audit_book_file(path, "2024-11-01", output)
    assert not result["zip_crc_verified"] and not result["structural_pass"]
    assert not output.exists()


def test_empty_or_snapshotless_archive_never_passes_structure(tmp_path):
    for rows in ([], [event(0, 10)]):
        result = audit_book_file(archive(tmp_path, rows), "2024-11-01")
        assert result["zip_crc_verified"] and not result["structural_pass"]
        assert not result["features_published"]


def test_late_publication_remains_in_prefix_availability(tmp_path):
    late = event(500, 11)
    late["ts"] = START + 5000
    rows = [event(0, 10, "snapshot"), late, event(800, 12), event(1500, 13), event(2500, 14)]
    output = tmp_path / "features.parquet"
    result = audit_book_file(archive(tmp_path, rows), "2024-11-01", output)
    assert result["structural_pass"]
    frame = pd.read_parquet(output)
    assert frame.available_at_ms.tolist() == [START + 5000, START + 5000]


def test_failed_reprocessing_disallows_old_features_in_manifest(tmp_path):
    rows = [event(0, 10, "snapshot"), event(1500, 11)]
    path, output = archive(tmp_path, rows), tmp_path / "features.parquet"
    good = audit_book_file(path, "2024-11-01", output)
    assert good["features_published"]
    content = output.read_bytes()
    path.write_bytes(path.read_bytes()[:-50])
    bad = audit_book_file(path, "2024-11-01", output)
    assert not bad["features_published"] and "features" not in bad
    assert output.read_bytes() == content  # Preserved evidence; manifest forbids its reuse.


def test_daily_file_overlap_is_audited_but_does_not_leak_into_day_features(tmp_path):
    end = 86400000
    rows = [event(end - 3000, 10, "snapshot")]
    rows.extend(
        event(end + offset, 11 + i) for i, offset in enumerate([-2500, -1500, -500, 500, 1500])
    )
    output = tmp_path / "features.parquet"
    result = audit_book_file(archive(tmp_path, rows), "2024-11-01", output)
    assert result["structural_pass"]
    assert result["counts"]["messages_outside_filename_day"] == 2
    assert result["raw_after_day_ms"] == 1500
    frame = pd.read_parquet(output)
    assert len(frame) == 2
    assert (frame.available_at_ms < START + end).all()
