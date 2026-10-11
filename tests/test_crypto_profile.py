import json

import numpy as np
import pandas as pd
import pytest

from experiments import crypto_profile as profile
from scripts import profile_crypto_microstructure as runner

DAY = "2024-11-01"
START = 1730419200000


def synthetic_book(seconds=None, segments=None):
    seconds = list(range(180) if seconds is None else seconds)
    n = len(seconds)
    buckets = np.array(seconds, dtype="int64") * 1000 + START
    frame = pd.DataFrame(
        {
            "bucket_ms": buckets,
            "available_at_ms": buckets + 1005,
            "cts_ms": buckets + 900,
            "segment": [0] * n if segments is None else segments,
            "bid_units": 10000 + np.arange(n),
            "ask_units": 10002 + np.arange(n),
            "spread_bps": [2.0] * n,
            "imbalance_l1": [0.2] * n,
            "visible_bid_levels": [500] * n,
            "visible_ask_levels": [500] * n,
            "delta_messages": np.arange(n) % 5,
            "positive_visible_depth_change_units": [200] * n,
            "negative_visible_depth_change_units": [100] * n,
        }
    )
    for side, size in (("bid", 1000), ("ask", 2000)):
        for levels in profile.DEPTH_LEVELS:
            frame[f"{side}_depth_{levels}_units"] = size * levels
        for bps in profile.DEPTH_BPS:
            frame[f"{side}_depth_{bps}bps_units"] = size * bps
            frame[f"{side}_covers_{bps}bps"] = True
    audit = {
        "features_published": True,
        "zip_crc_verified": True,
        "segments": [
            {
                "segment": int(segment),
                "first_cts_ms": int(rows.bucket_ms.min()),
                "last_cts_ms": int(rows.bucket_ms.max() + 1000),
            }
            for segment, rows in frame.groupby("segment")
        ],
    }
    return frame, audit


def synthetic_trades(seconds=None):
    seconds = list(range(180) if seconds is None else seconds)
    buckets = np.array(seconds, dtype="int64") * 1000 + START
    frame = pd.DataFrame(
        {
            "bucket_ms": buckets,
            "available_at_ms": buckets + 1000,
            "trade_count": [2] * len(seconds),
            "buy_quantity_units": [300] * len(seconds),
            "sell_quantity_units": [100] * len(seconds),
            "quote_volume_product_units": [4_000_000] * len(seconds),
        }
    )
    audit = {
        "features_published": True,
        "quality": "valid",
        "records_scan_complete": True,
        "gzip_integrity": {"status": "passed", "read_to_eof": True, "crc_verified": True},
        "first_timestamp_us": int(buckets[0] * 1000 + 250000) if seconds else None,
        "last_timestamp_us": int(buckets[-1] * 1000 + 750000) if seconds else None,
    }
    return frame, audit


def test_no_second_or_minute_return_crosses_gap_or_segment():
    raw, audit = synthetic_book([0, 1, 2, 4, 5, 6, 7], [0, 0, 0, 0, 0, 1, 1])
    book = profile.prepare_book(raw, audit, DAY)
    returns = profile.contiguous_returns(book, 1)
    assert ((returns.bucket_ms - START) // 1000).tolist() == [1, 2, 5, 7]
    raw, audit = synthetic_book([second for second in range(180) if second != 90])
    book = profile.prepare_book(raw, audit, DAY)
    returns = profile.contiguous_returns(book, 60)
    assert not returns.bucket_ms.between(START + 91000, START + 150000).any()
    minute = profile._return_profile(book)["60"]
    assert minute["log_return_bps"]["n"] == 1  # 119 -> 179; missing 90 blocks 59 -> 119.


def test_minute_series_is_nonoverlapping_and_constant_prices_have_zero_returns():
    raw, audit = synthetic_book()
    raw["bid_units"], raw["ask_units"] = 10000, 10002
    result = profile._return_profile(profile.prepare_book(raw, audit, DAY))
    assert result["1"]["log_return_bps"]["n"] == 179
    assert result["60"]["log_return_bps"]["n"] == 2
    assert result["1"]["unchanged_mid_fraction"] == 1.0
    assert result["1"]["autocorrelation"]["1"]["pearson"] is None
    assert result["1"]["autocorrelation"]["1"]["undefined_reason"] == "constant_series"


def test_depth_censoring_requires_both_sides_and_does_not_treat_missing_as_zero():
    raw, audit = synthetic_book(range(4))
    raw.loc[0, "bid_covers_10bps"] = False
    raw.loc[0, "bid_depth_10bps_units"] = 900_000  # Must not pollute covered summary.
    raw.loc[1, "ask_covers_10bps"] = False
    raw.loc[2, "visible_bid_levels"] = 3
    book = profile.prepare_book(raw, audit, DAY)
    stats = profile._depth_profile(book, 10, bps=True)
    assert stats["covered_seconds"] == stats["censored_seconds"] == 2
    assert stats["bid_depth_btc"]["mean"] == 10
    assert stats["imbalance"]["mean"] == pytest.approx(-1 / 3)
    assert profile._depth_profile(book, 5)["censored_seconds"] == 1


def test_covered_radius_with_no_quotes_has_undefined_imbalance():
    raw, audit = synthetic_book(range(2))
    raw["bid_depth_1bps_units"], raw["ask_depth_1bps_units"] = 0, 0
    stats = profile._depth_profile(profile.prepare_book(raw, audit, DAY), 1, bps=True)
    assert stats["covered_seconds"] == stats["covered_zero_total_depth_seconds"] == 2
    assert stats["imbalance"]["n"] == 0
    assert stats["bid_depth_btc"]["mean"] == 0


def test_trade_zero_fill_uses_full_internal_seconds_and_never_fills_book_gaps():
    raw, book_audit = synthetic_book([*range(8), 10, 11, 12])
    trades, trade_audit = synthetic_trades([0, 2, 7, 12])
    book = profile.prepare_book(raw, book_audit, DAY)
    frame, coverage = profile.covered_trade_seconds(book, trades, trade_audit)
    assert ((frame.bucket_ms - START) // 1000).tolist() == [1, 2, 3, 4, 5, 6, 7, 10, 11]
    assert coverage["zero_filled_seconds"] == 7
    assert coverage["book_seconds_excluded_by_trade_boundaries"] == 2
    assert frame.trade_count.sum() == 4
    assert frame.quote_volume_usdt.sum() == 80
    assert frame.signed_trade_quantity_btc.sum() == pytest.approx(0.4)


@pytest.mark.parametrize("corruption", ["partial_scan", "bad_crc", "invalid", "unpublished"])
def test_invalid_trade_archive_never_authorizes_zero_counts(corruption):
    raw, book_audit = synthetic_book(range(5))
    trades, audit = synthetic_trades([0, 4])
    if corruption == "partial_scan":
        audit["records_scan_complete"] = False
    elif corruption == "bad_crc":
        audit["gzip_integrity"]["crc_verified"] = False
    elif corruption == "invalid":
        audit["quality"] = "invalid"
    else:
        audit["features_published"] = False
    with pytest.raises(ValueError, match="fully scanned"):
        profile.covered_trade_seconds(profile.prepare_book(raw, book_audit, DAY), trades, audit)


def test_empty_valid_trade_tape_does_not_invent_all_day_zero_intensity():
    raw, book_audit = synthetic_book(range(5))
    trades, audit = synthetic_trades([])
    frame, coverage = profile.covered_trade_seconds(
        profile.prepare_book(raw, book_audit, DAY), trades, audit
    )
    assert frame.empty and coverage["zero_filled_seconds"] == 0
    assert coverage["book_seconds_excluded_by_trade_boundaries"] == 5


def test_future_targets_begin_strictly_after_signal_availability_and_skip_late_signals():
    raw, audit = synthetic_book(range(12))
    raw.loc[0, "available_at_ms"] = START + 5500
    book = profile.prepare_book(raw, audit, DAY)
    pairs = profile.future_return_pairs(book, book, 1)
    first = pairs.loc[0]
    assert first.target_start_bucket_ms == START + 6000
    assert first.target_end_bucket_ms == START + 7000
    assert (pairs.target_start_cts_ms > pairs.signal_available_at_ms).all()
    assert (pairs.target_end_cts_ms > pairs.signal_available_at_ms).all()
    expected = 10000 * np.log(book.loc[7, "mid_twice_units"] / book.loc[6, "mid_twice_units"])
    assert first.target_return_bps == pytest.approx(expected)
    assert 10 not in pairs.index and 11 not in pairs.index


def test_future_target_cannot_cross_segment_or_skip_a_missing_intermediate_second():
    raw, audit = synthetic_book(range(10), [0] * 4 + [1] * 6)
    book = profile.prepare_book(raw, audit, DAY)
    assert 1 not in profile.future_return_pairs(book, book, 1).index
    raw, audit = synthetic_book([*range(30), *range(31, 90)])
    book = profile.prepare_book(raw, audit, DAY)
    assert profile.future_return_pairs(book, book.iloc[:1], 60).empty


def test_exact_boundary_target_quote_is_not_strictly_future():
    raw, audit = synthetic_book(range(6))
    raw.loc[0, "available_at_ms"] = START + 2000
    raw.loc[2, "cts_ms"] = START + 2000
    book = profile.prepare_book(raw, audit, DAY)
    pairs = profile.future_return_pairs(book, book.iloc[:1], 1)
    assert pairs.iloc[0].target_start_bucket_ms == START + 3000


def test_count_blocks_require_all_seconds_and_lag_pairs_do_not_cross_gaps():
    raw, audit = synthetic_book([*range(5), *range(6, 20)])
    book = profile.prepare_book(raw, audit, DAY)
    stats = profile.count_clustering(book, "delta_messages")
    assert stats["block_seconds"]["10"]["complete_blocks"] == 1  # Only 10..19.
    assert stats["block_seconds"]["60"]["complete_blocks"] == 0
    assert stats["autocorrelation"]["1"]["pairs"] == 17
    assert stats["autocorrelation"]["5"]["pairs"] == 9


def test_partial_book_seconds_and_premature_availability_are_rejected():
    raw, audit = synthetic_book(range(5))
    audit["segments"][0]["first_cts_ms"] = START + 1
    with pytest.raises(ValueError, match="full valid segment"):
        profile.prepare_book(raw, audit, DAY)
    raw, audit = synthetic_book(range(5))
    raw.loc[0, "available_at_ms"] = START + 999
    with pytest.raises(ValueError, match="before its complete second"):
        profile.prepare_book(raw, audit, DAY)


def test_profile_contains_finite_json_and_explicit_interpretation():
    book, book_audit = synthetic_book()
    trades, trade_audit = synthetic_trades()
    report = profile.profile_day(book, trades, book_audit, trade_audit, split="train", day=DAY)
    json.dumps(report, allow_nan=False)
    assert report["trades_within_joint_coverage"]["trade_count"]["n"] == 178
    assert report["return_horizon_seconds"]["1"]["log_return_bps"]["n"] == 179
    assert report["observed_depth_updates"]["visible_increase_btc_per_second"]["mean"] == (
        pytest.approx(0.2)
    )
    assert (
        "lower bound"
        in report["trade_signal_future_return_horizon_seconds"]["1"]["clock_assumption"]
    )
    assert "not a validated" in profile.DEFINITIONS["scope"]


@pytest.mark.parametrize("split", ["event_test", "all", "test"])
def test_closed_split_rejected_before_any_disk_read(tmp_path, split):
    with pytest.raises(ValueError, match="closed"):
        profile.profile_study(tmp_path, tmp_path / "does-not-exist", split)
    with pytest.raises(ValueError, match="closed"):
        profile.profile_day(None, None, None, None, split=split, day="2024-11-05")


def test_study_uses_authorized_reader_and_preserves_provenance(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "study_id": "fixture",
                "splits": {
                    "train": [DAY],
                    "development": ["2024-11-04"],
                    "event_test": ["2024-11-05"],
                },
            }
        )
    )
    calls = []

    def read(root, config_path, split, day, kind):
        calls.append((root, config_path, split, day, kind))
        frame, audit = synthetic_book() if kind == "orderbook" else synthetic_trades()
        audit.update(
            audit_run_id="successful-run",
            input_sha256="raw-sha",
            features_path=f"{kind}.parquet",
            features_sha256=f"{kind}-sha",
            processing_source_sha256={"auditor.py": "auditor-sha"},
        )
        return frame, audit

    monkeypatch.setattr(profile.crypto_features, "read_feature_day", read)
    report = profile.profile_study(tmp_path, config, "train")
    assert [call[4] for call in calls] == ["orderbook", "trades"]
    assert all(call[2:4] == ("train", DAY) for call in calls)
    assert report["sources"][0]["audit_run_id"] == "successful-run"
    assert report["sources"][1]["features_sha256"] == "trades-sha"
    assert len(report["config_sha256"]) == len(report["profile_source_sha256"]) == 64
    calls.clear()
    with pytest.raises(ValueError, match="frozen"):
        profile.profile_study(tmp_path, config, "train", ["2024-11-05"])
    assert not calls


def test_cli_cannot_select_event_test_even_with_nonexistent_config(tmp_path):
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--split",
                "event_test",
                "--config",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    assert exc.value.code == 2
