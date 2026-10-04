import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from behavioral_market.data.archive import digest
from behavioral_market.data.calendar import sessions, visible_news
from behavioral_market.data.schemas import MarketEvent
from experiments.two_event_data import (
    date_window,
    parse_binance_klines,
    summarize_confirmation,
    summarize_daily,
    validate_bars,
)


def kline_zip(year, unit):
    start = pd.Timestamp(f"{year}-11-06T00:00:00Z")
    divisor = 1_000_000 if unit == "ms" else 1_000
    t0 = start.value // divisor
    t1 = (start + pd.Timedelta(days=1)).value // divisor - 1
    csv = f"{t0},100,102,99,101,0.125,{t1},12.5,3,0.075,7.5,0\n"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("sample.csv", csv)
    body = stream.getvalue()
    return body, f"{digest(body)}  sample.zip".encode()


@pytest.mark.parametrize("year,unit", [(2024, "ms"), (2025, "us")])
def test_crypto_archive_units_checksum_and_fractional_volume(year, unit):
    body, checksum = kline_zip(year, unit)
    frame = parse_binance_klines(body, checksum, "BTCUSDT", "1d")
    assert frame.session_date.iloc[0] == f"{year}-11-06"
    assert frame.bar_end_utc.iloc[0] == pd.Timestamp(f"{year}-11-07T00:00:00Z")
    assert frame.volume.iloc[0] == 0.125
    with pytest.raises(ValueError, match="checksum"):
        parse_binance_klines(body + b"changed", checksum, "BTCUSDT", "1d")


def test_crypto_calendar_includes_weekends_and_correct_confirmation_visibility():
    clock = sessions("2024-11-01", "2024-11-10", "24/7")
    assert len(clock) == 10
    assert clock.loc["2024-11-03"].market_open == pd.Timestamp("2024-11-03T00:00:00Z")
    assert clock.loc["2024-11-06"].market_close == pd.Timestamp("2024-11-07T00:00:00Z")
    event = MarketEvent(
        event_id="confirmation",
        first_seen_at="2024-11-06T10:34:00Z",
        published_at_et="2024-11-06T05:34:00-05:00",
        headline="Election called",
        neutral_summary="The result has been called.",
        related_symbols=["BTCUSDT"],
        source_urls=["https://example.com"],
    )
    assert not visible_news(
        [event], clock.loc["2024-11-05"].market_open, clock.loc["2024-11-06"].market_open
    )
    assert visible_news(
        [event], clock.loc["2024-11-06"].market_open, clock.loc["2024-11-07"].market_open
    ) == [event]
    assert len(sessions("2024-11-01", "2024-11-10")) == 6


def test_event_window_distinguishes_drawdown_from_monthly_loss():
    prices = pd.Series(
        [100.0, 120.0, 90.0, 110.0], index=["2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08"]
    )
    result = date_window(prices, "2024-11-06", "2024-11-09")
    assert result["return"] == pytest.approx(0.1)
    assert result["max_drawdown"] == pytest.approx(0.25)
    assert result["last_date"] == "2024-11-08"
    with pytest.raises(ValueError, match="baseline"):
        date_window(prices, "2024-11-05", "2024-11-08")


def test_confirmation_does_not_use_the_event_minute_close():
    opens = pd.date_range("2024-11-05", "2024-11-07T11:00", freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "bar_open_utc": opens,
            "bar_end_utc": opens + pd.Timedelta(minutes=1),
            "open": 100.0,
            "close": 100.0,
        }
    )
    event = pd.Timestamp("2024-11-06T10:34:00Z")
    frame.loc[frame.bar_open_utc >= event, "close"] = 120.0
    result = summarize_confirmation(frame, event, "2024-11-05T00:00:00Z")[0]
    assert result["pre_confirmation_price"] == 100
    assert result["return_after_60m"] == pytest.approx(0.2)
    assert result["return_episode_start_to_confirmation"] == 0
    # A bar not yet complete cannot affect the 24-hour result.
    frame.loc[frame.bar_open_utc >= event + pd.Timedelta(days=1), "close"] = 999.0
    assert summarize_confirmation(frame, event, "2024-11-05T00:00:00Z")[0] == result


def test_market_model_is_fitted_before_episode_and_rejects_data_gaps():
    dates = sessions("2026-01-01", "2026-04-01", "24/7").index
    rows = []
    for symbol, scale in [("A", 1), ("B", 0.6)]:
        price = 100 * np.exp(np.cumsum(scale * np.sin(np.arange(len(dates))) / 100))
        rows.extend(
            {
                "symbol": symbol,
                "session_date": day,
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1.5,
            }
            for day, p in zip(dates, price, strict=True)
        )
    frame = pd.DataFrame(rows)
    spec = {
        "symbols": ["A", "B"],
        "calendar": "24/7",
        "history_start": "2026-01-01",
        "history_end": "2026-04-01",
        "estimation_end": "2026-02-19",
        "anticipation_start": "2026-02-20",
        "event_at": "2026-03-01T12:00:00Z",
        "benchmark": "B",
    }
    original = summarize_daily(frame, spec)[0]
    altered = frame.copy()
    selected = altered.session_date >= "2026-03-01"
    altered.loc[selected, ["open", "high", "low", "close"]] *= 2
    revised = summarize_daily(altered, spec)[0]
    for field in ("alpha", "beta", "n", "estimation_last_return"):
        assert original["market_model"][field] == revised["market_model"][field]
    with pytest.raises(ValueError, match="coverage"):
        validate_bars(frame.iloc[1:], spec)
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(pd.concat([frame, frame.iloc[:1]]), spec)
