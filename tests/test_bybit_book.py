import copy

import pytest

from behavioral_market.data.bybit_book import BookGap, BookIntegrityError, BybitBook, units


def message(kind="snapshot", u=10, seq=100, b=None, a=None, offset=0):
    return {
        "topic": "orderbook.500.BTCUSDT",
        "type": kind,
        "ts": 1730678400005 + offset,
        "cts": 1730678400000 + offset,
        "data": {
            "s": "BTCUSDT",
            "u": u,
            "seq": seq,
            "b": [["99.00", "1.000"], ["98.00", "2.000"]] if b is None else b,
            "a": [["100.00", "0.002"], ["101.00", "0.008"]] if a is None else a,
        },
    }


def test_absolute_delta_zero_delete_and_cross_sequence_may_jump():
    book = BybitBook()
    book.apply(message())
    book.apply(
        message(
            "delta",
            11,
            851,
            b=[["99.00", "0"], ["98.50", "1.200"]],
            a=[["100.00", "0.003"]],
            offset=100,
        )
    )
    assert list(book.levels("bid")) == [(9850, 1200), (9800, 2000)]
    assert list(book.levels("ask"))[0] == (10000, 3)


def test_duplicate_is_idempotent_but_conflicting_repeated_id_invalidates():
    book = BybitBook()
    initial = message()
    book.apply(initial)
    assert book.apply(copy.deepcopy(initial)).kind == "duplicate"
    with pytest.raises(BookGap):
        book.apply(message("delta", 10, 100, b=[], a=[], offset=100))
    with pytest.raises(BookIntegrityError):
        book.sweep("buy", "0.001")


def test_caller_mutating_original_message_cannot_bypass_consistency_check():
    book = BybitBook()
    original = message()
    book.apply(original)
    original["data"]["a"][0][1] = "999.000"
    with pytest.raises(BookIntegrityError, match="Same-version snapshot"):
        book.apply(original)
    assert not book.valid


def test_malformed_level_invalidates_and_depth_coverage_is_explicit():
    book = BybitBook()
    book.apply(message())
    assert book.features()["ask_covers_25bps"]
    bad = message("delta", 11, 101, b=[["99", "1", "extra"]], a=[], offset=100)
    with pytest.raises(BookIntegrityError, match="Malformed"):
        book.apply(bad)
    assert not book.valid


def test_missing_message_requires_new_snapshot_and_snapshot_removes_old_levels():
    book = BybitBook()
    book.apply(message())
    with pytest.raises(BookGap):
        book.apply(message("delta", 12, 102, b=[], a=[], offset=200))
    with pytest.raises(BookGap):
        book.apply(message("delta", 13, 103, b=[], a=[], offset=300))
    book.apply(message("snapshot", 1, 110, b=[["99.50", "1"]], a=[["102", "2"]], offset=400))
    assert list(book.levels("bid")) == [(9950, 1000)]
    assert list(book.levels("ask")) == [(10200, 2000)]


@pytest.mark.parametrize(
    "change",
    [
        {"b": [["101", "1"]]},
        {"a": [["100", "-1"]]},
        {"a": [["100", "0.0001"]]},
        {"b": [["99", "1"], ["99", "2"]]},
        {"seq": 99},
        {"offset": -1},
    ],
)
def test_corruption_never_leaves_book_usable(change):
    book = BybitBook()
    book.apply(message())
    args = {"kind": "delta", "u": 11, "seq": 101, "offset": 100, "b": [], "a": []}
    args.update(change)
    with pytest.raises(BookIntegrityError):
        book.apply(message(**args))
    assert not book.valid


def test_visible_sweep_partial_limit_and_depth_do_not_mutate_history():
    book = BybitBook()
    book.apply(message())
    before = book.features()
    result = book.sweep("buy", "0.010")
    assert result["filled_units"] == 10 and result["average_price"] == "100.8"
    assert result["quote_notional"] == "1.008"
    assert book.sweep("buy", "0.020")["unfilled_units"] == 10
    assert book.sweep("buy", "0.010", "100.00")["filled_units"] == 2
    assert book.sweep("buy", "0.010", "99.99")["filled_units"] == 0
    assert book.sweep("sell", "1.500")["average_price"] is not None
    assert book.features() == before


def test_decimal_storage_rejects_rounding_and_nonfinite_values():
    assert units("0.001000", 3) == 1
    assert units("68740.30", 2) == 6874030
    for value in ("NaN", "inf", "1e3", "0.0009", " 1.0", 1.0):
        with pytest.raises(BookIntegrityError):
            units(value, 3)
