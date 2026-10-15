import copy

import pytest

from experiments.crypto_autonomous_flow import initial_features
from experiments.crypto_autonomous_targets import _collect_span, collect_train_day
from experiments.crypto_effective_flow import ObservedPrint
from experiments.crypto_execution_diagnostic import day_start_us

DAY = "2024-11-01"
START = day_start_us(DAY)


def messages(seconds=900, *, reset_index=None):
    for i in range(seconds * 10):
        t = START // 1000 + i * 100 + 10
        yield {
            "topic": "orderbook.500.BTCUSDT", "type": "snapshot"
            if i in (0, reset_index) else "delta", "cts": t, "ts": t + 1,
            "data": {"s": "BTCUSDT", "u": i + 1, "seq": i + 10,
                     "b": [["99.99", "1.000"], ["99.00", "2.000"]],
                     "a": [["100.00", "1.000"], ["101.00", "2.000"]]},
        }


def test_missing_start_state_remains_incomplete_minute_not_zero_return():
    result = _collect_span(messages(120), (), DAY, START + 120_000_000)
    first, second = result.minute_labels
    assert not first.complete and first.values[0] is None
    assert second.complete and second.values[0] == 0
    assert first.values[1:3] == (0, 0)
    assert result.observed_prints == result.observed_quantity_units == 0


def test_full_source_quantity_and_half_open_print_minute_allocation():
    tape = (
        ObservedPrint(START + 60_000_000 - 1, "Buy", 3, 10000, 1, "a"),
        ObservedPrint(START + 60_000_000, "Sell", 7, 9999, 2, "b"),
    )
    result = _collect_span(messages(120), tape, DAY, START + 120_000_000)
    assert result.observed_prints == 2 and result.observed_quantity_units == 10
    assert result.minute_labels[0].values[1:3] == (0.003, 1)
    assert result.minute_labels[1].values[1:3] == (0.007, 1)
    assert result.minute_labels[1].values[-1] == -1


def test_prefix_reset_inside_minute_is_not_a_complete_label():
    result = _collect_span(messages(120, reset_index=900), (), DAY, START + 120_000_000)
    assert "reset_or_gap_crossing" in result.minute_labels[1].issues
    assert not result.minute_labels[1].complete


def test_fixed_hour_ten_minute_initialization_is_past_only_and_whole_book():
    at = START + 600_000_000
    tape = (
        ObservedPrint(at - 1, "Buy", 3, 10000, 1, "past"),
        ObservedPrint(at, "Sell", 999, 9999, 2, "future"),
    )
    result = _collect_span(messages(660), tape, DAY, START + 660_000_000)
    init = result.initializations[0]
    assert init.start_us == at and not init.issues
    assert init.initial_book.bids == ((9999, 1000), (9900, 2000))
    assert len(init.feature_context) == len(init.prefix_context) == 51
    assert [p.print_id for p in init.past_prints] == ["past"]
    assert init.feature_context[-1].features == initial_features(
        init.initial_book, init.past_prints)
    assert all(row[3] < row[0] <= at for row in init.prefix_context)


def test_changing_future_after_initialization_leaves_all_generator_inputs_identical():
    at = START + 600_000_000
    inputs = (ObservedPrint(at - 1, "Buy", 1, 10000, 1, "past"),)
    before = copy.deepcopy(inputs)
    first = _collect_span(messages(660), (*inputs,
        ObservedPrint(at + 1, "Buy", 1, 10000, 2, "future")), DAY, START + 660_000_000)
    second = _collect_span(messages(660), (*inputs,
        ObservedPrint(at + 1, "Sell", 999, 9999, 2, "future")), DAY, START + 660_000_000)
    assert first.initializations == second.initializations
    assert first.minute_labels[-1].values != second.minute_labels[-1].values
    assert inputs == before


def test_gap_or_reset_past_context_preserves_initialization_failure_candidate():
    result = _collect_span(messages(660, reset_index=5980), (), DAY, START + 660_000_000)
    init = result.initializations[0]
    assert init.start_us == START + 600_000_000
    assert init.initial_book is init.feature_context is init.past_prints is None
    assert init.issues == ("invalid_stale_or_reset_past_context",)


def test_contiguous_five_minute_price_baseline_does_not_hide_first_failed_block():
    result = _collect_span(messages(600), (), DAY, START + 600_000_000)
    assert len(result.price_blocks) == 2
    assert result.price_blocks[0].issues and result.price_blocks[0].returns_bps[0] is None
    assert not result.price_blocks[1].issues
    assert result.price_blocks[1].returns_bps == (0,) * 5


@pytest.mark.parametrize("day", ["2024-11-04", "2024-11-05", "2024-10-14"])
def test_calendar_gate_precedes_any_stream_read(day):
    def forbidden():
        raise AssertionError("Unapproved raw must remain unread")
        yield None
    with pytest.raises(ValueError, match="closed"):
        collect_train_day(forbidden(), forbidden(), day)


def test_nonmonotone_tape_never_silently_resorts():
    tape = (ObservedPrint(START + 50000, "Buy", 1, 10000, 2, "a"),
            ObservedPrint(START + 40000, "Buy", 1, 10000, 1, "b"))
    with pytest.raises(ValueError, match="reassignment"):
        _collect_span(messages(60), tape, DAY, START + 60_000_000)
