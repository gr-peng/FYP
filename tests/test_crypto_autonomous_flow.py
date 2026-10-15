import builtins
import inspect
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction

import pytest

from experiments.crypto_autonomous_flow import (
    GRID_US,
    FeatureContext,
    InitialBook,
    KnownPrint,
    initial_features,
    simulate_steps,
    simulate_window,
)
from experiments.crypto_effective_flow import (
    FlowTemplate,
    Intent,
    RelativeLiquidity,
    TrainingRow,
    fit_template_pool,
)

START = 10_000_000
TRAIN_START = 1_730_419_200_000_000
DAY = "2024-11-01"
BOOK = InitialBook(((9900, 1000),), ((10000, 1000),))


def context(book=BOOK, prints=(), *, pressure_before=None):
    features = initial_features(book, prints)
    old = features if pressure_before is None else replace(features, trade_pressure=pressure_before)
    return tuple(
        FeatureContext(t, old if t < START else features)
        for t in range(START - 5_000_000, START + 1, GRID_US)
    )


def template(*, adds=(), removes=(), intents=()):
    return FlowTemplate("placeholder", tuple(adds), tuple(removes), tuple(intents))


def pool(templates, *, book=BOOK, features_by_row=None):
    base = initial_features(book)
    rows = []
    for i, flow in enumerate(templates):
        features = (base, base, base) if features_by_row is None else features_by_row[i]
        rows.append(
            TrainingRow(DAY, replace(flow, row_id=f"{DAY}:{TRAIN_START + i * GRID_US}"), features)
        )
    return fit_template_pool(tuple(rows))


def run(flow, *, book=BOOK, steps=1, ordering="S1", delay=0, seed=7, method="M0"):
    fitted = pool([flow], book=book)
    return simulate_steps(
        fitted,
        book,
        (),
        context(book),
        start_us=START,
        method=method,
        ordering=ordering,
        delay_ms=delay,
        seed=seed,
        steps=steps,
    )


def assert_conserved(receipt):
    conserved = receipt.conservation
    assert conserved.equity_error == 0
    assert conserved.signed_position_btc_sum == 0
    assert conserved.bid_quantity_balance_error_units == 0
    assert conserved.ask_quantity_balance_error_units == 0
    assert conserved.intent_quantity_balance_error_units == 0
    assert conserved.print_quantity_balance_error_units == 0


def test_two_ordering_hypotheses_distinguish_replenishment_after_execution():
    book = InitialBook(((9900, 1000),), ((10000, 1000), (10100, 1000)))
    flow = template(adds=[RelativeLiquidity("bid", 150, 1000)], intents=[Intent("Buy", 1000)])
    first = run(flow, book=book, ordering="S1")
    second = run(flow, book=book, ordering="S2")
    assert first.status == second.status == "complete"
    assert first.path[-1].best_bid_units == 9900
    assert second.path[-1].best_bid_units == 10050
    assert first.count("add_rejected_units") == 1000 and second.count("added_units") == 1000
    assert first.actions[0].reason == "passive_cross"
    assert first.prints[0].price_units == second.prints[0].price_units == 10000
    assert first.path[-1].mid_twice_units != second.path[-1].mid_twice_units
    assert_conserved(first)
    assert_conserved(second)


def test_temporary_empty_side_can_be_replenished_in_the_same_step():
    book = InitialBook(((9900, 1000),), ((10000, 1000), (10100, 1000)))
    flow = template(adds=[RelativeLiquidity("ask", 100, 1000)], intents=[Intent("Buy", 2000)])
    receipt = run(flow, book=book, ordering="S2")
    assert receipt.status == "complete"
    assert receipt.path[-1].asks == ((10100, 1000),)
    assert receipt.count("undefined_points") == 0
    assert receipt.count("intent_filled_units") == 2000
    assert receipt.count("added_units") == 1000
    assert_conserved(receipt)


def test_trade_consumption_is_once_and_same_price_fills_merge_per_intent():
    flow = template(adds=[RelativeLiquidity("ask", 0, 1000)], intents=[Intent("Buy", 1500)])
    receipt = run(flow)
    assert receipt.status == "complete"
    assert receipt.final_asks == ((10000, 500),)
    assert receipt.count("intent_count") == 1
    assert receipt.count("intent_filled_units") == 1500
    assert receipt.count("execution_fills") == 2 and receipt.count("output_prints") == 1
    assert receipt.prints[0].quantity_units == 1500
    assert len(receipt.prints[0].fill_ids) == 2
    assert receipt.prints[0].timestamp_us == START + 1000
    assert_conserved(receipt)
    unadded = run(
        template(intents=[Intent("Buy", 1000)]), book=InitialBook(((9900, 1000),), ((10000, 2000),))
    )
    assert unadded.final_asks == ((10000, 1000),)
    assert unadded.count("removed_units") == 0
    assert_conserved(unadded)


def test_mapping_stays_at_step_start_reference_after_inside_spread_add():
    book = InitialBook(((9900, 1000),), ((10100, 1000),))
    flow = template(adds=[RelativeLiquidity("bid", 50, 100), RelativeLiquidity("bid", 75, 100)])
    receipt = run(flow, book=book)
    assert receipt.path[-1].bids[:2] == ((9975, 100), (9950, 100))
    assert [a.price_units for a in receipt.actions] == [9975, 9950]  # Canonical descending bids.
    assert_conserved(receipt)


def test_removal_clips_to_existing_qty_and_keeps_fifo_cancellations():
    book = InitialBook(((9900, 2000),), ((10000, 1000),))
    flow = template(
        adds=[RelativeLiquidity("bid", 0, 1000)], removes=[RelativeLiquidity("bid", 0, 1500)]
    )
    receipt = run(flow, book=book, steps=2)
    removals = [a for a in receipt.actions if a.kind == "remove"]
    assert removals[0].order_ids == ("PXO00000001",)
    assert removals[1].order_ids == ("PXO00000001", "PXO00000003")
    assert receipt.final_bids == ((9900, 1000),)
    assert_conserved(receipt)
    clipped = run(
        template(
            adds=[RelativeLiquidity("bid", 0, 1000)], removes=[RelativeLiquidity("bid", 0, 3000)]
        )
    )
    assert clipped.count("removed_units") == 1000
    assert clipped.count("remove_unimplemented_units") == 2000
    assert clipped.count("clipped_remove_events") == 1
    assert_conserved(clipped)


def test_cash_is_finite_and_rejected_addition_does_not_mutate_book_or_accounts():
    repeated = run(template(adds=[RelativeLiquidity("bid", 0, 1000)]), steps=3)
    assert repeated.initial_cash_per_account == Decimal(199)
    assert sum(a.state.cash for a in repeated.accounts) == Decimal(796)
    assert repeated.count("added_units") == 1000
    assert repeated.count("add_rejected_units") == 2000
    assert repeated.final_bids == ((9900, 2000),)
    assert all(a.reason == "risk" for a in repeated.actions[1:])
    assert_conserved(repeated)
    baseline = run(template(intents=[Intent("Buy", 1)]))
    rejected = run(
        template(adds=[RelativeLiquidity("bid", 0, 2_000_000)], intents=[Intent("Buy", 1)])
    )
    assert rejected.actions[0].reason == "risk"
    assert rejected.final_bids == baseline.final_bids and rejected.final_asks == baseline.final_asks
    assert rejected.accounts == baseline.accounts and rejected.prints == baseline.prints
    assert rejected.actions[1].client_order_id == "step:0:intent:0"
    assert_conserved(rejected)


def test_invalid_prices_and_sizes_are_counted_without_silent_clamping():
    flow = template(
        adds=[RelativeLiquidity("bid", -9901, 1000), RelativeLiquidity("ask", 0, 2**63)],
        removes=[RelativeLiquidity("ask", -10001, 1000)],
    )
    receipt = run(flow)
    assert receipt.final_bids == BOOK.bids and receipt.final_asks == BOOK.asks
    assert {a.reason for a in receipt.actions} == {"invalid_price", "invalid_quantity"}
    assert receipt.count("add_rejected_units") == 1000 + 2**63
    assert receipt.count("remove_unimplemented_units") == 1000
    assert_conserved(receipt)


def test_missing_side_has_fixed_failure_denominator_and_no_future_replenishment():
    flow = template(intents=[Intent("Buy", 1500)])
    receipt = run(flow, steps=3)
    assert receipt.status == "failed_missing_side" and receipt.failure_at_us == START + GRID_US
    assert len(receipt.path) == 4 and receipt.count("requested_steps") == 3
    assert receipt.count("executed_steps") == 1 and receipt.count("undefined_points") == 3
    assert receipt.path[0].is_valid
    assert receipt.path[1].asks == () and receipt.path[1].mid_twice_units is None
    assert receipt.path[2].bids is None and receipt.path[2].spread_bps is None
    assert receipt.path[3].asks is None and receipt.path[3].mark_price_units is None
    assert (
        receipt.count("intent_filled_units") == 1000 and receipt.count("ioc_cancelled_units") == 500
    )
    assert receipt.final_asks == () and receipt.financial_mark_stale
    assert_conserved(receipt)


def test_unfinanceable_initial_depth_is_failure_not_capital_injection_or_hidden_trimming():
    book = InitialBook(((1, 1000),), ((10000, 1),))
    receipt = run(template(), book=book, steps=2)
    assert receipt.status == "failed_initialization"
    assert receipt.initial_cash_per_account == Decimal("0.11")
    assert len(receipt.path) == 3 and all(not p.is_valid for p in receipt.path)
    assert receipt.count("executed_steps") == 0
    assert receipt.initial_actions[-1].reason == "risk"
    assert receipt.conservation.initial_equity == Decimal("0.44")
    assert_conserved(receipt)


def test_mark_half_up_and_scoring_mid_keep_distinct_precision():
    book = InitialBook(((9999, 1000),), ((10000, 1000),))
    receipt = run(template(), book=book)
    assert receipt.path[-1].mid_twice_units == 19999
    assert receipt.path[-1].mark_price_units == receipt.final_financial_mark_units == 10000
    assert all(a.state.mark_price == Decimal(100) for a in receipt.accounts)
    assert receipt.path[-1].spread_bps == Fraction(20000, 19999)
    assert_conserved(receipt)


def test_state_features_use_generated_prints_and_only_the_requested_delayed_history():
    base = initial_features(BOOK)
    negative = replace(base, trade_pressure=Fraction(-1))
    positive = replace(base, trade_pressure=Fraction(1))
    buys = [template(intents=[Intent("Buy", 1)]) for _ in range(32)]
    sells = [template(intents=[Intent("Sell", 1)]) for _ in range(32)]
    features = [(base, negative, negative)] * 32 + [(positive, positive, positive)] * 32
    fitted = pool(buys + sells, features_by_row=features)
    past = context(pressure_before=Fraction(-1))
    immediate = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M1",
        ordering="S1",
        delay_ms=0,
        seed=4,
        steps=2,
    )
    delayed = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M1",
        ordering="S1",
        delay_ms=1000,
        seed=4,
        steps=2,
    )
    five = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M1",
        ordering="S1",
        delay_ms=5000,
        seed=4,
        steps=2,
    )
    assert [p.side for p in immediate.prints] == ["Buy", "Sell"]
    assert [p.side for p in delayed.prints] == ["Buy", "Buy"]
    assert [p.side for p in five.prints] == ["Buy", "Buy"]
    assert immediate.selections[1].features.trade_pressure == Fraction(1)
    assert delayed.selections[1].feature_us == START - 900_000
    assert delayed.selections[1].features.trade_pressure == Fraction(-1)
    assert all(
        not s.fallback for s in (*immediate.selections, *delayed.selections, *five.selections)
    )
    assert_conserved(immediate)
    assert_conserved(delayed)


@pytest.mark.parametrize("delay", [0, 1000, 5000])
def test_seed_reproducibility_small_cells_and_fixed_no_future_entry(delay, monkeypatch):
    flows = [template(), template(adds=[RelativeLiquidity("bid", 0, 1)])]
    fitted = pool(flows)
    past = context()
    first = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M1",
        ordering="S1",
        delay_ms=delay,
        seed=12,
        steps=5,
    )
    second = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M1",
        ordering="S1",
        delay_ms=delay,
        seed=12,
        steps=5,
    )
    assert first == second and first.count("fallbacks") == 5
    assert BOOK.bids == ((9900, 1000),) and BOOK.asks == ((10000, 1000),)
    assert "horizon" not in inspect.signature(simulate_window).parameters
    assert not {"future_price", "labels", "reader", "path", "q1"}.intersection(
        inspect.signature(simulate_window).parameters
    )
    with pytest.raises(TypeError):
        simulate_window(
            fitted,
            BOOK,
            (),
            past,
            start_us=START,
            method="M0",
            ordering="S1",
            delay_ms=delay,
            seed=12,
            future_price=123,
        )
    monkeypatch.setattr(builtins, "open", lambda *a, **k: pytest.fail("Generator opened a file"))
    pure = simulate_steps(
        fitted,
        BOOK,
        (),
        past,
        start_us=START,
        method="M0",
        ordering="S1",
        delay_ms=delay,
        seed=12,
        steps=1,
    )
    assert pure.status == "complete"
    with pytest.raises(FrozenInstanceError):
        pure.status = "changed"


def test_real_entry_is_always_five_minutes_even_when_the_seed_fails_early():
    fitted = pool([template(intents=[Intent("Buy", 1000)])])
    receipt = simulate_window(
        fitted, BOOK, (), context(), start_us=START, method="M0", ordering="S1", delay_ms=0, seed=5
    )
    assert receipt.end_us - receipt.start_us == 300_000_000
    assert receipt.requested_steps == 3000 and len(receipt.path) == 3001
    assert receipt.count("executed_steps") == 1 and receipt.count("undefined_points") == 3000
    assert_conserved(receipt)


def test_complete_fixed_synthetic_window_preserves_all_grid_endpoints_and_capital_policy():
    fitted = pool([template()])
    receipt = simulate_window(
        fitted,
        BOOK,
        (),
        context(),
        start_us=START,
        method="M1",
        ordering="S2",
        delay_ms=5000,
        seed=9,
    )
    assert receipt.status == "complete" and len(receipt.path) == 3001
    assert receipt.count("executed_steps") == receipt.count("fallbacks") == 3000
    assert receipt.path[-1].timestamp_us == START + 300_000_000
    assert receipt.path[-1].bids == BOOK.bids and receipt.path[-1].asks == BOOK.asks
    assert receipt.selections[-1].feature_us == receipt.selections[-1].decision_us - 5_000_000
    assert receipt.capital_configuration.max_leverage == 1
    assert receipt.capital_configuration.topups_allowed is False
    assert (
        receipt.capital_configuration.maker_fee_rate
        == receipt.capital_configuration.taker_fee_rate
        == 0
    )
    assert_conserved(receipt)


@pytest.mark.parametrize(
    "change",
    ["future_print", "old_print", "duplicate", "missing_grid", "future_grid", "start_mismatch"],
)
def test_initial_context_refuses_future_missing_or_inconsistent_inputs(change):
    prints = (KnownPrint(START - 500_000, "Buy", 1, "past"),)
    past = context(prints=prints)
    if change == "future_print":
        prints = (KnownPrint(START, "Buy", 1, "future"),)
    elif change == "old_print":
        prints = (KnownPrint(START - 1_000_001, "Buy", 1, "too-old"),)
    elif change == "duplicate":
        prints = prints + prints
    elif change == "missing_grid":
        past = past[1:]
    elif change == "future_grid":
        past = past[:-1] + (replace(past[-1], timestamp_us=START + GRID_US),)
    else:
        past = past[:-1] + (replace(past[-1], features=initial_features(BOOK)),)
    with pytest.raises(ValueError):
        simulate_steps(
            pool([template()]),
            BOOK,
            prints,
            past,
            start_us=START,
            method="M0",
            ordering="S1",
            delay_ms=0,
            seed=1,
            steps=1,
        )
