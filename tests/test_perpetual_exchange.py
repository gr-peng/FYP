from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from behavioral_market.market.order import OrderStatus
from behavioral_market.market.perpetual_exchange import ExperimentalPerpetualExchange

D = Decimal


def register(exchange, name, cash="1000", *, maker="0", taker="0", minimum="0", leverage="2"):
    return exchange.register(
        name,
        cash,
        maker_fee_rate=maker,
        taker_fee_rate=taker,
        min_open_notional=minimum,
        max_leverage=leverage,
    )


def submit(exchange, name, side, price, quantity, client, tif="DAY"):
    return exchange.submit_order(
        name, side, price, quantity, client_order_id=client, time_in_force=tif
    )


def snapshot(exchange, names):
    return (
        exchange.sequences,
        exchange.day,
        exchange.mark_price_units,
        exchange.fee_balance,
        exchange.resting_orders(),
        exchange.trades,
        tuple(
            (
                exchange.account_state(name),
                exchange.account_receipts(name),
                exchange.risk_snapshot(name),
            )
            for name in names
        ),
    )


def total_equity_and_fees(exchange, names):
    return sum((exchange.account_state(name).equity for name in names), D(0)) + exchange.fee_balance


def test_fractional_btc_fees_and_bilateral_equity_identity():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker", maker="0.0002")
    register(exchange, "taker", taker="0.00055")
    submit(exchange, "maker", "sell", 10123, 1, "offer")
    order, trades = submit(exchange, "taker", "buy", 10123, 1, "take", "IOC")
    assert order.status == OrderStatus.FILLED
    assert len(trades) == 1 and trades[0].quantity_units == 1
    maker, taker = exchange.account_state("maker"), exchange.account_state("taker")
    assert maker.position == D("-0.001") and taker.position == D("0.001")
    assert maker.average_entry == taker.average_entry == D("101.23")
    maker_fee = D("0.001") * D("101.23") * D("0.0002")
    taker_fee = D("0.001") * D("101.23") * D("0.00055")
    assert maker.cash == D(1000) - maker_fee
    assert taker.cash == D(1000) - taker_fee  # Contract notional is not a cash debit.
    assert exchange.fee_balance == maker_fee + taker_fee
    assert total_equity_and_fees(exchange, ["maker", "taker"]) == D(2000)
    assert trades[0].maker_fill.liquidity == "maker"
    assert trades[0].taker_fill.liquidity == "taker"
    assert trades[0].maker_fill.event_id != trades[0].taker_fill.event_id


def test_storage_price_conversion_is_exact_even_with_a_lower_decimal_context():
    # Ledger arithmetic follows its documented Decimal context; storage conversion
    # and the retained limit/fill price must not round merely during conversion.
    with localcontext() as context:
        context.prec = 6
        exchange = ExperimentalPerpetualExchange(10000)
        register(exchange, "maker", cash="1e15")
        register(exchange, "taker", cash="1e15")
        offer, _ = submit(exchange, "maker", "sell", 1234567899, 1, "large-price")
        _, trades = submit(exchange, "taker", "buy", 1234567899, 1, "large-take", "IOC")
        assert offer.price_units == trades[0].price_units == 1234567899
        assert trades[0].maker_fill.price == D("12345678.99")
        assert trades[0].taker_fill.signed_quantity == D("0.001")


@pytest.mark.parametrize(
    "side,older_price,better_price,active_side,limit_price",
    [
        ("buy", 1234567890, 1234567899, "sell", 1234567890),
        ("sell", 1234567899, 1234567890, "buy", 1234567899),
    ],
)
def test_low_precision_price_priority_uses_exact_both_side_keys(
    side, older_price, better_price, active_side, limit_price
):
    with localcontext() as context:
        context.prec = 6
        exchange = ExperimentalPerpetualExchange(10000)
        for name in ["older", "better", "taker"]:
            register(exchange, name, cash="1e15")
        submit(exchange, "older", side, older_price, 1, "older")
        submit(exchange, "better", side, better_price, 1, "better")
        _, trades = submit(exchange, "taker", active_side, limit_price, 1, "take", "IOC")
        assert len(trades) == 1
        assert trades[0].maker_id == "better" and trades[0].price_units == better_price
        assert exchange.resting_orders()[0].agent_id == "older"


def test_price_priority_fifo_and_ioc_partial_execution():
    exchange = ExperimentalPerpetualExchange(10000)
    for name in ["older", "first", "second", "taker"]:
        register(exchange, name)
    older, _ = submit(exchange, "older", "sell", 10100, 1000, "old-price")
    first, _ = submit(exchange, "first", "sell", 10000, 1000, "first-price")
    second, _ = submit(exchange, "second", "sell", 10000, 1000, "second-price")
    order, trades = submit(exchange, "taker", "buy", 10100, 3500, "sweep", "IOC")
    assert [(t.maker_id, t.price_units, t.quantity_units) for t in trades] == [
        ("first", 10000, 1000),
        ("second", 10000, 1000),
        ("older", 10100, 1000),
    ]
    assert order.filled_quantity_units == 3000
    assert order.cancelled_quantity_units == 500
    assert order.remaining_quantity_units == 0
    assert order.status == OrderStatus.CANCELLED
    assert exchange.resting_orders() == ()
    assert all(
        exchange.order(o.order_id).status == OrderStatus.FILLED for o in [older, first, second]
    )
    assert total_equity_and_fees(exchange, ["older", "first", "second", "taker"]) == D(4000)


def test_ioc_checks_visible_plan_and_remainder_cannot_rest():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker")
    register(exchange, "buyer", cash="60")
    submit(exchange, "maker", "sell", 10000, 1000, "offer")
    order, _ = submit(exchange, "buyer", "buy", 10000, 1500, "limited", "IOC")
    assert order.filled_quantity_units == 1000 and order.cancelled_quantity_units == 500
    assert exchange.account_state("buyer").position == D(1)
    assert exchange.risk_snapshot("buyer").reserved_collateral == 0
    assert exchange.risk_snapshot("buyer").sufficient
    empty, trades = submit(exchange, "buyer", "buy", 10000, 1, "empty", "IOC")
    assert trades == () and empty.cancelled_quantity_units == 1
    assert exchange.resting_orders() == ()


def test_remaining_day_reservation_failure_rolls_back_already_previewed_fill():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker")
    register(exchange, "buyer", cash="100")
    submit(exchange, "maker", "sell", 10000, 1000, "offer")
    before = snapshot(exchange, ["maker", "buyer"])
    with pytest.raises(ValueError, match="pending-order"):
        submit(exchange, "buyer", "buy", 10000, 3000, "failed")
    assert snapshot(exchange, ["maker", "buyer"]) == before
    with pytest.raises(ValueError, match="Unknown client"):
        exchange.order_for_client("failed")
    retried, _ = submit(exchange, "buyer", "buy", 10000, 1000, "failed", "IOC")
    assert retried.order_id == "PXO00000002"
    assert exchange.trades[0].trade_id == "PXT00000001"


def test_opposing_orders_do_not_net_and_partial_cancel_releases_budget():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker", cash="100")
    first, _ = submit(exchange, "maker", "buy", 9900, 1000, "bid1")
    submit(exchange, "maker", "buy", 9800, 1000, "bid2")
    assert exchange.risk_snapshot("maker").reserved_collateral == D(100)
    before = snapshot(exchange, ["maker"])
    with pytest.raises(ValueError, match="pending-order"):
        submit(exchange, "maker", "sell", 10100, 1000, "ask")
    assert snapshot(exchange, ["maker"]) == before
    cancelled = exchange.cancel_order(first.order_id, "maker", quantity_units=500)
    assert cancelled.remaining_quantity_units == cancelled.cancelled_quantity_units == 500
    assert exchange.risk_snapshot("maker").reserved_collateral == D(75)
    submit(exchange, "maker", "sell", 10100, 500, "ask")
    assert exchange.risk_snapshot("maker").reserved_collateral == D(100)
    exchange.cancel_order(first.order_id, "maker")
    assert exchange.risk_snapshot("maker").reserved_collateral == D(75)


def test_adverse_entry_loss_and_positive_fee_buffer_are_reserved():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "under", cash="151.49", maker="0.01", leverage="1")
    register(exchange, "exact", cash="151.50", maker="0.01", leverage="1")
    with pytest.raises(ValueError, match="pending-order"):
        submit(exchange, "under", "buy", 15000, 1000, "under")
    submit(exchange, "exact", "buy", 15000, 1000, "exact")
    risk = exchange.risk_snapshot("exact")
    assert risk.open_order_margin == D(100)
    assert risk.adverse_entry_buffer == D(50)
    assert risk.maker_fee_buffer == D("1.50")
    assert risk.sufficient and risk.free_collateral == 0


def test_complete_order_minimum_is_not_reapplied_to_partial_fills_or_pure_close():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker")
    register(exchange, "buyer", minimum="100")
    register(exchange, "other")
    submit(exchange, "maker", "sell", 10000, 500, "small-offer")
    bought, fills = submit(exchange, "buyer", "buy", 10000, 1000, "full-notional", "IOC")
    assert bought.filled_quantity_units == 500
    assert fills[0].taker_fill.opened_quantity == D("0.5")
    submit(exchange, "other", "buy", 10000, 600, "close-bid")
    before = snapshot(exchange, ["maker", "buyer", "other"])
    with pytest.raises(ValueError, match="minimum opening"):
        submit(exchange, "buyer", "sell", 9900, 600, "small-flip", "IOC")
    assert snapshot(exchange, ["maker", "buyer", "other"]) == before
    closed, fills = submit(exchange, "buyer", "sell", 9900, 500, "pure-close", "IOC")
    assert closed.status == OrderStatus.FILLED
    assert fills[0].taker_fill.closed_quantity == D("0.5")
    assert fills[0].taker_fill.opened_quantity == 0
    assert exchange.account_state("buyer").position == 0


def test_minimum_is_admission_only_and_a_reducing_resting_order_can_later_open():
    exchange = ExperimentalPerpetualExchange(10000)
    for name in ["maker", "closer", "opener"]:
        register(exchange, name)
    register(exchange, "account", minimum="100")
    submit(exchange, "maker", "sell", 10000, 500, "offer")
    submit(exchange, "account", "buy", 10000, 1000, "open", "IOC")
    # 0.5 BTC @110 is below 100 USDT, but at admission it only reduces the long.
    reducing, _ = submit(exchange, "account", "sell", 11000, 500, "reducing-day")
    submit(exchange, "closer", "buy", 10000, 500, "close-bid")
    submit(exchange, "account", "sell", 10000, 500, "close-first", "IOC")
    assert exchange.account_state("account").position == 0
    _, trades = submit(exchange, "opener", "buy", 11000, 500, "later-take", "IOC")
    assert trades[0].maker_order_id == reducing.order_id
    assert trades[0].maker_fill.opened_quantity == D("0.5")
    assert exchange.account_state("account").position == D("-0.5")


def test_close_and_flip_both_accounts_without_spot_notional_cash_transfers():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "a", leverage="5")
    register(exchange, "b", leverage="5")
    submit(exchange, "a", "sell", 10000, 2000, "open-short")
    submit(exchange, "b", "buy", 10000, 2000, "open-long", "IOC")
    submit(exchange, "a", "buy", 11000, 3000, "reverse-bid")
    _, fills = submit(exchange, "b", "sell", 11000, 3000, "reverse-sell", "IOC")
    a, b = exchange.account_state("a"), exchange.account_state("b")
    assert a.position == D(1) and b.position == D(-1)
    assert a.entry_cost == b.entry_cost == D(110)
    assert a.realized_pnl == D(-20) and b.realized_pnl == D(20)
    assert a.cash == D(980) and b.cash == D(1020)
    assert fills[0].maker_fill.closed_quantity == fills[0].taker_fill.closed_quantity == D(2)
    assert fills[0].maker_fill.opened_quantity == fills[0].taker_fill.opened_quantity == D(1)
    assert total_equity_and_fees(exchange, ["a", "b"]) == D(2000)
    exchange.set_mark_price(12000)
    assert total_equity_and_fees(exchange, ["a", "b"]) == D(2000)


def test_partial_cancel_preserves_fifo_and_separates_cancelled_from_filled():
    exchange = ExperimentalPerpetualExchange(10000)
    for name in ["a", "b", "seller"]:
        register(exchange, name)
    first, _ = submit(exchange, "a", "buy", 10000, 2000, "first")
    second, _ = submit(exchange, "b", "buy", 10000, 1000, "second")
    partial = exchange.cancel_order(first.order_id, "a", quantity_units=500)
    assert partial.timestamp == first.timestamp
    _, trades = submit(exchange, "seller", "sell", 10000, 2000, "sell", "IOC")
    assert [(t.maker_id, t.quantity_units) for t in trades] == [("a", 1500), ("b", 500)]
    first_after, second_after = exchange.order(first.order_id), exchange.order(second.order_id)
    assert first_after.filled_quantity_units == 1500 and first_after.cancelled_quantity_units == 500
    assert first_after.remaining_quantity_units == 0
    assert first_after.status == OrderStatus.CANCELLED
    assert (
        second_after.remaining_quantity_units == 500 and second_after.timestamp == second.timestamp
    )


def test_day_expiration_releases_pending_risk_without_account_mutation():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker")
    order, _ = submit(exchange, "maker", "buy", 9900, 1000, "day")
    state = exchange.account_state("maker")
    expired = exchange.end_day()
    assert len(expired) == 1 and expired[0].order_id == order.order_id
    assert expired[0].status == OrderStatus.EXPIRED
    assert expired[0].expired_quantity_units == 1000
    assert expired[0].cancelled_quantity_units == expired[0].remaining_quantity_units == 0
    assert exchange.day == 1 and exchange.resting_orders() == ()
    assert exchange.account_state("maker") == state
    assert exchange.risk_snapshot("maker").reserved_collateral == 0
    before = snapshot(exchange, ["maker"])
    with pytest.raises(ValueError, match="unique"):
        submit(exchange, "maker", "buy", 9900, 1, "day", "IOC")
    assert snapshot(exchange, ["maker"]) == before


def test_late_maker_risk_failure_rolls_back_all_fills_counters_fees_and_client_id():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "safe", maker="0.001")
    register(exchange, "late", cash="100", maker="0.001")
    register(exchange, "taker", taker="0.002")
    submit(exchange, "safe", "sell", 10000, 1000, "first")
    submit(exchange, "late", "sell", 10000, 1000, "second")
    exchange.set_mark_price(30000)
    assert not exchange.risk_snapshot("late").sufficient
    before = snapshot(exchange, ["safe", "late", "taker"])
    with pytest.raises(ValueError, match="leverage cap"):
        submit(exchange, "taker", "buy", 10000, 2000, "late-failure", "IOC")
    assert snapshot(exchange, ["safe", "late", "taker"]) == before
    with pytest.raises(ValueError, match="Unknown client"):
        exchange.order_for_client("late-failure")
    # Retry the same unconsumed client ID; the first valid fill has no sequence gap.
    order, trades = submit(exchange, "taker", "buy", 10000, 1000, "late-failure", "IOC")
    assert order.order_id == "PXO00000003" and trades[0].trade_id == "PXT00000001"
    assert exchange.order_for_client("late-failure") == order


def test_late_taker_failure_also_rolls_back_second_makers_preview():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "first", maker="0.001")
    register(exchange, "second", maker="0.001")
    register(exchange, "taker", cash="55", taker="0.002")
    submit(exchange, "first", "sell", 10000, 1000, "first")
    submit(exchange, "second", "sell", 12000, 1000, "second")
    before = snapshot(exchange, ["first", "second", "taker"])
    # The first 1 BTC execution passes; the second maker also passes, but the
    # taker's second opening exceeds its cap after the worse price and fees.
    with pytest.raises(ValueError, match="leverage cap"):
        submit(exchange, "taker", "buy", 12000, 2000, "late-taker", "IOC")
    assert snapshot(exchange, ["first", "second", "taker"]) == before
    assert exchange.account_receipts("second") == ()


def test_self_cross_is_rejected_before_external_fill_even_with_better_other_quote():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "owner")
    register(exchange, "other")
    submit(exchange, "owner", "sell", 10100, 1000, "own-ask")
    submit(exchange, "other", "sell", 10000, 1000, "other-ask")
    before = snapshot(exchange, ["owner", "other"])
    with pytest.raises(ValueError, match="self-crossing"):
        submit(exchange, "owner", "buy", 10100, 1500, "self", "IOC")
    assert snapshot(exchange, ["owner", "other"]) == before


def test_existing_margin_deficit_can_close_without_new_pending_risk():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "short", cash="100")
    register(exchange, "long")
    register(exchange, "counterparty")
    submit(exchange, "short", "sell", 10000, 1000, "short")
    submit(exchange, "long", "buy", 10000, 1000, "long", "IOC")
    exchange.set_mark_price(30000)
    assert exchange.account_state("short").equity == D(-100)
    submit(exchange, "counterparty", "sell", 30000, 1000, "close-offer")
    _, trades = submit(exchange, "short", "buy", 30000, 1000, "close", "IOC")
    assert trades[0].taker_fill.opened_quantity == 0
    assert exchange.account_state("short").position == 0
    assert exchange.account_state("short").cash == D(-100)
    assert exchange.risk_snapshot("short").sufficient is False
    before = snapshot(exchange, ["short", "long", "counterparty"])
    with pytest.raises(ValueError):
        submit(exchange, "short", "buy", 29000, 1, "new-risk")
    assert snapshot(exchange, ["short", "long", "counterparty"]) == before


def test_rebate_is_paid_only_from_finite_explicit_fee_funds():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "maker", maker="-0.01")
    register(exchange, "taker", taker="0")
    submit(exchange, "maker", "sell", 10000, 1000, "offer")
    before = snapshot(exchange, ["maker", "taker"])
    with pytest.raises(ValueError, match="finite fee balance"):
        submit(exchange, "taker", "buy", 10000, 1000, "take", "IOC")
    assert snapshot(exchange, ["maker", "taker"]) == before
    funded = ExperimentalPerpetualExchange(10000, initial_fee_balance="1")
    register(funded, "maker", maker="-0.01")
    register(funded, "taker")
    submit(funded, "maker", "sell", 10000, 1000, "offer")
    submit(funded, "taker", "buy", 10000, 1000, "take", "IOC")
    assert funded.fee_balance == 0
    assert funded.account_state("maker").cash == D(1001)
    assert total_equity_and_fees(funded, ["maker", "taker"]) == D(2001)


@pytest.mark.parametrize("value", [True, 1.0, D(1), 0, -1, 2**63])
def test_price_and_quantity_require_integer_storage_precision_and_do_not_consume_ids(value):
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "a")
    before = snapshot(exchange, ["a"])
    for price, quantity in [(value, 1), (10000, value)]:
        with pytest.raises(ValueError, match="storage-unit"):
            submit(exchange, "a", "buy", price, quantity, "bad")
        assert snapshot(exchange, ["a"]) == before


def test_duplicate_ids_and_invalid_partial_cancel_leave_state_unchanged():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "a")
    register(exchange, "b")
    order, _ = submit(exchange, "a", "buy", 9900, 1000, "client")
    before = snapshot(exchange, ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        register(exchange, "a")
    with pytest.raises(ValueError, match="unique"):
        submit(exchange, "b", "buy", 9900, 1, "client")
    for agent, quantity in [("b", 1), ("a", 1001), ("a", True)]:
        with pytest.raises(ValueError):
            exchange.cancel_order(order.order_id, agent, quantity_units=quantity)
    assert snapshot(exchange, ["a", "b"]) == before
    with pytest.raises(FrozenInstanceError):
        order.remaining_quantity_units = 0


def test_previews_do_not_copy_accumulated_execution_history():
    exchange = ExperimentalPerpetualExchange(10000)
    register(exchange, "a")
    register(exchange, "b")
    submit(exchange, "a", "sell", 10000, 1000, "open")
    submit(exchange, "b", "buy", 10000, 1000, "take", "IOC")

    class HistoryWithoutCopy(dict):
        def __deepcopy__(self, memo):
            pytest.fail("Copied accumulated execution history")

        def copy(self):
            pytest.fail("Copied accumulated execution history")

        def __iter__(self):
            pytest.fail("Iterated accumulated execution history during preview")

        def items(self):
            pytest.fail("Iterated accumulated execution history during preview")

    for account in exchange._accounts.values():
        account._events = HistoryWithoutCopy(account._events)
    submit(exchange, "a", "buy", 10000, 1000, "close-bid")
    submit(exchange, "b", "sell", 10000, 1000, "close-sell", "IOC")
    assert exchange.account_state("a").position == exchange.account_state("b").position == 0
    assert len(exchange.account_receipts("a")) == len(exchange.account_receipts("b")) == 2
