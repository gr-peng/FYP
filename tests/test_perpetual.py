from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from behavioral_market.market.perpetual import LinearPerpetualAccount

D = Decimal
SETTLEMENT = datetime(2024, 11, 1, 8, tzinfo=UTC)


def account(**overrides):
    parameters = {
        "initial_cash": "1000",
        "mark_price": "100",
        "maker_fee_rate": "0",
        "taker_fee_rate": "0",
        "min_open_notional": "0",
        "max_leverage": "2",
    }
    return LinearPerpetualAccount(**(parameters | overrides))


def fill(wallet, event_id, quantity, price, mark=None, liquidity="taker"):
    return wallet.apply_fill(
        event_id, quantity, price, mark_price=price if mark is None else mark, liquidity=liquidity
    )


def assert_cash_identity(wallet):
    state = wallet.state
    assert state.cash == (
        wallet.initial_cash + state.realized_pnl - state.fees_paid - state.funding_paid
    )
    assert state.equity == state.cash + state.unrealized_pnl


def test_open_increase_reduce_close_are_self_financing_without_notional_cash_transfer():
    wallet = account()
    fill(wallet, "open", "2", "100")
    assert wallet.state.cash == D("1000")
    assert wallet.state.position == D("2")
    assert wallet.required_margin == D("100")
    assert wallet.free_collateral == D("900")
    fill(wallet, "increase", "1", "130")
    assert wallet.state.average_entry == D("110")
    assert wallet.state.unrealized_pnl == D("60")
    assert wallet.state.equity == D("1060")
    reduction = fill(wallet, "reduce", "-1", "140")
    assert reduction.realized_pnl == D("30")
    assert wallet.state.cash == D("1030")
    assert wallet.state.entry_cost == D("220")
    assert wallet.state.unrealized_pnl == D("60")
    assert wallet.state.equity == D("1090")
    fill(wallet, "close", "-2", "90")
    assert wallet.state.position == wallet.state.entry_cost == wallet.state.average_entry == 0
    assert wallet.state.unrealized_pnl == 0
    assert wallet.state.realized_pnl == D("-10")
    assert wallet.state.cash == wallet.state.equity == D("990")
    assert_cash_identity(wallet)


def test_short_position_and_flip_realize_only_closed_quantity():
    wallet = account(taker_fee_rate="0.001")
    fill(wallet, "short", "-2", "100")
    assert wallet.state.cash == D("999.8")
    assert wallet.mark_to_market("80").unrealized_pnl == D("40")
    flip = fill(wallet, "flip", "3", "80")
    assert flip.closed_quantity == D("2")
    assert flip.opened_quantity == D("1")
    assert flip.realized_pnl == D("40")
    assert flip.fee_paid == D("0.24")
    assert wallet.state.position == D("1")
    assert wallet.state.average_entry == D("80")
    assert wallet.state.unrealized_pnl == 0
    assert wallet.state.cash == D("1039.56")
    assert_cash_identity(wallet)


def test_long_flip_to_short_resets_basis_at_flip_price():
    wallet = account()
    fill(wallet, "long", "2", "100")
    flip = fill(wallet, "flip", "-3", "120")
    assert flip.realized_pnl == D("40")
    assert wallet.state.position == D("-1")
    assert wallet.state.entry_cost == wallet.state.average_entry == D("120")
    assert wallet.mark_to_market("110").unrealized_pnl == D("10")
    assert wallet.state.equity == D("1050")
    assert_cash_identity(wallet)


@pytest.mark.parametrize(
    "position,rate,payment",
    [("2", "0.01", "2.2"), ("-2", "0.01", "-2.2"), ("2", "-0.01", "-2.2"), ("-2", "-0.01", "2.2")],
)
def test_funding_uses_signed_position_and_settlement_mark(position, rate, payment):
    wallet = account()
    fill(wallet, "open", position, "100")
    receipt = wallet.settle_funding("funding", rate, "110", settled_at=SETTLEMENT)
    assert receipt.funding_paid == D(payment)
    assert wallet.state.cash == D("1000") - D(payment)
    assert wallet.state.funding_paid == D(payment)
    assert wallet.state.realized_pnl == wallet.state.fees_paid == 0
    assert receipt.settled_at == SETTLEMENT
    assert_cash_identity(wallet)


def test_same_price_roundtrip_loses_only_exact_fees():
    wallet = account(maker_fee_rate="0.0002", taker_fee_rate="0.00055")
    fill(wallet, "buy", "0.123", "1234.56789", liquidity="maker")
    fill(wallet, "sell", "-0.123", "1234.56789", liquidity="taker")
    expected = D("0.123") * D("1234.56789") * D("0.00075")
    assert wallet.state.fees_paid == expected
    assert wallet.state.realized_pnl == wallet.state.unrealized_pnl == 0
    assert wallet.state.cash == D("1000") - expected
    assert_cash_identity(wallet)


def test_marking_does_not_realize_or_double_count_pnl():
    wallet = account()
    fill(wallet, "buy", "1", "100")
    for _ in range(3):
        wallet.mark_to_market("150")
    assert wallet.state.cash == D("1000")
    assert wallet.state.realized_pnl == 0
    assert wallet.state.equity == D("1050")
    fill(wallet, "sell", "-1", "150")
    assert wallet.state.equity == wallet.state.cash == D("1050")
    assert wallet.state.realized_pnl == D("50")
    assert_cash_identity(wallet)


def test_duplicate_ids_cannot_double_charge_or_change_position():
    wallet = account(taker_fee_rate="0.001")
    fill(wallet, "first", "1", "100")
    state, receipts = wallet.state, wallet.receipts
    with pytest.raises(ValueError, match="Duplicate"):
        fill(wallet, "first", "1", "100")
    with pytest.raises(ValueError, match="Duplicate"):
        wallet.settle_funding("first", "0.01", "100", settled_at=SETTLEMENT)
    assert (wallet.state, wallet.receipts) == (state, receipts)
    wallet.settle_funding("funding", "0.01", "100", settled_at=SETTLEMENT)
    state, receipts = wallet.state, wallet.receipts
    with pytest.raises(ValueError, match="Duplicate"):
        wallet.settle_funding("funding", "0.02", "120", settled_at=SETTLEMENT)
    assert (wallet.state, wallet.receipts) == (state, receipts)


def test_leverage_rejection_uses_post_fee_mark_equity_and_is_atomic():
    wallet = account(initial_cash="100", max_leverage="1", taker_fee_rate="0.01")
    state = wallet.state
    with pytest.raises(ValueError, match="leverage cap"):
        fill(wallet, "order", "1", "100", mark="110")
    assert wallet.state == state
    assert wallet.receipts == ()
    # The failed ID was never consumed; a smaller corrected proposal may use it.
    fill(wallet, "order", "0.5", "100", mark="110")
    assert wallet.state.cash == D("99.5")
    assert wallet.state.equity == D("104.5")
    assert_cash_identity(wallet)


def test_flip_cannot_bypass_margin_check_even_when_absolute_position_shrinks():
    wallet = account(initial_cash="100", max_leverage="1")
    fill(wallet, "long", "1", "100")
    wallet.mark_to_market("1")
    state, receipts = wallet.state, wallet.receipts
    with pytest.raises(ValueError, match="leverage cap"):
        fill(wallet, "flip", "-1.9", "1", mark="100")
    assert (wallet.state, wallet.receipts) == (state, receipts)


def test_admitted_order_can_execute_as_several_subminimum_fills():
    wallet = account(min_open_notional="100", taker_fee_rate="0.001")
    wallet.validate_order_notional("1.2", "100")
    for index, quantity in enumerate(("0.2", "0.3", "0.7")):
        fill(wallet, f"partial-{index}", quantity, "100")
    assert wallet.state.position == D("1.2")
    assert wallet.state.entry_cost == D("120")
    assert wallet.state.average_entry == D("100")
    assert wallet.state.cash == D("999.88")
    assert wallet.state.fees_paid == D("0.12")
    assert len(wallet.receipts) == 3
    assert_cash_identity(wallet)


def test_subminimum_opening_order_is_rejected_at_admission_without_effects():
    wallet = account(min_open_notional="100")
    state, receipts = wallet.state, wallet.receipts
    with pytest.raises(ValueError, match="minimum opening notional"):
        wallet.validate_order_notional("0.5", "100")
    assert (wallet.state, wallet.receipts) == (state, receipts)


def test_pure_closing_order_is_exempt_and_flip_checks_complete_order_notional():
    wallet = account(min_open_notional="100")
    wallet.validate_order_notional("1", "100")
    fill(wallet, "open", "1", "100")
    wallet.validate_order_notional("-0.5", "100")  # Pure close below the opening minimum.
    fill(wallet, "reduce", "-0.5", "100")
    # Full flip order is 100 USDT although its new short portion is only 50 USDT.
    wallet.validate_order_notional("-1", "100")
    fill(wallet, "flip", "-1", "100")
    assert wallet.state.position == D("-0.5")
    wallet.validate_order_notional("0.5", "100")
    fill(wallet, "close", "0.5", "100")
    assert wallet.state.position == 0


def test_one_times_short_can_have_negative_equity_and_still_reduce():
    wallet = account(initial_cash="100", max_leverage="1")
    fill(wallet, "short", "-1", "100")
    wallet.mark_to_market("250")
    assert wallet.state.equity == D("-50")
    assert not wallet.margin_sufficient
    assert wallet.state.position == D("-1")  # No invented automatic liquidation.
    fill(wallet, "reduce", "0.5", "250")
    assert wallet.state.cash == D("25")
    assert wallet.state.equity == D("-50")
    fill(wallet, "close", "0.5", "250")
    assert wallet.state.cash == wallet.state.equity == D("-50")
    assert wallet.state.position == 0
    assert not wallet.margin_sufficient
    assert_cash_identity(wallet)


def test_funding_can_exhaust_collateral_and_flat_funding_has_no_cash_effect():
    wallet = account(initial_cash="100", max_leverage="1")
    fill(wallet, "long", "1", "100")
    wallet.settle_funding("stress", "2", "100", settled_at=SETTLEMENT)
    assert wallet.state.cash == D("-100")
    assert not wallet.margin_sufficient
    fill(wallet, "close", "-1", "100")
    state = wallet.state
    zero = wallet.settle_funding("flat", "2", "100", settled_at=SETTLEMENT)
    assert zero.funding_paid == 0
    assert wallet.state == state
    assert_cash_identity(wallet)


def test_full_close_releases_average_entry_division_remainder():
    wallet = account()
    fill(wallet, "first", "1", "100")
    fill(wallet, "second", "2", "101")
    assert wallet.state.entry_cost == D("302")
    fill(wallet, "partial", "-1", "100")
    fill(wallet, "rest", "-2", "100")
    assert wallet.state.entry_cost == 0
    assert wallet.state.cash == D("998")
    assert wallet.state.realized_pnl == D("-2")


def test_explicit_maker_rebate_is_cash_credit():
    wallet = account(maker_fee_rate="-0.0001")
    fill(wallet, "maker", "1", "100", liquidity="maker")
    assert wallet.state.fees_paid == D("-0.01")
    assert wallet.state.cash == D("1000.01")
    assert_cash_identity(wallet)


@pytest.mark.parametrize("bad", [0.1, True, "NaN", "Infinity", "bad"])
def test_invalid_numeric_input_has_no_account_effect(bad):
    wallet = account()
    state = wallet.state
    with pytest.raises((TypeError, ValueError)):
        fill(wallet, "invalid", bad, "100")
    assert wallet.state == state
    assert wallet.receipts == ()


def test_funding_requires_aware_settlement_time_and_invalid_mark_is_atomic():
    wallet = account()
    fill(wallet, "open", "1", "100")
    state, receipts = wallet.state, wallet.receipts
    with pytest.raises(ValueError, match="timezone-aware"):
        wallet.settle_funding("funding", "0.01", "100", settled_at=datetime(2024, 11, 1))
    with pytest.raises(ValueError, match="positive"):
        wallet.mark_to_market("0")
    assert (wallet.state, wallet.receipts) == (state, receipts)
    with pytest.raises(FrozenInstanceError):
        wallet.state.cash = D("999999")
