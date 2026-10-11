"""Single-instrument USDT linear-perpetual accounting for execution experiments.

This is an experimental collateral constraint, not a historical venue margin model.
There is no liquidation, ADL, borrowing, collateral conversion, or order reservation.
Fees, minimum opening notional and maximum leverage must be supplied explicitly.
Arithmetic uses Decimal without float conversion or currency-unit quantization.
Division follows the active Decimal context; the remaining entry cost is retained
so a complete close releases all cost, including any allocation-rounding remainder.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

Number = Decimal | int | str
ZERO = Decimal(0)


def decimal_value(value: Number, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise TypeError(f"{name} must be Decimal, int or str; floats are not accepted")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return result


def positive(value: Number, name: str) -> Decimal:
    result = decimal_value(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class PerpetualState:
    """Cash excludes unrealized PnL; entry_cost is positive on either side."""

    cash: Decimal
    position: Decimal
    entry_cost: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal
    funding_paid: Decimal
    mark_price: Decimal

    @property
    def average_entry(self) -> Decimal:
        return self.entry_cost / abs(self.position) if self.position else ZERO

    @property
    def unrealized_pnl(self) -> Decimal:
        if not self.position:
            return ZERO
        sign = Decimal(1) if self.position > 0 else Decimal(-1)
        return self.position * self.mark_price - sign * self.entry_cost

    @property
    def equity(self) -> Decimal:
        return self.cash + self.unrealized_pnl

    @property
    def gross_notional(self) -> Decimal:
        return abs(self.position) * self.mark_price


@dataclass(frozen=True)
class FillReceipt:
    event_id: str
    signed_quantity: Decimal
    price: Decimal
    mark_price: Decimal
    liquidity: Literal["maker", "taker"]
    opened_quantity: Decimal
    closed_quantity: Decimal
    realized_pnl: Decimal
    fee_paid: Decimal
    position_after: Decimal


@dataclass(frozen=True)
class FundingReceipt:
    event_id: str
    settled_at: datetime
    rate: Decimal
    mark_price: Decimal
    position: Decimal
    funding_paid: Decimal


class LinearPerpetualAccount:
    """A single USDT wallet and signed BTC contract position, initially flat.

    Opening does not spend the contract's notional. Cash changes only through
    realized PnL, fees and funding. Margin is an equity requirement, not a second
    cash debit. A negative fee is an explicitly configured maker/taker rebate.
    State and receipts are immutable; failed operations have no account effects.
    Event IDs share one namespace and duplicates are rejected, not charged twice.
    The caller first validates the full order with validate_order_notional, then
    calls apply_fill for each partial execution with its own unique fill ID. This
    account does not track orders, remaining quantities or reserve order margin.
    """

    def __init__(
        self,
        initial_cash: Number,
        mark_price: Number,
        *,
        maker_fee_rate: Number,
        taker_fee_rate: Number,
        min_open_notional: Number,
        max_leverage: Number,
    ) -> None:
        cash = decimal_value(initial_cash, "initial_cash")
        minimum = decimal_value(min_open_notional, "min_open_notional")
        if cash < 0 or minimum < 0:
            raise ValueError("initial_cash and min_open_notional must be nonnegative")
        self.initial_cash = cash
        self.min_open_notional = minimum
        self.max_leverage = positive(max_leverage, "max_leverage")
        self.maker_fee_rate = decimal_value(maker_fee_rate, "maker_fee_rate")
        self.taker_fee_rate = decimal_value(taker_fee_rate, "taker_fee_rate")
        if abs(self.maker_fee_rate) >= 1 or abs(self.taker_fee_rate) >= 1:
            raise ValueError("Experimental fee-rate magnitude must be below one")
        self._state = PerpetualState(
            cash, ZERO, ZERO, ZERO, ZERO, ZERO, positive(mark_price, "mark_price")
        )
        self._events: dict[str, FillReceipt | FundingReceipt] = {}

    @property
    def state(self) -> PerpetualState:
        return self._state

    @property
    def receipts(self) -> tuple[FillReceipt | FundingReceipt, ...]:
        return tuple(self._events.values())

    @property
    def required_margin(self) -> Decimal:
        """Experimental initial-risk budget; this is not maintenance margin."""
        return self.state.gross_notional / self.max_leverage

    @property
    def free_collateral(self) -> Decimal:
        return self.state.equity - self.required_margin

    @property
    def margin_sufficient(self) -> bool:
        # Use the same multiplicative cap as admission; division may round at the boundary.
        return (
            self.state.equity >= 0
            and self.state.gross_notional <= self.state.equity * self.max_leverage
        )

    def _check_event_id(self, event_id: str) -> None:
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a nonempty string")
        if event_id in self._events:
            raise ValueError(f"Duplicate account event: {event_id}")

    def mark_to_market(self, mark_price: Number) -> PerpetualState:
        """Revalue only; adverse marks may report negative equity without liquidation."""
        updated = replace(self.state, mark_price=positive(mark_price, "mark_price"))
        self._state = updated
        return updated

    def validate_order_notional(self, signed_quantity: Number, price: Number) -> None:
        """Check a complete order's minimum notional before submitting it.

        Pure position reductions at the current account state are exempt. If any
        portion opens exposure, including a flip, check the entire order notional,
        not each eventual fill or just the opening portion. The caller supplies
        the order-admission valuation price; venue-specific price filters remain
        outside this account. This check neither reserves funds nor authorizes a
        later risk-increasing fill: apply_fill still checks its experimental cap.
        """
        quantity = decimal_value(signed_quantity, "signed_quantity")
        order_price = positive(price, "price")
        if quantity == 0:
            raise ValueError("signed_quantity must be nonzero")
        pure_reduction = quantity * self.state.position < 0 and abs(quantity) <= abs(
            self.state.position
        )
        if not pure_reduction and abs(quantity) * order_price < self.min_open_notional:
            raise ValueError("Order is below minimum opening notional")

    def apply_fill(
        self,
        fill_id: str,
        signed_quantity: Number,
        price: Number,
        *,
        mark_price: Number,
        liquidity: Literal["maker", "taker"],
    ) -> FillReceipt:
        """Book a partial hypothetical execution of a previously admitted order.

        Positive quantity buys; negative sells. Pure reductions may execute even
        with insufficient collateral. Any newly opened portion, including a flip,
        must satisfy the final gross-exposure cap at the supplied mark, after all
        fees and realized PnL. Minimum notional is checked on the full order by
        validate_order_notional, never on these potentially tiny partial fills.
        The caller owns admission and order tracking. This risk check describes
        hypothetical execution feasibility; it cannot reject an already executed
        live trade. An executor must check before committing an external fill.
        """
        self._check_event_id(fill_id)
        quantity = decimal_value(signed_quantity, "signed_quantity")
        fill_price = positive(price, "price")
        mark = positive(mark_price, "mark_price")
        if quantity == 0:
            raise ValueError("signed_quantity must be nonzero")
        if liquidity not in ("maker", "taker"):
            raise ValueError("liquidity must be maker or taker")
        prior = self.state
        closing = min(abs(quantity), abs(prior.position)) if quantity * prior.position < 0 else ZERO
        opening = abs(quantity) - closing
        # A full close consumes the residual exactly, avoiding rounded average-cost dust.
        released_cost = (
            prior.entry_cost
            if closing and closing == abs(prior.position)
            else prior.entry_cost * closing / abs(prior.position)
            if closing
            else ZERO
        )
        prior_sign = Decimal(1) if prior.position > 0 else Decimal(-1)
        realized = prior_sign * (closing * fill_price - released_cost)
        entry_cost = prior.entry_cost - released_cost + opening * fill_price
        position = prior.position + quantity
        fee_rate = self.maker_fee_rate if liquidity == "maker" else self.taker_fee_rate
        fee = abs(quantity) * fill_price * fee_rate
        updated = PerpetualState(
            cash=prior.cash + realized - fee,
            position=position,
            entry_cost=entry_cost,
            realized_pnl=prior.realized_pnl + realized,
            fees_paid=prior.fees_paid + fee,
            funding_paid=prior.funding_paid,
            mark_price=mark,
        )
        if opening:
            if updated.equity <= 0 or updated.gross_notional > updated.equity * self.max_leverage:
                raise ValueError("Insufficient equity for experimental leverage cap")
        receipt = FillReceipt(
            fill_id,
            quantity,
            fill_price,
            mark,
            liquidity,
            opening,
            closing,
            realized,
            fee,
            position,
        )
        self._state = updated
        self._events[fill_id] = receipt
        return receipt

    def settle_funding(
        self,
        event_id: str,
        rate: Number,
        mark_price: Number,
        *,
        settled_at: datetime,
    ) -> FundingReceipt:
        """Apply one explicitly supplied settlement, never a predicted future rate.

        The caller owns the event clock and must call only at the actual funding
        event. No interval, future settlement or rate forecast is inferred here.
        Positive funding_paid is a cash debit; a negative amount is a credit.
        Funding may exhaust collateral; it is still booked without fabricated ADL.
        """
        self._check_event_id(event_id)
        if not isinstance(settled_at, datetime) or settled_at.utcoffset() is None:
            raise ValueError("settled_at must be an explicit timezone-aware datetime")
        funding_rate = decimal_value(rate, "rate")
        mark = positive(mark_price, "mark_price")
        prior = self.state
        paid = prior.position * mark * funding_rate
        updated = replace(
            prior,
            cash=prior.cash - paid,
            funding_paid=prior.funding_paid + paid,
            mark_price=mark,
        )
        receipt = FundingReceipt(
            event_id, settled_at.astimezone(UTC), funding_rate, mark, prior.position, paid
        )
        self._state = updated
        self._events[event_id] = receipt
        return receipt
