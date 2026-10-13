"""Experimental, finite-collateral matching for one USDT linear perpetual.

Integer price/quantity storage scales are 100/1000, not historical tick/lot rules.
DAY orders reserve gross possible exposure at the caller's current mark, adverse
entry-price loss and positive maker fees. Opposing orders and reductions are not
netted in that reservation. Changes in the external mark can breach this budget;
there is no historical maintenance margin, liquidation, ADL or funding scheduler.

Submission previews only the live book and affected account states. It commits
neither fills nor counters until every partial fill and remaining reservation
passes. Accumulated receipt/trade histories are not copied for each submission.
Existing undercollateralized positions may be reduced by actual closing fills
without adding any resting risk, as in LinearPerpetualAccount. Rebate cash must
come from the explicitly funded, finite fee account.

Minimum notional is an admission check against the account's current position.
There is no reduce-only flag: another execution may change that position before
a formerly reducing DAY remainder fills, so that remainder may open exposure.
Admission does not guarantee future fills remain reductions or satisfy a venue's
minimum-opening policy. Marks are explicitly supplied integer storage units;
an own-book midpoint halfway between such units needs a future caller policy.
"""

from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .matching_engine import MatchingEngine
from .order import Order, OrderStatus, Side
from .order_book import OrderBook
from .perpetual import FillReceipt, LinearPerpetualAccount, Number, PerpetualState, decimal_value
from .trade import Trade

PRICE_SCALE = 100
QUANTITY_SCALE = 1000
MAX_STORAGE_UNITS = 2**63 - 1
ZERO = Decimal(0)


def _storage_units(value: int, name: str) -> int:
    if type(value) is not int or not 0 < value <= MAX_STORAGE_UNITS:
        raise ValueError(f"{name} must be a positive int64 storage-unit count")
    return value


def _price(units: int) -> Decimal:
    value = Decimal(units).as_tuple()
    return Decimal((value.sign, value.digits, -2))


def _quantity(units: int) -> Decimal:
    value = Decimal(units).as_tuple()
    return Decimal((value.sign, value.digits, -3))


def _signed_quantity(units: int, sign: int) -> Decimal:
    quantity = _quantity(units)
    return quantity if sign > 0 else quantity.copy_negate()


def _price_units(price: Decimal) -> int:
    numerator, denominator = price.as_integer_ratio()
    units, remainder = divmod(numerator * PRICE_SCALE, denominator)
    if remainder:
        raise RuntimeError("Internal limit price lost integer storage precision")
    return units


def _preview_account(original: LinearPerpetualAccount) -> LinearPerpetualAccount:
    """Small state-only bridge to the existing ledger; no historical event copy.

    The ledger currently has no public pure transition API. These two private
    fields are accessed only here and during the final commit. The immutable
    current state is shared until apply_fill replaces it; only new receipts enter
    the preview's empty event dictionary. Existing IDs are checked separately.
    """
    preview = LinearPerpetualAccount(
        original.initial_cash,
        original.state.mark_price,
        maker_fee_rate=original.maker_fee_rate,
        taker_fee_rate=original.taker_fee_rate,
        min_open_notional=original.min_open_notional,
        max_leverage=original.max_leverage,
    )
    preview._state = original.state
    return preview


@dataclass(frozen=True, slots=True)
class PerpetualOrderReceipt:
    order_id: str
    client_order_id: str
    agent_id: str
    side: Side
    price_units: int
    quantity_units: int
    filled_quantity_units: int
    remaining_quantity_units: int
    cancelled_quantity_units: int
    expired_quantity_units: int
    status: OrderStatus
    time_in_force: Literal["DAY", "IOC"]
    timestamp: int
    day: int


@dataclass(frozen=True, slots=True)
class PerpetualTradeReceipt:
    trade_id: str
    price_units: int
    quantity_units: int
    mark_price_units: int
    maker_id: str
    taker_id: str
    maker_order_id: str
    taker_order_id: str
    buyer_id: str
    seller_id: str
    maker_fill: FillReceipt
    taker_fill: FillReceipt
    timestamp: int
    day: int


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    equity: Decimal
    position_initial_margin: Decimal
    open_order_margin: Decimal
    adverse_entry_buffer: Decimal
    maker_fee_buffer: Decimal
    sufficient: bool

    @property
    def reserved_collateral(self) -> Decimal:
        return self.open_order_margin + self.adverse_entry_buffer + self.maker_fee_buffer

    @property
    def free_collateral(self) -> Decimal:
        return self.equity - self.position_initial_margin - self.reserved_collateral


@dataclass(slots=True)
class _OrderMetadata:
    client_order_id: str
    time_in_force: Literal["DAY", "IOC"]
    cancelled_units: int = 0
    expired_units: int = 0


def _risk(account: LinearPerpetualAccount, orders) -> RiskSnapshot:
    mark = account.state.mark_price
    open_notional = adverse = fees = ZERO
    for order in orders:
        quantity = _quantity(order.remaining_quantity)
        open_notional += quantity * mark
        price_loss = order.price - mark if order.side == Side.BUY else mark - order.price
        adverse += quantity * max(price_loss, ZERO)
        fees += quantity * order.price * max(account.maker_fee_rate, ZERO)
    position_notional = account.state.gross_notional
    available = account.state.equity - adverse - fees
    # A multiplicative comparison avoids division rounding at the leverage boundary.
    sufficient = available >= 0 and (
        position_notional + open_notional <= available * account.max_leverage
    )
    return RiskSnapshot(
        account.state.equity,
        position_notional / account.max_leverage,
        open_notional / account.max_leverage,
        adverse,
        fees,
        sufficient,
    )


class ExperimentalPerpetualExchange:
    """Atomic hypothetical fills; caller supplies funds, fees, leverage and mark.

    Incoming IOC orders check the actual visible fill plan, not a reservation for
    their subsequently cancelled remainder. DAY orders additionally reserve all
    remaining quantity. Full-order minimum notional is checked before either.
    The reduction exemption uses the submission-time position only; resting
    orders are not reduce-only and the exemption is not reassessed per fill.
    This is an experimental admission policy, not venue order-admission behavior.
    """

    def __init__(
        self,
        mark_price_units: int,
        *,
        symbol: str = "BTCUSDT",
        initial_fee_balance: Number = 0,
    ) -> None:
        self._mark_units = _storage_units(mark_price_units, "mark_price_units")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be nonempty")
        balance = decimal_value(initial_fee_balance, "initial_fee_balance")
        if balance < 0:
            raise ValueError("initial_fee_balance must be nonnegative")
        self.symbol = symbol
        self.initial_fee_balance = balance
        self._fee_balance = balance
        self._accounts: dict[str, LinearPerpetualAccount] = {}
        self._book = OrderBook()
        self._orders: dict[str, Order] = {}
        self._metadata: dict[str, _OrderMetadata] = {}
        self._client_ids: dict[str, str] = {}
        self._trades: list[PerpetualTradeReceipt] = []
        self._order_count = self._trade_count = self._clock = self._day = 0

    @property
    def mark_price_units(self) -> int:
        return self._mark_units

    @property
    def fee_balance(self) -> Decimal:
        return self._fee_balance

    @property
    def day(self) -> int:
        return self._day

    @property
    def sequences(self) -> tuple[int, int, int]:
        """Committed order, trade and event sequences; failed requests consume none."""
        return self._order_count, self._trade_count, self._clock

    @property
    def trades(self) -> tuple[PerpetualTradeReceipt, ...]:
        return tuple(self._trades)

    def _account(self, agent_id: str) -> LinearPerpetualAccount:
        if agent_id not in self._accounts:
            raise ValueError("Unknown account")
        return self._accounts[agent_id]

    def register(
        self,
        agent_id: str,
        initial_cash: Number,
        *,
        maker_fee_rate: Number,
        taker_fee_rate: Number,
        min_open_notional: Number,
        max_leverage: Number,
    ) -> PerpetualState:
        if not isinstance(agent_id, str) or not agent_id.strip() or agent_id in self._accounts:
            raise ValueError("Account ID must be unique and nonempty")
        account = LinearPerpetualAccount(
            initial_cash,
            _price(self._mark_units),
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
            min_open_notional=min_open_notional,
            max_leverage=max_leverage,
        )
        self._accounts[agent_id] = account
        return account.state

    def account_state(self, agent_id: str) -> PerpetualState:
        return self._account(agent_id).state

    def account_receipts(self, agent_id: str) -> tuple[FillReceipt, ...]:
        return self._account(agent_id).receipts

    def risk_snapshot(self, agent_id: str) -> RiskSnapshot:
        account = self._account(agent_id)
        orders = (o for o in self._book.resting_orders() if o.agent_id == agent_id)
        return _risk(account, orders)

    def set_mark_price(self, mark_price_units: int) -> None:
        """Caller-supplied revaluation may breach margin; no invented liquidation."""
        mark_price_units = _storage_units(mark_price_units, "mark_price_units")
        states = {}
        for agent_id, original in self._accounts.items():
            preview = _preview_account(original)
            states[agent_id] = preview.mark_to_market(_price(mark_price_units))
        for agent_id, state in states.items():
            self._accounts[agent_id]._state = state
        self._mark_units = mark_price_units
        self._clock += 1

    def _receipt(self, order: Order) -> PerpetualOrderReceipt:
        metadata = self._metadata[order.order_id]
        filled = (
            order.quantity
            - order.remaining_quantity
            - metadata.cancelled_units
            - metadata.expired_units
        )
        status = order.status
        if order.remaining_quantity == 0:
            if metadata.expired_units:
                status = OrderStatus.EXPIRED
            elif metadata.cancelled_units:
                status = OrderStatus.CANCELLED
        return PerpetualOrderReceipt(
            order.order_id,
            metadata.client_order_id,
            order.agent_id,
            order.side,
            _price_units(order.price),
            order.quantity,
            filled,
            order.remaining_quantity,
            metadata.cancelled_units,
            metadata.expired_units,
            status,
            metadata.time_in_force,
            order.timestamp,
            order.day,
        )

    def order(self, order_id: str) -> PerpetualOrderReceipt:
        if order_id not in self._orders:
            raise ValueError("Unknown order")
        return self._receipt(self._orders[order_id])

    def order_for_client(self, client_order_id: str) -> PerpetualOrderReceipt:
        if client_order_id not in self._client_ids:
            raise ValueError("Unknown client order ID")
        return self.order(self._client_ids[client_order_id])

    def resting_orders(self) -> tuple[PerpetualOrderReceipt, ...]:
        return tuple(self._receipt(o) for o in self._book.resting_orders())

    def _copy_live_book(self):
        book = OrderBook()
        # Both live sides already have exact price/FIFO order. Preserve it while
        # copying; repeatedly calling add would re-sort every growing prefix.
        book.bids = [copy(order) for order in self._book.bids]
        book.asks = [copy(order) for order in self._book.asks]
        orders = {order.order_id: order for order in book.resting_orders()}
        return book, orders

    def submit_order(
        self,
        agent_id: str,
        side: Side | str,
        price_units: int,
        quantity_units: int,
        *,
        client_order_id: str,
        time_in_force: Literal["DAY", "IOC"] = "DAY",
    ) -> tuple[PerpetualOrderReceipt, tuple[PerpetualTradeReceipt, ...]]:
        original_account = self._account(agent_id)
        side = Side(side)
        price_units = _storage_units(price_units, "price_units")
        quantity_units = _storage_units(quantity_units, "quantity_units")
        if time_in_force not in ("DAY", "IOC"):
            raise ValueError("Only DAY and IOC are supported")
        if (
            not isinstance(client_order_id, str)
            or not client_order_id.strip()
            or client_order_id in self._client_ids
        ):
            raise ValueError("Client order ID must be unique and nonempty")
        sign = 1 if side == Side.BUY else -1
        original_account.validate_order_notional(
            _signed_quantity(quantity_units, sign), _price(price_units)
        )
        book, live_orders = self._copy_live_book()
        clock, trade_count = self._clock + 1, self._trade_count
        incoming = Order(
            f"PXO{self._order_count + 1:08d}",
            agent_id,
            side,
            _price(price_units),
            quantity_units,
            clock,
            self._day,
        )
        live_orders[incoming.order_id] = incoming
        previews = {agent_id: _preview_account(original_account)}
        trades = []
        fee_balance = self._fee_balance

        def account_preview(account_id):
            if account_id not in previews:
                previews[account_id] = _preview_account(self._accounts[account_id])
            return previews[account_id]

        def execute(active: Order, resting: Order, quantity: int) -> Trade:
            nonlocal clock, trade_count, fee_balance
            trade_id = f"PXT{trade_count + 1:08d}"
            mark = _price(self._mark_units)
            active_sign = 1 if active.side == Side.BUY else -1
            receipts = []
            for order, role, direction in (
                (resting, "maker", -active_sign),
                (active, "taker", active_sign),
            ):
                fill_id = f"{trade_id}:{role}"
                if fill_id in self._accounts[order.agent_id]._events:
                    raise ValueError("Duplicate committed fill ID")
                receipt = account_preview(order.agent_id).apply_fill(
                    fill_id,
                    _signed_quantity(quantity, direction),
                    resting.price,
                    mark_price=mark,
                    liquidity=role,
                )
                receipts.append(receipt)
            new_fee_balance = fee_balance + sum((r.fee_paid for r in receipts), ZERO)
            if new_fee_balance < 0:
                raise ValueError("Insufficient finite fee balance for rebate")
            fee_balance = new_fee_balance
            maker, taker = receipts
            buyer = active if active.side == Side.BUY else resting
            seller = resting if active.side == Side.BUY else active
            clock += 1
            trade_count += 1
            trades.append(
                PerpetualTradeReceipt(
                    trade_id,
                    _price_units(resting.price),
                    quantity,
                    self._mark_units,
                    resting.agent_id,
                    active.agent_id,
                    resting.order_id,
                    active.order_id,
                    buyer.agent_id,
                    seller.agent_id,
                    maker,
                    taker,
                    clock,
                    self._day,
                )
            )
            return Trade(
                trade_id,
                buyer.agent_id,
                seller.agent_id,
                resting.price,
                quantity,
                buyer.order_id,
                seller.order_id,
                clock,
                self._day,
            )

        MatchingEngine(book).submit(incoming, execute)
        metadata = _OrderMetadata(client_order_id, time_in_force)
        if time_in_force == "IOC" and incoming.remaining_quantity:
            book.remove(incoming)
            metadata.cancelled_units = incoming.remaining_quantity
            incoming.close(OrderStatus.CANCELLED)
            incoming.remaining_quantity = 0
        remaining = book.resting_orders()
        for account_id, preview in previews.items():
            owned = [o for o in remaining if o.agent_id == account_id]
            if _risk(preview, owned).sufficient:
                continue
            # Preserve the ledger's closing escape without granting new resting risk.
            prior = self._accounts[account_id]
            prior_orders = {
                o.order_id: o.remaining_quantity
                for o in self._book.resting_orders()
                if o.agent_id == account_id
            }
            only_closes = bool(preview._events) and all(
                receipt.opened_quantity == 0 for receipt in preview._events.values()
            )
            reduced = abs(preview.state.position) < abs(prior.state.position)
            no_new_pending_risk = all(
                o.order_id in prior_orders and o.remaining_quantity <= prior_orders[o.order_id]
                for o in owned
            )
            if not (only_closes and reduced and no_new_pending_risk):
                raise ValueError(
                    "Insufficient collateral for conservative pending-order reservation"
                )

        # Nothing before this boundary changes live books/accounts/IDs/history.
        for account_id, preview in previews.items():
            actual = self._accounts[account_id]
            actual._state = preview.state
            actual._events.update(preview._events)
        self._book = book
        self._orders.update(live_orders)
        self._metadata[incoming.order_id] = metadata
        self._client_ids[client_order_id] = incoming.order_id
        self._fee_balance = fee_balance
        self._trades.extend(trades)
        self._order_count += 1
        self._trade_count, self._clock = trade_count, clock
        return self._receipt(incoming), tuple(trades)

    def cancel_order(
        self, order_id: str, agent_id: str, *, quantity_units: int | None = None
    ) -> PerpetualOrderReceipt:
        self._account(agent_id)
        order = self._orders.get(order_id)
        if order is None or order.agent_id != agent_id or not order.remaining_quantity:
            raise ValueError("Open order not found for account")
        quantity = (
            order.remaining_quantity
            if quantity_units is None
            else _storage_units(quantity_units, "quantity_units")
        )
        if quantity > order.remaining_quantity:
            raise ValueError("Cannot cancel more than the remaining quantity")
        if quantity == order.remaining_quantity:
            self._book.remove(order)
            order.close(OrderStatus.CANCELLED)
        order.remaining_quantity -= quantity
        self._metadata[order_id].cancelled_units += quantity
        self._clock += 1
        return self._receipt(order)

    def end_day(self) -> tuple[PerpetualOrderReceipt, ...]:
        """Expire DAY remainders and release their reservations; caller owns UTC clock."""
        expired = []
        for order in self._book.resting_orders():
            self._book.remove(order)
            self._metadata[order.order_id].expired_units += order.remaining_quantity
            order.close(OrderStatus.EXPIRED)
            order.remaining_quantity = 0
            expired.append(self._receipt(order))
            self._clock += 1
        self._day += 1
        return tuple(expired)
