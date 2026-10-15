"""Pure five-minute effective-flow generation with an independent finite book.

There are no paths, files, labels, historical readers or future-price arguments.
The caller supplies a fitted training pool and an explicitly known initial book
and past context. Generation changes only synthetic orders and accounts. This
is an experimental conditional mechanism, not historical Bybit margin, queue,
funding, liquidation or end-to-end information-delay validation.
"""

from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from random import Random

from behavioral_market.market.order import Side
from behavioral_market.market.perpetual import PerpetualState
from behavioral_market.market.perpetual_exchange import ExperimentalPerpetualExchange, RiskSnapshot
from experiments.crypto_effective_flow import (
    DELAYS_MS,
    GRID_US,
    MAX_UNITS,
    StateFeatures,
    TemplatePool,
    TrainingRow,
    select_template,
)

HORIZON_US = 300_000_000
STEPS = HORIZON_US // GRID_US
PAST_FEATURE_US = 5_000_000
PRESSURE_US = 1_000_000
DECIMAL_PRECISION = 80
AGENTS = ("bid-maker", "ask-maker", "buy-taker", "sell-taker")
COUNTERS = (
    "template_draws",
    "fallbacks",
    "add_events",
    "add_requested_units",
    "added_units",
    "add_rejected_units",
    "remove_events",
    "remove_requested_units",
    "removed_units",
    "remove_unimplemented_units",
    "clipped_remove_events",
    "intent_count",
    "intent_requested_units",
    "intent_filled_units",
    "intent_rejected_units",
    "ioc_cancelled_units",
    "execution_fills",
    "output_prints",
    "executed_steps",
    "requested_steps",
    "undefined_points",
)


def _integer(value, name, *, positive=False):
    if type(value) is not int or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be an exact {'positive' if positive else 'nonnegative'} int")


def _units(value, name):
    _integer(value, name, positive=True)
    if value > MAX_UNITS:
        raise ValueError(f"{name} exceeds int64 storage")


def _levels(levels, name):
    if type(levels) is not tuple or not levels or len(levels) > 500:
        raise ValueError("Initial sides require immutable, nonempty, at-most-500-level tuples")
    prices = []
    for level in levels:
        if type(level) is not tuple or len(level) != 2:
            raise ValueError("Initial levels must be exact price/quantity pairs")
        price, quantity = level
        _units(price, "price_units")
        _units(quantity, "quantity_units")
        prices.append(price)
    if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=name == "bid"):
        raise ValueError("Initial prices must be unique and in best-to-worst order")


@dataclass(frozen=True, slots=True)
class InitialBook:
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]

    def __post_init__(self):
        _levels(self.bids, "bid")
        _levels(self.asks, "ask")
        if self.bids[0][0] >= self.asks[0][0]:
            raise ValueError("Initial book must be strictly uncrossed")


@dataclass(frozen=True, slots=True)
class KnownPrint:
    """Already known, strictly past print context; no price input is needed."""

    timestamp_us: int
    side: str
    quantity_units: int
    print_id: str

    def __post_init__(self):
        _integer(self.timestamp_us, "timestamp_us")
        _units(self.quantity_units, "quantity_units")
        if (
            self.side not in ("Buy", "Sell")
            or not isinstance(self.print_id, str)
            or not self.print_id
        ):
            raise ValueError("Past print needs a side and nonempty identity")


@dataclass(frozen=True, slots=True)
class FeatureContext:
    timestamp_us: int
    features: StateFeatures

    def __post_init__(self):
        _integer(self.timestamp_us, "timestamp_us")
        if type(self.features) is not StateFeatures:
            raise ValueError("Context features must be exact immutable StateFeatures")


@dataclass(frozen=True, slots=True)
class SyntheticPrint:
    timestamp_us: int
    side: str
    quantity_units: int
    price_units: int
    print_id: str
    intent_id: str
    fill_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PathPoint:
    timestamp_us: int
    bids: tuple[tuple[int, int], ...] | None
    asks: tuple[tuple[int, int], ...] | None
    mid_twice_units: int | None
    spread_bps: Fraction | None
    visible_bid_depth_5bps_units: int | None
    visible_ask_depth_5bps_units: int | None
    bid_covers_5bps: bool | None
    ask_covers_5bps: bool | None
    model_bid_levels: int | None
    model_ask_levels: int | None
    mark_price_units: int | None
    status: str

    @property
    def best_bid_units(self):
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask_units(self):
        return self.asks[0][0] if self.asks else None

    @property
    def is_valid(self):
        return self.status == "defined"


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    step_index: int
    kind: str
    side: str
    price_units: int | None
    requested_units: int
    implemented_units: int
    unimplemented_units: int
    accepted: bool
    reason: str | None
    client_order_id: str | None
    order_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionReceipt:
    step_index: int
    decision_us: int
    feature_us: int
    features: StateFeatures
    row_id: str
    row_index: int
    state_cell: int
    fallback: bool


@dataclass(frozen=True, slots=True)
class AccountEnd:
    agent_id: str
    state: PerpetualState
    risk: RiskSnapshot


@dataclass(frozen=True, slots=True)
class CapitalConfiguration:
    account_names: tuple[str, ...]
    cash_per_account: Decimal
    max_leverage: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    min_open_notional: Decimal
    initial_fee_balance: Decimal
    topups_allowed: bool
    mark_policy: str
    reservation_policy: str


@dataclass(frozen=True, slots=True)
class Conservation:
    initial_equity: Decimal
    final_equity_plus_fees: Decimal
    equity_error: Decimal
    signed_position_btc_sum: Decimal
    bid_quantity_balance_error_units: int
    ask_quantity_balance_error_units: int
    intent_quantity_balance_error_units: int
    print_quantity_balance_error_units: int


@dataclass(frozen=True, slots=True)
class SimulationReceipt:
    start_us: int
    end_us: int
    requested_steps: int
    method: str
    ordering: str
    delay_ms: int
    seed: int
    pool_row_identity_sha256: str
    initial_book: InitialBook
    known_past_prints: tuple[KnownPrint, ...]
    known_feature_context: tuple[FeatureContext, ...]
    initial_cash_per_account: Decimal
    decimal_precision: int
    capital_configuration: CapitalConfiguration
    initial_actions: tuple[ActionReceipt, ...]
    path: tuple[PathPoint, ...]
    prints: tuple[SyntheticPrint, ...]
    actions: tuple[ActionReceipt, ...]
    selections: tuple[SelectionReceipt, ...]
    counters: tuple[tuple[str, int], ...]
    rejections: tuple[tuple[str, str, int, int], ...]
    accounts: tuple[AccountEnd, ...]
    final_bids: tuple[tuple[int, int], ...]
    final_asks: tuple[tuple[int, int], ...]
    fee_balance: Decimal
    final_financial_mark_units: int
    financial_mark_stale: bool
    conservation: Conservation
    status: str
    failure_at_us: int | None
    failure_reason: str | None

    def count(self, name):
        return dict(self.counters)[name]


def _aggregate(exchange):
    quantities = {Side.BUY: {}, Side.SELL: {}}
    for order in exchange.resting_orders():
        side = quantities[order.side]
        side[order.price_units] = side.get(order.price_units, 0) + order.remaining_quantity_units
    return (
        tuple(sorted(quantities[Side.BUY].items(), reverse=True)),
        tuple(sorted(quantities[Side.SELL].items())),
    )


def _half_up_mark(bids, asks):
    return (bids[0][0] + asks[0][0] + 1) // 2


def _pressure(prints):
    buy = sum(p.quantity_units for p in prints if p.side == "Buy")
    sell = sum(p.quantity_units for p in prints if p.side == "Sell")
    return Fraction(buy - sell, buy + sell) if buy + sell else Fraction(0)


def _features(bids, asks, prints):
    if not bids or not asks:
        raise ValueError("Features are undefined after a missing-side step")
    bid, bq = bids[0]
    ask, aq = asks[0]
    return StateFeatures(
        Fraction(bq - aq, bq + aq), Fraction(20000 * (ask - bid), ask + bid), _pressure(prints)
    )


def initial_features(initial_book, known_past_prints=()):
    """Pure own-state calculation; caller validates the actual past window."""
    return _features(initial_book.bids, initial_book.asks, known_past_prints)


def _point(timestamp_us, bids, asks, *, undefined=False, status=None):
    if undefined:
        return PathPoint(timestamp_us, *([None] * 11), status or "undefined_after_failure")
    view_bid, view_ask = bids[:500], asks[:500]
    if not bids or not asks:
        return PathPoint(
            timestamp_us,
            view_bid,
            view_ask,
            None,
            None,
            None,
            None,
            None,
            None,
            len(bids),
            len(asks),
            None,
            status or "missing_side",
        )
    mid = bids[0][0] + asks[0][0]

    def depth(levels):
        return sum(q for p, q in levels if abs(2 * p - mid) * 10000 <= 5 * mid)

    def covers(levels):
        return abs(2 * levels[-1][0] - mid) * 10000 >= 5 * mid

    return PathPoint(
        timestamp_us,
        view_bid,
        view_ask,
        mid,
        Fraction(20000 * (asks[0][0] - bids[0][0]), mid),
        depth(view_bid),
        depth(view_ask),
        covers(view_bid),
        covers(view_ask),
        len(bids),
        len(asks),
        _half_up_mark(bids, asks),
        status or "defined",
    )


def _validate_inputs(
    pool,
    initial_book,
    past_prints,
    past_features,
    start_us,
    method,
    ordering,
    delay_ms,
    seed,
    steps,
):
    _integer(start_us, "start_us")
    _integer(steps, "steps", positive=True)
    if start_us < PAST_FEATURE_US or start_us % GRID_US or steps > STEPS:
        raise ValueError("Start and engineering horizon require the fixed 100ms grid and <=300s")
    if (
        method not in ("M0", "M1")
        or ordering not in ("S1", "S2")
        or type(delay_ms) is not int
        or delay_ms not in DELAYS_MS
    ):
        raise ValueError("Unknown fixed method, ordering or information delay")
    if type(seed) is not int:
        raise ValueError("Seed must be an explicit integer")
    if type(initial_book) is not InitialBook:
        raise ValueError("Pass an immutable InitialBook, not a historical observation/reader")
    if (
        type(pool) is not TemplatePool
        or type(pool.rows) is not tuple
        or not pool.rows
        or any(type(row) is not TrainingRow for row in pool.rows)
        or type(pool.spread_median) is not Fraction
        or pool.spread_median <= 0
        or type(pool.indexes) is not tuple
        or len(pool.indexes) != len(DELAYS_MS)
        or any(type(cells) is not tuple or len(cells) != 18 for cells in pool.indexes)
        or any(
            type(indexes) is not tuple
            or any(type(i) is not int or not 0 <= i < len(pool.rows) for i in indexes)
            for cells in pool.indexes
            for indexes in cells
        )
    ):
        raise ValueError("Pass a fitted immutable training TemplatePool")
    if type(past_prints) is not tuple or any(type(p) is not KnownPrint for p in past_prints):
        raise ValueError("Past prints must be an immutable KnownPrint tuple")
    if (
        any(not start_us - PRESSURE_US <= p.timestamp_us < start_us for p in past_prints)
        or any(
            a.timestamp_us > b.timestamp_us
            for a, b in zip(past_prints, past_prints[1:], strict=False)
        )
        or len({p.print_id for p in past_prints}) != len(past_prints)
    ):
        raise ValueError("Past prints must be unique, ordered, and strictly before the start")
    expected = tuple(range(start_us - PAST_FEATURE_US, start_us + 1, GRID_US))
    if (
        type(past_features) is not tuple
        or any(type(p) is not FeatureContext for p in past_features)
        or tuple(p.timestamp_us for p in past_features) != expected
    ):
        raise ValueError("Known feature context must supply the complete [-5s,0] grid")
    if past_features[-1].features != initial_features(initial_book, past_prints):
        raise ValueError("Start context disagrees with the known initial book/past prints")


def _rejection(exc):
    detail = str(exc)
    if "Insufficient" in detail or "collateral" in detail or "leverage" in detail:
        return "risk"
    if "self-cross" in detail:
        return "self_cross"
    if "storage-unit" in detail:
        return "invalid_storage_units"
    return "admission"


def _mapped(events, bid, ask):
    mapped = [
        (event, (bid if event.side == "bid" else ask) + event.price_offset_units, i)
        for i, event in enumerate(events)
    ]
    return sorted(mapped, key=lambda x: (0, -x[1], x[2]) if x[0].side == "bid" else (1, x[1], x[2]))


class _Engine:
    def __init__(self, initial):
        product_units = sum(p * q for p, q in (*initial.bids, *initial.asks))
        value = Decimal(product_units).as_tuple()
        self.cash = Decimal((value.sign, value.digits, -5))
        self.exchange = ExperimentalPerpetualExchange(_half_up_mark(initial.bids, initial.asks))
        for name in AGENTS:
            self.exchange.register(
                name,
                self.cash,
                maker_fee_rate="0",
                taker_fee_rate="0",
                min_open_notional="0",
                max_leverage="1",
            )
        self.initial_actions = []
        self.actions = []
        self.prints = []
        self.fill_count = 0
        self.seed_error = None
        for side, levels, owner in (
            ("bid", initial.bids, AGENTS[0]),
            ("ask", initial.asks, AGENTS[1]),
        ):
            for rank, (price, quantity) in enumerate(levels):
                client = f"initial:{side}:{rank}"
                try:
                    order, trades = self.exchange.submit_order(
                        owner,
                        "buy" if side == "bid" else "sell",
                        price,
                        quantity,
                        client_order_id=client,
                    )
                except ValueError as exc:
                    self.initial_actions.append(
                        ActionReceipt(
                            -1,
                            "initial",
                            side,
                            price,
                            quantity,
                            0,
                            quantity,
                            False,
                            _rejection(exc),
                            client,
                        )
                    )
                    self.seed_error = f"initial_{side}_order_rejected:{_rejection(exc)}"
                    break
                if trades:
                    raise RuntimeError("Uncrossed initial seeding unexpectedly traded")
                self.initial_actions.append(
                    ActionReceipt(
                        -1,
                        "initial",
                        side,
                        price,
                        quantity,
                        quantity,
                        0,
                        True,
                        None,
                        client,
                        (order.order_id,),
                    )
                )
            if self.seed_error:
                break
        self.seeded_bids, self.seeded_asks = _aggregate(self.exchange)

    def removes(self, events, step):
        for event, price, _ordinal in events:
            remaining = event.quantity_units
            if not 0 < price <= MAX_UNITS:
                self.actions.append(
                    ActionReceipt(
                        step,
                        "remove",
                        event.side,
                        price,
                        remaining,
                        0,
                        remaining,
                        False,
                        "invalid_price",
                        None,
                    )
                )
                continue
            side = Side.BUY if event.side == "bid" else Side.SELL
            owner = AGENTS[0] if event.side == "bid" else AGENTS[1]
            orders = [
                o
                for o in self.exchange.resting_orders()
                if o.side == side and o.agent_id == owner and o.price_units == price
            ]
            ids = []
            for order in orders:  # Adapter output already preserves FIFO within each price.
                if not remaining:
                    break
                take = min(remaining, order.remaining_quantity_units)
                self.exchange.cancel_order(order.order_id, owner, quantity_units=take)
                ids.append(order.order_id)
                remaining -= take
            self.actions.append(
                ActionReceipt(
                    step,
                    "remove",
                    event.side,
                    price,
                    event.quantity_units,
                    event.quantity_units - remaining,
                    remaining,
                    remaining < event.quantity_units,
                    "quantity_unavailable" if remaining else None,
                    None,
                    tuple(ids),
                )
            )

    def adds(self, events, step):
        for event, price, ordinal in events:
            client = f"step:{step}:add:{ordinal}"
            bids, asks = _aggregate(self.exchange)
            reason = None
            if not 0 < price <= MAX_UNITS:
                reason = "invalid_price"
            elif event.quantity_units > MAX_UNITS:
                reason = "invalid_quantity"
            elif (event.side == "bid" and asks and price >= asks[0][0]) or (
                event.side == "ask" and bids and price <= bids[0][0]
            ):
                reason = "passive_cross"
            if reason:
                self.actions.append(
                    ActionReceipt(
                        step,
                        "add",
                        event.side,
                        price,
                        event.quantity_units,
                        0,
                        event.quantity_units,
                        False,
                        reason,
                        client,
                    )
                )
                continue
            owner = AGENTS[0] if event.side == "bid" else AGENTS[1]
            try:
                order, trades = self.exchange.submit_order(
                    owner,
                    "buy" if event.side == "bid" else "sell",
                    price,
                    event.quantity_units,
                    client_order_id=client,
                )
            except ValueError as exc:
                self.actions.append(
                    ActionReceipt(
                        step,
                        "add",
                        event.side,
                        price,
                        event.quantity_units,
                        0,
                        event.quantity_units,
                        False,
                        _rejection(exc),
                        client,
                    )
                )
                continue
            if trades:
                raise RuntimeError("Guarded passive addition unexpectedly traded")
            self.actions.append(
                ActionReceipt(
                    step,
                    "add",
                    event.side,
                    price,
                    event.quantity_units,
                    event.quantity_units,
                    0,
                    True,
                    None,
                    client,
                    (order.order_id,),
                )
            )

    def intents(self, intents, step, start_us, base_bid, base_ask):
        for ordinal, intent in enumerate(intents):
            client = f"step:{step}:intent:{ordinal}"
            bids, asks = _aggregate(self.exchange)
            opposite = asks if intent.side == "Buy" else bids
            limit = (
                opposite[-1][0] if opposite else (base_ask if intent.side == "Buy" else base_bid)
            )
            owner = AGENTS[2] if intent.side == "Buy" else AGENTS[3]
            try:
                order, trades = self.exchange.submit_order(
                    owner,
                    "buy" if intent.side == "Buy" else "sell",
                    limit,
                    intent.quantity_units,
                    client_order_id=client,
                    time_in_force="IOC",
                )
            except ValueError as exc:
                self.actions.append(
                    ActionReceipt(
                        step,
                        "intent",
                        intent.side,
                        limit,
                        intent.quantity_units,
                        0,
                        intent.quantity_units,
                        False,
                        _rejection(exc),
                        client,
                    )
                )
                continue
            self.fill_count += len(trades)
            self.actions.append(
                ActionReceipt(
                    step,
                    "intent",
                    intent.side,
                    limit,
                    intent.quantity_units,
                    order.filled_quantity_units,
                    order.cancelled_quantity_units,
                    True,
                    "ioc_remainder" if order.cancelled_quantity_units else None,
                    client,
                    (order.order_id,),
                )
            )
            groups = {}
            for trade in trades:
                group = groups.setdefault(trade.price_units, [0, []])
                group[0] += trade.quantity_units
                group[1].append(trade.trade_id)
            for price, (quantity, ids) in groups.items():  # First-execution order, not a new sort.
                self.prints.append(
                    SyntheticPrint(
                        start_us + 1000,
                        intent.side,
                        quantity,
                        price,
                        f"{client}:price:{price}",
                        client,
                        tuple(ids),
                    )
                )


def _finish(
    engine,
    pool,
    initial,
    past_prints,
    past_features,
    start_us,
    steps,
    method,
    ordering,
    delay_ms,
    seed,
    path,
    selections,
    status,
    failure_at,
    failure_reason,
):
    counts = Counter(dict.fromkeys(COUNTERS, 0))
    counts.update(
        template_draws=len(selections),
        fallbacks=sum(s.fallback for s in selections),
        execution_fills=engine.fill_count,
        output_prints=len(engine.prints),
        executed_steps=len(selections),
        requested_steps=steps,
        undefined_points=sum(not p.is_valid for p in path),
    )
    rejections = {}
    for action in engine.actions:
        kind = action.kind
        if kind == "add":
            counts.update(
                add_events=1,
                add_requested_units=action.requested_units,
                added_units=action.implemented_units,
                add_rejected_units=action.unimplemented_units,
            )
        elif kind == "remove":
            counts.update(
                remove_events=1,
                remove_requested_units=action.requested_units,
                removed_units=action.implemented_units,
                remove_unimplemented_units=action.unimplemented_units,
                clipped_remove_events=int(action.unimplemented_units > 0),
            )
        else:
            counts.update(
                intent_count=1,
                intent_requested_units=action.requested_units,
                intent_filled_units=action.implemented_units,
                intent_rejected_units=action.unimplemented_units if not action.accepted else 0,
                ioc_cancelled_units=action.unimplemented_units if action.accepted else 0,
            )
        if action.reason:
            value = rejections.setdefault((kind, action.reason), [0, 0])
            value[0] += 1
            value[1] += action.unimplemented_units
    bids, asks = _aggregate(engine.exchange)
    accounts = tuple(
        AccountEnd(name, engine.exchange.account_state(name), engine.exchange.risk_snapshot(name))
        for name in AGENTS
    )
    final_equity = sum((a.state.equity for a in accounts), Decimal(0)) + engine.exchange.fee_balance
    adds = {
        side: sum(a.implemented_units for a in engine.actions if a.kind == "add" and a.side == side)
        for side in ("bid", "ask")
    }
    removes = {
        side: sum(
            a.implemented_units for a in engine.actions if a.kind == "remove" and a.side == side
        )
        for side in ("bid", "ask")
    }
    consumed = {
        side: sum(p.quantity_units for p in engine.prints if p.side == side)
        for side in ("Buy", "Sell")
    }
    conservation = Conservation(
        4 * engine.cash,
        final_equity,
        final_equity - 4 * engine.cash,
        sum((a.state.position for a in accounts), Decimal(0)),
        sum(q for _, q in engine.seeded_bids)
        + adds["bid"]
        - removes["bid"]
        - consumed["Sell"]
        - sum(q for _, q in bids),
        sum(q for _, q in engine.seeded_asks)
        + adds["ask"]
        - removes["ask"]
        - consumed["Buy"]
        - sum(q for _, q in asks),
        counts["intent_requested_units"]
        - counts["intent_filled_units"]
        - counts["ioc_cancelled_units"]
        - counts["intent_rejected_units"],
        counts["intent_filled_units"] - sum(p.quantity_units for p in engine.prints),
    )
    return SimulationReceipt(
        start_us,
        start_us + steps * GRID_US,
        steps,
        method,
        ordering,
        delay_ms,
        seed,
        pool.row_identity_sha256,
        initial,
        past_prints,
        past_features,
        engine.cash,
        DECIMAL_PRECISION,
        CapitalConfiguration(
            AGENTS,
            engine.cash,
            Decimal(1),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            False,
            "Own midpoint half-up to price storage unit: (best_bid+best_ask+1)//2",
            "Adapter gross mark exposure, adverse entry loss and positive maker fees; no netting",
        ),
        tuple(engine.initial_actions),
        tuple(path),
        tuple(engine.prints),
        tuple(engine.actions),
        tuple(selections),
        tuple((name, counts[name]) for name in COUNTERS),
        tuple((kind, reason, n, q) for (kind, reason), (n, q) in sorted(rejections.items())),
        accounts,
        bids,
        asks,
        engine.exchange.fee_balance,
        engine.exchange.mark_price_units,
        not bids or not asks,
        conservation,
        status,
        failure_at,
        failure_reason,
    )


def _simulate(
    pool,
    initial_book,
    known_past_prints,
    past_features,
    start_us,
    method,
    ordering,
    delay_ms,
    seed,
    steps,
):
    _validate_inputs(
        pool,
        initial_book,
        known_past_prints,
        past_features,
        start_us,
        method,
        ordering,
        delay_ms,
        seed,
        steps,
    )
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        engine = _Engine(initial_book)
        selections, path = [], []
        history = {point.timestamp_us: point.features for point in past_features}
        past = deque(known_past_prints)
        rng = Random(seed)
        status, failure_at, failure_reason = "complete", None, None
        if engine.seed_error:
            status, failure_at, failure_reason = (
                "failed_initialization",
                start_us,
                engine.seed_error,
            )
            path = [
                _point(start_us + i * GRID_US, (), (), undefined=True, status="not_initialized")
                for i in range(steps + 1)
            ]
        else:
            path.append(_point(start_us, initial_book.bids, initial_book.asks))
            for step in range(steps):
                decision_us = start_us + step * GRID_US
                bids, asks = _aggregate(engine.exchange)
                while past and past[0].timestamp_us < decision_us - PRESSURE_US:
                    past.popleft()
                history[decision_us] = _features(bids, asks, past)
                engine.exchange.set_mark_price(_half_up_mark(bids, asks))
                feature_us = decision_us - delay_ms * 1000
                features = history[feature_us]
                selection = select_template(
                    pool, features, method=method, delay_ms=delay_ms, rng=rng
                )
                selections.append(
                    SelectionReceipt(
                        step,
                        decision_us,
                        feature_us,
                        features,
                        selection.template.row_id,
                        selection.row_index,
                        selection.state_cell,
                        selection.fallback,
                    )
                )
                base_bid, base_ask = bids[0][0], asks[0][0]
                adds = _mapped(selection.template.adds, base_bid, base_ask)
                removes = _mapped(selection.template.removes, base_bid, base_ask)
                first_print = len(engine.prints)
                if ordering == "S1":
                    engine.removes(removes, step)
                    engine.adds(adds, step)
                    engine.intents(
                        selection.template.intents, step, decision_us, base_bid, base_ask
                    )
                else:
                    engine.intents(
                        selection.template.intents, step, decision_us, base_bid, base_ask
                    )
                    engine.removes(removes, step)
                    engine.adds(adds, step)
                past.extend(engine.prints[first_print:])
                bids, asks = _aggregate(engine.exchange)
                end_us = decision_us + GRID_US
                path.append(_point(end_us, bids, asks))
                if not bids or not asks:
                    status, failure_at, failure_reason = (
                        "failed_missing_side",
                        end_us,
                        "missing_side",
                    )
                    path.extend(
                        _point(start_us + i * GRID_US, (), (), undefined=True)
                        for i in range(step + 2, steps + 1)
                    )
                    break
                engine.exchange.set_mark_price(_half_up_mark(bids, asks))
        return _finish(
            engine,
            pool,
            initial_book,
            known_past_prints,
            past_features,
            start_us,
            steps,
            method,
            ordering,
            delay_ms,
            seed,
            path,
            selections,
            status,
            failure_at,
            failure_reason,
        )


def simulate_window(
    pool,
    initial_book,
    known_past_prints,
    past_features,
    *,
    start_us,
    method,
    ordering,
    delay_ms,
    seed,
):
    """Fixed 300-second entry point; no future target or adjustable horizon input."""
    return _simulate(
        pool,
        initial_book,
        known_past_prints,
        past_features,
        start_us,
        method,
        ordering,
        delay_ms,
        seed,
        STEPS,
    )


def simulate_steps(
    pool,
    initial_book,
    known_past_prints,
    past_features,
    *,
    start_us,
    method,
    ordering,
    delay_ms,
    seed,
    steps,
):
    """Small synthetic engineering helper; never used as the study's real entry."""
    return _simulate(
        pool,
        initial_book,
        known_past_prints,
        past_features,
        start_us,
        method,
        ordering,
        delay_ms,
        seed,
        steps,
    )
