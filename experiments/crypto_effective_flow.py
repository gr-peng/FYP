"""Pure effective L2-flow decomposition and finite training-template pools.

There is no file reader, historical-path lookup or generator in this module.
Endpoint evidence retains observed prices; the generation template deliberately
contains only relative passive changes and direction/size IOC intentions.
Net residuals are not identified gross orders or cancellations. Public prints
need not be consumptions of displayed liquidity or identified parent orders.
"""

import hashlib
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from random import Random
from typing import Literal

GRID_US = 100_000
DAY_US = 86_400_000_000
MAX_AGE_US = 1_000_000
DELAYS_MS = (0, 1000, 5000)
TRAIN_DAYS = ("2024-11-01", "2024-11-02", "2024-11-03")
MIN_CELL_SIZE = 32
MAX_UNITS = 2**63 - 1
BookSide = Literal["bid", "ask"]
TradeSide = Literal["Buy", "Sell"]


def _integer(value, name, *, positive=False):
    if type(value) is not int or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _units(value, name):
    if _integer(value, name, positive=True) > MAX_UNITS:
        raise ValueError(f"{name} must fit positive int64 storage")
    return value


def _day_start(day):
    return (date.fromisoformat(day) - date(1970, 1, 1)).days * DAY_US


@dataclass(frozen=True, slots=True)
class BookObservation:
    """Immutable copy of the observer's at-most-500-level historical view.

    Invalid/no-state observations use empty sides and retain whatever clocks
    are known. segment_id increments on every valid snapshot; invalid_count is
    cumulative, so a gap inside the interval cannot hide behind valid endpoints.
    The caller constructs this only from the consumed raw-order prefix.
    """

    observed_at_us: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    valid: bool
    segment_id: int | None
    segment_start_us: int | None
    message_index: int | None
    engine_us: int | None
    available_us: int | None
    publication_prefix_ms: int | None
    invalid_count: int = 0

    def __post_init__(self):
        _integer(self.observed_at_us, "observed_at_us")
        _integer(self.invalid_count, "invalid_count")
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        for name in (
            "segment_id", "segment_start_us", "message_index", "engine_us",
            "available_us", "publication_prefix_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name)
        for side, levels in (("bid", self.bids), ("ask", self.asks)):
            if type(levels) is not tuple or any(type(level) is not tuple for level in levels):
                raise ValueError("Levels must be immutable tuples")
            if len(levels) > 500:
                raise ValueError("Historical observation exceeds its subscribed depth")
            for price, quantity in levels:
                _units(price, "price_units")
                _units(quantity, "quantity_units")
            prices = [p for p, _ in levels]
            if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bid"):
                raise ValueError("Observation levels must be unique and ordered")
        if self.valid:
            _integer(self.message_index, "message_index", positive=True)
            if not self.bids or not self.asks or self.bids[0][0] >= self.asks[0][0]:
                raise ValueError("Valid observation requires an uncrossed two-sided book")
            if any(
                getattr(self, field) is None
                for field in (
                    "segment_id", "segment_start_us", "message_index", "engine_us",
                    "available_us", "publication_prefix_ms",
                )
            ):
                raise ValueError("Valid observation requires complete prefix clocks")
            if self.engine_us % 1000 or self.segment_start_us % 1000:
                raise ValueError("Source CTS and snapshot origin must retain millisecond precision")
            if not (
                self.segment_start_us <= self.engine_us
                and self.engine_us + 1000 <= self.available_us
                and (self.publication_prefix_ms + 1) * 1000 <= self.available_us
                and self.available_us < self.observed_at_us
            ):
                raise ValueError("Observation violates the conservative available-prefix clock")
        elif self.bids or self.asks:
            raise ValueError("Invalid observations cannot expose usable levels")


@dataclass(frozen=True, slots=True)
class ObservedPrint:
    timestamp_us: int
    side: TradeSide
    quantity_units: int
    price_units: int
    source_row: int
    trade_id: str

    def __post_init__(self):
        _integer(self.timestamp_us, "timestamp_us")
        _integer(self.source_row, "source_row", positive=True)
        _units(self.quantity_units, "quantity_units")
        _units(self.price_units, "price_units")
        if (self.side not in {"Buy", "Sell"}
                or not isinstance(self.trade_id, str) or not self.trade_id):
            raise ValueError("Print requires a side and nonempty source identity")


@dataclass(frozen=True, slots=True)
class RelativeLiquidity:
    side: BookSide
    price_offset_units: int
    quantity_units: int

    def __post_init__(self):
        if self.side not in {"bid", "ask"} or type(self.price_offset_units) is not int:
            raise ValueError("Relative liquidity requires a side and exact integer offset")
        _integer(self.quantity_units, "quantity_units", positive=True)


@dataclass(frozen=True, slots=True)
class Intent:
    """Synthetic IOC representation of one print; observed price is absent."""

    side: TradeSide
    quantity_units: int

    def __post_init__(self):
        if self.side not in {"Buy", "Sell"}:
            raise ValueError("Unknown intention side")
        _units(self.quantity_units, "quantity_units")


@dataclass(frozen=True, slots=True)
class FlowTemplate:
    row_id: str
    adds: tuple[RelativeLiquidity, ...]
    removes: tuple[RelativeLiquidity, ...]
    intents: tuple[Intent, ...]

    def __post_init__(self):
        if not isinstance(self.row_id, str) or not self.row_id:
            raise ValueError("Template requires a stable evidence row identity")
        for name, expected in (("adds", RelativeLiquidity), ("removes", RelativeLiquidity),
                               ("intents", Intent)):
            items = getattr(self, name)
            if type(items) is not tuple or any(type(item) is not expected for item in items):
                raise ValueError("Template events must be immutable typed tuples")


@dataclass(frozen=True, slots=True)
class EffectiveLevel:
    side: BookSide
    price_units: int
    start_quantity_units: int
    end_quantity_units: int
    consumed_proxy_units: int
    residual_units: int


@dataclass(frozen=True, slots=True)
class PrintAlignment:
    source_row: int
    domain: str
    within_engine_span: bool | None
    absent_at_both_endpoints: bool | None


@dataclass(frozen=True, slots=True)
class FlowEvidence:
    day: str
    start: BookObservation
    end: BookObservation
    prints: tuple[ObservedPrint, ...]
    exclusion_reasons: tuple[str, ...]
    template: FlowTemplate | None
    levels: tuple[EffectiveLevel, ...]
    alignment: tuple[PrintAlignment, ...]
    # (side, endpoint, number of prices, quantity units), not inferred flow.
    censored_endpoints: tuple[tuple[str, str, int, int], ...]

    @property
    def eligible(self):
        return not self.exclusion_reasons

    @property
    def observed_quantity_units(self):
        return sum(t.quantity_units for t in self.prints)


def _check_prints(prints, start_us, end_us):
    if type(prints) is not tuple or any(type(t) is not ObservedPrint for t in prints):
        raise ValueError("Print buffer must be an immutable typed tuple")
    previous_time, previous_row = -1, 0
    ids = set()
    for print_ in prints:
        if not start_us <= print_.timestamp_us < end_us:
            raise ValueError("Print outside the half-open interval; never silently reassign it")
        if print_.timestamp_us < previous_time or print_.source_row <= previous_row:
            raise ValueError("Prints must retain source-row order, including timestamp ties")
        if print_.trade_id in ids:
            raise ValueError("Duplicate print identity in interval")
        previous_time, previous_row = print_.timestamp_us, print_.source_row
        ids.add(print_.trade_id)


def extract_interval(day, start, end, prints=()):
    """Decompose a 100ms interval; excluded candidates remain full evidence.

    Effective price events have fixed canonical order: bids descending, then
    asks ascending. Aggregation cannot recover their real intrainterval order.
    Timestamp ties in intentions retain the original tape row order instead.
    """
    if type(start) is not BookObservation or type(end) is not BookObservation:
        raise ValueError("Endpoint evidence requires exact immutable BookObservation objects")
    day_start = _day_start(day)
    if (
        start.observed_at_us % GRID_US
        or end.observed_at_us != start.observed_at_us + GRID_US
        or not day_start <= start.observed_at_us < end.observed_at_us <= day_start + DAY_US
    ):
        raise ValueError("Expected one UTC-aligned 100ms interval in its named day")
    _check_prints(prints, start.observed_at_us, end.observed_at_us)
    reasons = []
    for label, observation in (("start", start), ("end", end)):
        if not observation.valid:
            reasons.append(f"{label}_invalid_or_absent")
            continue
        if not (
            day_start <= observation.segment_start_us < day_start + DAY_US
            and day_start <= observation.engine_us < day_start + DAY_US
            and day_start <= observation.available_us < day_start + DAY_US
        ):
            reasons.append(f"{label}_outside_day_state")
        if observation.observed_at_us - observation.engine_us >= MAX_AGE_US:
            reasons.append(f"{label}_stale")
    if start.valid and end.valid:
        if (end.segment_id < start.segment_id or end.invalid_count < start.invalid_count):
            raise ValueError("Observer cumulative segment/error counters moved backwards")
        if start.message_index == end.message_index and any(
            getattr(start, key) != getattr(end, key)
            for key in (
                "bids", "asks", "segment_id", "segment_start_us", "engine_us", "available_us",
                "publication_prefix_ms", "invalid_count",
            )
        ):
            raise ValueError("Same consumed prefix cannot change its book or source identity")
        if start.segment_id == end.segment_id:
            if start.segment_start_us != end.segment_start_us:
                raise ValueError("Same segment cannot change its snapshot origin")
            if end.publication_prefix_ms < start.publication_prefix_ms:
                raise ValueError("Same segment publication prefix moved backwards")
        if start.segment_id != end.segment_id:
            reasons.append("snapshot_or_reset_crossing")
        if start.invalid_count != end.invalid_count:
            reasons.append("invalid_message_crossing")
        if (
            end.message_index < start.message_index
            or end.engine_us < start.engine_us
            or end.available_us < start.available_us
        ):
            raise ValueError("Observer prefix moved backwards")
    if reasons:
        return FlowEvidence(
            day, start, end, prints, tuple(reasons), None, (),
            tuple(PrintAlignment(t.source_row, "unusable_endpoints", None, None) for t in prints),
            (),
        )

    originals = {"bid": dict(start.bids), "ask": dict(start.asks)}
    finals = {"bid": dict(end.bids), "ask": dict(end.asks)}
    boundaries = {"bid": max(start.bids[-1][0], end.bids[-1][0]),
                  "ask": min(start.asks[-1][0], end.asks[-1][0])}

    def known(side, price):
        return price >= boundaries[side] if side == "bid" else price <= boundaries[side]

    consumes = {"bid": {}, "ask": {}}
    alignment = []
    for print_ in prints:
        side = "ask" if print_.side == "Buy" else "bid"
        price, quantity = print_.price_units, print_.quantity_units
        in_domain = known(side, price)
        if in_domain:
            consumes[side][price] = consumes[side].get(price, 0) + quantity
        alignment.append(PrintAlignment(
            print_.source_row,
            "common_visible_domain" if in_domain else "outside_common_visible_domain",
            start.engine_us <= print_.timestamp_us < end.engine_us,
            (price not in originals[side] and price not in finals[side]) if in_domain else None,
        ))
    adds, removes, levels, censored = [], [], [], []
    for side in ("bid", "ask"):
        old, new, consumed = originals[side], finals[side], consumes[side]
        best = (start.bids if side == "bid" else start.asks)[0][0]
        for endpoint, quantities in (("start", old), ("end", new)):
            outside = [quantity for price, quantity in quantities.items() if not known(side, price)]
            censored.append((side, endpoint, len(outside), sum(outside)))
        for price in sorted(old.keys() | new.keys() | consumed.keys(), reverse=side == "bid"):
            if not known(side, price):
                continue
            q0, q1, c = old.get(price, 0), new.get(price, 0), consumed.get(price, 0)
            residual = q1 - q0 + c
            levels.append(EffectiveLevel(side, price, q0, q1, c, residual))
            if residual:
                event = RelativeLiquidity(side, price - best, abs(residual))
                (adds if residual > 0 else removes).append(event)
    template = FlowTemplate(
        f"{day}:{start.observed_at_us}", tuple(adds), tuple(removes),
        tuple(Intent(t.side, t.quantity_units) for t in prints),
    )
    return FlowEvidence(day, start, end, prints, (), template, tuple(levels),
                        tuple(alignment), tuple(censored))


@dataclass(frozen=True, slots=True)
class StateFeatures:
    imbalance_l1: Fraction
    spread_bps: Fraction
    trade_pressure: Fraction

    def __post_init__(self):
        if any(type(x) is not Fraction for x in (
            self.imbalance_l1, self.spread_bps, self.trade_pressure,
        )):
            raise ValueError("Features must be exact Fractions")
        if not -1 <= self.imbalance_l1 <= 1 or not -1 <= self.trade_pressure <= 1:
            raise ValueError("Invalid signed imbalance")
        if self.spread_bps <= 0:
            raise ValueError("State requires a positive spread")


def state_features(observation, past_prints=()):
    """Only a causal book and the already completed past 1s of ideal-visible prints."""
    if type(observation) is not BookObservation:
        raise ValueError("Features require an exact immutable BookObservation")
    if not observation.valid or observation.observed_at_us - observation.engine_us >= MAX_AGE_US:
        raise ValueError("State features require a current valid observation")
    _check_prints(past_prints, observation.observed_at_us - 1_000_000,
                  observation.observed_at_us)
    bid, bq = observation.bids[0]
    ask, aq = observation.asks[0]
    buy = sum(t.quantity_units for t in past_prints if t.side == "Buy")
    sell = sum(t.quantity_units for t in past_prints if t.side == "Sell")
    return StateFeatures(Fraction(bq - aq, bq + aq),
                         Fraction(20000 * (ask - bid), ask + bid),
                         Fraction(buy - sell, buy + sell) if buy + sell else Fraction(0))


def _imbalance_bin(value):
    return 0 if value < Fraction(-1, 3) else (2 if value > Fraction(1, 3) else 1)


def state_cell(features, spread_median):
    if type(features) is not StateFeatures:
        raise ValueError("Cell input requires exact immutable StateFeatures")
    if type(spread_median) is not Fraction or spread_median <= 0:
        raise ValueError("Expected a frozen positive exact spread median")
    return ((_imbalance_bin(features.imbalance_l1) * 2
             + int(features.spread_bps > spread_median)) * 3
            + _imbalance_bin(features.trade_pressure))


@dataclass(frozen=True, slots=True)
class TrainingRow:
    day: str
    template: FlowTemplate
    features_by_delay: tuple[StateFeatures, StateFeatures, StateFeatures]

    def __post_init__(self):
        if self.day not in TRAIN_DAYS:
            raise ValueError("Development and event-test templates are closed")
        if type(self.template) is not FlowTemplate:
            raise ValueError("Training row requires the exact narrow immutable FlowTemplate")
        start = _day_start(self.day)
        prefix = f"{self.day}:"
        if not self.template.row_id.startswith(prefix):
            raise ValueError("Template identity differs from its approved train day")
        timestamp = int(self.template.row_id[len(prefix):])
        if (self.template.row_id != f"{self.day}:{timestamp}" or timestamp % GRID_US
                or not start <= timestamp < start + DAY_US):
            raise ValueError("Template identity is not an approved aligned training interval")
        if (type(self.features_by_delay) is not tuple or len(self.features_by_delay) != 3
                or any(type(x) is not StateFeatures for x in self.features_by_delay)):
            raise ValueError("Every training row requires all three frozen delay inputs")


@dataclass(frozen=True, slots=True)
class TemplatePool:
    rows: tuple[TrainingRow, ...]
    spread_median: Fraction
    # Delay, then the fixed 18 cells, then common-row indexes.
    indexes: tuple[tuple[tuple[int, ...], ...], ...]
    row_identity_sha256: str

    def __post_init__(self):
        if (type(self.rows) is not tuple or not self.rows
                or any(type(row) is not TrainingRow for row in self.rows)):
            raise ValueError("Pool requires immutable approved TrainingRow objects")
        identities = [row.template.row_id for row in self.rows]
        if identities != sorted(identities) or len(set(identities)) != len(identities):
            raise ValueError("Pool cohort must be unique and retain day/grid order")
        spreads = sorted(row.features_by_delay[0].spread_bps for row in self.rows)
        n = len(spreads)
        median = (spreads[(n - 1) // 2] + spreads[n // 2]) / 2
        if type(self.spread_median) is not Fraction or self.spread_median != median:
            raise ValueError("Pool threshold must be the exact common zero-delay train median")
        expected = []
        for delay in range(len(DELAYS_MS)):
            cells = [[] for _ in range(18)]
            for index, row in enumerate(self.rows):
                cells[state_cell(row.features_by_delay[delay], median)].append(index)
            expected.append(tuple(tuple(cell) for cell in cells))
        if type(self.indexes) is not tuple or self.indexes != tuple(expected):
            raise ValueError("Pool indexes differ from the fixed common-cohort state cells")
        digest = hashlib.sha256("\n".join(identities).encode()).hexdigest()
        if self.row_identity_sha256 != digest:
            raise ValueError("Pool row identity digest differs")


def fit_template_pool(rows):
    """Pure empirical fit on one common train-only cohort; no search or refit."""
    if type(rows) is not tuple or not rows or any(type(row) is not TrainingRow for row in rows):
        raise ValueError("Fit requires a nonempty immutable training cohort")
    identities = [row.template.row_id for row in rows]
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("Training cohort must be unique and retain day/grid order")
    spreads = sorted(row.features_by_delay[0].spread_bps for row in rows)
    n = len(spreads)
    median = (spreads[(n - 1) // 2] + spreads[n // 2]) / 2
    indexes = []
    for d in range(len(DELAYS_MS)):
        cells = [[] for _ in range(18)]
        for index, row in enumerate(rows):
            cells[state_cell(row.features_by_delay[d], median)].append(index)
        indexes.append(tuple(tuple(cell) for cell in cells))
    identity_hash = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    return TemplatePool(rows, median, tuple(indexes), identity_hash)


@dataclass(frozen=True, slots=True)
class TemplateSelection:
    template: FlowTemplate
    row_index: int
    state_cell: int
    fallback: bool


def select_template(pool, features, *, method, delay_ms, rng: Random):
    """Draw a joint row with one fixed-width parent draw in every strategy.

    randrange consumes variable entropy when pool sizes differ. A fixed 128-bit
    parent draw seeds a local sampler instead, preserving the shared parent
    stream across M0/M1, cells and fallbacks without floor-map sampling bias.
    Runtime Python/Random versions still belong in the later model/run receipt.
    """
    if type(pool) is not TemplatePool or type(features) is not StateFeatures:
        raise ValueError("Selection requires exact validated immutable pool and feature objects")
    if method not in {"M0", "M1"} or delay_ms not in DELAYS_MS:
        raise ValueError("Unknown method or frozen information delay")
    if not isinstance(rng, Random):
        raise ValueError("Pass the simulation's explicit seeded Random stream")
    cell = state_cell(features, pool.spread_median)
    conditioned = pool.indexes[DELAYS_MS.index(delay_ms)][cell]
    fallback = method == "M1" and len(conditioned) < MIN_CELL_SIZE
    local_rng = Random(rng.getrandbits(128))
    if method == "M0" or fallback:
        index = local_rng.randrange(len(pool.rows))
    else:
        index = conditioned[local_rng.randrange(len(conditioned))]
    return TemplateSelection(pool.rows[index].template, index, cell, fallback)
