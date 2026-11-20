"""Pure fixed TRAIN ensembles; no readers, future labels, files or worker pools.

The caller must establish the pool and initialization provenance before entry.
This module fixes the full run denominator and validates returned identities;
it does not fit, select a winner, score targets or repair failed simulations.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from experiments.crypto_autonomous_flow import (
    GRID_US,
    HORIZON_US,
    STEPS,
    FeatureContext,
    InitialBook,
    KnownPrint,
    PathPoint,
    SimulationReceipt,
    SyntheticPrint,
    initial_features,
    simulate_window,
)
from experiments.crypto_autonomous_metrics import BookPoint, PublishedPrint, SimulationPath
from experiments.crypto_autonomous_targets import WindowInitialization
from experiments.crypto_effective_flow import DELAYS_MS, TRAIN_DAYS, TemplatePool

METHODS = ("M0", "M1")
ORDERINGS = ("S1", "S2")
SEEDS = tuple(range(20))
TRAIN_WINDOW_STARTS = tuple(
    int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()) * 1_000_000
    + hour * 3_600_000_000
    + 600_000_000
    for day in TRAIN_DAYS
    for hour in range(24)
)


def _calendar(start_us):
    if type(start_us) is not int or start_us not in TRAIN_WINDOW_STARTS:
        raise ValueError("Only TRAIN November 1-3 hourly 10-15 minute windows are approved")


def _scenario(method, ordering, delay_ms, seed):
    if (
        method not in METHODS
        or ordering not in ORDERINGS
        or type(delay_ms) is not int
        or delay_ms not in DELAYS_MS
        or type(seed) is not int
        or seed not in SEEDS
    ):
        raise ValueError("Require a fixed method, ordering, delay and seed 0-19")


def _issues(issues):
    if type(issues) is not tuple or any(type(s) is not str or not s for s in issues):
        raise TypeError("Source issues must be an immutable tuple of nonempty strings")


def _initialization(initialization):
    # Calendar access precedes even inspecting the pool or creating a generator.
    if type(initialization) is not WindowInitialization:
        raise TypeError("Require the exact immutable WindowInitialization")
    _calendar(initialization.start_us)
    _issues(initialization.issues)
    for value, item_type in (
        (initialization.past_prints, KnownPrint),
        (initialization.feature_context, FeatureContext),
    ):
        if value is not None and (
            type(value) is not tuple or any(type(item) is not item_type for item in value)
        ):
            raise TypeError("Initialization context must contain exact immutable values")
    if initialization.initial_book is not None and type(initialization.initial_book) is not InitialBook:
        raise TypeError("Initialization requires the exact immutable InitialBook")
    references = initialization.prefix_context
    if type(references) is not tuple or any(
        type(row) is not tuple
        or len(row) != 8
        or any(value is not None and type(value) is not int for value in row)
        for row in references
    ):
        raise TypeError("Prefix context must retain immutable eight-field source references")
    if initialization.issues:
        return  # An explicit source failure is preserved, never sent to the generator.
    if (
        initialization.initial_book is None
        or initialization.past_prints is None
        or initialization.feature_context is None
    ):
        raise ValueError("Missing initialization values require explicit source issues")
    start = initialization.start_us
    expected = tuple(range(start - 5_000_000, start + 1, GRID_US))
    if (
        tuple(p.timestamp_us for p in initialization.feature_context) != expected
        or tuple(row[0] for row in references) != expected
        or any(
            not start - 1_000_000 <= p.timestamp_us < start
            for p in initialization.past_prints
        )
        or any(
            a.timestamp_us > b.timestamp_us
            for a, b in zip(
                initialization.past_prints, initialization.past_prints[1:], strict=False
            )
        )
        or len({p.print_id for p in initialization.past_prints}) != len(initialization.past_prints)
    ):
        raise ValueError("Initialization must contain the complete grid and strictly past tape")
    if initialization.feature_context[-1].features != initial_features(
        initialization.initial_book, initialization.past_prints
    ):
        raise ValueError("Initial book and past context disagree")
    day_start = start // 86_400_000_000 * 86_400_000_000
    for observed, index, engine, available, prefix, segment, origin, errors in references:
        if (
            any(v is None for v in (index, engine, available, prefix, segment, origin, errors))
            or index <= 0
            or segment < 0
            or errors < 0
            or not day_start <= origin <= engine < observed
            or not day_start <= available < observed
            or observed - engine >= 1_000_000
            or (prefix + 1) * 1000 > available
            or engine + 1000 > available
        ):
            raise ValueError("Unflagged initialization has an unavailable or stale source prefix")
    if len({(row[5], row[6], row[7]) for row in references}) != 1:
        raise ValueError("Unflagged initialization crosses a source reset or gap")


@dataclass(frozen=True, slots=True)
class RunSpecification:
    method: str
    ordering: str
    delay_ms: int
    seed: int

    def __post_init__(self):
        _scenario(self.method, self.ordering, self.delay_ms, self.seed)


# Ordering is part of this interface, including all 20 seeds within each scenario.
WORK_PLAN = tuple(
    RunSpecification(method, ordering, delay, seed)
    for method in METHODS
    for ordering in ORDERINGS
    for delay in DELAYS_MS
    for seed in SEEDS
)


def _book_side(levels, name, *, optional=False):
    if optional and levels is None:
        return
    if type(levels) is not tuple or len(levels) > 500 or any(
        type(level) is not tuple
        or len(level) != 2
        or any(type(value) is not int or value <= 0 for value in level)
        for level in levels
    ):
        raise ValueError("Receipt must preserve exact immutable top-500 book levels")
    prices = tuple(p for p, _ in levels)
    if len(set(prices)) != len(prices) or prices != tuple(sorted(prices, reverse=name == "bid")):
        raise ValueError("Receipt book levels are duplicated or out of price order")


def _receipt(receipt):
    if type(receipt) is not SimulationReceipt:
        raise TypeError("Require an actual immutable SimulationReceipt")
    _calendar(receipt.start_us)
    _scenario(receipt.method, receipt.ordering, receipt.delay_ms, receipt.seed)
    if (
        type(receipt.end_us) is not int
        or receipt.end_us != receipt.start_us + HORIZON_US
        or type(receipt.requested_steps) is not int
        or receipt.requested_steps != STEPS
        or type(receipt.path) is not tuple
        or len(receipt.path) != STEPS + 1
        or type(receipt.prints) is not tuple
    ):
        raise ValueError("Receipt must retain the fixed 3000 steps and all 3001 clock points")
    if receipt.status == "complete":
        if receipt.failure_at_us is not None or receipt.failure_reason is not None:
            raise ValueError("Complete receipt contains failure metadata")
        failed_index = None
    elif receipt.status in ("failed_initialization", "failed_missing_side"):
        if (
            type(receipt.failure_at_us) is not int
            or not receipt.start_us <= receipt.failure_at_us <= receipt.end_us
            or (receipt.failure_at_us - receipt.start_us) % GRID_US
            or type(receipt.failure_reason) is not str
            or not receipt.failure_reason
        ):
            raise ValueError("Failure receipt lacks a valid failure clock or reason")
        failed_index = (receipt.failure_at_us - receipt.start_us) // GRID_US
        if (receipt.status == "failed_initialization") != (failed_index == 0):
            raise ValueError("Failure status disagrees with its first undefined clock")
    else:
        raise ValueError("Unknown simulation receipt status")
    for index, point in enumerate(receipt.path):
        if (
            type(point) is not PathPoint
            or type(point.timestamp_us) is not int
            or point.timestamp_us != receipt.start_us + index * GRID_US
        ):
            raise ValueError("Receipt has a missing, duplicate or stale point clock")
        expected_status = (
            "not_initialized" if failed_index == 0
            else "defined" if failed_index is None or index < failed_index
            else "missing_side" if index == failed_index
            else "undefined_after_failure"
        )
        if point.status != expected_status:
            raise ValueError("Receipt point status loses its failure or undefined tail")
        _book_side(point.bids, "bid", optional=True)
        _book_side(point.asks, "ask", optional=True)
        if expected_status == "defined":
            if not point.bids or not point.asks or point.bids[0][0] >= point.asks[0][0]:
                raise ValueError("A defined receipt point requires two uncrossed sides")
        elif expected_status == "missing_side":
            if point.bids is None or point.asks is None or (point.bids and point.asks):
                raise ValueError("A missing-side point must retain its actual surviving levels")
        elif point.bids is not None or point.asks is not None:
            raise ValueError("An undefined tail cannot carry a book forward")
    previous = receipt.start_us
    identities = set()
    print_end = receipt.end_us if failed_index is None else receipt.failure_at_us
    for print_ in receipt.prints:
        if (
            type(print_) is not SyntheticPrint
            or type(print_.timestamp_us) is not int
            or not previous <= print_.timestamp_us < print_end
            or (print_.timestamp_us - receipt.start_us) % GRID_US != 1000
            or print_.side not in ("Buy", "Sell")
            or type(print_.quantity_units) is not int
            or print_.quantity_units <= 0
            or type(print_.price_units) is not int
            or print_.price_units <= 0
            or type(print_.print_id) is not str
            or not print_.print_id
            or print_.print_id in identities
            or type(print_.fill_ids) is not tuple
        ):
            raise ValueError("Receipt must preserve actual ordered, fixed-clock synthetic prints")
        previous = print_.timestamp_us
        identities.add(print_.print_id)


@dataclass(frozen=True, slots=True)
class EnsembleRun:
    start_us: int
    method: str
    ordering: str
    delay_ms: int
    seed: int
    source_issues: tuple[str, ...]
    receipt: SimulationReceipt | None

    def __post_init__(self):
        _calendar(self.start_us)
        _scenario(self.method, self.ordering, self.delay_ms, self.seed)
        _issues(self.source_issues)
        if self.source_issues:
            if self.receipt is not None:
                raise ValueError("A source failure cannot contain a simulated receipt")
            return
        _receipt(self.receipt)
        if any(
            getattr(self.receipt, name) != getattr(self, name)
            for name in ("start_us", "method", "ordering", "delay_ms", "seed")
        ):
            raise ValueError("Receipt identity differs from the requested ensemble run")

    @property
    def end_us(self):
        return self.start_us + HORIZON_US

    @property
    def status(self):
        return "failed_source_initialization" if self.source_issues else self.receipt.status


def iter_window_ensemble(pool, initialization):
    """Yield all fixed 240 runs, or propagate a code error to an incomplete job.

    Validation is eager; generation is lazy and holds only the current receipt.
    Explicit source issues yield 240 missing receipts. Valid model failures keep
    their receipts and do not shorten the remaining scenario/seed denominator.
    """
    _initialization(initialization)
    if type(pool) is not TemplatePool:
        raise TypeError("Require the exact immutable training TemplatePool")

    def runs():
        for specification in WORK_PLAN:
            receipt = None
            if not initialization.issues:
                receipt = simulate_window(
                    pool,
                    initialization.initial_book,
                    initialization.past_prints,
                    initialization.feature_context,
                    start_us=initialization.start_us,
                    method=specification.method,
                    ordering=specification.ordering,
                    delay_ms=specification.delay_ms,
                    seed=specification.seed,
                )
                if type(receipt) is not SimulationReceipt:
                    raise TypeError("Generator returned a non-receipt")
                if (
                    receipt.pool_row_identity_sha256 != pool.row_identity_sha256
                    or receipt.initial_book != initialization.initial_book
                    or receipt.known_past_prints != initialization.past_prints
                    or receipt.known_feature_context != initialization.feature_context
                ):
                    raise ValueError("Receipt belongs to a different pool or source initialization")
            yield EnsembleRun(
                initialization.start_us,
                specification.method,
                specification.ordering,
                specification.delay_ms,
                specification.seed,
                initialization.issues,
                receipt,
            )

    return runs()


def receipt_to_path(receipt):
    """Copy actual book views/prints; retain partial prints and undefined clocks.

    Metrics recalculate prices and depth from book units, not cached indicators.
    Overall status remains in EnsembleRun/SimulationReceipt; each invalid point
    carries its original point status in BookPoint.failure_reason.
    """
    _receipt(receipt)
    points = tuple(
        BookPoint(
            point.timestamp_us // 1000,
            point.bids,
            point.asks,
            point.is_valid,
            None if point.is_valid else point.status,
        )
        for point in receipt.path
    )
    prints = tuple(
        PublishedPrint(p.timestamp_us // 1000, p.side, p.quantity_units, p.price_units)
        for p in receipt.prints
    )
    return SimulationPath(receipt.seed, points, prints)


def run_to_path(run):
    """A source failure has an unknown stream, never a fabricated empty market."""
    if type(run) is not EnsembleRun:
        raise TypeError("Require the exact immutable EnsembleRun")
    if run.source_issues:
        return SimulationPath(run.seed, None, None)
    return receipt_to_path(run.receipt)
