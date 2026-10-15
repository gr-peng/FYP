"""Pure train-stream bridge for observed labels and strictly past initialization.

The future labels are never passed to the generator. This module has no file
reader: a later provenance runner must authorize, freeze and verify the complete
raw sources before calling collect_train_day. Engineering spans are synthetic.
"""

from collections import deque
from dataclasses import dataclass, replace

from .crypto_autonomous_flow import FeatureContext, InitialBook, KnownPrint
from .crypto_autonomous_metrics import BlockSummary, BookPoint, PublishedPrint, summarize_block
from .crypto_effective_flow import DAY_US, GRID_US, TRAIN_DAYS, ObservedPrint, state_features
from .crypto_effective_flow_reader import FlowBookCursor
from .crypto_execution_diagnostic import day_start_us


@dataclass(frozen=True, slots=True)
class WindowInitialization:
    start_us: int
    initial_book: InitialBook | None
    past_prints: tuple[KnownPrint, ...] | None
    feature_context: tuple[FeatureContext, ...] | None
    # observed time, consumed index, CTS, available, prefix TS, segment, origin, errors.
    prefix_context: tuple[tuple[int | None, ...], ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservedPriceBlock:
    start_us: int
    returns_bps: tuple[float | None, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DayTargets:
    day: str
    span_us: tuple[int, int]
    minute_labels: tuple[BlockSummary, ...]
    initializations: tuple[WindowInitialization, ...]
    price_blocks: tuple[ObservedPriceBlock, ...]
    observed_prints: int
    observed_quantity_units: int


def _point(observation, day_start):
    valid = observation.valid and (
        day_start <= observation.segment_start_us <= observation.engine_us
        and day_start <= observation.available_us < day_start + DAY_US
        and observation.engine_us < day_start + DAY_US
        and observation.observed_at_us - observation.engine_us < 1_000_000
    )
    return BookPoint(observation.observed_at_us // 1000,
                     observation.bids if valid else None,
                     observation.asks if valid else None, valid,
                     None if valid else "invalid_absent_stale_or_outside_day_prefix")


def _initialization(observations, tape, at, day_start):
    points = tuple(observations)
    references = tuple(tuple(getattr(point, key) for key in (
        "observed_at_us", "message_index", "engine_us", "available_us",
        "publication_prefix_ms", "segment_id", "segment_start_us", "invalid_count",
    )) for point in points)
    expected = tuple(range(at - 5_000_000, at + 1, GRID_US))
    issues = []
    if tuple(point.observed_at_us for point in points) != expected or at - 6_000_000 < day_start:
        issues.append("incomplete_known_past_grid_or_tape")
    if points and any(
        point.segment_id != points[-1].segment_id
        or point.invalid_count != points[-1].invalid_count
        or not _point(point, day_start).is_valid
        for point in points
    ):
        issues.append("invalid_stale_or_reset_past_context")
    if issues:
        return WindowInitialization(at, None, None, None, references, tuple(issues))
    context = tuple(FeatureContext(point.observed_at_us, state_features(
        point, tuple(t for t in tape
                     if point.observed_at_us - 1_000_000 <= t.timestamp_us < point.observed_at_us),
    )) for point in points)
    known = tuple(KnownPrint(t.timestamp_us, t.side, t.quantity_units, t.trade_id)
                  for t in tape if at - 1_000_000 <= t.timestamp_us < at)
    return WindowInitialization(at, InitialBook(points[-1].bids, points[-1].asks),
                                known, context, references, ())


def _collect_span(messages, prints, day, end_us):
    """Synthetic engineering helper; public real-source entry is always a full day."""
    start = day_start_us(day)
    if (day not in TRAIN_DAYS or type(end_us) is not int
            or end_us % 60_000_000 or not start < end_us <= start + DAY_US):
        raise ValueError("Only whole UTC minutes of the approved train calendar")
    prints = iter(prints)

    def next_print():
        print_ = next(prints, None)
        if print_ is not None and type(print_) is not ObservedPrint:
            raise ValueError("Bridge requires exact immutable audited ObservedPrint inputs")
        return print_

    pending = next_print()
    previous_time, previous_row = -1, 0
    cursor = FlowBookCursor(messages, day, 0)
    observation = cursor.observation(start)
    past_books, past_tape = deque(maxlen=51), deque()
    past_books.append(observation)
    block_points, block_sources, block_prints = [], [], []
    minutes, initializations = [], []
    print_count = quantity = 0
    for at in range(start, end_us, GRID_US):
        if (at - start) % 3_600_000_000 == 600_000_000:
            initializations.append(_initialization(past_books, past_tape, at, start))
        block_points.append(_point(observation, start))
        block_sources.append((observation.segment_id, observation.invalid_count))
        end = at + GRID_US
        while pending is not None and pending.timestamp_us < end:
            if (not at <= pending.timestamp_us < end or pending.timestamp_us < previous_time
                    or pending.source_row <= previous_row):
                raise ValueError("Audited tape day/grid/order differs; no silent reassignment")
            previous_time, previous_row = pending.timestamp_us, pending.source_row
            past_tape.append(pending)
            block_prints.append(PublishedPrint(pending.timestamp_us // 1000, pending.side,
                                              pending.quantity_units, pending.price_units))
            print_count += 1
            quantity += pending.quantity_units
            pending = next_print()
        observation = cursor.observation(end)
        past_books.append(observation)
        while past_tape and past_tape[0].timestamp_us < end - 6_000_000:
            past_tape.popleft()
        if (end - start) % 60_000_000 == 0:
            points = (*block_points, _point(observation, start))
            block_start_ms = (end - 60_000_000) // 1000
            summary = summarize_block(points, tuple(block_prints), start_ms=block_start_ms)
            sources = (*block_sources, (observation.segment_id, observation.invalid_count))
            if len(set(sources)) != 1:
                summary = replace(summary, issues=(*summary.issues, "reset_or_gap_crossing"))
            minutes.append(summary)
            block_points, block_sources, block_prints = [], [], []
    price_blocks = []
    for i in range(0, len(minutes) - 4, 5):
        group = minutes[i:i + 5]
        issues = tuple(f"minute_{j}:{issue}" for j, block in enumerate(group)
                       for issue in block.issues)
        price_blocks.append(ObservedPriceBlock(group[0].start_ms * 1000,
                                              tuple(block.values[0] for block in group), issues))
    if len(minutes) != (end_us - start) // 60_000_000:
        raise RuntimeError("Observed target collector lost a fixed candidate minute")
    return DayTargets(day, (start, end_us), tuple(minutes), tuple(initializations),
                      tuple(price_blocks), print_count, quantity)


def collect_train_day(messages, prints, day):
    """Collect 1440 candidates/24 initializations; caller must verify raw provenance."""
    if day not in TRAIN_DAYS:
        raise ValueError("Development and event-test target access is closed")
    return _collect_span(messages, prints, day, day_start_us(day) + DAY_US)
