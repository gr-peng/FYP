"""Pure, fixed scoring for the five-minute effective-flow prototype.

Inputs are immutable copied points and actually published prints, never readers
or generator callbacks. Visible 5 bps depth is measured inside the supplied top
500 view; band coverage is reported separately and unseen depth is not imputed.
These are per-window diagnostics. Seeds and blocks are not independent trading
days; this module has no overall-date aggregation, pass threshold or selection.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real

import numpy as np

PRICE_SCALE = 100
QUANTITY_SCALE = 1000
GRID_MS = 100
BLOCK_MS = 60_000
BLOCKS_PER_WINDOW = 5
LEFT_POINTS_PER_BLOCK = 600
ENSEMBLE_SIZE = 20
DEFAULT_SEEDS = tuple(range(ENSEMBLE_SIZE))
TRAIN_DAYS = ("2024-11-01", "2024-11-02", "2024-11-03")
DIMENSIONS = (
    "return_bps",
    "volume_btc",
    "published_print_count",
    "mean_spread_bps",
    "mean_visible_bid_depth_5bps_btc",
    "mean_visible_ask_depth_5bps_btc",
    "quantity_pressure",
)
TRAIN_STARTS = tuple(
    int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000) + minute * BLOCK_MS
    for day in TRAIN_DAYS
    for minute in range(1440)
)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _time(value):
    if not _integer(value) or value < 0:
        raise ValueError("Use nonnegative integer milliseconds")


def _tuple(value, name, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class BookPoint:
    """Observed top-500 view at one clock point, with exact stored units."""

    timestamp_ms: int
    bids: tuple[tuple[int, int], ...] | None
    asks: tuple[tuple[int, int], ...] | None
    is_valid: bool = True
    failure_reason: str | None = None

    def __post_init__(self):
        _time(self.timestamp_ms)
        _tuple(self.bids, "bids", optional=True)
        _tuple(self.asks, "asks", optional=True)
        for side in (self.bids, self.asks):
            if side is not None and any(type(level) is not tuple for level in side):
                raise TypeError("Price/quantity levels must be immutable tuples")
        if type(self.is_valid) is not bool:
            raise TypeError("is_valid must be bool")

    @property
    def mid_twice_units(self):
        values = _point_values(self)
        return None if values is None else values[0]

    @property
    def mid_price_usdt(self):
        value = self.mid_twice_units
        return None if value is None else value / (2 * PRICE_SCALE)


@dataclass(frozen=True, slots=True)
class PublishedPrint:
    timestamp_ms: int
    side: str
    quantity_units: int
    price_units: int

    def __post_init__(self):
        _time(self.timestamp_ms)


@dataclass(frozen=True, slots=True)
class SimulationPath:
    seed: int
    points: tuple[BookPoint, ...] | None
    prints: tuple[PublishedPrint, ...] | None

    def __post_init__(self):
        if not _integer(self.seed):
            raise ValueError("Seed must be an integer")
        _tuple(self.points, "points", optional=True)
        _tuple(self.prints, "prints", optional=True)


@dataclass(frozen=True, slots=True)
class BlockSummary:
    start_ms: int
    values: tuple[float | None, ...]
    issues: tuple[str, ...] = ()
    valid_left_points: int | None = None
    bid_5bps_coverage_fraction: float | None = None
    ask_5bps_coverage_fraction: float | None = None

    def __post_init__(self):
        _time(self.start_ms)
        if self.start_ms % BLOCK_MS:
            raise ValueError("Block starts must be UTC minute aligned")
        _tuple(self.values, "values")
        _tuple(self.issues, "issues")
        if len(self.values) != len(DIMENSIONS):
            raise ValueError("Require the fixed seven-dimensional order")

    @property
    def complete(self):
        # Nonfinite claimed values are not silently treated as absent labels.
        return not self.issues and all(value is not None for value in self.values)


@dataclass(frozen=True, slots=True)
class WindowSummary:
    start_ms: int
    blocks: tuple[BlockSummary, ...]

    def __post_init__(self):
        _time(self.start_ms)
        _tuple(self.blocks, "blocks")
        if len(self.blocks) != BLOCKS_PER_WINDOW or any(
            type(block) is not BlockSummary or block.start_ms != self.start_ms + i * BLOCK_MS
            for i, block in enumerate(self.blocks)
        ):
            raise ValueError("Require all five consecutive fixed 60-second blocks")


@dataclass(frozen=True, slots=True)
class FrozenMetrics:
    scales: tuple[float, ...]
    training_days: tuple[str, ...] = TRAIN_DAYS
    training_candidate_labels: int = len(TRAIN_STARTS)
    training_complete_labels: int = len(TRAIN_STARTS)

    def __post_init__(self):
        _tuple(self.scales, "scales")
        if (
            len(self.scales) != len(DIMENSIONS)
            or not all(_finite(value) and value > 0 for value in self.scales)
            or self.training_days != TRAIN_DAYS
            or not _integer(self.training_candidate_labels)
            or not _integer(self.training_complete_labels)
            or self.training_candidate_labels != len(TRAIN_STARTS)
            or not 2 <= self.training_complete_labels <= len(TRAIN_STARTS)
        ):
            raise ValueError("Invalid fixed training scale or calendar metadata")


@dataclass(frozen=True, slots=True)
class PricePath:
    """P may supply only five returns; it has no seven-dimensional interface."""

    seed: int
    returns_bps: tuple[float | None, ...]

    def __post_init__(self):
        if not _integer(self.seed):
            raise ValueError("Seed must be an integer")
        _tuple(self.returns_bps, "returns_bps")
        if len(self.returns_bps) != BLOCKS_PER_WINDOW:
            raise ValueError("Require all five price returns, including missing ones")


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def transform_vector(values):
    """Fixed float64 transform; no centering or fit occurs here."""
    if len(values) != len(DIMENSIONS) or not all(_finite(v) for v in values):
        raise ValueError("Seven finite raw values are required")
    data = np.array(values, dtype=np.float64, copy=True)
    if (data[1:6] < 0).any() or abs(data[6]) > 1 or data[2] != math.floor(data[2]):
        raise ValueError("Invalid volume/count/spread/depth/pressure")
    data[1:6] = np.log1p(data[1:6])
    return data


def fit_metrics(training_labels, *, split):
    """Train only, all 4,320 UTC minute candidates; incomplete labels stay counted.

    The later provenance pipeline must prove that these supplied labels came
    from every approved source minute. This pure function cannot prove origins.
    """
    if split != "train":
        raise ValueError("Development/event/unknown splits cannot fit metrics")
    _tuple(training_labels, "training_labels")
    expected = set(TRAIN_STARTS)
    starts = []
    for label in training_labels:
        if type(label) is not BlockSummary or label.start_ms not in expected:
            raise ValueError("Only November 1-3 UTC minute training labels are allowed")
        starts.append(label.start_ms)
    if len(starts) != len(expected) or set(starts) != expected:
        raise ValueError("Supply the complete candidate calendar; do not select windows")
    complete = [transform_vector(label.values) for label in training_labels if label.complete]
    if len(complete) < 2:
        raise ValueError("Not enough complete training labels")
    with np.errstate(over="ignore", invalid="ignore"):
        scales = np.asarray(complete, dtype=np.float64).std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Nonfinite or zero population training standard deviation")
    return FrozenMetrics(
        tuple(float(value) for value in scales), training_complete_labels=len(complete)
    )


def _point_values(point):
    if point is None or not point.is_valid:
        return None
    for name, levels in (("bid", point.bids), ("ask", point.asks)):
        if (
            not levels
            or len(levels) > 500
            or any(
                len(level) != 2 or not all(_integer(value) and value > 0 for value in level)
                for level in levels
            )
        ):
            return None
        prices = [level[0] for level in levels]
        if any(
            (a <= b if name == "bid" else a >= b) for a, b in zip(prices, prices[1:], strict=False)
        ):
            return None
    bid, ask = int(point.bids[0][0]), int(point.asks[0][0])
    if bid >= ask:
        return None
    mid_twice = bid + ask  # The half storage unit is never rounded away.
    try:
        spread = 20_000 * (ask - bid) / mid_twice
        bid_depth = (
            sum(int(q) for p, q in point.bids if 20_000 * int(p) >= 9995 * mid_twice)
            / QUANTITY_SCALE
        )
        ask_depth = (
            sum(int(q) for p, q in point.asks if 20_000 * int(p) <= 10005 * mid_twice)
            / QUANTITY_SCALE
        )
        if not all(math.isfinite(v) for v in (spread, bid_depth, ask_depth)):
            return None
    except (OverflowError, ZeroDivisionError):
        return None
    return (
        mid_twice,
        spread,
        bid_depth,
        ask_depth,
        20_000 * int(point.bids[-1][0]) <= 9995 * mid_twice,
        20_000 * int(point.asks[-1][0]) >= 10005 * mid_twice,
    )


def _point_index(points):
    _tuple(points, "points", optional=True)
    index, duplicates = {}, set()
    if points is not None:
        for point in points:
            if type(point) is not BookPoint:
                raise TypeError("Bridge must copy into this module's BookPoint")
            if point.timestamp_ms in index:
                duplicates.add(point.timestamp_ms)
            index[point.timestamp_ms] = point
    return index, duplicates


def _return_bps(start, end):
    if start is None or end is None:
        return None
    change = end[0] - start[0]
    if 2 * abs(change) < start[0]:
        value = 10_000 * math.log1p(change / start[0])
    else:
        value = 10_000 * (math.log(end[0]) - math.log(start[0]))
    return value if math.isfinite(value) else None


def _block(index, duplicates, prints, start_ms):
    end_ms = start_ms + BLOCK_MS
    times = tuple(range(start_ms, end_ms, GRID_MS))
    measures = [None if t in duplicates else _point_values(index.get(t)) for t in times]
    end = None if end_ms in duplicates else _point_values(index.get(end_ms))
    valid = [value for value in measures if value is not None]
    issues = []
    if len(valid) != LEFT_POINTS_PER_BLOCK:
        issues.append("missing_invalid_or_duplicate_left_point")
    if end is None:
        issues.append("missing_invalid_or_duplicate_return_endpoint")
    # Off-grid observations cannot be averaged as if they were fixed clock points.
    if any(start_ms <= t <= end_ms and (t - start_ms) % GRID_MS for t in index):
        issues.append("off_grid_point")
    means = (None, None, None)
    if len(valid) == LEFT_POINTS_PER_BLOCK:
        means = tuple(math.fsum(row[i] / LEFT_POINTS_PER_BLOCK for row in valid) for i in (1, 2, 3))
    volume = count = pressure = None
    if prints is None:
        issues.append("missing_print_stream")
    else:
        _tuple(prints, "prints")
        if any(type(row) is not PublishedPrint for row in prints):
            raise TypeError("Bridge must copy into this module's PublishedPrint")
        selected = [row for row in prints if start_ms <= row.timestamp_ms < end_ms]
        if any(a.timestamp_ms > b.timestamp_ms for a, b in zip(prints, prints[1:], strict=False)):
            issues.append("nonmonotone_print_stream")
        if any(
            row.side not in {"Buy", "Sell"}
            or not _integer(row.quantity_units)
            or row.quantity_units <= 0
            or not _integer(row.price_units)
            or row.price_units <= 0
            for row in selected
        ):
            issues.append("invalid_print")
        else:
            buy = sum(int(row.quantity_units) for row in selected if row.side == "Buy")
            sell = sum(int(row.quantity_units) for row in selected if row.side == "Sell")
            try:
                volume, count = (buy + sell) / QUANTITY_SCALE, float(len(selected))
                pressure = (buy - sell) / (buy + sell) if buy + sell else 0.0
                if not math.isfinite(volume):
                    volume = count = pressure = None
                    issues.append("nonfinite_print_quantity")
            except OverflowError:
                volume = count = pressure = None
                issues.append("nonfinite_print_quantity")
    values = (_return_bps(measures[0], end), volume, count, *means, pressure)
    return BlockSummary(
        start_ms,
        values,
        tuple(issues),
        len(valid),
        sum(row[4] for row in valid) / LEFT_POINTS_PER_BLOCK
        if len(valid) == LEFT_POINTS_PER_BLOCK
        else None,
        sum(row[5] for row in valid) / LEFT_POINTS_PER_BLOCK
        if len(valid) == LEFT_POINTS_PER_BLOCK
        else None,
    )


def summarize_block(points, prints, *, start_ms):
    _time(start_ms)
    if start_ms % BLOCK_MS:
        raise ValueError("Use a UTC-aligned 60-second block")
    index, duplicates = _point_index(points)
    return _block(index, duplicates, prints, start_ms)


def summarize_window(points, prints, *, start_ms):
    _time(start_ms)
    if start_ms % BLOCK_MS:
        raise ValueError("Use a UTC minute aligned five-minute window")
    index, duplicates = _point_index(points)
    return WindowSummary(
        start_ms,
        tuple(_block(index, duplicates, prints, start_ms + i * BLOCK_MS) for i in range(5)),
    )


def fair_energy_score(predictions, target):
    """Fixed K=20 fair ES. The pair denominator is K(K-1), not K squared."""
    data, actual = np.asarray(predictions, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (
        data.ndim != 2
        or data.shape[0] != ENSEMBLE_SIZE
        or data.shape[1] not in (1, 7)
        or actual.shape != (data.shape[1],)
    ):
        raise ValueError("Require exactly 20 finite one- or seven-dimensional predictions")
    if not np.isfinite(data).all() or not np.isfinite(actual).all():
        raise ValueError("Energy score inputs must be finite")
    first = math.fsum(math.dist(row, actual) for row in data) / ENSEMBLE_SIZE
    second = math.fsum(math.dist(data[k], data[j]) for k in range(20) for j in range(k)) / (20 * 19)
    value = first - second
    if not math.isfinite(value):
        raise ValueError("Nonfinite energy score")
    return value


def _interval(values, observed):
    finite = [_finite(value) for value in values]
    report = {
        "expected_predictions": 20,
        "valid_predictions": sum(finite),
        "lower": None,
        "upper": None,
        "covered": None,
    }
    if len(values) == 20 and all(finite) and _finite(observed):
        low, high = np.quantile(
            np.asarray(values, dtype=np.float64), (0.05, 0.95), method="inverted_cdf"
        )
        report.update(lower=float(low), upper=float(high), covered=bool(low <= observed <= high))
    return report


def _seed_index(paths, expected_seeds, path_type):
    _tuple(paths, "paths")
    _tuple(expected_seeds, "expected_seeds")
    if (
        len(expected_seeds) != 20
        or len(set(expected_seeds)) != 20
        or not all(_integer(seed) for seed in expected_seeds)
    ):
        raise ValueError("Freeze exactly 20 unique expected seed IDs")
    if any(type(path) is not path_type for path in paths):
        raise TypeError(f"This entry accepts only {path_type.__name__}")
    index, duplicate = {}, set()
    for path in paths:
        if path.seed in index:
            duplicate.add(path.seed)
        index[path.seed] = path
    unexpected = set(index) - set(expected_seeds)
    exact = len(paths) == 20 and set(index) == set(expected_seeds) and not duplicate
    return index, exact, tuple(sorted(duplicate)), tuple(sorted(unexpected))


def _scores(metrics, observed, values, *, exact, complete_flags):
    if type(metrics) is not FrozenMetrics:
        raise TypeError("Use frozen training scales")
    main = {"complete": False, "energy_score": None, "numerical_issue": None}
    price = {"complete": False, "energy_score": None, "numerical_issue": None}
    if exact and observed.complete and all(complete_flags):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                z = np.asarray([transform_vector(row) for row in values]) / metrics.scales
                y = transform_vector(observed.values) / metrics.scales
            main.update(complete=True, energy_score=fair_energy_score(z, y))
        except (ValueError, OverflowError) as exc:
            main["numerical_issue"] = str(exc)
    returns = [row[0] for row in values]
    if exact and all(_finite(value) for value in returns) and _finite(observed.values[0]):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                z = np.asarray(returns, dtype=np.float64)[:, None] / metrics.scales[0]
                y = np.array([observed.values[0]], dtype=np.float64) / metrics.scales[0]
            price.update(complete=True, energy_score=fair_energy_score(z, y))
        except (ValueError, OverflowError) as exc:
            price["numerical_issue"] = str(exc)
    return main, price


def score_window(metrics, observed, paths, *, expected_seeds=DEFAULT_SEEDS):
    """All 20 seeds and five blocks are retained, including failed paths."""
    if type(observed) is not WindowSummary:
        raise TypeError("Use a fixed observed WindowSummary")
    index, exact, duplicates, unexpected = _seed_index(paths, expected_seeds, SimulationPath)
    summaries = {
        seed: summarize_window(path.points, path.prints, start_ms=observed.start_ms)
        for seed, path in index.items()
    }
    blocks, failed = [], set()
    for i, actual in enumerate(observed.blocks):
        predictions, flags, failures = [], [], []
        for seed in expected_seeds:
            summary = summaries.get(seed)
            block = None if summary is None or seed in duplicates else summary.blocks[i]
            values = (None,) * 7 if block is None else block.values
            complete = block is not None and block.complete
            predictions.append(values)
            flags.append(complete)
            if not complete:
                failed.add(seed)
                failures.append(
                    {
                        "seed": int(seed),
                        "issues": ("missing_or_duplicate_seed",) if block is None else block.issues,
                        "values": values,
                    }
                )
        main, price = _scores(metrics, actual, predictions, exact=exact, complete_flags=flags)
        blocks.append(
            {
                "start_ms": actual.start_ms,
                "expected_predictions": 20,
                "complete_predictions": sum(flags),
                "observed_complete": actual.complete,
                "observed_issues": actual.issues,
                "joint": main,
                "price": price,
                "failed_seeds": tuple(failures),
                "intervals_90": {
                    name: _interval([row[j] for row in predictions], actual.values[j])
                    if exact
                    else _interval([], actual.values[j])
                    for j, name in enumerate(DIMENSIONS)
                },
            }
        )
    complete = all(block["joint"]["complete"] for block in blocks)
    price_complete = all(block["price"]["complete"] for block in blocks)
    return {
        "start_ms": observed.start_ms,
        "status": "complete_diagnostic" if complete else "not_completely_evaluable",
        "expected_seed_ids": expected_seeds,
        "received_seed_ids": tuple(int(path.seed) for path in paths),
        "expected_paths": 20,
        "expected_blocks": 5,
        "expected_path_blocks": 100,
        "incomplete_expected_paths": len(failed),
        "duplicate_seed_ids": duplicates,
        "unexpected_seed_ids": unexpected,
        "complete": complete,
        "energy_score": math.fsum(block["joint"]["energy_score"] for block in blocks) / 5
        if complete
        else None,
        "price_complete": price_complete,
        "price_energy_score": math.fsum(block["price"]["energy_score"] for block in blocks) / 5
        if price_complete
        else None,
        "blocks": tuple(blocks),
        "interpretation": (
            "Five blocks equally weighted; incomplete cases have no replacement penalty. "
            "No whole-date generalization, independence claim, pass threshold or winner selection."
        ),
    }


def score_price_window(metrics, observed, paths, *, expected_seeds=DEFAULT_SEEDS):
    """Price-only P input cannot be routed into the seven-dimensional score."""
    if type(observed) is not WindowSummary or type(metrics) is not FrozenMetrics:
        raise TypeError("Use frozen scales and a fixed observed window")
    index, exact, duplicates, unexpected = _seed_index(paths, expected_seeds, PricePath)
    blocks = []
    for i, actual in enumerate(observed.blocks):
        returns = [
            None if seed not in index or seed in duplicates else index[seed].returns_bps[i]
            for seed in expected_seeds
        ]
        _, price = _scores(
            metrics,
            actual,
            [(value, None, None, None, None, None, None) for value in returns],
            exact=exact,
            complete_flags=[False] * 20,
        )
        blocks.append(
            {
                "start_ms": actual.start_ms,
                **price,
                "interval_90": _interval(returns, actual.values[0])
                if exact
                else _interval([], actual.values[0]),
            }
        )
    complete = all(block["complete"] for block in blocks)
    return {
        "start_ms": observed.start_ms,
        "scope": "price_only",
        "complete": complete,
        "status": "complete_diagnostic" if complete else "not_completely_evaluable",
        "expected_seed_ids": expected_seeds,
        "received_seed_ids": tuple(int(path.seed) for path in paths),
        "expected_paths": 20,
        "expected_blocks": 5,
        "duplicate_seed_ids": duplicates,
        "unexpected_seed_ids": unexpected,
        "price_energy_score": math.fsum(block["energy_score"] for block in blocks) / 5
        if complete
        else None,
        "blocks": tuple(blocks),
    }
