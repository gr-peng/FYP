"""Conservative descriptive profiles of audited training/development observations.

This module neither fits a model nor supplies a counterfactual execution simulator.
The sole disk-data entry point is ``crypto_features.read_feature_day``; the pure
frame functions also support small synthetic tests. All time comparisons use the
archive's exchange clocks, not a measured client receipt/decision clock.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import crypto_features

OPEN_SPLITS = {"train", "development"}
DEPTH_BPS = (1, 5, 10, 25)
DEPTH_LEVELS = (1, 5, 10, 50)
FLOW_COLUMNS = (
    "delta_messages",
    "positive_visible_depth_change_units",
    "negative_visible_depth_change_units",
)
TRADE_COLUMNS = (
    "trade_count",
    "buy_quantity_units",
    "sell_quantity_units",
    "quote_volume_product_units",
)

DEFINITIONS = {
    "population": "One exported end-of-second observed book state per covered second; "
    "equal weight per retained second, not event-time or whole-day weighting.",
    "depth": "BTC depth at N levels requires both sides to contain N levels. Bps depth "
    "and imbalance use only seconds whose visible outer prices cover the radius on "
    "BOTH sides. Excluded radii are censored, not zero depth. Coverage conditioning "
    "can select market states; retained and censored counts are reported.",
    "net_updates": "Positive/negative visible depth changes sum positive/negative "
    "magnitudes of per-price L2 quantity differences, across BOTH sides. Snapshots "
    "are excluded upstream. These are neither true order-add/cancel intensities "
    "nor directional order-flow imbalance; trades and visibility changes contribute.",
    "trade_zero_fill": "An absent trade second is set to zero only within exported "
    "full book seconds and full seconds strictly bounded by the first/last trade "
    "timestamps of a completely scanned, CRC-verified valid archive. No book state, "
    "price or missing book second is interpolated. Archive validity does not prove "
    "exchange-side trade completeness; zeros are conditional on that assumption.",
    "returns": "10000 * log(end midpoint / start midpoint), using the last observed "
    "quote in each second. All intervening seconds must exist in the same segment "
    "and UTC day. The 1s series uses consecutive second samples; the 60s series "
    "uses non-overlapping returns ending in UTC second 59. Actual quote timestamps "
    "need not be exactly 1000/60000 ms apart. Zero returns compare integer mid sums.",
    "autocorrelation": "Pearson correlation of pairs at exact calendar lag in the "
    "same uninterrupted run. Missing/constant/insufficient samples produce null, "
    "not zero. Pair counts are provided; no significance or stationarity claim.",
    "clustering": "Sample variance / mean of trade counts or delta-message counts "
    "in complete UTC-aligned 1s/10s/60s blocks, plus exact-lag correlations. Changes "
    "in activity over time can cause overdispersion without self-excitation.",
    "future_relationships": "Descriptive correlations, not a fitted prediction. "
    "For a signal available at A, the target starts at the first observed second "
    "whose bucket starts >= A and whose quote timestamp is > A; it ends 1 or 60 "
    "seconds later within the same uninterrupted run. Both target quotes are "
    "strictly after A. Available-at includes book prefix/confirmation publication "
    "and trade bucket end. Trade dissemination and client network latency are "
    "unobserved, so this is not proof of executable live predictability.",
    "scope": "Per-day results only; no cross-day, train/development, or event-test "
    "pooling. No passive queue fills, causal impact, liquidation rules or LLM "
    "realism are established. The old spot fixed-LP model is not a validated "
    "same-market perpetual baseline.",
}


def _open_split(split):
    if split not in OPEN_SPLITS:
        raise ValueError("Only train/development profiles are allowed; event_test is closed")


def summary(values):
    """JSON-safe finite-value summary; n excludes undefined observations."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if not n:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
            "zero_fraction": None,
        }
    return {
        "n": n,
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if n > 1 else None,
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "zero_fraction": float(np.mean(array == 0)),
    }


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        value, reason = None, "fewer_than_three_pairs"
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        value, reason = None, "constant_series"
    else:
        value, reason = float(np.corrcoef(x, y)[0, 1]), None
    return {"pairs": len(x), "pearson": value, "undefined_reason": reason}


def _require_frame(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    buckets = frame["bucket_ms"]
    if (
        buckets.isna().any()
        or not buckets.is_unique
        or not buckets.is_monotonic_increasing
        or (buckets % 1000 != 0).any()
    ):
        raise ValueError(f"{name} seconds must be unique, increasing and aligned")
    if (frame["available_at_ms"] < buckets + 1000).any():
        raise ValueError(f"{name} feature is available before its complete second")


def prepare_book(book, audit, day):
    """Check full-segment coverage, then identify runs without filling any holes."""
    columns = [
        "bucket_ms",
        "available_at_ms",
        "cts_ms",
        "segment",
        "bid_units",
        "ask_units",
        "spread_bps",
        "imbalance_l1",
        "visible_bid_levels",
        "visible_ask_levels",
        *FLOW_COLUMNS,
    ]
    columns += [f"{side}_depth_{n}_units" for side in ("bid", "ask") for n in DEPTH_LEVELS]
    columns += [
        f"{side}_{field}_{n}bps{suffix}"
        for side in ("bid", "ask")
        for n in DEPTH_BPS
        for field, suffix in (("depth", "_units"), ("covers", ""))
    ]
    _require_frame(book, columns, "Book")
    if not audit.get("features_published") or not audit.get("zip_crc_verified"):
        raise ValueError("Book requires published features and a CRC-verified archive")
    result = book.copy().reset_index(drop=True)
    start = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    ends = {s["segment"]: s["last_cts_ms"] for s in audit["segments"]}
    starts = {s["segment"]: s["first_cts_ms"] for s in audit["segments"]}
    invalid = (
        result.segment.map(starts).isna()
        | (result.bucket_ms < result.segment.map(starts))
        | (result.bucket_ms + 1000 > result.segment.map(ends))
        | (result.bucket_ms < start)
        | (result.available_at_ms >= start + 86_400_000)
        | (result.cts_ms < result.bucket_ms)
        | (result.cts_ms >= result.bucket_ms + 1000)
        | (result.bid_units <= 0)
        | (result.ask_units <= result.bid_units)
    )
    if invalid.any():
        raise ValueError("Book feature is outside a full valid segment/day or has invalid quotes")
    breaks = result.bucket_ms.diff().ne(1000) | result.segment.ne(result.segment.shift())
    result["run_id"] = breaks.cumsum().astype("int64")
    result["mid_twice_units"] = result.bid_units + result.ask_units
    return result


def covered_trade_seconds(book, trades, audit):
    """Attach conditional zero counts only to internally covered book seconds."""
    integrity = audit.get("gzip_integrity", {})
    if (
        audit.get("quality") != "valid"
        or not audit.get("records_scan_complete")
        or not audit.get("features_published")
        or integrity.get("status") != "passed"
        or not integrity.get("read_to_eof")
        or not integrity.get("crc_verified")
    ):
        raise ValueError("Trade zero filling requires a fully scanned valid CRC-verified tape")
    _require_frame(trades, ["bucket_ms", "available_at_ms", *TRADE_COLUMNS], "Trades")
    if (trades[list(TRADE_COLUMNS)] < 0).any().any() or (trades.trade_count <= 0).any():
        raise ValueError("Sparse trade features must represent positive counts")
    first, last = audit.get("first_timestamp_us"), audit.get("last_timestamp_us")
    if first is None or last is None:
        if len(trades):
            raise ValueError("Nonempty trade features require archive time bounds")
        lower, upper = 0, 0
    else:
        if first > last:
            raise ValueError("Invalid trade archive time bounds")
        lower = ((first + 999_999) // 1_000_000) * 1000
        upper = (last // 1_000_000) * 1000
    selected = book[(book.bucket_ms >= lower) & (book.bucket_ms + 1000 <= upper)].copy()
    frame = selected.merge(
        trades[["bucket_ms", "available_at_ms", *TRADE_COLUMNS]].rename(
            columns={"available_at_ms": "trade_available_at_ms"}
        ),
        on="bucket_ms",
        how="left",
        validate="one_to_one",
    )
    frame["zero_filled"] = frame.trade_count.isna()
    for column in TRADE_COLUMNS:
        frame[column] = frame[column].fillna(0)
    frame["available_at_ms"] = np.maximum(
        frame.available_at_ms,
        frame.trade_available_at_ms.fillna(frame.bucket_ms + 1000),
    )
    frame["signed_trade_quantity_btc"] = (
        frame.buy_quantity_units - frame.sell_quantity_units
    ) / 1000
    frame["trade_quantity_btc"] = (frame.buy_quantity_units + frame.sell_quantity_units) / 1000
    frame["quote_volume_usdt"] = frame.quote_volume_product_units / 100_000
    return frame, {
        "full_trade_interval_start_ms": lower,
        "full_trade_interval_end_ms": upper,
        "book_seconds_excluded_by_trade_boundaries": len(book) - len(frame),
        "joint_covered_seconds": len(frame),
        "zero_filled_seconds": int(frame.zero_filled.sum()),
    }


def contiguous_returns(book, horizon_seconds):
    """Rolling endpoint returns whose entire sampled path stays in one run."""
    if type(horizon_seconds) is not int or horizon_seconds <= 0:
        raise ValueError("Return horizon must be a positive integer number of seconds")
    prior = book.shift(horizon_seconds)
    keep = book.run_id.eq(prior.run_id) & book.bucket_ms.sub(prior.bucket_ms).eq(
        horizon_seconds * 1000
    )
    result = book.loc[keep, ["bucket_ms", "run_id"]].copy()
    result["return_bps"] = 10000 * np.log(
        book.loc[keep, "mid_twice_units"] / prior.loc[keep, "mid_twice_units"]
    )
    result["unchanged"] = book.loc[keep, "mid_twice_units"].eq(prior.loc[keep, "mid_twice_units"])
    return result


def lag_correlation(frame, column, lag_seconds, absolute=False):
    left = frame[["bucket_ms", "run_id", column]].copy()
    right = left.copy()
    right["bucket_ms"] += lag_seconds * 1000
    pairs = left.merge(right, on=["bucket_ms", "run_id"], suffixes=("_now", "_prior"))
    x, y = pairs[f"{column}_now"], pairs[f"{column}_prior"]
    return correlation(x.abs() if absolute else x, y.abs() if absolute else y)


def count_clustering(frame, column):
    blocks = {}
    for seconds in (1, 10, 60):
        groups = (
            frame.assign(block=frame.bucket_ms // (seconds * 1000))
            .groupby(["run_id", "block"])[column]
            .agg(["size", "sum"])
        )
        values = groups.loc[groups["size"] == seconds, "sum"]
        stats = summary(values)
        blocks[str(seconds)] = {
            "complete_blocks": len(values),
            "count_summary": stats,
            "variance_to_mean": float(values.var(ddof=1) / values.mean())
            if len(values) > 1 and values.mean() > 0
            else None,
        }
    return {
        "block_seconds": blocks,
        "autocorrelation": {str(lag): lag_correlation(frame, column, lag) for lag in (1, 5, 30)},
    }


def future_return_pairs(book, signals, horizon_seconds):
    """Return auditable target timestamps; never target pre-availability movement."""
    if type(horizon_seconds) is not int or horizon_seconds <= 0:
        raise ValueError("Return horizon must be a positive integer number of seconds")
    columns = [
        "signal_bucket_ms",
        "signal_available_at_ms",
        "target_start_bucket_ms",
        "target_end_bucket_ms",
        "target_start_cts_ms",
        "target_end_cts_ms",
        "target_return_bps",
        "signal_to_target_start_ms",
    ]
    if book.empty or signals.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="signal_index"))
    available = signals.available_at_ms.to_numpy()
    positions = np.searchsorted(book.bucket_ms.to_numpy(), available, side="left")
    safe = np.minimum(positions, len(book) - 1)
    # Even exact-boundary quotes must be strictly later than signal availability.
    positions += (book.cts_ms.to_numpy()[safe] <= available).astype(int)
    ends = positions + horizon_seconds
    inside = ends < len(book)
    safe_start, safe_end = np.minimum(positions, len(book) - 1), np.minimum(ends, len(book) - 1)
    runs = book.run_id.to_numpy()
    keep = (
        inside
        & (runs[safe_start] == signals.run_id.to_numpy())
        & (runs[safe_end] == signals.run_id.to_numpy())
    )
    start_rows, end_rows = book.iloc[positions[keep]], book.iloc[ends[keep]]
    result = pd.DataFrame(index=signals.index[keep])
    result.index.name = "signal_index"
    result["signal_bucket_ms"] = signals.loc[keep, "bucket_ms"]
    result["signal_available_at_ms"] = available[keep]
    for label, rows in (("start", start_rows), ("end", end_rows)):
        result[f"target_{label}_bucket_ms"] = rows.bucket_ms.to_numpy()
        result[f"target_{label}_cts_ms"] = rows.cts_ms.to_numpy()
    result["target_return_bps"] = 10000 * np.log(
        end_rows.mid_twice_units.to_numpy() / start_rows.mid_twice_units.to_numpy()
    )
    result["signal_to_target_start_ms"] = result.target_start_cts_ms - available[keep]
    return result[columns]


def _depth_profile(book, n, bps=False):
    if bps:
        keep = book[f"bid_covers_{n}bps"] & book[f"ask_covers_{n}bps"]
        suffix = f"{n}bps"
    else:
        keep = (book.visible_bid_levels >= n) & (book.visible_ask_levels >= n)
        suffix = str(n)
    bid, ask = (book.loc[keep, f"{side}_depth_{suffix}_units"] / 1000 for side in ("bid", "ask"))
    total = bid + ask
    return {
        "covered_seconds": int(keep.sum()),
        "censored_seconds": int((~keep).sum()),
        "covered_fraction": float(keep.mean()) if len(keep) else None,
        "bid_depth_btc": summary(bid),
        "ask_depth_btc": summary(ask),
        "imbalance": summary((bid - ask) / total.where(total > 0)),
        "covered_zero_total_depth_seconds": int((total == 0).sum()),
    }


def _return_profile(book):
    result = {}
    for horizon, lags in ((1, (1, 5, 30)), (60, (60, 300))):
        frame = contiguous_returns(book, horizon)
        if horizon == 60:
            frame = frame[frame.bucket_ms % 60000 == 59000]
        result[str(horizon)] = {
            "log_return_bps": summary(frame.return_bps),
            "unchanged_mid_fraction": float(frame.unchanged.mean()) if len(frame) else None,
            "autocorrelation": {
                str(lag): lag_correlation(frame, "return_bps", lag) for lag in lags
            },
            "absolute_return_autocorrelation": {
                str(lag): lag_correlation(frame, "return_bps", lag, absolute=True) for lag in lags
            },
        }
    return result


def _future_relationships(book, signals, columns):
    result = {}
    for horizon in (1, 60):
        pairs = future_return_pairs(book, signals, horizon)
        result[str(horizon)] = {
            "clock_assumption": "Ideal exchange publication / zero client-network latency; "
            "trade bucket-end availability is a lower bound, not observed publication. "
            "Book ts is server publication, not client receipt.",
            "inference_limit": "Targets can overlap or repeat; pairs are not independent "
            "trials. No p-values, causal or tradable prediction claim.",
            "candidate_signal_seconds": len(signals),
            "eligible_pairs": len(pairs),
            "distinct_target_windows": len(
                pairs[["target_start_bucket_ms", "target_end_bucket_ms"]].drop_duplicates()
            ),
            "ineligible_signal_seconds": len(signals) - len(pairs),
            "signal_to_target_start_ms": summary(pairs.signal_to_target_start_ms),
            "associations": {
                column: {
                    "signed_return": correlation(
                        signals.loc[pairs.index, column], pairs.target_return_bps
                    ),
                    "absolute_return": correlation(
                        signals.loc[pairs.index, column], pairs.target_return_bps.abs()
                    ),
                }
                for column in columns
            },
        }
    return result


def profile_day(book, trades, book_audit, trade_audit, *, split, day):
    _open_split(split)
    book = prepare_book(book, book_audit, day)
    joint, coverage = covered_trade_seconds(book, trades, trade_audit)
    runs = book.groupby("run_id").size()
    return {
        "split": split,
        "day": day,
        "coverage": {
            "book_seconds": len(book),
            "uninterrupted_runs": len(runs),
            "run_length_seconds": summary(runs),
            **coverage,
        },
        "spread_bps": summary(book.spread_bps),
        "imbalance_l1": summary(book.imbalance_l1),
        "state_age_at_bucket_end_ms": summary(book.bucket_ms + 1000 - book.cts_ms),
        "book_availability_after_bucket_end_ms": summary(
            book.available_at_ms - book.bucket_ms - 1000
        ),
        "depth_levels": {str(n): _depth_profile(book, n) for n in DEPTH_LEVELS},
        "depth_bps": {str(n): _depth_profile(book, n, bps=True) for n in DEPTH_BPS},
        "observed_depth_updates": {
            "delta_messages_per_second": summary(book.delta_messages),
            "visible_increase_btc_per_second": summary(
                book.positive_visible_depth_change_units / 1000
            ),
            "visible_decrease_btc_per_second": summary(
                book.negative_visible_depth_change_units / 1000
            ),
            "delta_message_clustering": count_clustering(book, "delta_messages"),
        },
        "trades_within_joint_coverage": {
            column: summary(joint[column])
            for column in (
                "trade_count",
                "trade_quantity_btc",
                "signed_trade_quantity_btc",
                "quote_volume_usdt",
            )
        },
        "trade_count_clustering": count_clustering(joint, "trade_count"),
        "return_horizon_seconds": _return_profile(book),
        "book_signal_future_return_horizon_seconds": _future_relationships(
            book, book, ["imbalance_l1", *FLOW_COLUMNS]
        ),
        "trade_signal_future_return_horizon_seconds": _future_relationships(
            book, joint, ["trade_count", "signed_trade_quantity_btc", "trade_quantity_btc"]
        ),
    }


def profile_study(root: Path, config_path: Path, split: str, days=None):
    """Read authorized features only; never open event-test data or audit reports."""
    _open_split(split)  # Gate before even opening the config.
    root, config_path = Path(root), Path(config_path)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    selected = list(config["splits"][split] if days is None else days)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(day not in config["splits"][split] for day in selected)
    ):
        raise ValueError("Requested days must be unique and belong to the frozen selected split")
    profiles, sources = [], []
    for day in selected:
        book, book_audit = crypto_features.read_feature_day(
            root, config_path, split, day, "orderbook"
        )
        trades, trade_audit = crypto_features.read_feature_day(
            root, config_path, split, day, "trades"
        )
        profiles.append(profile_day(book, trades, book_audit, trade_audit, split=split, day=day))
        for kind, audit in (("orderbook", book_audit), ("trades", trade_audit)):
            sources.append(
                {
                    "day": day,
                    "kind": kind,
                    **{
                        key: audit[key]
                        for key in (
                            "audit_run_id",
                            "input_sha256",
                            "features_path",
                            "features_sha256",
                            "processing_source_sha256",
                        )
                    },
                }
            )
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "split": split,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "profile_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reader_source_sha256": hashlib.sha256(
            Path(crypto_features.__file__).read_bytes()
        ).hexdigest(),
        "definitions": DEFINITIONS,
        "sources": sources,
        "days": profiles,
        "next_experiments": [
            "Fit a simple activity baseline on training days only, then a small "
            "state-conditioned count/quantity model using available spread, covered "
            "depth and imbalance. Freeze bins/parameters before development scoring.",
            "Compare development count dispersion/autocorrelation, depth/coverage "
            "distributions, return zero rates and absolute-return correlation jointly. "
            "Only add self-excitation if it improves frozen development likelihood "
            "and these joint diagnostics over the simpler baseline.",
            "Treat a generated L2-change stream as observable depth dynamics, not "
            "identified individual-order arrivals/cancellations. Keep matching, "
            "accounting and reaction/impact assumptions explicit; let the LLM issue "
            "sparse high-level intent only after these components are validated.",
        ],
    }
