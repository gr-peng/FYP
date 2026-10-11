"""A frozen NB2 activity probe, not a generative market or execution simulator.

Model A uses deterministic UTC clock and latest known completed trade activity.
Model B adds known book state. Both fit and score on identical eligible rows.
All standardization and dispersion estimates use training data only.
"""

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln

from experiments import crypto_features, crypto_profile

FIXED_SPLITS = {
    "train": ["2024-11-01", "2024-11-02", "2024-11-03"],
    "development": ["2024-11-04"],
}
DELAYS_MS = (0, 1000, 5000)
BASELINE_FEATURES = (
    "utc_sin",
    "utc_cos",
    "log1p_last_trade_count",
    "log1p_last_trade_quantity_btc",
)
BOOK_FEATURES = (
    "log1p_spread_bps",
    "imbalance_l1",
    "log1p_bid_depth_5_btc",
    "log1p_ask_depth_5_btc",
)
SAMPLE_COLUMNS = (
    "split",
    "day",
    "run_id",
    "signal_bucket_ms",
    "signal_available_at_ms",
    "decision_at_ms",
    "activity_bucket_ms",
    "activity_available_at_ms",
    "delayed_activity_available_at_ms",
    "target_bucket_ms",
    "target_end_ms",
    "target_count",
    *BASELINE_FEATURES,
    *BOOK_FEATURES,
)
ASSUMPTIONS = {
    "target": "First complete UTC second starting at or after signal available_at plus "
    "the prescribed delay. An exact-boundary decision may target the second beginning "
    "there. Signal, last known activity and target must be in one uninterrupted joint "
    "book/trade-coverage run in one day. No interpolation or gap crossing.",
    "information": "Activity is the latest fully observed trade second whose ideal "
    "bucket-end availability is <= signal available_at. Both activity and book inputs "
    "receive the same extra delay; no newer information is added during that delay. "
    "UTC features describe the already known target clock, not future market state.",
    "clock": "Book ts is server publication, not client receipt. Trade CSV records "
    "matching time only; bucket-end availability is an ideal publication lower bound. "
    "Additional 0/1000/5000 ms delays are assumptions, not measured latency.",
    "sampling": "Only signals with five visible levels on each side are retained, "
    "for BOTH models. Duplicate target seconds keep the latest decision, then latest "
    "signal bucket. This uses timestamps/coverage, never target values. Targets are "
    "unique non-overlapping seconds within each delay scenario; serial dependence "
    "remains. Across scenarios samples can differ and must be evaluated separately.",
    "zeros": crypto_profile.DEFINITIONS["trade_zero_fill"],
    "model": "NB2 mean exp(intercept + standardized features @ slopes), variance "
    "mean + alpha * mean^2. One train-fitted dispersion per model/scenario; fixed "
    "slope L2 penalty on mean negative log likelihood. Intercept/dispersion unpenalized. "
    "NB2 is a marginal conditional count approximation, not identified order arrivals "
    "or self-excitation. Numerical dispersion bounds and optimizer status are recorded.",
    "evaluation": "Higher mean count log probability is better (proper log score); "
    "MAE is secondary. Per-day/three-hour calibration uses the same rows for both "
    "models. No significance intervals from one development day, no causal or "
    "tradable-prediction conclusion. Development was previously descriptively profiled; "
    "this is a development diagnostic, not untouched final evaluation.",
}


class FitError(RuntimeError):
    """Optimization did not meet the fixed acceptance criteria; no fallback model."""


def _selection(split, day):
    if split not in FIXED_SPLITS or day not in FIXED_SPLITS[split]:
        raise ValueError(
            "Only the fixed November train/development days are allowed; test is closed"
        )


def _finite(values, name):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _counts(values):
    result = _finite(values, "Counts")
    if (result < 0).any() or (result != np.floor(result)).any():
        raise ValueError("Counts must be nonnegative integers")
    return result


def build_activity_samples(joint, *, split, day, delay_ms):
    """Build common-sample pairs from profile.covered_trade_seconds output only."""
    _selection(split, day)
    if type(delay_ms) is not int or delay_ms not in DELAYS_MS:
        raise ValueError("Delay must be one of the frozen 0/1000/5000 ms scenarios")
    work = joint.reset_index(drop=True).copy()
    n = len(work)
    coverage = {
        "joint_seconds": n,
        "depth_rejected_seconds": 0,
        "depth_rejected_fraction": None if not n else 0.0,
        "missing_target_seconds": 0,
        "broken_path_seconds": 0,
        "duplicate_target_signals_removed": 0,
        "samples": 0,
    }
    if work.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS), coverage
    numeric = [
        "bucket_ms",
        "available_at_ms",
        "run_id",
        "trade_count",
        "trade_quantity_btc",
        "spread_bps",
        "imbalance_l1",
        "visible_bid_levels",
        "visible_ask_levels",
        "bid_depth_5_units",
        "ask_depth_5_units",
    ]
    _finite(work[numeric], "Joint features")
    _counts(work.trade_count)
    buckets = work.bucket_ms.to_numpy()
    available = work.available_at_ms.to_numpy()
    day_start = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    if (buckets < day_start).any() or (buckets + 1000 > day_start + 86_400_000).any():
        raise ValueError("Joint buckets must belong to the stated UTC day")
    if (np.diff(buckets) <= 0).any() or (buckets % 1000 != 0).any():
        raise ValueError("Joint seconds must be unique, sorted and UTC-second aligned")
    if (available < buckets + 1000).any() or (available != np.floor(available)).any():
        raise ValueError("Signal availability must follow its complete second")
    if (
        work[["trade_quantity_btc", "spread_bps", "bid_depth_5_units", "ask_depth_5_units"]] < 0
    ).any().any() or (work.imbalance_l1.abs() > 1).any():
        raise ValueError("Invalid nonnegative activity/depth or book imbalance")
    trade_available = work.trade_available_at_ms.fillna(work.bucket_ms + 1000).to_numpy()
    _finite(trade_available, "Trade availability")
    if (
        (trade_available < buckets + 1000).any()
        or (np.diff(trade_available) < 0).any()
        or (trade_available != np.floor(trade_available)).any()
    ):
        raise ValueError("Trade availability must follow complete seconds and be monotone")
    runs = ((work.bucket_ms.diff() != 1000) | (work.run_id != work.run_id.shift())).cumsum()
    run_values = runs.to_numpy()
    # Common latency affects every feature. It does not permit refreshing the
    # baseline from trades observed during that additional latency interval.
    activity_positions = np.searchsorted(trade_available, available, side="right") - 1
    decisions = available.astype("int64") + delay_ms
    target_buckets = ((decisions + 999) // 1000) * 1000
    targets = pd.Index(buckets).get_indexer(target_buckets)
    safe_activity, safe_targets = np.maximum(activity_positions, 0), np.maximum(targets, 0)
    depth_ok = ((work.visible_bid_levels >= 5) & (work.visible_ask_levels >= 5)).to_numpy()
    target_exists = targets >= 0
    same_run = (
        (activity_positions >= 0)
        & (run_values[safe_targets] == run_values)
        & (run_values[safe_activity] == run_values)
    )
    keep = depth_ok & target_exists & same_run
    coverage.update(
        depth_rejected_seconds=int((~depth_ok).sum()),
        depth_rejected_fraction=float((~depth_ok).mean()),
        missing_target_seconds=int((depth_ok & ~target_exists).sum()),
        broken_path_seconds=int((depth_ok & target_exists & ~same_run).sum()),
    )
    signal, activity, target = (
        work.loc[keep],
        work.iloc[activity_positions[keep]],
        work.iloc[targets[keep]],
    )
    samples = pd.DataFrame(
        {
            "split": split,
            "day": day,
            "run_id": run_values[keep],
            "signal_bucket_ms": signal.bucket_ms.to_numpy(),
            "signal_available_at_ms": available[keep].astype("int64"),
            "decision_at_ms": decisions[keep],
            "activity_bucket_ms": activity.bucket_ms.to_numpy(),
            "activity_available_at_ms": trade_available[activity_positions[keep]].astype("int64"),
            "delayed_activity_available_at_ms": (
                trade_available[activity_positions[keep]].astype("int64") + delay_ms
            ),
            "target_bucket_ms": target_buckets[keep],
            "target_end_ms": target_buckets[keep] + 1000,
            "target_count": target.trade_count.to_numpy().astype("int64"),
        }
    )
    angle = 2 * np.pi * (samples.target_bucket_ms % 86_400_000) / 86_400_000
    samples["utc_sin"], samples["utc_cos"] = np.sin(angle), np.cos(angle)
    samples["log1p_last_trade_count"] = np.log1p(activity.trade_count.to_numpy())
    samples["log1p_last_trade_quantity_btc"] = np.log1p(activity.trade_quantity_btc.to_numpy())
    samples["log1p_spread_bps"] = np.log1p(signal.spread_bps.to_numpy())
    samples["imbalance_l1"] = signal.imbalance_l1.to_numpy()
    for side in ("bid", "ask"):
        samples[f"log1p_{side}_depth_5_btc"] = np.log1p(
            signal[f"{side}_depth_5_units"].to_numpy() / 1000
        )
    before = len(samples)
    samples = (
        samples.sort_values(
            ["target_bucket_ms", "decision_at_ms", "signal_bucket_ms"], kind="stable"
        )
        .drop_duplicates("target_bucket_ms", keep="last")
        .reset_index(drop=True)
    )
    coverage.update(duplicate_target_signals_removed=before - len(samples), samples=len(samples))
    return samples[list(SAMPLE_COLUMNS)], coverage


def nb2_log_probability(y, log_mean, alpha):
    """NB2 log PMF including all constants, including when observed count is zero."""
    y, eta = _counts(y), _finite(log_mean, "Log mean")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("NB2 alpha must be finite and positive")
    r = 1 / alpha
    log_total = np.logaddexp(np.log(r), eta)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1)
        + r * (np.log(r) - log_total)
        + y * (eta - log_total)
    )


def nb2_objective(parameters, design, counts, penalty):
    """Mean negative log likelihood + fixed half-L2 slope penalty, and gradient."""
    beta, log_alpha = parameters[:-1], parameters[-1]
    eta, r = design @ beta, np.exp(-log_alpha)
    log_r = -log_alpha
    log_total = np.logaddexp(log_r, eta)
    logp = (
        gammaln(counts + r)
        - gammaln(r)
        - gammaln(counts + 1)
        + r * (log_r - log_total)
        + counts * (eta - log_total)
    )
    slope_penalty = beta.copy()
    slope_penalty[0] = 0
    objective = -logp.mean() + penalty * np.dot(slope_penalty, slope_penalty) / 2
    score_eta = counts - (counts + r) * expit(eta - log_r)
    beta_gradient = -(design.T @ score_eta) / len(counts) + penalty * slope_penalty
    score_r = (
        digamma(counts + r) - digamma(r) + log_r - log_total + 1 - (counts + r) * np.exp(-log_total)
    )
    gradient = np.r_[beta_gradient, (r * score_r).mean()]
    return float(objective), gradient


@dataclass(frozen=True)
class NB2Fit:
    feature_names: tuple
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float
    diagnostics: dict

    def log_mean(self, frame):
        values = _finite(frame[list(self.feature_names)], "Prediction features")
        standardized = (values - self.mean) / self.scale
        return _finite(self.coefficients[0] + standardized @ self.coefficients[1:], "Log mean")

    def to_dict(self):
        return {
            "feature_names": list(self.feature_names),
            "training_mean": self.mean.tolist(),
            "training_scale": self.scale.tolist(),
            "intercept": float(self.coefficients[0]),
            "standardized_slopes": self.coefficients[1:].tolist(),
            "alpha": self.alpha,
            "diagnostics": self.diagnostics,
        }


def fit_nb2(training, feature_names, settings):
    if "split" in training and set(training.split) != {"train"}:
        raise ValueError("NB2 fitting accepts training rows only")
    names = tuple(feature_names)
    values = _finite(training[list(names)], "Training features")
    y = _counts(training.target_count)
    if len(y) <= len(names) + 2 or y.sum() == 0:
        raise FitError("Insufficient training observations or all-zero training counts")
    mean, raw_scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    constant = np.ptp(values, axis=0) == 0
    # Repeated floating values can have a tiny nonzero computed standard deviation.
    scale = np.where(constant, 1.0, raw_scale)
    design = np.column_stack([np.ones(len(y)), (values - mean) / scale])
    lower, upper = settings["alpha_bounds"]
    initial_alpha = np.clip((y.var() - y.mean()) / y.mean() ** 2, lower, upper)
    initial = np.r_[np.log(y.mean()), np.zeros(len(names)), np.log(initial_alpha)]
    bounds = [(None, None)] * (len(names) + 1) + [(np.log(lower), np.log(upper))]
    options = settings["optimizer"]
    result = minimize(
        nb2_objective,
        initial,
        args=(design, y, settings["l2_penalty"]),
        method=options["method"],
        jac=True,
        bounds=bounds,
        options={key: options[key] for key in ("maxiter", "ftol", "gtol")},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise FitError(f"NB2 optimizer failed: {result.message}")
    objective, gradient = nb2_objective(result.x, design, y, settings["l2_penalty"])
    projected = gradient.copy()
    if result.x[-1] <= bounds[-1][0] + 1e-8 and projected[-1] > 0:
        projected[-1] = 0
    if result.x[-1] >= bounds[-1][1] - 1e-8 and projected[-1] < 0:
        projected[-1] = 0
    gradient_norm = float(np.max(np.abs(projected)))
    if (
        not np.isfinite(objective)
        or not np.isfinite(gradient).all()
        or gradient_norm > options["accepted_projected_gradient_inf_norm"]
    ):
        raise FitError("NB2 optimizer did not meet the fixed projected-gradient acceptance limit")
    alpha = float(np.exp(result.x[-1]))
    return NB2Fit(
        names,
        mean,
        scale,
        result.x[:-1],
        alpha,
        {
            "success": True,
            "message": str(result.message),
            "iterations": int(result.nit),
            "penalized_mean_negative_log_likelihood": objective,
            "projected_gradient_inf_norm": gradient_norm,
            "training_samples": len(y),
            "l2_penalty": settings["l2_penalty"],
            "constant_training_features": [
                name for name, is_constant in zip(names, constant, strict=True) if is_constant
            ],
            "alpha_bounds": [lower, upper],
            "alpha_at_numerical_bound": bool(
                abs(result.x[-1] - np.log(lower)) < 1e-6 or abs(result.x[-1] - np.log(upper)) < 1e-6
            ),
        },
    )


def _scored_rows(samples, models):
    frame = samples[["day", "target_bucket_ms", "target_count"]].copy()
    y = frame.target_count.to_numpy()
    for label, model in models.items():
        eta = model.log_mean(samples)
        with np.errstate(over="ignore", under="ignore"):
            mu = np.exp(eta)
        if not np.isfinite(mu).all() or (mu <= 0).any():
            raise FitError("Nonfinite or underflowed prediction; no clipping/fallback is allowed")
        frame[f"{label}_mean"] = mu
        frame[f"{label}_logp"] = nb2_log_probability(y, eta, model.alpha)
        frame[f"{label}_zero"] = np.exp(nb2_log_probability(np.zeros(len(y)), eta, model.alpha))
    return frame


def _score_summary(frame):
    n = len(frame)
    if not n:
        raise ValueError("Cannot evaluate an empty common sample")
    result = {
        "samples": n,
        "observed_mean": float(frame.target_count.mean()),
        "observed_zero_fraction": float((frame.target_count == 0).mean()),
    }
    for label in ("baseline", "book"):
        mean = frame[f"{label}_mean"]
        result[label] = {
            "mean_log_probability": float(frame[f"{label}_logp"].mean()),
            "mae": float((frame.target_count - mean).abs().mean()),
            "predicted_mean": float(mean.mean()),
            "observed_to_predicted_mean_ratio": float(frame.target_count.mean() / mean.mean()),
            "predicted_zero_fraction": float(frame[f"{label}_zero"].mean()),
        }
    result["book_minus_baseline_log_score"] = float((frame.book_logp - frame.baseline_logp).mean())
    result["baseline_minus_book_mae"] = result["baseline"]["mae"] - result["book"]["mae"]
    return result


def evaluate_pair(samples, baseline, book, block_hours):
    frame = _scored_rows(samples, {"baseline": baseline, "book": book})
    block_ms = block_hours * 3_600_000
    frame["utc_block_ms"] = frame.target_bucket_ms // block_ms * block_ms
    return {
        "overall": _score_summary(frame),
        "by_day": [
            {"day": day, **_score_summary(rows)} for day, rows in frame.groupby("day", sort=True)
        ],
        "by_utc_block": [
            {
                "day": day,
                "start_ms": int(start),
                "end_ms": int(start + block_ms),
                **_score_summary(rows),
            }
            for (day, start), rows in frame.groupby(["day", "utc_block_ms"], sort=True)
        ],
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample_sha(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if (
        protocol["protocol_version"] != 1
        or protocol["splits"] != FIXED_SPLITS
        or protocol["additional_delay_ms"] != list(DELAYS_MS)
        or protocol["target_seconds"] != 1
        or protocol["minimum_visible_levels_per_side"] != 5
        or protocol["baseline_features"] != list(BASELINE_FEATURES)
        or protocol["additional_book_features"] != list(BOOK_FEATURES)
        or protocol["development_previously_profiled"] is not True
    ):
        raise ValueError("Protocol differs from this implemented frozen diagnostic; test is closed")
    model, optimizer = protocol["model"], protocol["model"]["optimizer"]
    if (
        model["family"] != "NB2"
        or model["l2_penalty"] != 0.01
        or model["objective"] != "mean_negative_log_probability_plus_half_l2_slopes"
        or model["alpha_bounds"] != [0.0001, 1000.0]
        or optimizer
        != {
            "method": "L-BFGS-B",
            "maxiter": 500,
            "ftol": 1e-10,
            "gtol": 1e-7,
            "accepted_projected_gradient_inf_norm": 0.001,
        }
        or protocol["calibration_block_hours"] != 3
    ):
        raise ValueError("Frozen NB2 specification changed; create and review a new protocol")
    return protocol


def run_probe(root: Path, protocol_path: Path):
    """Fit only fixed training days; development is passed solely to evaluation."""
    root, protocol_path = Path(root), Path(protocol_path)
    protocol_hash = _sha(protocol_path)
    implementation_paths = {
        "experiments/crypto_activity_model.py": __file__,
        "experiments/crypto_profile.py": crypto_profile.__file__,
        "experiments/crypto_features.py": crypto_features.__file__,
    }
    implementation_hashes = {name: _sha(path) for name, path in implementation_paths.items()}
    protocol = read_protocol(protocol_path)
    source_config = root / protocol["source_config"]
    if _sha(source_config) != protocol["source_config_sha256"]:
        raise ValueError("Source configuration does not match the frozen archive study")
    source = json.loads(source_config.read_text())
    if any(source["splits"].get(split) != days for split, days in FIXED_SPLITS.items()):
        raise ValueError("Source train/development split differs from the fixed probe")
    frames, provenance, joint_coverage = {}, [], []
    for split, days in FIXED_SPLITS.items():
        for day in days:
            book, book_audit = crypto_features.read_feature_day(
                root, source_config, split, day, "orderbook"
            )
            trades, trade_audit = crypto_features.read_feature_day(
                root, source_config, split, day, "trades"
            )
            prepared = crypto_profile.prepare_book(book, book_audit, day)
            joint, coverage = crypto_profile.covered_trade_seconds(prepared, trades, trade_audit)
            frames[(split, day)] = joint
            joint_coverage.append({"split": split, "day": day, **coverage})
            for kind, audit in (("orderbook", book_audit), ("trades", trade_audit)):
                provenance.append(
                    {
                        "split": split,
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
    scenarios = []
    for delay in DELAYS_MS:
        datasets, coverage = {}, []
        for split, days in FIXED_SPLITS.items():
            pieces = []
            for day in days:
                samples, counts = build_activity_samples(
                    frames[(split, day)], split=split, day=day, delay_ms=delay
                )
                if samples.empty:
                    raise ValueError(f"No eligible common samples: {split}/{day}/{delay}ms")
                pieces.append(samples)
                coverage.append({"split": split, "day": day, **counts})
            datasets[split] = pd.concat(pieces, ignore_index=True)
        baseline = fit_nb2(datasets["train"], BASELINE_FEATURES, protocol["model"])
        book_model = fit_nb2(
            datasets["train"], BASELINE_FEATURES + BOOK_FEATURES, protocol["model"]
        )
        scenarios.append(
            {
                "additional_delay_ms": delay,
                "coverage": coverage,
                "sample_sha256": {split: _sample_sha(frame) for split, frame in datasets.items()},
                "models": {"baseline": baseline.to_dict(), "book": book_model.to_dict()},
                "evaluation": {
                    split: evaluate_pair(
                        frame, baseline, book_model, protocol["calibration_block_hours"]
                    )
                    for split, frame in datasets.items()
                },
            }
        )
    if (
        _sha(protocol_path) != protocol_hash
        or _sha(source_config) != protocol["source_config_sha256"]
        or any(
            _sha(path) != implementation_hashes[name] for name, path in implementation_paths.items()
        )
    ):
        raise ValueError("Protocol or implementation changed during the diagnostic run")
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "diagnostic_computed",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "source_config_sha256": protocol["source_config_sha256"],
        "source_sha256": implementation_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "assumptions": ASSUMPTIONS,
        "sources": provenance,
        "joint_coverage": joint_coverage,
        "scenarios": scenarios,
    }
