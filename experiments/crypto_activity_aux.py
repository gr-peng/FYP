"""Fixed auxiliary count-distribution experiment: pure frames, fitting and scoring.

There is no data reader, CLI, file publication or development access in this core.
The two-stage runner owns provenance and access receipts. NB2 numerical helpers
are reused from an explicitly hash-bound, unchanged v1 implementation.
"""

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betainc, betaincc, expit, logsumexp

from experiments import crypto_activity_model as v1

FIXED_SPLITS = {
    "train": [f"2024-10-{day:02d}" for day in range(14, 21)],
    "development": [f"2024-10-{day:02d}" for day in range(21, 28)],
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
NOISE_FEATURES = tuple(f"noise_{index}" for index in range(4))
MODEL_FEATURES = {
    "N0": BASELINE_FEATURES,
    "N1": BASELINE_FEATURES + BOOK_FEATURES,
    "H0": BASELINE_FEATURES,
    "H1": BASELINE_FEATURES + BOOK_FEATURES,
    "Hnoise": BASELINE_FEATURES + NOISE_FEATURES,
}
NOISE_SPEC = {
    "seed": 20261011,
    "features": list(NOISE_FEATURES),
    "hash": "sha256",
    "serialization": "ASCII crypto-activity-aux-noise-v1|{seed}|{signal_bucket_ms}|"
    "{feature_index}; "
    "base-10 integers; no newline",
    "uniforms": "u1=((big_endian_uint64(digest[0:8]) >> 12)+0.5)/2**52; "
    "u2=((big_endian_uint64(digest[8:16]) >> 12)+0.5)/2**52",
    "transform": "sqrt(-2*log(u1))*cos(2*pi*u2)",
    "feature_index": [0, 1, 2, 3],
}
NB2_HELPER_SHA256 = "dd196f4c56835c1262864ce0e656fbb0e91daada908d1117710e7fb825c5c0af"
NUMERICAL_SPEC = {
    "l2_penalty": 0.01,
    "alpha_bounds": [0.0001, 1000.0],
    "optimizer": {
        "method": "L-BFGS-B",
        "maxiter": 500,
        "ftol": 1e-10,
        "gtol": 1e-7,
        "accepted_projected_gradient_inf_norm": 0.001,
    },
}
COUNT_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, None),
)
COMPARISONS = (("N1", "N0"), ("H0", "N0"), ("H1", "H0"), ("Hnoise", "H0"))
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
    *NOISE_FEATURES,
)


class FitError(RuntimeError):
    """Numerical failure, never replaced by an unreported fallback model."""


def assert_helper_source():
    if hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest() != NB2_HELPER_SHA256:
        raise ValueError("The frozen v1 numerical helper source hash changed")


def _finite(values, name):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _counts(values):
    result = _finite(values, "Counts")
    if (result < 0).any() or (result != np.floor(result)).any():
        raise ValueError("Counts must be nonnegative integers")
    return result


def _selection(split, day):
    if split not in FIXED_SPLITS or day not in FIXED_SPLITS[split]:
        raise ValueError(
            "Only fixed auxiliary train/development dates are allowed; event is closed"
        )


def noise_features(signal_bucket_ms):
    """Fixed SHA256 / 52-bit midpoint uniforms / Box-Muller cosine construction."""
    buckets = _finite(signal_bucket_ms, "Noise timestamps")
    if buckets.ndim != 1 or (buckets % 1000 != 0).any() or (buckets < 0).any():
        raise ValueError("Noise requires a vector of nonnegative aligned millisecond timestamps")
    result = np.empty((len(buckets), 4))
    for row, bucket in enumerate(buckets):
        for index in range(4):
            message = f"crypto-activity-aux-noise-v1|20261011|{int(bucket)}|{index}".encode("ascii")
            digest = hashlib.sha256(message).digest()
            u1 = ((int.from_bytes(digest[:8], "big") >> 12) + 0.5) / 2**52
            u2 = ((int.from_bytes(digest[8:16], "big") >> 12) + 0.5) / 2**52
            result[row, index] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return result


def build_activity_samples(joint, *, split, day, delay_ms):
    """v1 timing rules with an explicit auxiliary date gate, without monkeypatching."""
    _selection(split, day)
    if type(delay_ms) is not int or delay_ms not in DELAYS_MS:
        raise ValueError("Only the fixed 0/1000/5000 ms delays are supported")
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
    if not n:
        return pd.DataFrame(columns=SAMPLE_COLUMNS), coverage
    _finite(
        work[
            [
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
        ],
        "Joint features",
    )
    _counts(work.trade_count)
    buckets, available = work.bucket_ms.to_numpy(), work.available_at_ms.to_numpy()
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
        raise ValueError("Invalid activity, depth or imbalance")
    trade_available = work.trade_available_at_ms.fillna(work.bucket_ms + 1000).to_numpy()
    _finite(trade_available, "Trade availability")
    if (
        (trade_available < buckets + 1000).any()
        or (np.diff(trade_available) < 0).any()
        or (trade_available != np.floor(trade_available)).any()
    ):
        raise ValueError("Trade availability must be integer, complete-second and monotone")
    runs = ((work.bucket_ms.diff() != 1000) | (work.run_id != work.run_id.shift())).cumsum()
    runs = runs.to_numpy()
    activity_indices = np.searchsorted(trade_available, available, side="right") - 1
    decisions = available.astype("int64") + delay_ms
    target_buckets = ((decisions + 999) // 1000) * 1000
    target_indices = pd.Index(buckets).get_indexer(target_buckets)
    depth_ok = ((work.visible_bid_levels >= 5) & (work.visible_ask_levels >= 5)).to_numpy()
    target_exists = target_indices >= 0
    same_run = (
        (activity_indices >= 0)
        & (runs[np.maximum(target_indices, 0)] == runs)
        & (runs[np.maximum(activity_indices, 0)] == runs)
    )
    keep = depth_ok & target_exists & same_run
    coverage.update(
        depth_rejected_seconds=int((~depth_ok).sum()),
        depth_rejected_fraction=float((~depth_ok).mean()),
        missing_target_seconds=int((depth_ok & ~target_exists).sum()),
        broken_path_seconds=int((depth_ok & target_exists & ~same_run).sum()),
    )
    signal, last, target = (
        work.loc[keep],
        work.iloc[activity_indices[keep]],
        work.iloc[target_indices[keep]],
    )
    samples = pd.DataFrame(
        {
            "split": split,
            "day": day,
            "run_id": runs[keep],
            "signal_bucket_ms": signal.bucket_ms.to_numpy(),
            "signal_available_at_ms": available[keep].astype("int64"),
            "decision_at_ms": decisions[keep],
            "activity_bucket_ms": last.bucket_ms.to_numpy(),
            "activity_available_at_ms": trade_available[activity_indices[keep]].astype("int64"),
            "delayed_activity_available_at_ms": (
                trade_available[activity_indices[keep]].astype("int64") + delay_ms
            ),
            "target_bucket_ms": target_buckets[keep],
            "target_end_ms": target_buckets[keep] + 1000,
            "target_count": target.trade_count.to_numpy().astype("int64"),
        }
    )
    angle = 2 * np.pi * (samples.target_bucket_ms % 86_400_000) / 86_400_000
    samples["utc_sin"], samples["utc_cos"] = np.sin(angle), np.cos(angle)
    samples["log1p_last_trade_count"] = np.log1p(last.trade_count.to_numpy())
    samples["log1p_last_trade_quantity_btc"] = np.log1p(last.trade_quantity_btc.to_numpy())
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
    samples[list(NOISE_FEATURES)] = noise_features(samples.signal_bucket_ms)
    coverage.update(duplicate_target_signals_removed=before - len(samples), samples=len(samples))
    return samples[list(SAMPLE_COLUMNS)], coverage


def log1mexp(log_probability):
    """Stable log(1-exp(a)) for finite a<0; no probability clipping."""
    value = _finite(log_probability, "Log probability")
    if (value >= 0).any():
        raise FitError("Complement probability is numerically zero or invalid")
    return np.log(-np.expm1(value))


def nb2_log_zero(log_mean, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("NB2 alpha must be finite and positive")
    return -np.logaddexp(0, _finite(log_mean, "Log mean") + np.log(alpha)) / alpha


def logistic_objective(beta, design, positive, penalty):
    eta = design @ beta
    loss = np.where(positive > 0, np.logaddexp(0, -eta), np.logaddexp(0, eta))
    slopes = beta.copy()
    slopes[0] = 0
    return (
        float(loss.mean() + penalty * np.dot(slopes, slopes) / 2),
        design.T @ (expit(eta) - positive) / len(positive) + penalty * slopes,
    )


def truncated_nb2_objective(parameters, design, positive_counts, total_count, penalty):
    """Positive truncated NB likelihood divided by ALL training N, not positive N."""
    if len(positive_counts) > total_count or (np.asarray(positive_counts) <= 0).any():
        raise ValueError("Truncated objective requires positive observations and full N")
    beta, log_alpha = parameters[:-1], parameters[-1]
    eta, alpha = design @ beta, np.exp(log_alpha)
    log_zero = nb2_log_zero(eta, alpha)
    log_nonzero = log1mexp(log_zero)
    objective, gradient = v1.nb2_objective(parameters, design, positive_counts, 0.0)
    z = eta + log_alpha
    log_fraction = -np.logaddexp(0, -z)
    # Derivatives of log(1-f(0)); log-space products avoid infinity * zero.
    correction_eta = np.exp(-log_alpha + log_zero - log_nonzero + log_fraction)
    difference = np.maximum(np.logaddexp(0, z) - expit(z), 0)
    correction_alpha = np.zeros_like(eta)
    valid = difference > 0
    correction_alpha[valid] = -np.exp(
        -log_alpha + log_zero[valid] - log_nonzero[valid] + np.log(difference[valid])
    )
    gradient += np.r_[design.T @ correction_eta / len(positive_counts), correction_alpha.mean()]
    weight = len(positive_counts) / total_count
    slopes = beta.copy()
    slopes[0] = 0
    return (
        float(weight * (objective + log_nonzero.mean()) + penalty * np.dot(slopes, slopes) / 2),
        weight * gradient + np.r_[penalty * slopes, 0.0],
    )


def hurdle_log_probability(y, log_mean, alpha, gate_logits):
    y = _counts(y)
    gate = _finite(gate_logits, "Gate logits")
    log_positive = -np.logaddexp(0, -gate)
    log_zero = -np.logaddexp(0, gate)
    conditional = v1.nb2_log_probability(y, log_mean, alpha) - log1mexp(
        nb2_log_zero(log_mean, alpha)
    )
    return np.where(y == 0, log_zero, log_positive + conditional)


def _optimize(objective, initial, args, bounds, spec):
    settings = spec["optimizer"]
    result = minimize(
        objective,
        initial,
        args=args,
        method=settings["method"],
        jac=True,
        bounds=bounds,
        options={k: settings[k] for k in ("maxiter", "ftol", "gtol")},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise FitError(f"Optimizer failed: {result.message}")
    value, gradient = objective(result.x, *args)
    projected = gradient.copy()
    for i, (lower, upper) in enumerate(bounds):
        if (lower is not None and result.x[i] <= lower + 1e-8 and projected[i] > 0) or (
            upper is not None and result.x[i] >= upper - 1e-8 and projected[i] < 0
        ):
            projected[i] = 0
    norm = float(np.max(np.abs(projected)))
    if (
        not np.isfinite(value)
        or not np.isfinite(gradient).all()
        or norm > settings["accepted_projected_gradient_inf_norm"]
    ):
        raise FitError("Optimizer failed fixed finite/projected-gradient acceptance")
    return result.x, {
        "success": True,
        "iterations": int(result.nit),
        "message": str(result.message),
        "objective": float(value),
        "projected_gradient_inf_norm": norm,
    }


def _validate_samples(samples, training=False):
    if samples.empty:
        raise ValueError("An empty common sample cannot be fitted or scored")
    if len(set(samples.split)) != 1 or (training and set(samples.split) != {"train"}):
        raise ValueError("Fitting requires train only; scoring requires one split")
    for split, day in samples[["split", "day"]].drop_duplicates().itertuples(index=False):
        _selection(split, day)
    buckets = _finite(samples.target_bucket_ms, "Target times")
    if (buckets % 1000 != 0).any() or samples.duplicated(["day", "target_bucket_ms"]).any():
        raise ValueError("Targets must be aligned unique seconds, one delay scenario at a time")
    actual_days = pd.to_datetime(buckets, unit="ms", utc=True).strftime("%Y-%m-%d")
    if not np.array_equal(actual_days, samples.day.to_numpy()):
        raise ValueError("Target UTC day differs from sample label")
    _finite(samples[list(BASELINE_FEATURES + BOOK_FEATURES + NOISE_FEATURES)], "Features")
    _counts(samples.target_count)


def fit_models(samples, spec):
    """Fit exactly five models on the same train rows; return JSON-only parameters."""
    assert_helper_source()
    if spec != NUMERICAL_SPEC:
        raise ValueError("The auxiliary numerical specification differs from the frozen design")
    _validate_samples(samples, training=True)
    y = _counts(samples.target_count)
    positive = y > 0
    if positive.sum() <= max(map(len, MODEL_FEATURES.values())) + 2 or positive.all():
        raise FitError("Require enough positive observations and both zero/nonzero classes")
    models = {}
    for model_id, names in MODEL_FEATURES.items():
        values = samples[list(names)].to_numpy(dtype=float)
        mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
        constant = np.ptp(values, axis=0) == 0
        scale[constant] = 1.0
        design = np.column_stack([np.ones(len(y)), (values - mean) / scale])
        hurdle = model_id.startswith("H")
        selected_y = y[positive] if hurdle else y
        initial_mean = max(selected_y.mean() - (0.5 if hurdle else 0.0), 0.1)
        lower, upper = spec["alpha_bounds"]
        initial_alpha = np.clip(
            (selected_y.var() - selected_y.mean()) / selected_y.mean() ** 2, lower, upper
        )
        initial = np.r_[np.log(initial_mean), np.zeros(len(names)), np.log(initial_alpha)]
        bounds = [(None, None)] * (len(names) + 1) + [(np.log(lower), np.log(upper))]
        if hurdle:
            parameters, count_status = _optimize(
                truncated_nb2_objective,
                initial,
                (design[positive], y[positive], len(y), spec["l2_penalty"]),
                bounds,
                spec,
            )
            probability = positive.mean()
            gate_initial = np.r_[np.log(probability / (1 - probability)), np.zeros(len(names))]
            gate, gate_status = _optimize(
                logistic_objective,
                gate_initial,
                (design, positive.astype(float), spec["l2_penalty"]),
                [(None, None)] * len(gate_initial),
                spec,
            )
        else:
            parameters, count_status = _optimize(
                v1.nb2_objective,
                initial,
                (design, y, spec["l2_penalty"]),
                bounds,
                spec,
            )
            gate, gate_status = None, None
        alpha = float(np.exp(parameters[-1]))
        models[model_id] = {
            "family": "hurdle_NB2" if hurdle else "NB2",
            "feature_names": list(names),
            "training_mean": mean.tolist(),
            "training_scale": scale.tolist(),
            "count_coefficients": parameters[:-1].tolist(),
            "gate_coefficients": None if gate is None else gate.tolist(),
            "alpha": alpha,
            "nb2_helper_sha256": NB2_HELPER_SHA256,
            "diagnostics": {
                "status": "accepted_fit",
                "training_samples": len(y),
                "training_positive_samples": int(positive.sum()),
                "training_days": sorted(set(samples.day)),
                "normalization": "full_training_N",
                "l2_penalty": spec["l2_penalty"],
                "alpha_bounds": [lower, upper],
                "alpha_at_numerical_bound": bool(
                    abs(parameters[-1] - np.log(lower)) < 1e-6
                    or abs(parameters[-1] - np.log(upper)) < 1e-6
                ),
                "constant_training_features": [
                    name for name, flag in zip(names, constant, strict=True) if flag
                ],
                "count_optimizer": count_status,
                "gate_optimizer": gate_status,
            },
        }
    validate_models(models)
    return models


def _validate_models(models):
    """Validate a complete numerical model pack, not a declaration of simulation realism."""
    assert_helper_source()
    if not isinstance(models, dict) or set(models) != set(MODEL_FEATURES):
        raise ValueError("Require exactly N0/N1/H0/H1/Hnoise model parameters")
    populations = []
    for model_id, names in MODEL_FEATURES.items():
        model = models[model_id]
        hurdle = model_id.startswith("H")
        if (
            model["family"] != ("hurdle_NB2" if hurdle else "NB2")
            or model["feature_names"] != list(names)
            or model["nb2_helper_sha256"] != NB2_HELPER_SHA256
        ):
            raise ValueError("Model family/features/helper source differ from the frozen design")
        for key, size in (
            ("training_mean", len(names)),
            ("training_scale", len(names)),
            ("count_coefficients", len(names) + 1),
        ):
            if _finite(model[key], key).shape != (size,):
                raise ValueError(f"Invalid {key} shape")
        if (_finite(model["training_scale"], "Scale") <= 0).any():
            raise ValueError("Training scales must be positive")
        if not hurdle and model["gate_coefficients"] is not None:
            raise ValueError("NB2 must not contain hurdle coefficients")
        if hurdle and _finite(model["gate_coefficients"], "Gate coefficients").shape != (
            len(names) + 1,
        ):
            raise ValueError("Invalid hurdle gate shape")
        alpha = model["alpha"]
        if (
            isinstance(alpha, bool)
            or not np.isscalar(alpha)
            or not np.isfinite(alpha)
            or not 0.0001 * (1 - 1e-12) <= alpha <= 1000 * (1 + 1e-12)
        ):
            raise ValueError("Invalid or out-of-bounds dispersion")
        status = model["diagnostics"]
        n, positives = status["training_samples"], status["training_positive_samples"]
        if (
            status["status"] != "accepted_fit"
            or type(n) is not int
            or type(positives) is not int
            or not len(names) + 2 < positives < n
            or status["normalization"] != "full_training_N"
            or status["l2_penalty"] != 0.01
            or status["alpha_bounds"] != [0.0001, 1000.0]
            or not status["training_days"]
            or any(day not in FIXED_SPLITS["train"] for day in status["training_days"])
        ):
            raise ValueError("Invalid training population, status or frozen numerical settings")
        populations.append((n, positives, tuple(status["training_days"])))
        optimizers = [status["count_optimizer"]]
        if hurdle:
            optimizers.append(status["gate_optimizer"])
        elif status["gate_optimizer"] is not None:
            raise ValueError("NB2 has an unexpected gate optimizer")
        for optimizer in optimizers:
            if (
                optimizer["success"] is not True
                or type(optimizer["iterations"]) is not int
                or optimizer["iterations"] < 0
                or not np.isfinite(optimizer["objective"])
                or not 0 <= optimizer["projected_gradient_inf_norm"] <= 0.001
            ):
                raise ValueError("Unaccepted optimizer in model artifact")
    if len(set(populations)) != 1:
        raise ValueError("Models were not fitted on matching training populations")


def validate_models(models):
    """Raise ValueError for incomplete, malformed or unaccepted five-model artifacts."""
    try:
        _validate_models(models)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"Malformed model artifact: {type(exc).__name__}") from exc


def nb2_cdf_sf(k, log_mean, alpha):
    """CDF and SF via complementary beta functions, retaining small tail probabilities."""
    k, eta = np.asarray(k), np.asarray(log_mean)
    z = eta + np.log(alpha)
    q, p, r = expit(z), expit(-z), 1 / alpha
    cdf = np.where(p <= 0.5, betainc(r, k + 1, p), betaincc(k + 1, r, q))
    sf = np.where(q <= 0.5, betainc(k + 1, r, q), betaincc(r, k + 1, p))
    if np.any(k == 0):
        zero_logp = nb2_log_zero(eta, alpha)
        cdf = np.where(k == 0, np.exp(zero_logp), cdf)
        sf = np.where(k == 0, -np.expm1(zero_logp), sf)
    if not np.isfinite(cdf).all() or not np.isfinite(sf).all():
        raise FitError("Count CDF/SF numerical evaluation failed")
    return cdf, sf


def distribution_median(log_mean, alpha, gate_logits=None):
    """Integer inf{k:F(k)>=.5}; hurdle conversion uses NB SF threshold .5*S0/p."""
    eta = np.atleast_1d(_finite(log_mean, "Log mean"))
    positive_prob = (
        np.ones(len(eta)) if gate_logits is None else expit(_finite(gate_logits, "Gate logits"))
    )
    need_positive = positive_prob > 0.5 if gate_logits is not None else np.ones(len(eta), bool)
    threshold = np.full(len(eta), 0.5)
    if gate_logits is not None:
        threshold[need_positive] *= (
            -np.expm1(nb2_log_zero(eta[need_positive], alpha)) / (positive_prob[need_positive])
        )
    if (threshold <= 0).any():
        raise FitError("Median survival threshold is numerically zero")
    low = np.full(len(eta), -1, dtype="int64")
    high = np.ones(len(eta), dtype="int64")
    high[~need_positive] = 0
    for _ in range(53):
        grow = need_positive & (nb2_cdf_sf(high, eta, alpha)[1] > threshold)
        if not grow.any():
            break
        if (high[grow] > 2**51).any():
            raise FitError("Median exceeds the exact-integer numerical search range")
        high[grow] = high[grow] * 2 + 1
    else:
        raise FitError("Median bracketing failed")
    for _ in range(53):
        active = high - low > 1
        if not active.any():
            return high
        mid = (low[active] + high[active]) // 2
        left = nb2_cdf_sf(mid, eta[active], alpha)[1] <= threshold[active]
        indices = np.flatnonzero(active)
        high[indices[left]], low[indices[~left]] = mid[left], mid[~left]
    raise FitError("Median integer search failed")


def count_bin_probabilities(log_mean, alpha, gate_logits=None):
    eta = np.atleast_1d(_finite(log_mean, "Log mean"))
    log_zero = nb2_log_zero(eta, alpha)
    values = [np.exp(log_zero)]
    for _, lower, upper in COUNT_BINS[1:]:
        cdf_lower, sf_lower = nb2_cdf_sf(lower - 1, eta, alpha)
        if upper is None:
            mass = sf_lower
        else:
            cdf_upper, sf_upper = nb2_cdf_sf(upper, eta, alpha)
            mass = np.where(cdf_lower > 0.5, sf_lower - sf_upper, cdf_upper - cdf_lower)
        values.append(mass)
    result = np.column_stack(values)
    if gate_logits is not None:
        gate = _finite(gate_logits, "Gate logits")
        survival = -np.expm1(log_zero)
        if (survival <= 0).any():
            raise FitError("Truncated normalization is numerically zero")
        result[:, 0] = expit(-gate)
        result[:, 1:] *= (expit(gate) / survival)[:, None]
    if (
        not np.isfinite(result).all()
        or (result < 0).any()
        or np.max(np.abs(result.sum(axis=1) - 1)) > 1e-8
    ):
        raise FitError("Count-bin probabilities are invalid or do not sum to one")
    return result


def _predictions(samples, model):
    values = samples[model["feature_names"]].to_numpy(dtype=float)
    design = np.column_stack(
        [
            np.ones(len(samples)),
            (values - np.array(model["training_mean"])) / model["training_scale"],
        ]
    )
    eta = _finite(design @ model["count_coefficients"], "Log mean")
    alpha, y = model["alpha"], samples.target_count.to_numpy()
    nb_zero = nb2_log_zero(eta, alpha)
    nb_nonzero = log1mexp(nb_zero)
    gate = None
    if model["family"] == "hurdle_NB2":
        gate = _finite(design @ model["gate_coefficients"], "Gate logits")
        log_zero, log_positive = -np.logaddexp(0, gate), -np.logaddexp(0, -gate)
        log_mean = log_positive + eta - nb_nonzero
    else:
        log_zero, log_positive, log_mean = nb_zero, nb_nonzero, eta
    binary = np.where(y == 0, log_zero, log_positive)
    contribution = np.where(y > 0, v1.nb2_log_probability(y, eta, alpha) - nb_nonzero, 0.0)
    with np.errstate(over="ignore", under="ignore"):
        mean = np.exp(log_mean)
        conditional_mean = np.exp(eta - nb_nonzero)
    if not np.isfinite(mean).all() or (mean <= 0).any():
        raise FitError("Mean overflow/underflow; no prediction clipping is allowed")
    if not np.isfinite(conditional_mean).all():
        raise FitError("Positive conditional mean overflow")
    full_logp = binary + contribution
    _finite(full_logp, "Full log probability")
    return {
        "mean": mean,
        "log_mean": log_mean,
        "positive_conditional_mean": conditional_mean,
        "log_positive_probability": log_positive,
        "median": distribution_median(eta, alpha, gate),
        "zero": np.exp(log_zero),
        "logp": full_logp,
        "binary": binary,
        "positive_contribution": contribution,
        "bins": count_bin_probabilities(eta, alpha, gate),
    }


def _summary(y, prediction, indices):
    observed, mu = y[indices], prediction["mean"][indices]
    zero, positive = prediction["zero"][indices], observed > 0
    bins = prediction["bins"][indices]
    contributions = prediction["positive_contribution"][indices]
    positive_mixture = float(
        np.exp(
            logsumexp(prediction["log_mean"][indices])
            - logsumexp(prediction["log_positive_probability"][indices])
        )
    )
    if not np.isfinite(positive_mixture) or positive_mixture <= 0:
        raise FitError("Pooled positive conditional mean overflow/underflow")
    return {
        "mean_log_probability": float(prediction["logp"][indices].mean()),
        "mean_binary_log_probability": float(prediction["binary"][indices].mean()),
        "mean_positive_log_probability_contribution": float(contributions.mean()),
        "positive_only_mean_log_probability": float(contributions[positive].mean())
        if positive.any()
        else None,
        "zero_brier": float(np.mean((zero - (observed == 0)) ** 2)),
        "predicted_zero_fraction": float(zero.mean()),
        "predicted_mean": float(mu.mean()),
        "observed_to_predicted_mean_ratio": float(observed.mean() / mu.mean()),
        "observed_positive_mean": float(observed[positive].mean()) if positive.any() else None,
        "predicted_positive_mean_on_observed_positive_rows": float(
            prediction["positive_conditional_mean"][indices][positive].mean()
        )
        if positive.any()
        else None,
        "predicted_positive_mean_mixture": positive_mixture,
        "observed_to_predicted_positive_mean_ratio": (
            float(observed[positive].mean() / positive_mixture) if positive.any() else None
        ),
        "mae_mean": float(np.mean(np.abs(observed - mu))),
        "mae_median": float(np.mean(np.abs(observed - prediction["median"][indices]))),
        "count_bins": {
            label: {
                "observed_fraction": float(
                    np.mean((observed >= lower) & (True if upper is None else observed <= upper))
                ),
                "predicted_fraction": float(bins[:, index].mean()),
            }
            for index, (label, lower, upper) in enumerate(COUNT_BINS)
        },
        "max_probability_mass_error": float(np.max(np.abs(bins.sum(axis=1) - 1))),
    }


def score_models(samples, models):
    """Evaluate frozen parameters only; no fitting, normalization or calibration update."""
    validate_models(models)
    _validate_samples(samples)
    samples = samples.reset_index(drop=True)
    y = samples.target_count.to_numpy()
    groups = [("overall", {}, np.arange(len(samples)))]
    groups += [
        ("by_day", {"day": day}, np.asarray(indices))
        for day, indices in samples.groupby("day", sort=True).indices.items()
    ]
    grouping = samples.assign(block=samples.target_bucket_ms // 10_800_000 * 10_800_000)
    groups += [
        (
            "by_utc_block",
            {"day": day, "start_ms": int(block), "end_ms": int(block + 10_800_000)},
            np.asarray(indices),
        )
        for (day, block), indices in grouping.groupby(["day", "block"], sort=True).indices.items()
    ]
    sections = [
        {
            **identity,
            "samples": len(indices),
            "observed_mean": float(y[indices].mean()),
            "observed_zero_fraction": float(np.mean(y[indices] == 0)),
            "models": {},
        }
        for _, identity, indices in groups
    ]
    for model_id, model in models.items():
        prediction = _predictions(samples, model)
        for section, (_, _, indices) in zip(sections, groups, strict=True):
            section["models"][model_id] = _summary(y, prediction, indices)
    for section in sections:
        comparisons = {}
        for candidate, baseline in COMPARISONS:
            a, b = section["models"][candidate], section["models"][baseline]
            comparisons[f"{candidate}_minus_{baseline}"] = {
                "log_score_gain": a["mean_log_probability"] - b["mean_log_probability"],
                "binary_log_score_gain": a["mean_binary_log_probability"]
                - b["mean_binary_log_probability"],
                "positive_log_contribution_gain": a["mean_positive_log_probability_contribution"]
                - b["mean_positive_log_probability_contribution"],
                "zero_brier_reduction": b["zero_brier"] - a["zero_brier"],
                "mae_mean_reduction": b["mae_mean"] - a["mae_mean"],
                "mae_median_reduction": b["mae_median"] - a["mae_median"],
            }
        section["paired_comparisons"] = comparisons
    report = {"overall": sections[0], "by_day": [], "by_utc_block": []}
    for section, (kind, _, _) in zip(sections[1:], groups[1:], strict=True):
        report[kind].append(section)
    report["daily_log_score_gain_summary"] = {}
    for candidate, baseline in COMPARISONS:
        label = f"{candidate}_minus_{baseline}"
        gains = np.array(
            [s["paired_comparisons"][label]["log_score_gain"] for s in report["by_day"]]
        )
        report["daily_log_score_gain_summary"][label] = {
            "days": len(gains),
            "mean_daily_gain": float(gains.mean()),
            "worst_daily_gain": float(gains.min()),
            "best_daily_gain": float(gains.max()),
            "positive_days": int((gains > 0).sum()),
            "negative_days": int((gains < 0).sum()),
            "zero_days": int((gains == 0).sum()),
        }
    return report
