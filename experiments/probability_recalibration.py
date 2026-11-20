"""Fixed monotone probability correction: pure functions, no loaders or transport."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from experiments.market_quote_baselines import weights
from scripts.validate_memory_information import grouped_folds

EPS = 1e-6
RIDGE = 0.01
MODES = ("intercept", "monotone")
FIELDS = {"target_id", "session", "participant", "order_id", "observed_buy", "probability_buy"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(records):
    require(bool(records), "empty probability records")
    ids = []
    for row in records:
        require(set(row) == FIELDS, "unexpected probability record fields")
        for key in ("target_id", "session", "participant", "order_id"):
            require(isinstance(row[key], str) and bool(row[key].strip()), "missing identity")
        require(row["target_id"] == row["session"] + ":" + row["order_id"], "inconsistent identity")
        require(type(row["observed_buy"]) is int and row["observed_buy"] in (0, 1), "invalid label")
        p = row["probability_buy"]
        require(type(p) in (int, float) and np.isfinite(p) and 0 <= p <= 1, "invalid probability")
        ids.append(row["target_id"])
    require(len(ids) == len(set(ids)), "duplicate target")


def scoring_records(records):
    return [
        {
            "provenance": {k: row[k] for k in ("session", "participant", "order_id")},
            "target": {"buy": row["observed_buy"]},
        }
        for row in records
    ]


def arrays(records):
    validate(records)
    p = np.array([r["probability_buy"] for r in records], dtype=float)
    y = np.array([r["observed_buy"] for r in records], dtype=float)
    return logit(np.clip(p, EPS, 1 - EPS)), y, weights(scoring_records(records))


def objective(theta, x, y, w, mode):
    require(mode in MODES, "unknown correction")
    a, b = (theta[0], theta[1]) if mode == "monotone" else (theta[0], 1.0)
    z = a + b * x
    error = w * (expit(z) - y)
    loss = w @ (np.logaddexp(0, z) - y * z)
    gradient = [error.sum()]
    if mode == "monotone":
        loss += RIDGE / 2 * (b - 1) ** 2
        gradient.append(error @ x + RIDGE * (b - 1))
    return float(loss), np.array(gradient)


def fit(records, mode):
    require(mode in MODES, "unknown correction")
    x, y, w = arrays(records)
    require(set(y) == {0, 1}, "constant-label training complement")
    result = minimize(
        objective,
        np.array([0.0, 1.0]) if mode == "monotone" else np.array([0.0]),
        args=(x, y, w, mode),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0, None)] if mode == "monotone" else [(None, None)],
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    require(result.success and np.isfinite(result.x).all(), "probability correction fit failed")
    return {
        "mode": mode,
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]) if mode == "monotone" else 1.0,
        "ridge": RIDGE,
        "clip": EPS,
        "train_constant": float(w @ y),
        "training_records": len(records),
        "training_sessions": sorted({r["session"] for r in records}),
    }


def predict(fitted, probabilities):
    p = np.asarray(probabilities, dtype=float)
    require(p.ndim == 1 and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "invalid input")
    require(
        fitted["mode"] in MODES
        and fitted["ridge"] == RIDGE
        and fitted["clip"] == EPS
        and np.isfinite([fitted["intercept"], fitted["slope"]]).all()
        and fitted["slope"] >= 0
        and (fitted["mode"] != "intercept" or fitted["slope"] == 1),
        "invalid fitted correction",
    )
    return expit(fitted["intercept"] + fitted["slope"] * logit(np.clip(p, EPS, 1 - EPS)))


def components(records):
    validate(records)
    groups = grouped_folds(scoring_records(records))["components"]
    # Check every complement before the first fit, including late components.
    for group in groups:
        require(
            {r["observed_buy"] for r in records if r["session"] not in group} == {0, 1},
            "constant-label training complement",
        )
    return groups


def cross_validate(records):
    groups = components(records)
    raw = np.array([r["probability_buy"] for r in records], dtype=float)
    predictions = {"raw": raw.copy()}
    predictions.update({k: np.full(len(records), np.nan) for k in (*MODES, "training_constant")})
    coverage = np.zeros(len(records), dtype=int)
    models = []
    for index, group in enumerate(groups):
        held = [i for i, r in enumerate(records) if r["session"] in group]
        train = [r for r in records if r["session"] not in group]
        require(
            {r["participant"] for r in train}.isdisjoint({records[i]["participant"] for i in held}),
            "recorded participant crossed components",
        )
        fitted = {mode: fit(train, mode) for mode in MODES}
        for mode, model in fitted.items():
            predictions[mode][held] = predict(model, raw[held])
        predictions["training_constant"][held] = fitted["intercept"]["train_constant"]
        coverage[held] += 1
        models.append({"component": index, "held_sessions": group, "models": fitted})
    require(
        np.all(coverage == 1)
        and all(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all() for p in predictions.values()),
        "incomplete corrected predictions",
    )
    return {"components": groups, "models": models, "predictions": predictions}
