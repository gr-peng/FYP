"""Draft fixed training baselines for the extended personal-memory observation."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.market_quote_baselines import FEATURES as BASE_FEATURES
from experiments.market_quote_baselines import features as base_features
from experiments.market_quote_baselines import weights

FEATURES = (
    *BASE_FEATURES,
    "earlier_period_buy_fraction",
    "log_earlier_period_orders",
    "earlier_repetition_buy_fraction",
    "log_earlier_repetition_orders",
    "last_completed_period_buy_fraction",
    "log_last_completed_period_orders",
)


def features(state):
    memory = state["own_completed_period_memory"]
    groups = [
        memory["earlier_periods_this_repetition"],
        memory["previous_repetitions"],
        memory["last_completed_period_with_own_orders"] or {"buy": 0, "sell": 0},
    ]
    extended = []
    for counts in groups:
        n = counts["buy"] + counts["sell"]
        extended.extend([(counts["buy"] + 1) / (n + 2), math.log1p(n)])
    return np.concatenate([base_features(state), extended])


def fit(records, ridge=0.01):
    x = np.array([features(r["state"]) for r in records])
    y = np.array([r["target"]["buy"] for r in records], dtype=float)
    w = weights(records)
    mean = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0))
    scale[scale < 1e-12] = 1
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])

    def objective(beta):
        logits = z @ beta
        loss = np.sum(w * (np.logaddexp(0, logits) - y * logits))
        loss += ridge / 2 * np.sum(beta[1:] ** 2)
        gradient = z.T @ (w * (expit(logits) - y))
        gradient[1:] += ridge * beta[1:]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(z.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise ValueError("memory baseline failed to converge")
    return {
        "features": list(FEATURES),
        "ridge": ridge,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "beta": result.x.tolist(),
        "train_constant": float(w @ y),
        "training_records": len(records),
    }


def predict(model, records):
    if model["features"] != list(FEATURES):
        raise ValueError("incompatible feature schema")
    x = np.array([features(r["state"]) for r in records])
    z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    history = []
    for r in records:
        s = r["state"]
        m = s["own_completed_period_memory"]
        counts = [
            s["own_earlier_current_period_submissions"],
            m["earlier_periods_this_repetition"],
            m["previous_repetitions"],
        ]
        buys = sum(c["buy"] for c in counts)
        total = buys + sum(c["sell"] for c in counts)
        history.append((buys + 4 * model["train_constant"]) / (total + 4))
    return {
        "training_constant": np.full(len(records), model["train_constant"]),
        "lifetime_history_prior4": np.array(history),
        "memory_ridge_logistic": expit(z @ model["beta"]),
    }
