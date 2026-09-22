"""Predeclared statistical baselines for conditional sell-any probabilities."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

FEATURES = (
    "unrealized_return",
    "gain_indicator",
    "last_return",
    "three_step_return",
    "portfolio_weight",
    "cash_fraction",
    "period_fraction",
    "log_quantity",
)


def features(item):
    state, asset = item["state"], item["asset"]
    price = state["current_prices"][asset]
    quantity = state["holdings"][asset]
    basis = state["average_acquisition_prices"][asset]
    history = state["price_history"]
    wealth = state["cash"] + sum(
        state["holdings"][a] * p for a, p in state["current_prices"].items()
    )
    return np.array(
        [
            price / basis - 1,
            float(price > basis),
            price / history[-2][asset] - 1,
            price / history[-4][asset] - 1,
            quantity * price / wealth,
            state["cash"] / wealth,
            state["main_period"] / state["total_main_periods"],
            math.log1p(quantity),
        ]
    )


def person_weights(items):
    people = [x["provenance"]["participant"] for x in items]
    counts = {p: people.count(p) for p in set(people)}
    return np.array([1 / (len(counts) * counts[p]) for p in people])


def gain_bin(item):
    state, asset = item["state"], item["asset"]
    gap = state["current_prices"][asset] - state["average_acquisition_prices"][asset]
    return "gain" if gap > 1e-9 else "loss" if gap < -1e-9 else "flat"


def fit(items):
    x = np.array([features(item) for item in items])
    y = np.array([item["sell_any"] for item in items])
    w = person_weights(items)
    mean = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * w[:, None], axis=0))
    scale[scale < 1e-12] = 1
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = 0.01

    def objective(beta):
        logits = z @ beta
        loss = np.sum(w * (np.logaddexp(0, logits) - y * logits))
        loss += penalty / 2 * np.sum(beta[1:] ** 2)
        gradient = z.T @ (w * (expit(logits) - y))
        gradient[1:] += penalty * beta[1:]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(z.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise ValueError(f"statistical fit failed: {result.message}")
    bins = {}
    for name in ("gain", "loss", "flat"):
        mask = np.array([gain_bin(item) == name for item in items])
        # One pseudo-observation at the person-balanced global mean; train only.
        global_p = float(np.sum(w * y))
        effective = w * len(items)
        bins[name] = float(
            (np.sum(effective[mask] * y[mask]) + global_p) / (np.sum(effective[mask]) + 1)
        )
    return {
        "features": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "beta": result.x.tolist(),
        "ridge_penalty": penalty,
        "train_constant": float(np.sum(w * y)),
        "gain_bins": bins,
        "training_opportunities": len(items),
        "training_people": len({x["provenance"]["participant"] for x in items}),
    }


def predict(model, items):
    x = np.array([features(item) for item in items])
    z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    return {
        "hold": np.zeros(len(items)),
        "coin": np.full(len(items), 0.5),
        "train_constant": np.full(len(items), model["train_constant"]),
        "gain_bins": np.array([model["gain_bins"][gain_bin(item)] for item in items]),
        "ridge_logistic": expit(z @ model["beta"]),
    }


def metrics(items, predictions):
    y = np.array([x["sell_any"] for x in items])
    w = person_weights(items)
    report = {}
    for name, p in predictions.items():
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        report[name] = {
            "brier_person_balanced": float(np.sum(w * (p - y) ** 2)),
            "brier_opportunity_mean": float(np.mean((p - y) ** 2)),
            "log_loss_person_balanced": float(
                -np.sum(w * (y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
            ),
            "predicted_sell_rate_person_balanced": float(np.sum(w * p)),
        }
    return {
        "opportunities": len(items),
        "people": len({x["provenance"]["participant"] for x in items}),
        "sessions": len({x["provenance"]["session"] for x in items}),
        "human_sell_rate_person_balanced": float(np.sum(w * y)),
        "methods": report,
    }
