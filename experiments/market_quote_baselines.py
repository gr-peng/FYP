"""Session/person-balanced statistical controls on partial financial observations."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

FEATURES = (
    "cash_fraction_at_fundamental_value",
    "log_fundamental_wealth",
    "log_shares",
    "period_fraction",
    "elapsed_fraction",
    "second_repetition",
    "log_public_trade_count",
    "mean_price_over_fv",
    "last_second_mean_over_fv",
    "no_prior_trade",
    "previous_period_mean_over_fv",
    "no_previous_period_price",
    "own_buy_fraction_laplace",
    "log_own_submissions",
    "intense_video",
    "congruent_stroop",
    "incongruent_stroop",
)


def features(state):
    cash, shares = state["cash"], state["shares"]
    fv = state["displayed_expected_remaining_dividends_per_share"]
    wealth = cash + shares * fv
    history = state["earlier_current_period_trades"]
    own = state["own_earlier_current_period_submissions"]
    previous = (
        state["earlier_periods"][-1]["mean_trade_price"] if state["earlier_periods"] else None
    )
    task = state["pre_market_task"]
    if cash < 0 or shares < 0 or fv <= 0:
        raise ValueError("invalid financial observation")
    return np.array(
        [
            cash / max(1, wealth),
            math.log1p(wealth),
            math.log1p(shares),
            state["period"] / 10,
            state["elapsed_seconds"] / 120,
            float(state["repetition"] == 2),
            math.log1p(history["count"]),
            history["mean_price"] / fv if history["mean_price"] is not None else 0,
            history["mean_price_in_last_completed_trade_second"] / fv
            if history["mean_price_in_last_completed_trade_second"] is not None
            else 0,
            float(history["count"] == 0),
            previous / fv if previous is not None else 0,
            float(previous is None),
            (own["buy"] + 1) / (own["buy"] + own["sell"] + 2),
            math.log1p(own["buy"] + own["sell"]),
            float(task == {"task": "video", "condition": "intense"}),
            float(task == {"task": "stroop", "condition": "congruent"}),
            float(task == {"task": "stroop", "condition": "incongruent"}),
        ]
    )


def weights(records):
    sessions, counts = {}, {}
    for record in records:
        p = record["provenance"]
        sessions.setdefault(p["session"], set()).add(p["participant"])
        key = p["session"], p["participant"]
        counts[key] = counts.get(key, 0) + 1
    return np.array(
        [
            1
            / (
                len(sessions)
                * len(sessions[r["provenance"]["session"]])
                * counts[r["provenance"]["session"], r["provenance"]["participant"]]
            )
            for r in records
        ]
    )


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
        raise ValueError(f"financial baseline fit failed: {result.message}")
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
    x = np.array([features(r["state"]) for r in records])
    z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    own = [r["state"]["own_earlier_current_period_submissions"] for r in records]
    return {
        "constant_half": np.full(len(records), 0.5),
        "training_constant": np.full(len(records), model["train_constant"]),
        "own_history_prior4": np.array(
            [(r["buy"] + 4 * model["train_constant"]) / (r["buy"] + r["sell"] + 4) for r in own]
        ),
        "ridge_logistic": expit(z @ model["beta"]),
    }


def metrics(records, predictions):
    y = np.array([r["target"]["buy"] for r in records])
    w = weights(records)
    result = {}
    for name, p in predictions.items():
        p = np.asarray(p)
        if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("invalid financial probabilities")
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        result[name] = {
            "brier_session_person_balanced": float(w @ (p - y) ** 2),
            "log_loss_session_person_balanced": float(
                -w @ (y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
            ),
            "predicted_buy_rate_session_person_balanced": float(w @ p),
        }
    return {
        "records": len(records),
        "human_buy_rate_session_person_balanced": float(w @ y),
        "methods": result,
    }
