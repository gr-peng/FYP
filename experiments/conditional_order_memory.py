"""Explicit own-action information masks for conditional BUY/SELL prediction.

Pure in-memory functions: no loaders, network, credentials or empirical runner.
Fifteen features omit explicit own-action counts, not all traces of the past.
Temporal validity of aggregates still depends on the separately checked exporter.
"""

import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments import market_quote_baselines as base
from experiments import market_quote_memory_baselines as extended

CURRENT_FEATURES = {"own_buy_fraction_laplace", "log_own_submissions"}
ARMS = {
    "account_public_task_15": tuple(name for name in base.FEATURES if name not in CURRENT_FEATURES),
    "plus_current_period_17": base.FEATURES,
    "plus_completed_history_23": extended.FEATURES,
}
COMMON_FIELDS = (
    "cash",
    "shares",
    "displayed_expected_remaining_dividends_per_share",
    "earlier_current_period_trades",
    "earlier_periods",
    "pre_market_task",
    "period",
    "elapsed_seconds",
    "repetition",
)
CURRENT = "own_earlier_current_period_submissions"
COMPLETED = "own_completed_period_memory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(value):
    require(isinstance(value, dict), "missing or unknown history is not zero history")
    require(all(name in value for name in ("buy", "sell")), "incomplete history counts")
    require(
        all(type(value[name]) is int and value[name] >= 0 for name in ("buy", "sell")),
        "history counts must be nonnegative integers",
    )
    return {name: value[name] for name in ("buy", "sell")}


def project(state, arm):
    require(arm in ARMS, "unknown information arm")
    result = {name: copy.deepcopy(state[name]) for name in COMMON_FIELDS}
    if arm != "account_public_task_15":
        require(CURRENT in state, "missing current history")
        result[CURRENT] = counts(state[CURRENT])
    if arm == "plus_completed_history_23":
        require(
            COMPLETED in state and isinstance(state[COMPLETED], dict), "missing completed history"
        )
        m = state[COMPLETED]
        require("last_completed_period_with_own_orders" in m, "missing last-period history")
        earlier = counts(m.get("earlier_periods_this_repetition"))
        previous = counts(m.get("previous_repetitions"))
        last = m["last_completed_period_with_own_orders"]
        recent = earlier if sum(earlier.values()) else previous
        if last is None:
            require(
                sum(recent.values()) == 0, "inconsistent completed history: missing last period"
            )
        else:
            last = counts(last)
            require(
                sum(last.values()) > 0
                and all(last[side] <= recent[side] for side in ("buy", "sell")),
                "inconsistent completed history: last period exceeds most recent nonempty group",
            )
        result[COMPLETED] = {
            "earlier_periods_this_repetition": earlier,
            "previous_repetitions": previous,
            # None is the exporter's explicit known absence of any earlier period
            # with own orders. An absent key or unknown counts is rejected above.
            "last_completed_period_with_own_orders": last,
        }
    return result


def features(state, arm):
    selected = project(state, arm)
    if arm == "account_public_task_15":
        # These two placeholder-derived values are discarded before fitting.
        # The actual own-history field is neither read nor retained in this arm.
        selected[CURRENT] = {"buy": 0, "sell": 0}
        values = base.features(selected)
        values = values[[i for i, name in enumerate(base.FEATURES) if name not in CURRENT_FEATURES]]
    elif arm == "plus_current_period_17":
        values = base.features(selected)
    else:
        values = extended.features(selected)
    require(
        values.shape == (len(ARMS[arm]),) and np.isfinite(values).all(), "invalid feature vector"
    )
    return values


def record_keys(records, *, labels):
    require(bool(records), "empty records")
    keys = []
    for row in records:
        p = row["provenance"]
        require(
            all(isinstance(p.get(k), str) and p[k] for k in ("session", "participant", "order_id")),
            "missing target identity",
        )
        keys.append((p["session"], p["order_id"]))
        if labels:
            require(
                type(row["target"]["buy"]) is int and row["target"]["buy"] in (0, 1),
                "invalid conditional order label",
            )
    require(len(set(keys)) == len(keys), "duplicate targets")
    return keys


def fit(records, arm):
    """Refit each declared arm and its scaler on training records only; ridge fixed .01."""
    record_keys(records, labels=True)
    x = np.array([features(r["state"], arm) for r in records])
    y = np.array([r["target"]["buy"] for r in records], dtype=float)
    w = base.weights(records)
    mean = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0))
    scale[scale < 1e-12] = 1
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])

    def objective(beta):
        logits = z @ beta
        loss = w @ (np.logaddexp(0, logits) - y * logits) + 0.01 / 2 * np.sum(beta[1:] ** 2)
        gradient = z.T @ (w * (expit(logits) - y))
        gradient[1:] += 0.01 * beta[1:]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(z.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    require(result.success and np.isfinite(result.x).all(), "memory contrast fit failed")
    return {
        "arm": arm,
        "features": list(ARMS[arm]),
        "ridge": 0.01,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "beta": result.x.tolist(),
        "train_constant": float(w @ y),
        "training_records": len(records),
    }


def predict(model, records):
    record_keys(records, labels=False)
    arm = model["arm"]
    require(
        arm in ARMS and model["features"] == list(ARMS[arm]) and model["ridge"] == 0.01,
        "incompatible model or information mask",
    )
    mean, scale, beta = [np.array(model[k], dtype=float) for k in ("mean", "scale", "beta")]
    n = len(ARMS[arm])
    require(
        mean.shape == scale.shape == (n,)
        and beta.shape == (n + 1,)
        and all(np.isfinite(a).all() for a in (mean, scale, beta))
        and (scale > 0).all(),
        "invalid fitted model",
    )
    x = np.array([features(r["state"], arm) for r in records])
    p = expit(np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ beta)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "invalid probabilities")
    return p


def split_predictions(training, evaluation):
    """Engineering helper only; upstream must verify globally scoped participant IDs.

    Both repetitions of a person/session must stay together. A future empirical
    runner must separately establish linkage and approved train-only membership.
    This helper does not authorize opening any dataset or certify a fold map.
    """
    record_keys(training, labels=True)
    record_keys(evaluation, labels=True)
    for field in ("session", "participant"):
        left = {r["provenance"][field] for r in training}
        right = {r["provenance"][field] for r in evaluation}
        require(left.isdisjoint(right), "training/evaluation " + field + " crossing")
    models = {arm: fit(training, arm) for arm in ARMS}
    predictions = {arm: predict(model, evaluation) for arm, model in models.items()}
    prior = models["account_public_task_15"]["train_constant"]
    predictions["fold_training_constant"] = np.full(len(evaluation), prior)
    predictions["constant_half"] = np.full(len(evaluation), 0.5)
    short, long = [], []
    for row in evaluation:
        s = project(row["state"], "plus_completed_history_23")
        m = s[COMPLETED]
        short.append((s[CURRENT]["buy"] + 4 * prior) / (sum(s[CURRENT].values()) + 4))
        history = [s[CURRENT], m["earlier_periods_this_repetition"], m["previous_repetitions"]]
        buys = sum(c["buy"] for c in history)
        total = sum(sum(c.values()) for c in history)
        long.append((buys + 4 * prior) / (total + 4))
    predictions["current_history_prior4"] = np.array(short)
    predictions["combined_history_prior4"] = np.array(long)
    require(
        all(len(p) == len(evaluation) for p in predictions.values()), "incomplete common coverage"
    )
    return models, predictions
