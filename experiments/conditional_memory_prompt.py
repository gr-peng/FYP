"""Controlled information for conditional BUY/SELL probability questions.

Pure construction only: no dataset loader, model client, credentials, or fitting.
Arm suffixes count computed features, not all visible observation fields.
Only ``messages`` belongs in a future request; other fields are local bookkeeping.
"""

import json
import math

from experiments import conditional_order_memory as memory

ARMS = tuple(memory.ARMS)
MENUS = {"buy_first": ("BUY", "SELL"), "sell_first": ("SELL", "BUY")}
INSTRUCTIONS = (
    "Forecast the BUY versus SELL side of an active order, conditional on the participant "
    "already having decided to submit one. HOLD is not an option. This is a probability "
    "forecast, not an optimal action or a trading recommendation. The supplied observations "
    "may be synthetic research scenarios. Do not treat them as verified human records. "
    "The specified task is a continuous double auction with ten 120-second periods per "
    "repetition and two repetitions with account resets. Each share pays a period dividend "
    "of 0 or 10 with equal probabilities. Expected remaining dividends are displayed. "
    "Shares expire after period 10; there is no guaranteed terminal resale value. Cash "
    "and shares are total balances, not necessarily available balances. Resting orders, "
    "cancellations, and the current order book are unobserved. Public history and personal "
    "submission counts, when supplied, refer only to earlier observations. Personal counts "
    "are order submissions, not completed trades. Last-completed-period counts are a subset "
    "of the completed-history totals; do not add them again. Omitted history is withheld or "
    "unknown, not evidence of zero activity. Account balances and public history still "
    "reflect past events even when explicit personal counts are withheld. Computed features "
    "are deterministic summaries of the supplied observations, not extra examples or answers. "
    "Use the supplied menu labels. Return only a JSON object with keys A and B and numeric "
    "values, each a probability from 0 to 1, summing to 1. Do not output HOLD, quantities, prices, "
    "strategy names, explanations, or a predicted realized label."
)
require = memory.require


def number(value, name, *, low=0, high=None, integer=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= low
        and (high is None or value <= high)
        and (not integer or value == int(value)),
        "invalid " + name,
    )
    return int(value) if integer else float(value)


def price(value, name):
    if value is None:
        return None
    result = number(value, name)
    require(result > 0, "nonpositive " + name)
    return result


def mapping(value, name):
    require(isinstance(value, dict), "invalid " + name)
    return value


def public_state(state):
    """Reconstruct every selected nested value; arbitrary extras are never copied."""
    state = mapping(state, "state")
    period = number(state["period"], "period", low=1, high=10, integer=True)
    repetition = number(state["repetition"], "repetition", low=1, high=2, integer=True)
    elapsed = number(state["elapsed_seconds"], "elapsed_seconds", high=120, integer=True)
    cash = number(state["cash"], "cash")
    shares = number(state["shares"], "shares", integer=True)
    fv = number(state["displayed_expected_remaining_dividends_per_share"], "expected dividends")
    require(fv == 5 * (11 - period), "expected dividends inconsistent with specified task")
    task = mapping(state["pre_market_task"], "pre-market task")
    pair = (task["task"], task["condition"])
    require(
        pair
        in (
            ("video", "calm"),
            ("video", "intense"),
            ("stroop", "congruent"),
            ("stroop", "incongruent"),
        ),
        "invalid task category",
    )
    trades = mapping(state["earlier_current_period_trades"], "public trade history")
    count = number(trades["count"], "public trade count", integer=True)
    mean = price(trades["mean_price"], "mean price")
    last_mean = price(trades["mean_price_in_last_completed_trade_second"], "last mean price")
    last_second = trades["last_completed_trade_second"]
    if count == 0:
        require(mean is None and last_mean is None and last_second is None, "empty trade history")
    else:
        last_second = number(last_second, "last trade second", high=120, integer=True)
        require(
            mean is not None and last_mean is not None and last_second < elapsed,
            "public trade history not strictly earlier",
        )
        require(
            (count != 1 and last_second != 0) or mean == last_mean,
            "public means differ although all trades share the last trade second",
        )
    earlier = state["earlier_periods"]
    require(isinstance(earlier, list), "invalid earlier periods")
    public_periods = []
    for row in earlier:
        row = mapping(row, "earlier period")
        previous = number(row["period"], "earlier period", low=1, high=10, integer=True)
        dividend = number(row["dividend"], "realized earlier dividend", integer=True)
        require(dividend in (0, 10), "invalid earlier dividend")
        public_periods.append(
            {
                "period": previous,
                "mean_trade_price": price(row["mean_trade_price"], "earlier mean price"),
                "closing_price": price(row["closing_price"], "earlier closing price"),
                "dividend": dividend,
            }
        )
    require(
        [r["period"] for r in public_periods] == list(range(1, period)),
        "earlier periods must cover completed periods in order, never current or future",
    )
    return {
        "cash": cash,
        "shares": shares,
        "displayed_expected_remaining_dividends_per_share": fv,
        "period": period,
        "elapsed_seconds": elapsed,
        "repetition": repetition,
        "pre_market_task": {"task": pair[0], "condition": pair[1]},
        "earlier_current_period_trades": {
            "count": count,
            "mean_price": mean,
            "last_completed_trade_second": last_second,
            "mean_price_in_last_completed_trade_second": last_mean,
        },
        "earlier_periods": public_periods,
    }


def observation(state, arm):
    require(arm in ARMS, "unknown information arm")
    selected = public_state(state)
    if arm != ARMS[0]:
        selected[memory.CURRENT] = memory.counts(state.get(memory.CURRENT))
        require(
            selected["elapsed_seconds"] > 0 or sum(selected[memory.CURRENT].values()) == 0,
            "own current history before period begins",
        )
    if arm == ARMS[2]:
        completed = mapping(state.get(memory.COMPLETED), "completed history")
        last = completed["last_completed_period_with_own_orders"]
        selected[memory.COMPLETED] = {
            "earlier_periods_this_repetition": memory.counts(
                completed.get("earlier_periods_this_repetition")
            ),
            "previous_repetitions": memory.counts(completed.get("previous_repetitions")),
            "last_completed_period_with_own_orders": None if last is None else memory.counts(last),
        }
        # Preserve the accepted aggregate consistency checks; no extra fields survive.
        selected = memory.project(selected, arm)
        m = selected[memory.COMPLETED]
        require(
            selected["period"] > 1 or sum(m["earlier_periods_this_repetition"].values()) == 0,
            "completed current-repetition history before first period",
        )
        require(
            selected["repetition"] > 1 or sum(m["previous_repetitions"].values()) == 0,
            "previous repetition history before first repetition",
        )
        require(
            selected["period"] != 2
            or sum(m["earlier_periods_this_repetition"].values()) == 0
            or m["last_completed_period_with_own_orders"] == m["earlier_periods_this_repetition"],
            "single completed current-repetition period must equal its total",
        )
    values = memory.features(selected, arm).tolist()
    return {
        "partial_observation": selected,
        "computed_features": dict(zip(memory.ARMS[arm], values, strict=True)),
    }


def build_question(state, arm, order):
    require(order in MENUS, "unknown menu order")
    sides = MENUS[order]
    payload = {
        "observation": observation(state, arm),
        "choices": {
            label: f"Submit an active {side} order."
            for label, side in zip(("A", "B"), sides, strict=True)
        },
    }
    content = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": content},
        ],
        "buy_label": "A" if order == "buy_first" else "B",
        "arm": arm,
        "menu_order": order,
        "scope": "construction_only_no_data_or_model_validation",
    }


def build_triplet(state, order):
    """All three must validate before any triplet is returned; does no I/O."""
    return {arm: build_question(state, arm, order) for arm in ARMS}
