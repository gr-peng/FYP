"""Partial financial observations before identifiable active order submissions.

This is a conditional buy-versus-sell target. It does not infer HOLD opportunities,
unobserved order books, available funds, or cancelled-order timestamps.
"""

from collections import Counter

import numpy as np
import pandas as pd

GROUP = ["session", "repeat", "group", "period"]
TREATMENTS = {
    1: {"task": "video", "condition": "calm"},
    2: {"task": "video", "condition": "intense"},
    3: {"task": "stroop", "condition": "congruent"},
    4: {"task": "stroop", "condition": "incongruent"},
}


def prefix(values):
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def quote_states(orders, linked_trades, ledger, market_periods, markets, selected_ids=None):
    """Return separated observations/targets with conservative second-level eligibility.

    `orders` is the audited normalized table; accounting has already been checked.
    Current/end-of-period balances and target order attributes never enter state.
    `selected_ids` only narrows output, not histories or tie/ambiguity checks.
    """
    actual = orders[orders.is_replacement == 0].copy()
    grouped_orders = {k: g for k, g in actual.groupby(GROUP)}
    grouped_trades = {
        k: g.sort_values("created", kind="stable") for k, g in linked_trades.groupby(GROUP)
    }
    accounts = ledger.set_index(GROUP + ["player"])
    treatment = markets.set_index(["slab", "mid"])["cond"].to_dict()
    histories = {
        k: g.sort_values("period")
        for k, g in market_periods.groupby(["session", "repeat", "group"])
    }
    wanted = set(selected_ids) if selected_ids is not None else None
    results, exclusions = [], Counter()
    empty = linked_trades.iloc[:0]
    for key, group in sorted(grouped_orders.items()):
        trades = grouped_trades.get(key, empty)
        times = trades.created.to_numpy()
        prices = trades.price.to_numpy()
        price_prefix = prefix(prices)
        second_counts = group.groupby("created").size()
        own_orders = {
            p: g.sort_values("created", kind="stable") for p, g in group.groupby("player")
        }
        own_trade_prefixes = {}
        for player in group.player.unique():
            signs = trades.bid_player.eq(player).astype(int) - trades.ask_player.eq(player).astype(
                int
            )
            own_trade_prefixes[player] = (
                prefix(-signs.to_numpy() * trades.quantity.to_numpy() * prices),
                prefix(signs.to_numpy() * trades.quantity.to_numpy()),
            )
        period_history = histories[key[:3]]
        # Only complete earlier periods. Current/future prices and dividends stay out.
        previous = period_history[period_history.period < key[3]]
        public_history = [
            {
                "period": int(row.period),
                "mean_trade_price": None
                if pd.isna(row.reported_average)
                else float(row.reported_average),
                "closing_price": None if pd.isna(row.reported_close) else float(row.reported_close),
                "dividend": int(row.dividend),
            }
            for row in previous.itertuples(index=False)
        ]
        for target in group.itertuples(index=False):
            if wanted is not None and target.uuid not in wanted:
                continue
            if not 0 <= target.created <= 120:
                exclusions["outside_nominal_period"] += 1
                continue
            if second_counts.loc[target.created] != 1:
                exclusions["multiple_submissions_in_target_second"] += 1
                continue
            start = int(np.searchsorted(times, target.created, side="left"))
            end = int(np.searchsorted(times, target.created, side="right"))
            simultaneous = trades.iloc[start:end]
            # Allow the fill caused by this market submission; every other trade in
            # the same second could precede the decision and makes the state ambiguous.
            if len(simultaneous) and not (
                len(simultaneous) == 1
                and target.kind == "market"
                and (
                    simultaneous.ask_uuid.eq(target.uuid) | simultaneous.bid_uuid.eq(target.uuid)
                ).all()
            ):
                exclusions["other_trade_in_target_second"] += 1
                continue
            account = accounts.loc[(*key, target.player)]
            cash_flow, asset_flow = own_trade_prefixes[target.player]
            cash = float(account.start_cash + cash_flow[start])
            assets = float(account.start_assets + asset_flow[start])
            if cash < 0 or assets < 0:
                exclusions["negative_reconstructed_account"] += 1
                continue
            own = own_orders[target.player]
            own_past = own[own.created < target.created]
            last_second = int(times[start - 1]) if start else None
            if start:
                last_begin = int(np.searchsorted(times, last_second, side="left"))
                last_mean = float(
                    (price_prefix[start] - price_prefix[last_begin]) / (start - last_begin)
                )
            else:
                last_mean = None
            state = {
                "information_scope": "Recorded partial observation; current order book, pending "
                "orders and cancellations are unavailable.",
                "repetition": int(key[1]),
                "period": int(key[3]),
                "total_periods": 10,
                "elapsed_seconds": int(target.created),
                "period_seconds": 120,
                "cash": cash,
                "shares": assets,
                "displayed_expected_remaining_dividends_per_share": float(5 * (11 - key[3])),
                "pre_market_task": dict(TREATMENTS[int(treatment[(key[0], key[2])])]),
                "earlier_periods": public_history,
                "earlier_current_period_trades": {
                    "count": start,
                    "mean_price": float(price_prefix[start] / start) if start else None,
                    "last_completed_trade_second": last_second,
                    "mean_price_in_last_completed_trade_second": last_mean,
                },
                "own_earlier_current_period_submissions": {
                    "buy": int(own_past.side.eq("bid").sum()),
                    "sell": int(own_past.side.eq("ask").sum()),
                },
            }
            results.append(
                {
                    "provenance": {
                        "order_id": target.uuid,
                        "session": key[0],
                        "repeat": int(key[1]),
                        "group": int(key[2]),
                        "period": int(key[3]),
                        "player": int(target.player),
                        "participant": account.participant,
                        "second": int(target.created),
                    },
                    "state": state,
                    "target": {
                        "buy": int(target.side == "bid"),
                        "kind": target.kind,
                        "quantity": int(target.quantity),
                        "price": int(target.price),
                    },
                }
            )
    return results, dict(exclusions)
