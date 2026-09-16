import numpy as np


def path_metrics(prices):
    values = np.asarray(prices, dtype=float)
    returns = values[1:] / values[:-1] - 1
    drawdowns = 1 - values / np.maximum.accumulate(values)
    return {
        "return": float(values[-1] / values[0] - 1),
        "daily_volatility": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "max_drawdown": float(drawdowns.max()),
        "final_price": float(values[-1]),
    }


def run_metrics(market, orders, switches, initial_price, initial_shares):
    prices = [initial_price, *market.close.astype(float)]
    result = path_metrics(prices)
    submitted = int(orders.quantity.sum())
    result.update(
        buy_rate=float((orders.action == "buy").mean()),
        sell_rate=float((orders.action == "sell").mean()),
        hold_rate=float((orders.action == "hold").mean()),
        switch_rate=float((switches.decision == "switch").mean()) if len(switches) else 0.0,
        switching_opportunities=len(switches),
        submitted_quantity=submitted,
        filled_order_quantity=int(orders.filled_quantity.sum()),
        order_execution_rate=float(orders.filled_quantity.sum() / submitted) if submitted else 0.0,
        constraint_violation_rate=float(
            (orders.validation_status == "constraint_violation").mean()
        ),
        invalid_order_rate=float(
            (~orders.validation_status.isin(["ok", "mock", "rule"])).mean()
        ),
        price_band_violation_rate=float(
            (orders.validation_status == "price_band_violation").mean()
        ),
        fallback_rate=float(orders.fallback.mean()),
        mean_order_quantity=float(orders.quantity.mean()),
        mean_aggressiveness=float(orders.aggressiveness.mean()),
        turnover=float(market.simulated_volume.sum() / initial_shares),
        mean_abs_fundamental_deviation=float(
            (market.close / market.fundamental_value - 1).abs().mean()
        ),
        mean_order_imbalance=float(market.order_imbalance.mean()),
    )
    if "prior_majority_action" in orders:
        eligible = orders.prior_majority_action.isin(["buy", "sell"])
        active = eligible & orders.action.isin(["buy", "sell"])
        followed = orders.action == orders.prior_majority_action
        result.update(
            majority_signal_opportunities=int(eligible.sum()),
            follow_majority_rate=float(followed[eligible].mean()) if eligible.any() else None,
            active_follow_majority_rate=float(followed[active].mean()) if active.any() else None,
        )
    if "seen_best_ask" in orders:
        buys = orders[(orders.action == "buy") & orders.seen_best_ask.notna()]
        sells = orders[(orders.action == "sell") & orders.seen_best_bid.notna()]
        result.update(
            mean_bid_distance_to_ask_bps=float(buys.bid_distance_to_ask_bps.mean())
            if len(buys)
            else None,
            mean_ask_distance_to_bid_bps=float(sells.ask_distance_to_bid_bps.mean())
            if len(sells)
            else None,
            mean_seen_spread_bps=float(orders.seen_spread_bps.mean())
            if orders.seen_spread_bps.notna().any()
            else None,
        )
    group_key = "session_date" if "session_date" in orders else orders.index.to_series() * 0
    by_day = orders.groupby(group_key).action.agg(
        lambda values: {"buy", "sell"}.issubset(set(values))
    )
    result["both_sides_session_rate"] = float(by_day.mean()) if len(by_day) else 0.0
    for day in ("2026-07-01", "2026-07-02", "2026-07-10"):
        indices = market.index[market.session_date == day]
        if len(indices):
            i = indices[0]
            previous = prices[i]
            result[f"return_{day}"] = prices[i + 1] / previous - 1
            result[f"three_session_drawdown_{day}"] = max(
                0.0, 1 - min(prices[i + 1 : i + 4]) / previous
            )
    after = market.loc[market.session_date >= "2026-07-02", "close"].astype(float)
    before = market.loc[market.session_date < "2026-07-02", "close"].astype(float)
    if len(after) and len(before):
        anchor = before.iloc[-1]
        trough = int(np.argmin(after.to_numpy()))
        recovered = np.where(after.iloc[trough:].to_numpy() >= anchor)[0]
        result["recovery_sessions_from_trough"] = int(recovered[0]) if len(recovered) else None
        result["post_shock_recovery_slope"] = (
            float(np.polyfit(np.arange(len(after)), after / anchor, 1)[0])
            if len(after) > 1
            else None
        )
    return result


def event_study(bars, symbol, benchmark="QQQ", event="2026-07-01"):
    prices = (
        bars.pivot(index="session_date", columns="symbol", values="close")
        .astype(float)
        .sort_index()
    )
    returns = prices.pct_change()
    pre = returns.loc[returns.index < "2026-06-15", [symbol, benchmark]].dropna()
    beta, alpha = np.polyfit(pre[benchmark], pre[symbol], 1)
    abnormal = returns[symbol] - (alpha + beta * returns[benchmark])
    center = list(prices.index).index(event)
    result = {
        "symbol": symbol,
        "event": event,
        "benchmark": benchmark,
        "estimation_window": "2026-04-01..2026-06-12",
        "beta": float(beta),
        "alpha": float(alpha),
        "event_day_return": float(returns[symbol].loc[event]),
    }
    for radius in (1, 3):
        window = returns[symbol].iloc[center - radius : center + radius + 1]
        result[f"cumulative_return_{radius}"] = float((1 + window).prod() - 1)
        result[f"CAR_{radius}"] = float(abnormal.iloc[center - radius : center + radius + 1].sum())
    sample = prices.loc["2026-06-12":"2026-07-17", symbol]
    result.update(path_metrics(sample))
    volume = bars.loc[bars.symbol == symbol].set_index("session_date").volume.astype(float)
    result["event_volume_ratio_previous20"] = float(
        volume.loc[event] / volume.loc[volume.index < event].tail(20).mean()
    )
    return result
