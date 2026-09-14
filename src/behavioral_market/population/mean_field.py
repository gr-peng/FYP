def compute_mean_field(agents, decisions=(), as_of_session=None):
    n = len(agents)
    if not n:
        raise ValueError("population must be nonempty")
    buy = sum(d.quantity for d in decisions if d.action == "buy")
    sell = sum(d.quantity for d in decisions if d.action == "sell")
    return {
        "as_of_session": as_of_session,
        "fundamental_share": sum(a.current_style == "fundamental" for a in agents) / n,
        "technical_share": sum(a.current_style == "technical" for a in agents) / n,
        "positive_sentiment_share": sum(
            d.sentiment in {"bullish", "strongly_bullish"} for d in decisions
        )
        / n,
        "negative_sentiment_share": sum(
            d.sentiment in {"bearish", "strongly_bearish"} for d in decisions
        )
        / n,
        "buy_sell_imbalance": (buy - sell) / (buy + sell) if buy + sell else 0.0,
        "average_pnl": sum(float(a.portfolio.total_equity) / a.initial_equity - 1 for a in agents)
        / n,
    }
