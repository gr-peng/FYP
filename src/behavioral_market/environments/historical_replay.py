from decimal import Decimal


class HistoricalReplayEnvironment:
    """End-of-session OHLC touch at limit-price settlement; orders have no price impact."""

    def __init__(self, bars, start):
        self.bars = bars.copy(deep=True).sort_values("session_date")
        self.history = self.bars.loc[self.bars.session_date < start].copy()

    def settle(self, session, agents, decisions, arrival_order):
        row = self.bars.loc[self.bars.session_date == session].iloc[0]
        fills, orders = [], []
        for index in arrival_order:
            agent, decision = agents[index], decisions[index]
            order_id = f"H-{session}-{agent.persona.agent_id}"
            filled = (
                decision.action == "buy"
                and decision.limit_price >= float(row.low)
                or decision.action == "sell"
                and decision.limit_price <= float(row.high)
            )
            if filled:
                apply = (
                    agent.portfolio.apply_buy
                    if decision.action == "buy"
                    else agent.portfolio.apply_sell
                )
                apply(Decimal(str(decision.limit_price)), decision.quantity)
                fills.append(
                    {
                        "trade_id": order_id,
                        "session_date": session,
                        "buyer_id": agent.persona.agent_id
                        if decision.action == "buy"
                        else "external_market",
                        "seller_id": agent.persona.agent_id
                        if decision.action == "sell"
                        else "external_market",
                        "price": decision.limit_price,
                        "quantity": decision.quantity,
                    }
                )
            orders.append(
                {
                    "agent_id": agent.persona.agent_id,
                    "order_id": order_id,
                    "status": "filled"
                    if filled
                    else ("hold" if decision.action == "hold" else "expired"),
                    "filled_quantity": decision.quantity if filled else 0,
                }
            )
        for agent in agents:
            agent.portfolio.mark_to_market(row.close)
        self.history = self.bars.loc[self.bars.session_date <= session].copy()
        return float(row.close), int(row.volume), orders, fills
