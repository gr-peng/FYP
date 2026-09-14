from dataclasses import asdict

import pandas as pd

from behavioral_market.market import Exchange


class EndogenousEventEnvironment:
    """Own only pre-start real bars; all subsequent bars come from the CDA."""

    def __init__(self, bars, start, symbol, agents):
        self.history = (
            bars.loc[bars.session_date < start].copy(deep=True).sort_values("session_date")
        )
        self.exchange = Exchange(symbol, self.history.close.iloc[-1])
        for agent in agents:
            self.exchange.register(agent.persona.agent_id, agent.portfolio)
        self.initial_cash = sum(a.portfolio.cash for a in agents)
        self.initial_shares = sum(a.portfolio.position for a in agents)

    def settle(self, session, agents, decisions, arrival_order):
        opening = float(self.exchange.market_price)
        orders, trades = [], []
        for index in arrival_order:
            agent, decision = agents[index], decisions[index]
            if decision.action == "hold":
                orders.append(
                    {
                        "agent_id": agent.persona.agent_id,
                        "order_id": None,
                        "status": "hold",
                        "filled_quantity": 0,
                    }
                )
                continue
            order, fills = self.exchange.submit_order(
                agent.persona.agent_id, decision.action, decision.limit_price, decision.quantity
            )
            trades.extend(
                {**asdict(fill), "price": float(fill.price), "session_date": session}
                for fill in fills
            )
            orders.append({"agent_id": agent.persona.agent_id, "order": order})
        closing, volume = float(self.exchange.market_price), self.exchange.day_volume
        self.exchange.end_day()
        for row in orders:
            if "order" in row:
                order = row.pop("order")
                row.update(
                    order_id=order.order_id,
                    status=order.status.value,
                    filled_quantity=order.quantity - order.remaining_quantity,
                )
        # Dollar cash and share conservation are exchange invariants; marked equity may change.
        if (
            sum(a.portfolio.cash for a in agents) != self.initial_cash
            or sum(a.portfolio.position for a in agents) != self.initial_shares
        ):
            raise RuntimeError("CDA conservation violated")
        prices = [opening, *[t["price"] for t in trades]]
        self.history = pd.concat(
            [
                self.history,
                pd.DataFrame(
                    [
                        {
                            "session_date": session,
                            "symbol": self.exchange.asset,
                            "open": opening,
                            "close": closing,
                            "high": max(prices),
                            "low": min(prices),
                            "volume": volume,
                            "source": "endogenous_cda",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        return closing, volume, orders, trades
