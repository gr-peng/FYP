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
        self._session = None
        self._opening = None
        self._orders = []
        self._trades = []

    def book_context(self):
        """Return only book state formed by orders already submitted this session."""
        bid = self.exchange.book.bids[0] if self.exchange.book.bids else None
        ask = self.exchange.book.asks[0] if self.exchange.book.asks else None
        best_bid = float(bid.price) if bid else None
        best_ask = float(ask.price) if ask else None
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid if bid and ask else None,
            "spread_bps": (best_ask / best_bid - 1) * 10_000 if bid and ask else None,
            "last_trade": float(self.exchange.market_price),
            "bid_depth": sum(order.remaining_quantity for order in self.exchange.book.bids),
            "ask_depth": sum(order.remaining_quantity for order in self.exchange.book.asks),
            "scope": "current_session_prior_batches",
        }

    def start_session(self, session):
        if self._session is not None:
            raise RuntimeError("previous endogenous session is still open")
        self._session = session
        self._opening = float(self.exchange.market_price)
        self._orders, self._trades = [], []

    def submit(self, agent, decision):
        if self._session is None:
            raise RuntimeError("start_session must be called before submit")
        if decision.action == "hold":
            self._orders.append(
                {
                    "agent_id": agent.persona.agent_id,
                    "order_id": None,
                    "status": "hold",
                    "filled_quantity": 0,
                }
            )
            return
        order, fills = self.exchange.submit_order(
            agent.persona.agent_id, decision.action, decision.limit_price, decision.quantity
        )
        self._trades.extend(
            {**asdict(fill), "price": float(fill.price), "session_date": self._session}
            for fill in fills
        )
        self._orders.append({"agent_id": agent.persona.agent_id, "order": order})

    def finish_session(self, agents):
        if self._session is None:
            raise RuntimeError("no endogenous session is open")
        session, opening = self._session, self._opening
        closing, volume = float(self.exchange.market_price), self.exchange.day_volume
        self.exchange.end_day()
        for row in self._orders:
            if "order" in row:
                order = row.pop("order")
                row.update(
                    order_id=order.order_id,
                    status=order.status.value,
                    filled_quantity=order.quantity - order.remaining_quantity,
                )
        if (
            sum(a.portfolio.cash for a in agents) != self.initial_cash
            or sum(a.portfolio.position for a in agents) != self.initial_shares
        ):
            raise RuntimeError("CDA conservation violated")
        prices = [opening, *[trade["price"] for trade in self._trades]]
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
        result = (closing, volume, self._orders, self._trades)
        self._session, self._opening, self._orders, self._trades = None, None, [], []
        return result

    def settle(self, session, agents, decisions, arrival_order):
        self.start_session(session)
        for index in arrival_order:
            self.submit(agents[index], decisions[index])
        return self.finish_session(agents)
