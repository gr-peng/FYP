"""Single-asset exchange, including order risk checks and atomic settlement."""

from decimal import Decimal

from .matching_engine import MatchingEngine
from .money import money
from .order import Order, OrderStatus, Side
from .order_book import OrderBook
from .portfolio import Portfolio
from .trade import Trade


class Exchange:
    def __init__(self, asset: str, initial_price: Decimal | int | float | str) -> None:
        self.asset = asset
        self.market_price = money(initial_price)
        if not asset or self.market_price <= 0:
            raise ValueError("asset and initial price must be valid")
        self.day = 0
        self._clock = 0
        self._order_count = 0
        self._trade_count = 0
        self.portfolios: dict[str, Portfolio] = {}
        self.orders: list[Order] = []
        self.trades: list[Trade] = []
        self.book = OrderBook()
        self.matching_engine = MatchingEngine(self.book)
        self.day_volume = 0

    def register(self, agent_id: str, portfolio: Portfolio) -> None:
        if not agent_id or agent_id in self.portfolios:
            raise ValueError("agent ID must be unique and nonempty")
        portfolio.mark_to_market(self.market_price)
        self.portfolios[agent_id] = portfolio

    def _available_cash(self, agent_id: str) -> Decimal:
        reserved = sum(
            (
                order.price * order.remaining_quantity
                for order in self.book.bids
                if order.agent_id == agent_id
            ),
            Decimal("0"),
        )
        return self.portfolios[agent_id].cash - reserved

    def _available_shares(self, agent_id: str) -> int:
        reserved = sum(
            order.remaining_quantity for order in self.book.asks if order.agent_id == agent_id
        )
        return self.portfolios[agent_id].position - reserved

    def submit_order(
        self, agent_id: str, side: Side | str, price: Decimal | int | float | str, quantity: int
    ) -> tuple[Order, list[Trade]]:
        if agent_id not in self.portfolios:
            raise ValueError("unknown agent")
        side = Side(side)
        price = money(price)
        if (
            price <= 0
            or isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            raise ValueError("price and quantity must be positive")
        if side == Side.BUY and price * quantity > self._available_cash(agent_id):
            raise ValueError("insufficient available cash for limit order")
        if side == Side.SELL and quantity > self._available_shares(agent_id):
            raise ValueError("insufficient available shares for limit order")

        # The sequence is the authoritative time priority; caller-supplied times are excluded.
        self._clock += 1
        self._order_count += 1
        order = Order(
            f"O{self._order_count:08d}", agent_id, side, price, quantity, self._clock, self.day
        )
        trades = self.matching_engine.submit(order, self._execute)
        self.orders.append(order)
        return order, trades

    def _execute(self, incoming: Order, resting: Order, quantity: int) -> Trade:
        buyer = incoming if incoming.side == Side.BUY else resting
        seller = incoming if incoming.side == Side.SELL else resting
        price = resting.price
        buyer_portfolio = self.portfolios[buyer.agent_id]
        seller_portfolio = self.portfolios[seller.agent_id]
        if buyer_portfolio.cash < price * quantity or seller_portfolio.position < quantity:
            raise RuntimeError("order reservations violated")
        buyer_portfolio.apply_buy(price, quantity)
        seller_portfolio.apply_sell(price, quantity)
        self._trade_count += 1
        self._clock += 1
        trade = Trade(
            f"T{self._trade_count:08d}",
            buyer.agent_id,
            seller.agent_id,
            price,
            quantity,
            buyer.order_id,
            seller.order_id,
            self._clock,
            self.day,
        )
        self.trades.append(trade)
        self.market_price = price
        self.day_volume += quantity
        for portfolio in self.portfolios.values():
            portfolio.mark_to_market(price)
        return trade

    def cancel_order(self, order_id: str, agent_id: str) -> Order:
        order = next(
            (item for item in self.book.resting_orders() if item.order_id == order_id), None
        )
        if order is None or order.agent_id != agent_id:
            raise ValueError("open order not found for agent")
        self.book.remove(order)
        order.close(OrderStatus.CANCELLED)
        return order

    def end_day(self) -> list[Order]:
        expired = list(self.book.resting_orders())
        for order in expired:
            self.book.remove(order)
            order.close(OrderStatus.EXPIRED)
        self.day += 1
        self.day_volume = 0
        return expired
