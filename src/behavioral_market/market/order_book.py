"""Price-time-priority resting order book for one asset."""

from .order import Order, Side


class OrderBook:
    def __init__(self) -> None:
        self.bids: list[Order] = []
        self.asks: list[Order] = []

    def add(self, order: Order) -> None:
        side = self.bids if order.side == Side.BUY else self.asks
        side.append(order)
        if order.side == Side.BUY:
            side.sort(key=lambda item: (-item.price, item.timestamp))
        else:
            side.sort(key=lambda item: (item.price, item.timestamp))

    def best_opposite(self, side: Side) -> Order | None:
        opposite = self.asks if side == Side.BUY else self.bids
        return opposite[0] if opposite else None

    def remove(self, order: Order) -> None:
        side = self.bids if order.side == Side.BUY else self.asks
        side.remove(order)

    def resting_orders(self) -> tuple[Order, ...]:
        return (*self.bids, *self.asks)

    def crosses(self, incoming: Order, resting: Order) -> bool:
        if incoming.side == Side.BUY:
            return incoming.price >= resting.price
        return incoming.price <= resting.price
