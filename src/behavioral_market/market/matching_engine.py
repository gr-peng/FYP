"""Continuous matching at the resting order's limit price."""

from collections.abc import Callable

from .order import Order
from .order_book import OrderBook
from .trade import Trade


class MatchingEngine:
    def __init__(self, book: OrderBook) -> None:
        self.book = book

    def submit(self, incoming: Order, execute: Callable[[Order, Order, int], Trade]) -> list[Trade]:
        # Reject self-crosses before mutating accounts or the book.
        if any(
            resting.agent_id == incoming.agent_id and self.book.crosses(incoming, resting)
            for resting in self.book.resting_orders()
            if resting.side != incoming.side
        ):
            raise ValueError("self-crossing orders are not allowed")

        trades: list[Trade] = []
        while incoming.remaining_quantity:
            resting = self.book.best_opposite(incoming.side)
            if resting is None or not self.book.crosses(incoming, resting):
                break
            quantity = min(incoming.remaining_quantity, resting.remaining_quantity)
            trade = execute(incoming, resting, quantity)
            incoming.fill(quantity)
            resting.fill(quantity)
            trades.append(trade)
            if resting.remaining_quantity == 0:
                self.book.remove(resting)
        if incoming.remaining_quantity:
            self.book.add(incoming)
        return trades
