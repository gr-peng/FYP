"""Order representation; only the exchange creates order identifiers and timestamps."""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from .money import money


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(slots=True)
class Order:
    order_id: str
    agent_id: str
    side: Side
    price: Decimal
    quantity: int
    timestamp: int
    day: int
    remaining_quantity: int = field(init=False)
    status: OrderStatus = field(init=False, default=OrderStatus.OPEN)

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.price = money(self.price)
        if self.price <= 0:
            raise ValueError("limit price must be positive")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("quantity must be a positive integer")
        if self.timestamp < 0 or self.day < 0:
            raise ValueError("timestamp and day must be nonnegative")
        self.remaining_quantity = self.quantity

    def fill(self, quantity: int) -> None:
        if quantity <= 0 or quantity > self.remaining_quantity:
            raise ValueError("invalid fill quantity")
        self.remaining_quantity -= quantity
        self.status = (
            OrderStatus.FILLED if self.remaining_quantity == 0 else OrderStatus.PARTIALLY_FILLED
        )

    def close(self, status: OrderStatus) -> None:
        if status not in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
            raise ValueError("invalid closing status")
        if self.remaining_quantity == 0:
            raise ValueError("filled order cannot be closed")
        self.status = status
