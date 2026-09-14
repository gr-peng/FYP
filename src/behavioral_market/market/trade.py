"""Immutable execution record."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    buyer_id: str
    seller_id: str
    price: Decimal
    quantity: int
    buy_order_id: str
    sell_order_id: str
    timestamp: int
    day: int
