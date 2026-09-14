"""Cash and position accounting without margin or short selling."""

from dataclasses import dataclass
from decimal import Decimal

from .money import money


@dataclass(slots=True)
class Portfolio:
    cash: Decimal
    position: int = 0
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_equity: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        self.cash = money(self.cash)
        self.avg_cost = money(self.avg_cost)
        if self.cash < 0 or self.position < 0 or self.avg_cost < 0:
            raise ValueError("negative initial portfolio state")
        if self.position == 0 and self.avg_cost != 0:
            raise ValueError("empty position must have zero average cost")
        if self.position > 0 and self.avg_cost <= 0:
            raise ValueError("positive position must have positive average cost")
        self.total_equity = self.cash + self.position * self.avg_cost

    def apply_buy(self, price: Decimal | int | float | str, quantity: int) -> None:
        price = money(price)
        if price <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        cost = price * quantity
        if cost > self.cash:
            raise ValueError("insufficient cash")
        old_cost = self.avg_cost * self.position
        self.cash -= cost
        self.position += quantity
        self.avg_cost = (old_cost + cost) / self.position
        self.mark_to_market(price)

    def apply_sell(self, price: Decimal | int | float | str, quantity: int) -> None:
        price = money(price)
        if price <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        if quantity > self.position:
            raise ValueError("insufficient shares")
        self.cash += price * quantity
        self.realized_pnl += (price - self.avg_cost) * quantity
        self.position -= quantity
        if self.position == 0:
            self.avg_cost = Decimal("0")
        self.mark_to_market(price)

    def mark_to_market(self, price: Decimal | int | float | str) -> None:
        price = money(price)
        if price <= 0:
            raise ValueError("mark price must be positive")
        self.unrealized_pnl = (price - self.avg_cost) * self.position
        self.total_equity = self.cash + price * self.position
