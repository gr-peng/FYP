"""Exact decimal conversion for prices and cash."""

from decimal import Decimal


def money(value: Decimal | int | float | str) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise ValueError("amount must be finite")
    return amount
