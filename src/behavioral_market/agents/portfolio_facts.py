"""Draft deterministic account facts; no forecast, news interpretation or trade mandate."""

from decimal import Decimal


def portfolio_facts(portfolio, price, lot_size, target_weight=None):
    price = Decimal(str(price))
    lot_size = Decimal(str(lot_size))
    if not price.is_finite() or not lot_size.is_finite() or price <= 0 or lot_size <= 0:
        raise ValueError("positive price and lot size required")
    asset_value = portfolio.position * price
    equity = portfolio.cash + asset_value
    if equity <= 0:
        raise ValueError("positive equity required")
    result = {
        "asset_value_in_quote_currency": float(asset_value),
        "cash_value_in_quote_currency": float(portfolio.cash),
        "total_equity_in_quote_currency": float(equity),
        "actual_asset_weight_fraction": round(float(asset_value / equity), 8),
        "actual_cash_weight_fraction": round(float(portfolio.cash / equity), 8),
        "underlying_asset_units_held": float(portfolio.position * lot_size),
        "price_per_underlying_asset_unit": float(price / lot_size),
        "interpretation": "Computed current account values; weights are fractions from 0 to 1.",
    }
    if target_weight is not None:
        target = Decimal(str(target_weight))
        if not 0 <= target <= 1:
            raise ValueError("target allocation must be a fraction from zero to one")
        gap = target * equity - asset_value
        result.update(
            target_asset_weight_fraction=float(target),
            target_minus_actual_asset_weight_percentage_points=round(float(gap / equity * 100), 6),
            asset_value_change_to_target_in_quote_currency=float(gap),
            signed_lot_change_to_target_at_last_trade=round(float(gap / price), 6),
        )
    return result
