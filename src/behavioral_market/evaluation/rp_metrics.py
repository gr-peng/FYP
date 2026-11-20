"""Explicit RP slide11 measures. Preserve older metrics and their historical results."""

from __future__ import annotations

import math
from decimal import Decimal


def available(value):
    return {"status": "available", "value": value, "reason": None}


def unknown(reason):
    return {"status": "unknown", "value": None, "reason": reason}


def series(values, name):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if any(isinstance(v, bool) or not isinstance(v, (int, float, Decimal)) for v in values):
        raise ValueError(f"{name} must contain numeric values, excluding booleans")
    return [float(v) for v in values]


def ratio(numerator, denominator, reason):
    return available(numerator / denominator) if denominator else unknown(reason)


def execution_metrics(accepted_orders=None, trades=None):
    """Lists assert complete evidence for one interval; None means unavailable evidence.

    Each fill counts once for its buy order and once for its sell order in the
    quantity rate. Trade volume counts that matched quantity once. The accepted
    flag is an input convention, not external authentication or proof of settlement.
    """
    if accepted_orders is None:
        return {
            "accepted_order_count": None,
            "submitted_side_quantity": None,
            "executed_order_fraction": unknown("accepted_order_evidence_not_supplied"),
            "filled_side_quantity_fraction": unknown("accepted_order_evidence_not_supplied"),
            "fully_filled_order_fraction": unknown("accepted_order_evidence_not_supplied"),
            "single_side_trade_volume": None,
            "filled_side_quantity": None,
            "orders_with_any_fill": None,
            "fully_filled_orders": None,
        }
    if not isinstance(accepted_orders, list):
        raise ValueError("accepted order evidence must be a list")
    orders = {}
    for row in accepted_orders:
        if not isinstance(row, dict):
            raise ValueError("accepted order evidence must contain objects")
        oid, side, quantity = row.get("order_id"), row.get("side"), row.get("quantity")
        if not (
            isinstance(oid, str)
            and oid.strip() == oid
            and bool(oid)
            and oid not in orders
            and row.get("accepted") is True
            and side in ("buy", "sell")
            and type(quantity) is int
            and quantity > 0
            and row.get("status") not in ("hold", "rejected", "invalid")
            and row.get("action") != "hold"
        ):
            raise ValueError("unique accepted orders with positive quantities required")
        orders[oid] = (side, quantity)
    count, submitted = len(orders), sum(quantity for _, quantity in orders.values())
    base = {"accepted_order_count": count, "submitted_side_quantity": submitted}
    if trades is None:
        return {
            **base,
            "executed_order_fraction": unknown("complete_trade_evidence_not_supplied"),
            "filled_side_quantity_fraction": unknown("complete_trade_evidence_not_supplied"),
            "fully_filled_order_fraction": unknown("complete_trade_evidence_not_supplied"),
            "single_side_trade_volume": None,
            "filled_side_quantity": None,
            "orders_with_any_fill": None,
            "fully_filled_orders": None,
        }
    if not isinstance(trades, list):
        raise ValueError("complete trade evidence must be a list or None")
    filled, trade_ids, volume = dict.fromkeys(orders, 0), set(), 0
    for row in trades:
        if not isinstance(row, dict):
            raise ValueError("trade evidence must contain objects")
        tid, buy, sell, quantity = (
            row.get(k) for k in ("trade_id", "buy_order_id", "sell_order_id", "quantity")
        )
        if not (
            isinstance(tid, str)
            and tid.strip() == tid
            and bool(tid)
            and tid not in trade_ids
            and isinstance(buy, str)
            and isinstance(sell, str)
            and buy in orders
            and sell in orders
            and orders[buy][0] == "buy"
            and orders[sell][0] == "sell"
            and type(quantity) is int
            and quantity > 0
        ):
            raise ValueError("unique trades linked to accepted buy/sell orders required")
        trade_ids.add(tid)
        filled[buy] += quantity
        filled[sell] += quantity
        if filled[buy] > orders[buy][1] or filled[sell] > orders[sell][1]:
            raise ValueError("cumulative fills exceed accepted order quantity")
        volume += quantity
    any_fill = sum(quantity > 0 for quantity in filled.values())
    full = sum(filled[oid] == quantity for oid, (_, quantity) in orders.items())
    return {
        **base,
        "orders_with_any_fill": any_fill,
        "fully_filled_orders": full,
        "single_side_trade_volume": volume,
        "filled_side_quantity": sum(filled.values()),
        "executed_order_fraction": ratio(any_fill, count, "zero_accepted_orders"),
        "fully_filled_order_fraction": ratio(full, count, "zero_accepted_orders"),
        "filled_side_quantity_fraction": ratio(
            sum(filled.values()), submitted, "zero_submitted_quantity"
        ),
    }


def market_metrics(
    prices, reference_values=None, *, accepted_orders=None, trades=None, evidence_kind="unspecified"
):
    """Compute explicitly named definitions, never infer realism from synthetic traces."""
    price = series(prices, "prices")
    reference = series(reference_values, "reference_values")
    if reference is not None and price is not None and len(reference) != len(price):
        raise ValueError("price and reference observations must be aligned one to one")
    invalid_price = price is None or not price or any(not math.isfinite(v) or v <= 0 for v in price)
    reason = (
        "prices_not_supplied"
        if price is None
        else "empty_price_path"
        if not price
        else "nonpositive_or_nonfinite_price"
    )
    if invalid_price:
        log_returns, cumulative, drawdown = unknown(reason), unknown(reason), unknown(reason)
    elif len(price) < 2:
        log_returns = cumulative = drawdown = unknown("at_least_two_price_observations_required")
    else:
        values = [
            math.log(current) - math.log(previous)
            for previous, current in zip(price[:-1], price[1:], strict=True)
        ]
        log_returns, cumulative = available(values), available(math.fsum(values))
        peak, maximum = price[0], 0.0
        for value in price:
            peak = max(peak, value)
            maximum = max(maximum, 1 - value / peak)
        drawdown = available(maximum)
    if invalid_price:
        mse = unknown(reason)
    elif reference is None:
        mse = unknown("reference_values_not_supplied")
    elif any(not math.isfinite(value) for value in reference):
        mse = unknown("nonfinite_reference_value")
    else:
        squared = [(p - f) * (p - f) for p, f in zip(price, reference, strict=True)]
        mean = math.fsum(value / len(price) for value in squared)
        mse = available(mean) if math.isfinite(mean) else unknown("numeric_overflow")
    return {
        "definition_version": "rp-slide11-explicit-v1",
        "evidence_kind": evidence_kind,
        "price_observations": len(price) if price is not None else None,
        "log_returns": log_returns,
        "cumulative_log_return": cumulative,
        "maximum_price_drawdown_fraction": drawdown,
        "mean_squared_price_reference_deviation": mse,
        "execution": execution_metrics(accepted_orders, trades),
        "definitions": {
            "log_returns": "ln(Pt)-ln(Pt-1), per observed interval",
            "deviation": "mean((price-reference)^2), raw price squared units",
            "drawdown": "max(1-price/running_peak); reporting requires at least2 observations",
            "executed_order_fraction": "accepted orders with any fill / accepted orders",
            "filled_side_quantity_fraction": (
                "filled buy+sell side quantity / submitted side quantity"
            ),
            "single_side_trade_volume": "sum trade quantity, each matched quantity counted once",
        },
        "limits": [
            "Supplied references are not authenticated true asset values.",
            "Order acceptance and interval completeness are input evidence conventions, "
            "not external authentication or reconstructed cash settlement.",
            "HOLD/rejected intentions and rows without order IDs cannot supply order counts.",
            "Synthetic diagnostics do not establish human alignment or market realism.",
        ],
    }
