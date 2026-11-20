"""Hand-calculated synthetic traces, independent denominators and malformed evidence."""

import math
from copy import deepcopy

import pytest

from behavioral_market.evaluation.rp_metrics import execution_metrics, market_metrics


def evidence():
    orders = [
        {"order_id": oid, "side": side, "quantity": quantity, "accepted": True}
        for oid, side, quantity in [
            ("B1", "buy", 4),
            ("S1", "sell", 2),
            ("S2", "sell", 3),
            ("B2", "buy", 5),
        ]
    ]
    trades = [
        {"trade_id": "T1", "buy_order_id": "B1", "sell_order_id": "S1", "quantity": 2},
        {"trade_id": "T2", "buy_order_id": "B1", "sell_order_id": "S2", "quantity": 1},
    ]
    return orders, trades


def test_rp_price_formulae_match_hand_calculation_without_changing_old_definitions():
    result = market_metrics(
        [100, 110, 99, 108.9], [100, 100, 100, 100], evidence_kind="synthetic_diagnostic"
    )
    assert result["log_returns"]["value"] == pytest.approx(
        [math.log(1.1), math.log(0.9), math.log(1.1)]
    )
    assert result["cumulative_log_return"]["value"] == pytest.approx(math.log(1.089))
    assert result["maximum_price_drawdown_fraction"]["value"] == pytest.approx(0.1)
    assert result["mean_squared_price_reference_deviation"]["value"] == pytest.approx(45.0525)
    assert result["execution"]["executed_order_fraction"]["status"] == "unknown"
    assert result["evidence_kind"] == "synthetic_diagnostic"


def test_order_quantity_and_trade_volume_denominators_are_distinct():
    orders, trades = evidence()
    original = deepcopy((orders, trades))
    result = execution_metrics(orders, trades)
    assert result["accepted_order_count"] == 4 and result["submitted_side_quantity"] == 14
    assert result["orders_with_any_fill"] == 3 and result["fully_filled_orders"] == 1
    assert result["single_side_trade_volume"] == 3 and result["filled_side_quantity"] == 6
    assert result["executed_order_fraction"]["value"] == 0.75
    assert result["fully_filled_order_fraction"]["value"] == 0.25
    assert result["filled_side_quantity_fraction"]["value"] == pytest.approx(3 / 7)
    assert (orders, trades) == original


def test_explicit_zero_trades_differs_from_missing_trade_evidence():
    orders, _ = evidence()
    missing = execution_metrics(orders, None)
    assert missing["executed_order_fraction"]["value"] is None
    observed = execution_metrics(orders, [])
    assert observed["executed_order_fraction"]["value"] == 0
    assert observed["filled_side_quantity_fraction"]["value"] == 0
    empty = execution_metrics([], [])
    assert empty["accepted_order_count"] == 0
    assert empty["executed_order_fraction"]["value"] is None
    assert empty["filled_side_quantity_fraction"]["value"] is None


@pytest.mark.parametrize(
    "prices", [None, [], [100], [100, 0], [100, -1], [100, float("nan")], [100, float("inf")]]
)
def test_unavailable_returns_and_drawdown_are_unknown_not_zero_risk(prices):
    result = market_metrics(prices)
    assert result["log_returns"]["status"] == "unknown"
    assert result["maximum_price_drawdown_fraction"]["value"] is None


def test_single_pair_mse_defined_and_log_arithmetic_handles_extreme_positive_ratio():
    one = market_metrics([100], [90])
    assert one["mean_squared_price_reference_deviation"]["value"] == 100
    assert one["log_returns"]["value"] is None
    extreme = market_metrics([1e-300, 1e300])
    assert math.isfinite(extreme["log_returns"]["value"][0])
    overflow = market_metrics([1e300], [0])
    assert overflow["mean_squared_price_reference_deviation"]["reason"] == "numeric_overflow"


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_order",
        "duplicate_trade",
        "unknown_order",
        "wrong_side",
        "overfill",
        "zero_quantity",
        "boolean_quantity",
        "hold",
        "rejected",
        "not_accepted",
        "missing_order_id",
    ],
)
def test_bad_or_incomplete_order_identity_cannot_become_execution_evidence(change):
    orders, trades = evidence()
    if change == "duplicate_order":
        orders.append(deepcopy(orders[0]))
    elif change == "duplicate_trade":
        trades.append(deepcopy(trades[0]))
    elif change == "unknown_order":
        trades[0]["buy_order_id"] = "not-accepted"
    elif change == "wrong_side":
        orders[0]["side"] = "sell"
    elif change == "overfill":
        trades[1]["quantity"] = 3
    elif change == "zero_quantity":
        orders[0]["quantity"] = 0
    elif change == "boolean_quantity":
        trades[0]["quantity"] = True
    elif change in ("hold", "rejected"):
        orders[0]["status"] = change
    elif change == "not_accepted":
        orders[0]["accepted"] = False
    else:
        del orders[0]["order_id"]
    with pytest.raises(ValueError):
        execution_metrics(orders, trades)


def test_misaligned_reference_and_nonnumeric_input_are_rejected():
    with pytest.raises(ValueError):
        market_metrics([100, 110], [100])
    with pytest.raises(ValueError):
        market_metrics([True, 100])
    with pytest.raises(ValueError):
        market_metrics(["100", 110])
    result = market_metrics([100, 110], [100, float("nan")])
    assert result["log_returns"]["status"] == "available"
    assert result["mean_squared_price_reference_deviation"]["value"] is None
