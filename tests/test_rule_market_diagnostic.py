"""Literal capacity/valuation checks and rejection of doubled or mismatched fills."""

from copy import deepcopy

import pandas as pd
import pytest

from behavioral_market.agents.state import population
from scripts.diagnose_rule_market import (
    capacity_detail,
    nullable_records,
    reconstruct_personas,
    valuation,
    verify_fills,
)


def test_unknown_optional_fields_compare_equal_and_remain_null_not_zero():
    left = pd.DataFrame({"quantity": [0], "prior_majority_action": [float("nan")]})
    right = pd.DataFrame({"quantity": [0], "prior_majority_action": [float("nan")]})
    assert (
        nullable_records(left)
        == nullable_records(right)
        == [{"quantity": 0, "prior_majority_action": None}]
    )


def order(side, qty, **extra):
    return {
        "requested_action": side,
        "requested_quantity": qty,
        "action": side,
        "limit_price": {"buy": 100.3, "sell": 99.7, "hold": None}[side],
        "rationale": "rule portfolio bound" if side == "hold" else "rule",
        **extra,
    }


def test_conditional_buy_sell_capacities_are_not_preclipping_observations():
    before = {"cash": "1000", "position": 2}
    buy = capacity_detail(order("buy", 5), before, 100, 0.5)
    sell = capacity_detail(order("sell", 2), before, 100, 0.5)
    assert buy["nominal_quantity_if_active"] == sell["nominal_quantity_if_active"] == 5
    assert buy["conditional_buy_capacity_at_derived_limit"] == 9
    assert sell["conditional_sell_capacity"] == 2
    assert buy["known_active_size_clipped"] is False
    assert sell["known_active_size_clipped"] is True
    held = capacity_detail(order("hold", 0), {"cash": "1000", "position": 0}, 100, 0.5)
    assert held["desired_side_before_clipping"] is None
    assert held["raw_policy_noise"] is None and held["known_active_size_clipped"] is None
    with pytest.raises(ValueError, match="size differs"):
        capacity_detail(order("sell", 5), before, 100, 0.5)


def test_cash_constraint_and_later_rejection_are_distinct():
    clipped = capacity_detail(order("buy", 1), {"cash": "150", "position": 2}, 100, 0.5)
    assert clipped["known_active_size_clipped"] and not clipped["later_validation_rejected"]
    rejected = capacity_detail(
        order("buy", 5, action="hold", limit_price=None), {"cash": "1000", "position": 2}, 100, 0.5
    )
    assert rejected["later_validation_rejected"] and not rejected["known_active_size_clipped"]


def test_persona_reconstruction_matches_only_population_rng_not_order_rng():
    for seed in [17, 29, 43]:
        config = {
            "experiment": {"seed": seed},
            "population": {
                "n_agents": 20,
                "initial_fundamental_share": 0.5,
                "initial_cash": "1000",
                "initial_shares": 2,
            },
        }
        reconstructed = reconstruct_personas(config)
        actual = population(config, "rule_abm", "100")
        assert reconstructed == {
            a.persona.agent_id: {
                "risk_tolerance": a.persona.risk_tolerance,
                "initial_style": a.persona.initial_style,
            }
            for a in actual
        }


def test_realized_profit_does_not_equal_total_wealth_change():
    # Two accounts initially each500cash/5shares at100. A buys3 from B at105.
    # At an independently supplied mark101: realized15, unrealized-5, gain10.
    accounts = [
        {"cash": 185, "position": 8, "total_equity": 993, "realized_pnl": 0, "unrealized_pnl": -7},
        {"cash": 815, "position": 2, "total_equity": 1017, "realized_pnl": 15, "unrealized_pnl": 2},
    ]
    result = valuation(accounts, 1000, 10, 100, 101)
    assert result["sum_realized_pnl"] == "15"
    assert result["wealth_change_from_price_mark"] == "10"
    assert result["net_cash_creation"] == "0"
    broken = deepcopy(accounts)
    broken[0]["cash"] += 1
    with pytest.raises(ValueError, match="not conserved"):
        valuation(broken, 1000, 10, 100, 101)


def test_one_trade_is_three_units_not_six_and_order_owners_must_match():
    orders = [
        {
            "order_id": "B1",
            "agent_id": "A",
            "session_date": "day1",
            "action": "buy",
            "quantity": 3,
            "filled_quantity": 3,
        },
        {
            "order_id": "S1",
            "agent_id": "B",
            "session_date": "day1",
            "action": "sell",
            "quantity": 3,
            "filled_quantity": 3,
        },
    ]
    trades = [
        {
            "session_date": "day1",
            "quantity": 3,
            "buyer_id": "A",
            "seller_id": "B",
            "buy_order_id": "B1",
            "sell_order_id": "S1",
        }
    ]
    market = [{"session_date": "day1", "volume": 3}]
    assert verify_fills(orders, trades, market)["day1"] == 3
    with pytest.raises(ValueError, match="single-side"):
        verify_fills(orders, trades, [{"session_date": "day1", "volume": 6}])
    wrong = deepcopy(trades)
    wrong[0]["buyer_id"] = "B"
    with pytest.raises(ValueError, match="identity"):
        verify_fills(orders, wrong, market)


def test_empty_trade_day_is_retained():
    orders = [
        {
            "order_id": None,
            "agent_id": "A",
            "session_date": "day1",
            "action": "hold",
            "quantity": 0,
            "filled_quantity": 0,
        }
    ]
    assert verify_fills(orders, [], [{"session_date": "day1", "volume": 0}])["day1"] == 0
