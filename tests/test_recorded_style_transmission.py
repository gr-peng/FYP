"""Synthetic records; actual recorded model inputs are never opened by these tests."""

from copy import deepcopy
from itertools import product

import pytest

from scripts import build_style_review_economic_controls as materials
from scripts import replay_recorded_style_choices as tool


@pytest.fixture
def case():
    return materials.make_case("fundamental", "positive")


def choice(case, target):
    return {
        "slot_id": "FICTIONAL_NOT_A_MODEL_RECORD",
        "agent_id": "A000",
        "from_style": "fundamental",
        "case_id": case["case_id"],
        "compatible": True,
        "condition": "positive",
        "choice": {"target_style": target},
    }


def test_buy_sell_hold_hand_accounts_and_matching_engine(case):
    buy = tool.simulate(case, "fundamental", 0.1, -0.1)
    sell = tool.simulate(case, "technical", 0.1, -0.1)
    hold = tool.simulate(case, "fundamental", 0, 0.1)
    assert buy["decision"]["action"] == "buy" and buy["agent_filled_quantity"] == 5
    assert buy["agent_cash_after"] == "499.75" and buy["agent_shares_after"] == 7
    assert buy["agent_equity_after"] == "1200.10"
    assert sell["decision"]["action"] == "sell" and sell["agent_filled_quantity"] == 2
    assert sell["agent_cash_after"] == "1199.90" and sell["agent_shares_after"] == 0
    assert hold["decision"]["action"] == "hold" and not hold["agent_order_submitted"]
    assert hold["agent_cash_after"] == "1000.00" and hold["agent_shares_after"] == 2
    assert all(row["cash_and_share_conservation"] for row in (buy, sell, hold))
    assert buy["metrics"]["execution"]["single_side_trade_volume"] == 5
    assert buy["metrics"]["execution"]["filled_side_quantity"] == 10


def test_paired_control_keeps_all_nine_states_including_three_no_change_states(case):
    source = deepcopy(case)
    switched = [
        tool.paired_branch(case, choice(case, "technical"), f, t)
        for f, t in product(tool.LEVELS, repeat=2)
    ]
    stayed = [
        tool.paired_branch(case, choice(case, "fundamental"), f, t)
        for f, t in product(tool.LEVELS, repeat=2)
    ]
    assert len(switched) == len(stayed) == 9
    assert sum(row["action_changed"] for row in switched) == 6
    assert sum(row["style_changed"] for row in switched) == 9
    assert all(not row["action_changed"] and not row["holdings_changed"] for row in stayed)
    assert all(
        not row["action_changed"]
        for row in switched
        if row["fundamental_signal"] == row["technical_signal"]
    )
    assert case == source


def test_incompatible_record_or_foreign_account_cannot_supply_a_style(case):
    for field, value in (
        ("compatible", False),
        ("agent_id", "foreign"),
        ("from_style", "technical"),
    ):
        row = choice(case, "technical")
        row[field] = value
        with pytest.raises(ValueError):
            tool.paired_branch(case, row, 0.1, -0.1)
    with pytest.raises(ValueError):
        tool.simulate(case, "fundamental", 0.2, 0)
