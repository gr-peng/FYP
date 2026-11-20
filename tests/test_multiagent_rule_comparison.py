"""A null outcome must survive analysis; pre-review divergence must fail."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_multiagent_rule_comparison import compare_pair


@pytest.fixture
def tables(monkeypatch):
    dates = pd.date_range("2026-01-01", "2026-01-10").strftime("%Y-%m-%d")
    tables = {}
    for arm in ["fixed", "adaptive"]:
        tables[arm] = {
            "market": pd.DataFrame({"session_date": dates, "close": 100.0}),
            "orders": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "agents": pd.DataFrame(
                [
                    {
                        "session_date": date,
                        "agent_id": f"A{i:03d}",
                        "cash": 1000.0,
                        "position": 2,
                        "total_equity": 1200.0,
                        "current_style": "fundamental" if i < 10 else "technical",
                    }
                    for date in dates
                    for i in range(20)
                ]
            ),
            "style_switches": pd.DataFrame(
                [
                    {
                        "session_date": date,
                        "agent_id": f"A{i:03d}",
                        "from_style": "fundamental" if i < 10 else "technical",
                        "decision": "stay",
                        "observed_account_return": 0.0,
                        "current_rule_block_return": 0.0,
                        "alternative_rule_block_return": 0.0,
                        "paired_return_gap": 0.0,
                    }
                    for date in ["2026-01-05", "2026-01-10"]
                    for i in range(20)
                ]
            ),
        }

    def read(path):
        arm = path.parent.name.split("_")[-1]
        return tables[arm][path.stem].copy()

    monkeypatch.setattr(pd, "read_parquet", read)
    return tables


def test_empty_trade_exports_and_null_outcome_are_retained(tables):
    tables["adaptive"]["trades"] = pd.DataFrame(columns=["session_date", "quantity"])
    result = compare_pair(Path("unused"), 17)
    assert result["first_block_equal"]
    assert result["max_absolute_close_difference"] == 0
    assert len(result["adaptive_minus_fixed_close"]) == 5
    for arm in result["arms"].values():
        assert arm["second_block_single_side_trade_quantity"] == 0
        assert arm["first_review_switches"] == arm["terminal_review_switches"] == 0
        assert len(arm["final_account_wealth"]) == 20


def test_later_trade_does_not_invalidate_empty_first_block(tables):
    for arm in ["fixed", "adaptive"]:
        tables[arm]["trades"] = pd.DataFrame([{"session_date": "2026-01-06", "quantity": 3}])
    result = compare_pair(Path("unused"), 17)
    assert result["arms"]["adaptive"]["second_block_single_side_trade_quantity"] == 3
    assert result["no_first_switch_later_equality_checked"]


@pytest.mark.parametrize("table", ["market", "orders", "trades", "agents"])
def test_zero_first_switch_cannot_explain_later_path_difference(tables, table):
    if table in ["orders", "trades"]:
        tables["adaptive"][table] = pd.DataFrame([{"session_date": "2026-01-06", "quantity": 3}])
    elif table == "market":
        tables["adaptive"][table].loc[5, "close"] += 1
    else:
        # Preserve total cash so the unrelated conservation check still passes.
        frame = tables["adaptive"][table]
        frame.loc[100, "cash"] += 1
        frame.loc[101, "cash"] -= 1
    with pytest.raises(AssertionError):
        compare_pair(Path("unused"), 17)


def test_first_switch_permits_later_difference_without_claiming_it_is_realistic(tables):
    tables["adaptive"]["style_switches"].loc[0, "decision"] = "switch"
    tables["adaptive"]["trades"] = pd.DataFrame([{"session_date": "2026-01-06", "quantity": 3}])
    result = compare_pair(Path("unused"), 17)
    assert not result["no_first_switch_later_equality_checked"]
    assert result["arms"]["adaptive"]["first_review_switches"] == 1


@pytest.mark.parametrize("table", ["market", "trades", "agents", "style_switches"])
def test_pre_review_difference_is_not_called_a_policy_effect(tables, table):
    frame = tables["adaptive"][table]
    if table == "trades":
        tables["adaptive"][table] = pd.DataFrame([{"session_date": "2026-01-01", "quantity": 1}])
    else:
        field = {"market": "close", "agents": "cash", "style_switches": "paired_return_gap"}[table]
        frame.loc[0, field] += 1
    with pytest.raises(AssertionError):
        compare_pair(Path("unused"), 17)


def test_terminal_switch_is_reported_separately(tables):
    frame = tables["adaptive"]["style_switches"]
    frame.loc[frame.session_date == "2026-01-10", "decision"] = "switch"
    result = compare_pair(Path("unused"), 17)
    assert result["arms"]["adaptive"]["first_review_switches"] == 0
    assert result["arms"]["adaptive"]["terminal_review_switches"] == 20
    assert result["max_absolute_close_difference"] == 0
