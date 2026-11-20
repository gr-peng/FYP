"""Finite returns and all visible facts come from constrained chronological replay."""

from decimal import Decimal

import pytest

from scripts import build_finite_economic_controls as tool


@pytest.mark.parametrize("style", ["fundamental", "technical"])
@pytest.mark.parametrize(
    "condition,expected_equity", [("positive", "1249.50"), ("negative", "1149.75")]
)
def test_new_material_uses_finite_trade_returns_and_chronological_facts(
    style, condition, expected_equity
):
    case = tool.make_case(style, condition)
    user = tool.source.strict_json(case["payload"]["messages"][1]["content"])
    state, obs = user["agent_state"], user["observation"]
    cf = state["counterfactual"]
    evidence = obs["counterfactual_evidence"]
    replay = evidence["alternative_strategy_replay"]
    expected_return = Decimal(expected_equity) / 1200 - 1
    assert cf["method"] == evidence["method"] == tool.METHOD
    assert cf["actual_block_return"] == 0
    assert cf["alternative_block_return"] == pytest.approx(float(expected_return))
    assert Decimal(replay["final_equity"]) == Decimal(expected_equity)
    assert [d["decision"]["action"] for d in replay["daily"]] == (
        ["buy", "sell", "hold", "hold", "hold"]
        if condition == "positive"
        else ["hold", "buy", "hold", "hold", "hold"]
    )
    assert replay["daily"][0]["decision"]["quantity"] == (5 if condition == "positive" else 0)
    assert len(state["memory"]) == 5
    assert all(m["action"] == "hold" and m["filled_quantity"] == 0 for m in state["memory"])
    assert state["portfolio"]["cash"] == 1000 and state["portfolio"]["position"] == 2
    assert obs["session_date"] == obs["history_last_session"] == "2026-01-09"
    assert obs["last_close"] == 100
    assert "mean_field" not in obs and obs["mean_field_availability"] == "withheld_by_experiment"
    assert evidence["hypothetical_separate_replay_not_actual_personal_trades"] is True
    text = tool.old.canonical(case["payload"])
    assert "long_cash_signal_proxy" not in text and "frictionless long/cash proxy" not in text
    assert "rule_style_reference" not in text and "construction_not_sent" not in text
    assert case["payload"]["model"] == "DeepSeek-V4.1-Flash"
    assert case["planned_payload_sha256"] == tool.old.digest(case["payload"])


def test_new_cases_are_materials_not_model_results_or_human_validation():
    package = tool.build_materials()
    assert package["actual_remote_calls"] == 0 and package["status"] == "materials_only"
    assert len(package["cases"]) == 4
    assert all(c["status"] == "unsubmitted" for c in package["cases"])
