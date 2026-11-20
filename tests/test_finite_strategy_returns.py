"""Hand-derived finite-account returns; no model, original data or credential access."""

from decimal import Decimal

import pytest

from scripts import audit_finite_strategy_returns as tool


@pytest.fixture
def package():
    return tool.source.materials.build_materials()


@pytest.mark.parametrize(
    "spread,positive_equity,negative_equity",
    [
        ("0.00", "1200.00", "1150.00"),
        ("0.05", "1199.75", "1149.75"),
        ("1.00", "1200.00", "1200.00"),
    ],
)
def test_full_five_day_holdings_and_execution_match_hand_calculation(
    package, spread, positive_equity, negative_equity
):
    for case in package["cases"]:
        original = case["original_style"]
        alternative = "technical" if original == "fundamental" else "fundamental"
        current = tool.replay(case, original, spread)
        other = tool.replay(case, alternative, spread)
        assert Decimal(current["final_equity"]) == 1200
        assert all(d["decision"]["action"] == "hold" for d in current["daily"])
        assert Decimal(other["final_equity"]) == Decimal(
            positive_equity if case["economic_condition"] == "positive" else negative_equity
        )
        assert other["final_shares"] == (2 if spread == "1.00" else 7)
        assert sum(d["filled_quantity"] for d in other["daily"]) == (0 if spread == "1.00" else 5)
        assert sum(d["expired_orders"] for d in other["daily"]) == (2 if spread == "1.00" else 0)
        for d in other["daily"]:
            assert Decimal(d["cash_after"]) + Decimal(d["counterparty_cash_after"]) == 11000
            assert d["shares_after"] + d["counterparty_shares_after"] == 102
            assert d["history_last_session"] < d["session"]
            assert d["decision"]["quantity"] <= 5


def test_zero_signal_does_not_liquidate_a_previously_bought_position(package):
    case = next(c for c in package["cases"] if c["case_id"] == "A000.positive")
    result = tool.replay(case, "technical", "0.00")
    assert result["daily"][0]["decision"]["action"] == "buy"
    assert Decimal(result["daily"][0]["equity_after"]) == 1270
    assert all(
        d["decision"]["action"] == "hold" and d["shares_after"] == 7 for d in result["daily"][1:]
    )
    assert Decimal(result["return"]) == 0
    assert case["construction_not_sent"]["final_state"]["counterfactual"][
        "alternative_block_return"
    ] == pytest.approx(0.1)


def test_proxy_labels_are_preserved_without_imposing_their_sign_on_finite_returns(package):
    result = tool.audit(package)
    assert len(result["rows"]) == 12 and result["new_model_POSTs"] == 0
    assert all(Decimal(r["finite_alternative_minus_original"]) <= 0 for r in result["rows"])
    assert {r["original_proxy_condition"] for r in result["rows"]} == {"positive", "negative"}
    assert all(
        r["proxy_alternative_return"] > 0
        for r in result["rows"]
        if r["original_proxy_condition"] == "positive"
    )
