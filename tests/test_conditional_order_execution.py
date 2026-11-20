"""Invented finite-account witnesses for the probability-to-execution bridge."""

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import conditional_order_execution as tool


def callbacks(a=1.0, b=0.0):
    return dict(zip(tool.AGENTS, (lambda _: a, lambda _: b), strict=True))


def full(a=1.0, b=0.0, arm=tool.prompt.ARMS[2], config=tool.DEFAULT_CONFIG):
    return tool.run_demo(callbacks(a, b), arm, config)


def economic(result):
    result = deepcopy(result)
    result.pop("arm")
    for event in result["events"]:
        for entry in event["opportunities"]:
            entry.pop("question")
    return result


def test_partial_fill_hand_calculation_then_expiry_dividend_and_next_period_memory():
    result = full()
    a, b = tool.AGENTS
    first, second, close, next_period = result["events"][:4]
    assert [(r["price"], r["quantity"]) for r in result["trades"]] == [("50", 2), ("50", 1)]
    assert (
        first["accounts_after_trading"][a]["cash"],
        first["accounts_after_trading"][a]["shares"],
    ) == ("400", 6)
    assert (
        first["accounts_after_trading"][b]["cash"],
        first["accounts_after_trading"][b]["shares"],
    ) == ("600", 2)
    assert first["orders_after_trading"][1]["status"] == "partially_filled"
    assert first["orders_after_trading"][1]["remaining_quantity"] == 1
    assert (
        second["accounts_after_trading"][a]["cash"],
        second["accounts_after_trading"][a]["shares"],
    ) == ("350", 7)
    assert second["accounts_after_trading"][b]["cash"] == "650"
    assert second["opportunities"][1]["status"] == "blocked_external_opportunity"
    assert close["accounts_before"][a]["available_cash"] == "300"
    assert close["expired_order_ids"] == ["O00000003"]
    assert close["period_close_second"] == 120
    assert close["period_close"]["second"] == 120
    assert close["period_close"]["accounts_before"][a]["cash"] == "350"
    assert close["period_close"]["accounts_after"][a]["cash"] == "420"
    assert (close["accounts_end"][a]["cash"], close["accounts_end"][b]["cash"]) == ("420", "660")
    assert sum(Decimal(r["cash_created"]) for r in result["dividends"]) == 80
    assert result["accounts_final"][a]["shares"] == 7
    assert result["accounts_final"][b]["shares"] == 1
    next_state = next_period["opportunities"][0]["state_before"]
    assert next_state[tool.prompt.memory.CURRENT] == {"buy": 0, "sell": 0}
    assert next_state[tool.prompt.memory.COMPLETED] == {
        "earlier_periods_this_repetition": {"buy": 2, "sell": 0},
        "previous_repetitions": {"buy": 0, "sell": 0},
        "last_completed_period_with_own_orders": {"buy": 2, "sell": 0},
    }
    assert next_state["earlier_periods"] == [
        {"period": 1, "mean_trade_price": 50.0, "closing_price": 50.0, "dividend": 10}
    ]
    assert next_state["earlier_current_period_trades"]["count"] == 0
    assert next_state["displayed_expected_remaining_dividends_per_share"] == 45


def test_execution_denominators_are_accepted_orders_and_two_sided_quantities():
    result = full()
    assert result["denominators"] == {
        "potential_opportunities": 12,
        "externally_offered": 12,
        "blocked": 8,
        "probability_callbacks": 4,
        "accepted_submissions": 4,
        "rejected_submissions": 0,
    }
    metric = result["rp_metrics"]["execution"]
    assert metric["submitted_side_quantity"] == 9
    assert metric["single_side_trade_volume"] == 3
    assert metric["filled_side_quantity"] == 6
    assert metric["executed_order_fraction"]["value"] == 3 / 4
    assert metric["fully_filled_order_fraction"]["value"] == 1 / 2
    assert metric["filled_side_quantity_fraction"]["value"] == 2 / 3
    assert result["human_alignment"]["status"] == "unknown"
    assert result["human_probability_error"]["value"] is None
    assert result["direct_model_POSTs"] == 0
    assert result["callback_external_POSTs"] is None
    assert "model_POSTs" not in result


def test_same_half_callback_has_identical_economic_paths_across_all_memory_arms():
    paths = [full(0.5, 0.5, arm=arm) for arm in tool.prompt.ARMS]
    assert economic(paths[0]) == economic(paths[1]) == economic(paths[2])
    messages = [r["events"][0]["opportunities"][0]["question"]["messages"] for r in paths]
    assert messages[0] != messages[1] != messages[2]


def test_all_buy_has_zero_trades_but_unfilled_submissions_still_enter_history():
    result = full(1, 1)
    assert result["trades"] == []
    assert result["denominators"]["accepted_submissions"] == 4
    assert all(r["status"] == "expired" and r["filled_quantity"] == 0 for r in result["orders"])
    assert result["rp_metrics"]["execution"]["executed_order_fraction"]["value"] == 0
    assert result["events"][3]["opportunities"][0]["state_before"][tool.prompt.memory.COMPLETED][
        "earlier_periods_this_repetition"
    ] == {"buy": 1, "sell": 0}
    assert result["events"][3]["opportunities"][0]["state_before"]["earlier_periods"] == [
        {"period": 1, "mean_trade_price": None, "closing_price": None, "dividend": 10}
    ]


def test_no_external_opportunities_never_calls_predictor_or_fabricates_hold():
    def forbidden(_):
        raise AssertionError("no opportunity must not call a predictor")

    config = replace(tool.DemoConfig(), opportunities=(False,) * 12)
    result = tool.run_demo(dict.fromkeys(tool.AGENTS, forbidden), tool.prompt.ARMS[2], config)
    assert result["orders"] == result["trades"] == []
    assert result["denominators"]["blocked"] == 0
    assert result["denominators"]["probability_callbacks"] == 0
    assert result["rp_metrics"]["execution"]["executed_order_fraction"]["status"] == "unknown"
    assert all(
        e["status"] == "not_offered" for row in result["events"] for e in row["opportunities"]
    )


@pytest.mark.parametrize(
    "config,reason",
    [
        (replace(tool.DemoConfig(), cash=(99, 500)), "insufficient_available_cash_for_buy"),
        (replace(tool.DemoConfig(), shares=(1, 4)), "insufficient_available_shares_for_sell"),
    ],
)
def test_external_gate_requires_both_directions_before_asking(config, reason):
    demo = tool.ConditionalExecutionDemo(callbacks(), tool.prompt.ARMS[2], config)
    event = demo.step()
    entry = event["opportunities"][0]
    assert reason in entry["gate_reasons"]
    assert entry["question"] is None and "probability_buy" not in entry
    assert entry["status"] == "blocked_external_opportunity"


def test_resting_order_self_cross_blocks_without_labeling_it_hold():
    result = full()
    a = result["events"][2]["opportunities"][0]
    assert a["gate_reasons"] == ["one_direction_would_cross_own_resting_order"]
    assert a["question"] is None
    assert a["state_before"][tool.prompt.memory.CURRENT] == {"buy": 2, "sell": 0}


def test_blocking_or_disabling_an_earlier_slot_never_shifts_later_uniforms():
    plain = full(0.5, 0.5)
    config = replace(tool.DemoConfig(), opportunities=(False,) + (True,) * 11)
    changed = full(0.5, 0.5, config=config)
    assert changed["preallocated_uniforms"] == plain["preallocated_uniforms"]
    for x, y in zip(plain["events"], changed["events"], strict=True):
        for a, b in zip(x["opportunities"], y["opportunities"], strict=True):
            assert (a["slot"], a["uniform"]) == (b["slot"], b["uniform"])


def test_all_inputs_precede_current_batch_actions_and_future_dividends():
    original = full()
    changed = full(0, 1, config=replace(tool.DemoConfig(), dividends=(0, 10)))
    for i in range(2):
        x = original["events"][0]["opportunities"][i]
        y = changed["events"][0]["opportunities"][i]
        assert x["question"] == y["question"]
        assert x["state_before"]["cash"] == 500 and x["state_before"]["shares"] == 4
        assert x["state_before"]["earlier_current_period_trades"]["count"] == 0
        wire = json.dumps(x["question"]["messages"])
        assert not any(term in wire for term in (*tool.AGENTS, "uniform", "slot", "priority"))
    next_state = original["events"][1]["opportunities"][0]["state_before"]
    assert next_state["earlier_current_period_trades"]["last_completed_trade_second"] == 0


@pytest.mark.parametrize(
    "invalid", [True, float("nan"), float("inf"), -0.01, 1.01, "0.5", {"BUY": 0.5}]
)
def test_bad_second_probability_never_partially_applies_this_batch_and_stops(invalid):
    demo = tool.ConditionalExecutionDemo(callbacks(1, invalid), tool.prompt.ARMS[2])
    before = demo.accounts()
    with pytest.raises(ValueError, match="finite native"):
        demo.step()
    assert demo.accounts() == before
    assert demo.exchange.orders == demo.exchange.trades == demo.accepted == []
    assert demo.next_tick == 0 and demo.failed
    with pytest.raises(ValueError, match="cannot advance"):
        demo.step()
    with pytest.raises(ValueError, match="complete"):
        demo.result()


def test_callback_mutating_messages_cannot_change_retained_question_or_account():
    def mutate(messages):
        messages[0]["content"] = "MUTATED"
        messages.clear()
        return 0.5

    result = tool.run_demo(dict.fromkeys(tool.AGENTS, mutate), tool.prompt.ARMS[2])
    expected = full(0.5, 0.5)
    assert result == expected


def test_rejected_intent_is_excluded_from_accepted_history_and_execution_denominator(monkeypatch):
    demo = tool.ConditionalExecutionDemo(callbacks(), tool.prompt.ARMS[2])
    original = demo.exchange.submit_order
    calls = 0

    def reject_first(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected pre-acceptance rejection")
        return original(*args)

    monkeypatch.setattr(demo.exchange, "submit_order", reject_first)
    first = demo.step()
    assert first["opportunities"][0]["status"] == "rejected"
    second = demo.step()
    assert second["opportunities"][0]["state_before"][tool.prompt.memory.CURRENT] == {
        "buy": 0,
        "sell": 0,
    }
    for _ in range(4):
        demo.step()
    result = demo.result()
    counts = result["denominators"]
    assert counts["rejected_submissions"] == 1
    assert counts["probability_callbacks"] == counts["accepted_submissions"] + 1
    assert (
        result["rp_metrics"]["execution"]["accepted_order_count"] == counts["accepted_submissions"]
    )


def test_only_next_scheduled_observation_allowed_and_cannot_extend_after_completion():
    demo = tool.ConditionalExecutionDemo(callbacks(), tool.prompt.ARMS[2])
    with pytest.raises(ValueError, match="next scheduled"):
        demo.observation(tool.AGENTS[0], 2, 0)
    for _ in tool.TICKS:
        demo.step()
    with pytest.raises(ValueError, match="cannot advance"):
        demo.step()


@pytest.mark.parametrize(
    "config",
    [
        replace(tool.DemoConfig(), cash=(True, 500)),
        replace(tool.DemoConfig(), shares=(-1, 4)),
        replace(tool.DemoConfig(), quantities=(0, 3)),
        replace(tool.DemoConfig(), dividends=(5, 0)),
        replace(tool.DemoConfig(), opportunities=(True,)),
        replace(tool.DemoConfig(), seed=True),
    ],
)
def test_invalid_config_rejected_before_any_callbacks(config):
    with pytest.raises(ValueError):
        tool.ConditionalExecutionDemo(callbacks(), tool.prompt.ARMS[2], config)
