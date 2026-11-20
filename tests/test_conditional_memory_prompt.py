"""Information-boundary tests use invented records only, with external access blocked."""

import copy
import json

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import conditional_memory_prompt as tool


def state():
    return {
        "cash": 900,
        "shares": 8,
        "period": 2,
        "repetition": 2,
        "elapsed_seconds": 40,
        "displayed_expected_remaining_dividends_per_share": 45,
        "pre_market_task": {"task": "video", "condition": "calm"},
        "earlier_current_period_trades": {
            "count": 3,
            "mean_price": 47,
            "last_completed_trade_second": 35,
            "mean_price_in_last_completed_trade_second": 48,
        },
        "earlier_periods": [
            {"period": 1, "mean_trade_price": 49, "closing_price": 48, "dividend": 10}
        ],
        tool.memory.CURRENT: {"buy": 3, "sell": 1},
        tool.memory.COMPLETED: {
            "earlier_periods_this_repetition": {"buy": 1, "sell": 2},
            "previous_repetitions": {"buy": 8, "sell": 2},
            "last_completed_period_with_own_orders": {"buy": 1, "sell": 2},
        },
    }


def payload(question):
    return json.loads(question["messages"][1]["content"])


def messages(s, arm):
    return tool.build_question(s, arm, "buy_first")["messages"]


@pytest.mark.parametrize("order", tool.MENUS)
def test_three_arms_share_all_common_information_and_only_selected_features(order):
    original = state()
    before = copy.deepcopy(original)
    triplet = tool.build_triplet(original, order)
    public = tool.public_state(original)
    for arm, question in triplet.items():
        assert question["messages"][0] == {"role": "system", "content": tool.INSTRUCTIONS}
        visible = payload(question)["observation"]
        assert {k: visible["partial_observation"][k] for k in public} == public
        features = visible["computed_features"]
        assert set(features) == set(tool.memory.ARMS[arm])
        np.testing.assert_array_equal(
            [features[k] for k in tool.memory.ARMS[arm]], tool.memory.features(original, arm)
        )
        expected = set(public)
        if arm != tool.ARMS[0]:
            expected.add(tool.memory.CURRENT)
        if arm == tool.ARMS[2]:
            expected.add(tool.memory.COMPLETED)
        assert set(visible["partial_observation"]) == expected
        assert list(payload(question)["choices"]) == ["A", "B"]
    assert original == before


def test_all_unselected_keys_including_nested_poison_do_not_reach_wire():
    clean, poisoned = state(), state()
    poison = "FORBIDDEN_SOURCE_LABEL_FUTURE_SENTINEL"
    poisoned.update({"target": {"buy": 1}, "provenance": poison, "future": poison})
    poisoned["computed_features"] = {"hidden_own_history": poison}
    for row in [
        poisoned["pre_market_task"],
        poisoned["earlier_current_period_trades"],
        *poisoned["earlier_periods"],
        poisoned[tool.memory.CURRENT],
        poisoned[tool.memory.COMPLETED],
        *poisoned[tool.memory.COMPLETED].values(),
    ]:
        row["leaked_answer_or_identity"] = poison
    for arm in tool.ARMS:
        assert messages(poisoned, arm) == messages(clean, arm)
        assert poison not in json.dumps(messages(poisoned, arm))


def test_withheld_histories_cannot_change_earlier_arms_but_selected_history_can():
    s, changed = state(), state()
    changed[tool.memory.CURRENT] = {"buy": 1, "sell": 9}
    assert messages(s, tool.ARMS[0]) == messages(changed, tool.ARMS[0])
    for arm in tool.ARMS[1:]:
        assert messages(s, arm) != messages(changed, arm)
    changed = state()
    changed[tool.memory.COMPLETED]["previous_repetitions"] = {"buy": 2, "sell": 8}
    for arm in tool.ARMS[:2]:
        assert messages(s, arm) == messages(changed, arm)
    assert messages(s, tool.ARMS[2]) != messages(changed, tool.ARMS[2])


def test_omitted_history_is_not_zero_and_not_required_for_excluded_arm():
    s = state()
    del s[tool.memory.CURRENT]
    del s[tool.memory.COMPLETED]
    assert messages(s, tool.ARMS[0]) == messages(state(), tool.ARMS[0])
    with pytest.raises(ValueError):
        messages(s, tool.ARMS[1])
    s[tool.memory.CURRENT] = {"buy": 0, "sell": 0}
    messages(s, tool.ARMS[1])
    with pytest.raises(ValueError):
        messages(s, tool.ARMS[2])
    assert "Omitted history is withheld or unknown" in tool.INSTRUCTIONS


def test_menu_flip_only_changes_choices_and_local_buy_mapping():
    for arm in tool.ARMS:
        a = tool.build_question(state(), arm, "buy_first")
        b = tool.build_question(state(), arm, "sell_first")
        assert a["buy_label"] == "A" and b["buy_label"] == "B"
        assert a["messages"][0] == b["messages"][0]
        x, y = payload(a), payload(b)
        assert x["observation"] == y["observation"]
        assert x["choices"]["A"] == y["choices"]["B"]
        assert x["choices"]["B"] == y["choices"]["A"]
        assert set(x) == {"observation", "choices"}
        assert arm not in a["messages"][1]["content"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("cash",), float("nan")),
        (("shares",), 1.5),
        (("shares",), True),
        (("elapsed_seconds",), 121),
        (("repetition",), 3),
        (("displayed_expected_remaining_dividends_per_share",), 46),
        (("pre_market_task", "condition"), "current answer is BUY"),
        (("earlier_current_period_trades", "last_completed_trade_second"), 40),
        (("earlier_current_period_trades", "mean_price"), float("inf")),
        (("earlier_current_period_trades", "count"), 0),
        (("earlier_periods", 0, "period"), 2),
        (("earlier_periods", 0, "dividend"), 5),
        ((tool.memory.CURRENT, "buy"), None),
        ((tool.memory.COMPLETED, "last_completed_period_with_own_orders", "buy"), 2),
    ],
)
def test_invalid_selected_information_stops_entire_triplet(path, value):
    s = state()
    parent = s
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    with pytest.raises((ValueError, KeyError)):
        tool.build_triplet(s, "buy_first")


def test_empty_observed_history_and_first_period_reset_boundaries():
    s = state()
    s.update(period=1, repetition=1, displayed_expected_remaining_dividends_per_share=50)
    s["earlier_periods"] = []
    s["earlier_current_period_trades"] = {
        "count": 0,
        "mean_price": None,
        "last_completed_trade_second": None,
        "mean_price_in_last_completed_trade_second": None,
    }
    s[tool.memory.COMPLETED] = {
        "earlier_periods_this_repetition": {"buy": 0, "sell": 0},
        "previous_repetitions": {"buy": 0, "sell": 0},
        "last_completed_period_with_own_orders": None,
    }
    tool.build_triplet(s, "buy_first")
    for group in ("earlier_periods_this_repetition", "previous_repetitions"):
        bad = copy.deepcopy(s)
        bad[tool.memory.COMPLETED][group] = {"buy": 1, "sell": 0}
        bad[tool.memory.COMPLETED]["last_completed_period_with_own_orders"] = {"buy": 1, "sell": 0}
        with pytest.raises(ValueError):
            tool.build_triplet(bad, "buy_first")


def test_questions_are_fresh_values_and_never_call_old_full_memory_wrapper(monkeypatch):
    from experiments import financial_probability

    monkeypatch.setattr(financial_probability, "visible", lambda *_: pytest.fail("old wrapper"))
    triplet = tool.build_triplet(state(), "buy_first")
    triplet[tool.ARMS[0]]["messages"][0]["content"] = "mutated locally"
    assert triplet[tool.ARMS[1]]["messages"][0]["content"] == tool.INSTRUCTIONS
    assert (
        tool.build_triplet(state(), "buy_first")[tool.ARMS[0]]["messages"][0]["content"]
        == tool.INSTRUCTIONS
    )


@pytest.mark.parametrize("contradiction", ["time_zero", "one_trade", "all_time_zero", "one_period"])
def test_visible_cross_field_contradictions_rejected(contradiction):
    s = state()
    if contradiction == "time_zero":
        s["elapsed_seconds"] = 0
        s["earlier_current_period_trades"] = {
            "count": 0,
            "mean_price": None,
            "last_completed_trade_second": None,
            "mean_price_in_last_completed_trade_second": None,
        }
        # Hidden own counts do not constrain the first arm's displayed state.
        messages(s, tool.ARMS[0])
    elif contradiction == "one_trade":
        s["earlier_current_period_trades"]["count"] = 1
    elif contradiction == "all_time_zero":
        s["earlier_current_period_trades"]["last_completed_trade_second"] = 0
    else:
        s[tool.memory.COMPLETED]["last_completed_period_with_own_orders"]["sell"] = 1
    with pytest.raises(ValueError):
        tool.build_triplet(s, "buy_first")


def test_valid_boundary_cases_and_multiple_period_subset_remain_allowed():
    s = state()
    s["elapsed_seconds"] = 1
    s["earlier_current_period_trades"].update(
        count=1,
        mean_price=48,
        last_completed_trade_second=0,
    )
    tool.build_triplet(s, "buy_first")
    s["earlier_current_period_trades"]["count"] = 2
    tool.build_triplet(s, "buy_first")
    s.update(period=3, displayed_expected_remaining_dividends_per_share=40)
    s["earlier_periods"].append(
        {"period": 2, "mean_trade_price": 47, "closing_price": 46, "dividend": 0}
    )
    s[tool.memory.COMPLETED]["last_completed_period_with_own_orders"]["sell"] = 1
    tool.build_triplet(s, "buy_first")
