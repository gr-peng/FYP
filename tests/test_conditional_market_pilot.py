"""Synthetic paired-menu construction and fixed-list sampling witnesses."""

from copy import deepcopy

import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import conditional_market_pilot as tool


def messages():
    demo = tool.execution.ConditionalExecutionDemo(
        dict.fromkeys(tool.execution.AGENTS, lambda _: 0.5), tool.ARMS[2]
    )
    state = demo.observation(tool.execution.AGENTS[0], 1, 0)
    return tool.execution.prompt.build_question(state, tool.ARMS[2], "buy_first")["messages"]


def test_paired_menus_change_only_choices_and_never_send_local_identity():
    original = messages()
    pair = tool.request_pair(original, tool.ARMS[2], tool.execution.AGENTS[0], 0)
    a, b = [tool.probability.strict_json(r["payload"]["messages"][1]["content"]) for r in pair]
    assert a["observation"] == b["observation"]
    assert a["choices"]["A"] == b["choices"]["B"]
    assert a["choices"]["B"] == b["choices"]["A"]
    for r in pair:
        wire = tool.canonical(r["payload"])
        assert not any(x in wire for x in (*tool.execution.AGENTS, tool.RUN_ID, "slot", "tick"))
        assert r["wire_sha256"] == tool.digest(r["payload"])
    assert original == messages()


@pytest.mark.parametrize(
    "values", [[0.2], [0.2, 0.4, 0.6], [True, 0.5], [float("nan"), 0.5], [-0.1, 1.1]]
)
def test_pair_aggregation_rejects_missing_or_invalid_menu_values(values):
    with pytest.raises(ValueError):
        tool.pair_probability(values)


def test_pair_average_is_semantic_buy_and_keeps_extremes():
    assert tool.pair_probability([0, 1]) == 0.5
    assert tool.pair_probability([0.2, 0.8]) == 0.5


def test_future_local_tick_changes_schedule_order_but_not_observation_wire_set():
    first = tool.request_pair(messages(), tool.ARMS[0], tool.execution.AGENTS[0], 0)
    later = tool.request_pair(messages(), tool.ARMS[1], tool.execution.AGENTS[0], 0)
    assert [r["menu_order"] for r in first] == list(reversed([r["menu_order"] for r in later]))
    assert {r["wire_sha256"] for r in first} == {r["wire_sha256"] for r in later}


def test_exact_fixed_list_convolution_with_zero_one_and_half_probabilities():
    pairs = []
    for arm in tool.ARMS:
        for p, u in ((0, 0.2), (1, 0.2), (0.5, 0.6), (0.5, 0.1)):
            pairs.append({"arm": arm, "mean_buy_probability": p, "uniform": u})
    result = tool.fixed_probability_sampling_diagnostic({"menu_pairs": pairs})
    for branch in result["branches"].values():
        assert branch["reference_buy_count_distribution"] == [0, 0.25, 0.5, 0.25, 0]
        assert branch["reference_expected_buy_count"] == 2
        assert branch["reference_buy_count_variance"] == 0.5
        assert branch["recorded_buy_count"] == 2


def test_hidden_provenance_cannot_be_smuggled_into_pair_payload():
    poison = deepcopy(messages())
    payload = tool.probability.strict_json(poison[1]["content"])
    payload["future"] = "FORBIDDEN"
    poison[1]["content"] = tool.canonical(payload)
    with pytest.raises(ValueError):
        tool.request_pair(poison, tool.ARMS[0], tool.execution.AGENTS[0], 0)
