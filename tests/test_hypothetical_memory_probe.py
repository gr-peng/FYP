"""Invented cue contrasts and strict probability parsing, never external requests."""

import copy
import json

import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from experiments import hypothetical_memory_probe as tool


def response(content='{"A":0.7,"B":0.3}'):
    return {
        "id": "synthetic-response",
        "created": 1,
        "model": tool.MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
    }


def test_schedule_is_mirrored_and_hidden_inputs_are_identical():
    package = tool.materials()
    requests = package["requests"]
    assert len(requests) == 48
    assert len({r["request_hash"] for r in requests}) == 48
    assert len({r["wire_sha256"] for r in requests}) == 14
    for i, r in enumerate(requests[:24]):
        other = requests[47 - i]
        assert r["repeat"] == 1 and other["repeat"] == 2
        assert r["payload"] == other["payload"]
        assert r["position"] + other["position"] == 49
    common = None
    for request in requests:
        text = tool.canonical(request["payload"])
        for local in (tool.RUN_ID, "buy_heavy", "sell_heavy", request["arm"]):
            assert local not in text
        obs = json.loads(request["payload"]["messages"][1]["content"])["observation"]
        state = obs["partial_observation"]
        public = {
            k: v
            for k, v in state.items()
            if k not in (tool.prompt.memory.CURRENT, tool.prompt.memory.COMPLETED)
        }
        common = public if common is None else common
        assert public == common
        if request["arm"] == tool.ARMS[2]:
            past = state[tool.prompt.memory.COMPLETED]
            assert (
                past["last_completed_period_with_own_orders"]
                == past["earlier_periods_this_repetition"]
            )
            assert past["previous_repetitions"] == {"buy": 0, "sell": 0}
    for menu in tool.MENUS:
        for arm in tool.ARMS:
            for current in tool.CUES:
                wires = {
                    r["wire_sha256"]
                    for r in requests
                    if r["menu_order"] == menu and r["arm"] == arm and r["current"] == current
                }
                assert len(wires) == (2 if arm == tool.ARMS[2] else 1)


def test_menu_mapping_preserves_raw_probabilities_without_normalization():
    for order in tool.MENUS:
        req = next(r for r in tool.materials()["requests"] if r["menu_order"] == order)
        parsed = tool.parse_response(response(), req)
        assert parsed["probability_buy"] == (0.7 if order == "buy_first" else 0.3)
        parsed = tool.parse_response(response('{"A":0.7,"B":0.300000001}'), req)
        assert parsed["normalized"] is False
        assert parsed["probabilities"]["B"] == 0.300000001
        assert parsed["raw_probability_mass"] != 1


@pytest.mark.parametrize(
    "content",
    [
        '{"A":0.2,"A":0.7,"B":0.3}',
        '{"A":NaN,"B":0.3}',
        '{"A":1e999,"B":0}',
        '{"A":true,"B":0}',
        '{"A":0.8,"B":0.3}',
        '{"A":0.7,"B":0.3,"answer":"BUY"}',
        '{"probabilities":{"A":0.7,"B":0.3}}',
        '```json\n{"A":0.7,"B":0.3}\n```',
        '{"A":-0.1,"B":1.1}',
    ],
)
def test_invalid_probability_text_rejected_without_repair(content):
    with pytest.raises(ValueError):
        tool.parse_response(response(content), tool.materials()["requests"][0])


@pytest.mark.parametrize(
    "kind",
    ["model", "id", "created", "choice_count", "stop", "index", "role", "tools", "reasoning"],
)
def test_wrong_response_identity_mode_or_truncation_rejected(kind):
    body = response()
    if kind in {"model", "id", "created"}:
        body[kind] = {"model": "old-model", "id": "", "created": True}[kind]
    elif kind == "choice_count":
        body["choices"].append(copy.deepcopy(body["choices"][0]))
    elif kind == "stop":
        body["choices"][0]["finish_reason"] = "length"
    elif kind == "index":
        body["choices"][0]["index"] = 1
    else:
        message = body["choices"][0]["message"]
        message[{"role": "role", "tools": "tool_calls", "reasoning": "reasoning_content"}[kind]] = (
            "unexpected"
        )
    with pytest.raises(ValueError):
        tool.parse_response(body, tool.materials()["requests"][0])


def rows_for(package, function):
    return [
        {
            "position": r["position"],
            "request_hash": r["request_hash"],
            "probability_buy": function(r),
        }
        for r in package["requests"]
    ]


def test_primary_subtracts_hidden_contrast_and_keeps_menu_effects():
    package = tool.materials()

    def probability(r):
        current = 1 if r["current"] == "buy_heavy" else -1
        completed = 1 if r["completed"] == "buy_heavy" else -1
        # Intentionally fabricate hidden-factor variation to test subtraction,
        # not model behavior or a correct answer for these invented records.
        return (
            0.5
            + 0.01 * current
            + 0.02 * completed
            + (0.1 * current if r["arm"] != tool.ARMS[0] else 0)
            + (0.05 * completed if r["arm"] == tool.ARMS[2] else 0)
            + (0.03 if r["menu_order"] == "buy_first" else -0.03)
        )

    result = tool.summarize(rows_for(package, probability), package)
    main = result["primary_descriptive_differences"]
    assert main["current_visible_minus_hidden_cue_effect"]["equal_menu_average"] == pytest.approx(
        0.2
    )
    assert main["completed_visible_minus_hidden_cue_effect"]["equal_menu_average"] == pytest.approx(
        0.1
    )
    assert all(
        r["difference"] == pytest.approx(0.06)
        for r in result["menu_differences_buy_first_minus_sell_first"]
    )
    assert len(result["same_wire_groups"]) == 14
    assert result["independent_generation_count"] is None
    assert len(result["cells"]) == 24


def test_menu_only_and_zero_response_are_not_reported_as_memory_effect():
    package = tool.materials()
    for function in (lambda r: 0.5, lambda r: 0.3 if r["menu_order"] == "buy_first" else 0.7):
        result = tool.summarize(rows_for(package, function), package)
        assert all(
            abs(value) < 1e-12
            for c in result["primary_descriptive_differences"].values()
            for value in c.values()
        )
        assert all(group["range"] == 0 for group in result["same_wire_groups"])


def test_partial_foreign_rows_or_modified_package_never_produce_contrasts():
    package = tool.materials()
    rows = rows_for(package, lambda r: 0.5)
    with pytest.raises(ValueError, match="incomplete"):
        tool.summarize(rows[:-1], package)
    changed = copy.deepcopy(rows)
    changed[0]["request_hash"] = changed[1]["request_hash"]
    with pytest.raises(ValueError, match="identity"):
        tool.summarize(changed, package)
    package["requests"][0]["payload"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="material"):
        tool.summarize(rows, package)
