"""Frozen invented memory-cue questions and descriptive probability contrasts.

No human data, scores, fitting, network, or credentials. Direction is not accuracy.
"""

import copy
import hashlib
import json
import math
import random
from collections import defaultdict

import numpy as np

from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode
from experiments import conditional_memory_prompt as prompt

PROTOCOL = "hypothetical-memory-cue-probability-probe-v1"
RUN_ID = "hypothetical-memory-20261107-001"
MODEL = "DeepSeek-V4.1-Flash"
COUNT, PRIOR = 48, 349
CUES = ("buy_heavy", "sell_heavy")
ARMS, MENUS = prompt.ARMS, tuple(prompt.MENUS)
OPTIONS = {
    "model": MODEL,
    "temperature": 0.7,
    "max_tokens": 256,
    "enable_thinking": False,
    "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    "cache": {"no-cache": True, "no-store": True},
    "response_format": {"type": "json_object"},
}
require = prompt.require


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def scenario(current, completed):
    require(current in CUES and completed in CUES, "unknown invented cue")
    counts = {"buy_heavy": {"buy": 8, "sell": 2}, "sell_heavy": {"buy": 2, "sell": 8}}
    earlier = copy.deepcopy(counts[completed])
    return {
        "cash": 900,
        "shares": 8,
        "period": 2,
        "repetition": 1,
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
        prompt.memory.CURRENT: copy.deepcopy(counts[current]),
        prompt.memory.COMPLETED: {
            "earlier_periods_this_repetition": earlier,
            "previous_repetitions": {"buy": 0, "sell": 0},
            "last_completed_period_with_own_orders": copy.deepcopy(earlier),
        },
    }


def materials():
    slots = [
        (current, completed, arm, order)
        for current in CUES
        for completed in CUES
        for arm in ARMS
        for order in MENUS
    ]
    random.Random(20261107).shuffle(slots)
    requests = []
    for repeat, schedule in ((1, slots), (2, list(reversed(slots)))):
        for current, completed, arm, order in schedule:
            question = prompt.build_question(scenario(current, completed), arm, order)
            payload = {**copy.deepcopy(OPTIONS), "messages": question["messages"]}
            identity = {
                "run_id": RUN_ID,
                "position": len(requests) + 1,
                "current": current,
                "completed": completed,
                "arm": arm,
                "menu_order": order,
                "repeat": repeat,
                "buy_label": question["buy_label"],
            }
            requests.append(
                {
                    **identity,
                    "payload": payload,
                    "wire_sha256": digest(payload),
                    "request_hash": digest({"identity": identity, "payload": payload}),
                }
            )
    require(
        len(requests) == COUNT and len({r["wire_sha256"] for r in requests}) == 14,
        "wrong fixed coverage or hidden information leaked",
    )
    for order in MENUS:
        for arm in ARMS:
            selected = [r for r in requests if r["arm"] == arm and r["menu_order"] == order]
            require(
                len({r["wire_sha256"] for r in selected})
                == {ARMS[0]: 1, ARMS[1]: 2, ARMS[2]: 4}[arm],
                "hidden-factor payload differs",
            )
    return {
        "protocol": PROTOCOL,
        "data_origin": "entirely_invented_no_human_rows",
        "requests": requests,
        "options": copy.deepcopy(OPTIONS),
        "expected_unique_wire_bodies": 14,
        "same_wire_repetition_is_intentional": True,
    }


def parse_response(body, request):
    require(request["menu_order"] in MENUS, "unknown local menu mapping")
    require(isinstance(body, dict) and body.get("model") == MODEL, "returned model differs")
    require(
        isinstance(body.get("id"), str)
        and body["id"].strip()
        and type(body.get("created")) is int
        and body["created"] >= 0,
        "missing response identity",
    )
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "require one response choice")
    choice = choices[0]
    require(
        isinstance(choice, dict)
        and type(choice.get("index")) is int
        and choice["index"] == 0
        and choice.get("finish_reason") == "stop",
        "incomplete response choice",
    )
    msg = choice.get("message")
    require(
        isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("content"), str)
        and not msg.get("tool_calls")
        and not msg.get("function_call")
        and not msg.get("refusal")
        and returned_mode(body) == "no_returned_reasoning_observed",
        "incompatible response mode",
    )
    p = strict_json(msg["content"])
    require(isinstance(p, dict) and set(p) == {"A", "B"}, "exact A/B probability object required")
    require(
        all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in p.values())
        and abs(sum(p.values()) - 1) <= 1e-8,
        "invalid probability values or mass",
    )
    require(
        request["buy_label"] == ("A" if request["menu_order"] == "buy_first" else "B"),
        "local menu mapping differs",
    )
    return {
        "probabilities": p,
        "probability_buy": p[request["buy_label"]],
        "raw_probability_mass": sum(p.values()),
        "normalized": False,
    }


def summarize(rows, package):
    require(canonical(package) == canonical(materials()), "material package differs")
    expected = package["requests"]
    require(len(rows) == COUNT, "incomplete batch cannot produce primary contrasts")
    groups, by_wire = defaultdict(list), defaultdict(list)
    for row, request in zip(rows, expected, strict=True):
        require(
            row["position"] == request["position"]
            and row["request_hash"] == request["request_hash"],
            "response row identity differs",
        )
        p = row["probability_buy"]
        require(
            type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1,
            "invalid scored probability",
        )
        key = tuple(request[k] for k in ("current", "completed", "arm", "menu_order"))
        groups[key].append(p)
        by_wire[request["wire_sha256"]].append(
            {"position": request["position"], "probability_buy": p}
        )
    require(
        len(groups) == 24 and all(len(v) == 2 for v in groups.values()), "cell coverage differs"
    )
    means = {key: float(np.mean(p)) for key, p in groups.items()}

    def effect(factor, arm, menu):
        index = 0 if factor == "current" else 1
        return float(
            np.mean([v for k, v in means.items() if k[2:] == (arm, menu) and k[index] == CUES[0]])
            - np.mean([v for k, v in means.items() if k[2:] == (arm, menu) and k[index] == CUES[1]])
        )

    effects = {
        factor: {arm: {menu: effect(factor, arm, menu) for menu in MENUS} for arm in ARMS}
        for factor in ("current", "completed")
    }
    for factor in effects.values():
        for values in factor.values():
            values["equal_menu_average"] = float(np.mean(list(values.values())))
    primary = {}
    for factor, visible, hidden in (("current", ARMS[1], ARMS[0]), ("completed", ARMS[2], ARMS[1])):
        primary[factor + "_visible_minus_hidden_cue_effect"] = {
            menu: effects[factor][visible][menu] - effects[factor][hidden][menu]
            for menu in (*MENUS, "equal_menu_average")
        }
    return {
        "protocol": PROTOCOL,
        "valid_responses": len(rows),
        "data_origin": "entirely_invented_no_human_rows",
        "primary_descriptive_differences": primary,
        "all_directional_effects": effects,
        "cells": [
            {
                "current": k[0],
                "completed": k[1],
                "arm": k[2],
                "menu_order": k[3],
                "probabilities_buy": groups[k],
                "mean_probability_buy": v,
            }
            for k, v in sorted(means.items())
        ],
        "menu_differences_buy_first_minus_sell_first": [
            {
                "current": c,
                "completed": h,
                "arm": arm,
                "difference": means[c, h, arm, MENUS[0]] - means[c, h, arm, MENUS[1]],
            }
            for c in CUES
            for h in CUES
            for arm in ARMS
        ],
        "same_wire_groups": [
            {
                "wire_sha256": key,
                "rows": values,
                "range": max(r["probability_buy"] for r in values)
                - min(r["probability_buy"] for r in values),
            }
            for key, values in sorted(by_wire.items())
        ],
        "independent_generation_count": None,
        "limits": "Directional response is not correctness. Two repetitions are not two humans. "
        "No human labels, calibration score, significance or market realism claim. Raw counts "
        "and corresponding computed features change together; "
        "representation effects are not isolated.",
    }
