"""Offline, explicitly hypothetical contrasts; this module never calls a model."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from copy import deepcopy
from typing import Any

from behavioral_market.evaluation.style_probe_intake import strict_json

STYLES = ("fundamental", "technical")
FACTORS = {
    "current_style": STYLES,
    "reference_condition": ("neutral", "above_price"),
    "benchmark_favored_style": STYLES,
    "majority_style": STYLES,
}
SYSTEM = (
    "You are a simulated retail investor choosing a strategy for the next block. "
    "All supplied prices, accounts, histories, reference values, benchmark returns and "
    "population counts are hypothetical. Use only this information. "
    "Fundamental style emphasizes price relative to a supplied reference value; technical "
    "style emphasizes price trends. Neither style has a prescribed choice in this task. "
    "Return one JSON object with decision, target_style and rationale. "
    "This task requests a strategy choice. Keep rationale under 60 words."
)
INSTRUCTION = (
    "The 10-session block has ended. Choose a style for subsequent sessions. "
    "Stay must target current_style; switch must target the other style. "
    "The reference value is an assigned scenario input, not true firm value. "
    "The two named benchmark returns are externally assigned historical "
    "examples, not generated strategy rules, forecasts or your personal counterfactual "
    "returns. Your own account and actions are separate from those benchmarks. "
    "History style labels are assigned for this hypothetical setting; both labels can "
    "accompany the same hold actions. Population counts exclude you and refer to session 9."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["stay", "switch"]},
        "target_style": {"type": "string", "enum": list(STYLES)},
        "rationale": {"type": "string", "maxLength": 2000},
    },
    "required": ["decision", "target_style", "rationale"],
}
ENVELOPE = {
    "model": "DeepSeek-V4-Flash",
    "temperature": 0.2,
    "max_tokens": 512,
    "enable_thinking": False,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
}
MATERIAL_TYPE = "fully_synthetic_matched_style_design_not_executed"
USER_PREFIX = "messages.1.content."


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Empty containers remain visible; no input field disappears from an inventory."""
    if isinstance(value, dict) and value:
        return {
            p: v
            for key, child in value.items()
            for p, v in flatten(
                child, f"{prefix}.{str(key).replace('~', '~0').replace('.', '~1')}".lstrip(".")
            ).items()
        }
    if isinstance(value, list) and value:
        return {
            p: v
            for index, child in enumerate(value)
            for p, v in flatten(child, f"{prefix}.{index}".lstrip(".")).items()
        }
    return {prefix: value}


def semantic_payload(payload: dict) -> dict:
    result = deepcopy(payload)
    result["messages"][1]["content"] = strict_json(result["messages"][1]["content"])
    return result


def changed_paths(left: Any, right: Any) -> list[str]:
    a, b = flatten(left), flatten(right)
    return sorted(k for k in a.keys() | b.keys() if k not in a or k not in b or a[k] != b[k])


def allowed_paths() -> dict[str, list[str]]:
    paths = {
        "current_style": ["agent_state.current_style"]
        + [f"agent_state.memory.{i}.style" for i in range(10)],
        "reference_condition": ["observation.reference.value"],
        "benchmark_favored_style": [
            f"observation.style_benchmarks.returns.{style}" for style in STYLES
        ]
        + ["observation.style_benchmarks.fundamental_minus_technical_return"],
        "majority_style": [
            f"observation.mean_field.{style}_{field}"
            for style in STYLES
            for field in ("count", "share")
        ],
    }
    return {key: sorted(USER_PREFIX + p for p in values) for key, values in paths.items()}


def build_case(factors: dict[str, str]) -> dict:
    if set(factors) != set(FACTORS) or any(factors[k] not in v for k, v in FACTORS.items()):
        raise ValueError("unknown or incomplete factors")
    prices = [100.0 - 0.5 * i for i in range(11)]
    current = factors["current_style"]
    benchmarks = {}
    for style in STYLES:
        weight = 0.1 if style == factors["benchmark_favored_style"] else 0.9
        position = int(20000 * weight / prices[0])
        cash = 20000 - position * prices[0]
        final_equity = cash + position * prices[-1]
        benchmarks[style] = {
            "initial_stock_weight": weight,
            "initial_cash": cash,
            "position": position,
            "initial_equity": 20000.0,
            "final_equity": final_equity,
            "return": final_equity / 20000 - 1,
        }
    document = {
        "scenario": {"hypothetical": True, "asset": "SYNTHETIC_ASSET", "review_session": 10},
        "agent_state": {
            "persona": {"agent_id": "SYNTHETIC_AGENT"},
            "current_style": current,
            "initial_portfolio": {"cash": 10000.0, "position": 100, "avg_cost": 100.0},
            "portfolio": {
                "cash": 10000.0,
                "position": 100,
                "avg_cost": 100.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": -500.0,
                "total_equity": 19500.0,
            },
            "memory": [
                {
                    "session": i,
                    "style": current,
                    "action": "hold",
                    "filled_quantity": 0,
                    "pnl_change": -50.0,
                    "market_return": prices[i] / prices[i - 1] - 1,
                    "total_equity": 20000.0 - 50 * i,
                }
                for i in range(1, 11)
            ],
            "actual_block_return": -0.025,
            "drawdown": 0.025,
        },
        "observation": {
            "session": 10,
            "price_history": [{"session": i, "close": p} for i, p in enumerate(prices)],
            "last_close": prices[-1],
            "return_1d": prices[-1] / prices[-2] - 1,
            "return_5d": prices[-1] / prices[-6] - 1,
            "ma5": sum(prices[-5:]) / 5,
            "reference": {
                "value": 95.0 if factors["reference_condition"] == "neutral" else 105.0,
                "method": "externally_assigned_hypothetical_input_not_true_value",
            },
            "style_benchmarks": {
                "method": "externally_assigned_returns_not_personal_counterfactual",
                "start_session": 0,
                "end_session": 10,
                "returns": {style: benchmarks[style]["return"] for style in STYLES},
                "fundamental_minus_technical_return": (
                    benchmarks["fundamental"]["return"] - benchmarks["technical"]["return"]
                ),
            },
            "mean_field": {
                "as_of_session": 9,
                "population_size_excluding_self": 100,
                **{
                    f"{style}_{field}": (80 if field == "count" else 0.8)
                    if style == factors["majority_style"]
                    else (20 if field == "count" else 0.2)
                    for style in STYLES
                    for field in ("count", "share")
                },
            },
        },
        "decision_instruction": INSTRUCTION,
        "output_schema": deepcopy(SCHEMA),
    }
    payload = {
        **deepcopy(ENVELOPE),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(document)},
        ],
    }
    code = "".join(str(FACTORS[k].index(factors[k])) for k in FACTORS)
    return {
        "case_id": f"synthetic_{code}",
        "material_type": MATERIAL_TYPE,
        "factors": dict(factors),
        "private_construction_ledger_not_sent": {
            "method": (
                "initial_long_cash_weights_then_hold; realizability_witness_not_strategy_rules"
            ),
            "portfolios": benchmarks,
        },
        "payload": payload,
        "payload_sha256": digest(payload),
    }


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("inconsistent material")


def _close(actual: Any, expected: float) -> None:
    _require(isinstance(actual, (int, float)) and not isinstance(actual, bool))
    _require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9))


def _same_structure(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_structure(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_structure(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return not isinstance(actual, (dict, list))


def validate_case(row: dict) -> list[str]:
    """Inconsistent or unknown materials are ineligible; nothing is filled or repaired."""
    try:
        _require(
            set(row)
            == {
                "case_id",
                "material_type",
                "factors",
                "payload",
                "payload_sha256",
                "private_construction_ledger_not_sent",
            }
        )
        factors, payload = row["factors"], row["payload"]
        _require(set(factors) == set(FACTORS))
        _require(all(factors[k] in v for k, v in FACTORS.items()))
        code = "".join(str(FACTORS[k].index(factors[k])) for k in FACTORS)
        _require(row["case_id"] == f"synthetic_{code}")
        _require(row["material_type"] == MATERIAL_TYPE)
        _require(row["payload_sha256"] == digest(payload))
        _require(
            canonical({k: v for k, v in payload.items() if k != "messages"}) == canonical(ENVELOPE)
        )
        _require(len(payload["messages"]) == 2)
        _require(payload["messages"][0] == {"role": "system", "content": SYSTEM})
        _require(set(payload["messages"][1]) == {"role", "content"})
        _require(payload["messages"][1]["role"] == "user")
        d = strict_json(payload["messages"][1]["content"])
        _require(d["decision_instruction"] == INSTRUCTION and d["output_schema"] == SCHEMA)
        _require(
            canonical(d["scenario"])
            == canonical({"hypothetical": True, "asset": "SYNTHETIC_ASSET", "review_session": 10})
        )
        a, o = d["agent_state"], d["observation"]
        _require(a["persona"] == {"agent_id": "SYNTHETIC_AGENT"})
        _require(a["current_style"] == factors["current_style"])
        _require(a["initial_portfolio"] == {"cash": 10000.0, "position": 100, "avg_cost": 100.0})
        prices = [100.0 - 0.5 * i for i in range(11)]
        _require(o["price_history"] == [{"session": i, "close": p} for i, p in enumerate(prices)])
        _require(o["session"] == 10)
        _close(o["last_close"], prices[-1])
        _close(o["return_1d"], prices[-1] / prices[-2] - 1)
        _close(o["return_5d"], prices[-1] / prices[-6] - 1)
        _close(o["ma5"], sum(prices[-5:]) / 5)
        _require(len(a["memory"]) == 10)
        equity = [20000.0]
        for i, memory in enumerate(a["memory"], 1):
            _require(memory["session"] == i and memory["style"] == a["current_style"])
            _require(memory["action"] == "hold" and memory["filled_quantity"] == 0)
            _close(memory["pnl_change"], 100 * (prices[i] - prices[i - 1]))
            _close(memory["market_return"], prices[i] / prices[i - 1] - 1)
            _close(memory["total_equity"], 10000 + 100 * prices[i])
            equity.append(memory["total_equity"])
        portfolio = a["portfolio"]
        _require(portfolio["cash"] == 10000 and portfolio["position"] == 100)
        _require(portfolio["avg_cost"] == 100 and portfolio["realized_pnl"] == 0)
        _close(portfolio["unrealized_pnl"], 100 * (prices[-1] - portfolio["avg_cost"]))
        _close(portfolio["total_equity"], portfolio["cash"] + 100 * prices[-1])
        _close(a["actual_block_return"], equity[-1] / equity[0] - 1)
        peak, drawdown = equity[0], 0.0
        for value in equity:
            peak = max(peak, value)
            drawdown = max(drawdown, 1 - value / peak)
        _close(a["drawdown"], drawdown)
        _close(o["reference"]["value"], 95 if factors["reference_condition"] == "neutral" else 105)
        _require(
            o["reference"]["method"] == "externally_assigned_hypothetical_input_not_true_value"
        )
        benchmarks = o["style_benchmarks"]
        _require(benchmarks["method"] == "externally_assigned_returns_not_personal_counterfactual")
        _require(benchmarks["start_session"] == 0 and benchmarks["end_session"] == 10)
        private = row["private_construction_ledger_not_sent"]
        _require(
            private["method"]
            == ("initial_long_cash_weights_then_hold; realizability_witness_not_strategy_rules")
        )
        _require(set(private) == {"method", "portfolios"})
        _require(set(private["portfolios"]) == set(STYLES))
        for style in STYLES:
            b = private["portfolios"][style]
            _require(
                set(b)
                == {
                    "initial_stock_weight",
                    "initial_cash",
                    "position",
                    "initial_equity",
                    "final_equity",
                    "return",
                }
            )
            weight = 0.1 if style == factors["benchmark_favored_style"] else 0.9
            _close(b["initial_stock_weight"], weight)
            _close(b["initial_equity"], 20000)
            _close(b["position"], 20000 * weight / prices[0])
            _close(b["initial_cash"], 20000 - b["position"] * prices[0])
            _close(b["final_equity"], b["initial_cash"] + b["position"] * prices[-1])
            _close(b["return"], b["final_equity"] / b["initial_equity"] - 1)
            _close(benchmarks["returns"][style], b["return"])
        _close(
            benchmarks["fundamental_minus_technical_return"],
            benchmarks["returns"]["fundamental"] - benchmarks["returns"]["technical"],
        )
        m = o["mean_field"]
        _require(m["as_of_session"] == 9 and m["as_of_session"] < o["session"])
        _require(m["population_size_excluding_self"] == 100)
        for style in STYLES:
            count = 80 if style == factors["majority_style"] else 20
            _require(type(m[f"{style}_count"]) is int and m[f"{style}_count"] == count)
            _close(m[f"{style}_share"], count / m["population_size_excluding_self"])
        # Reject undeclared extra signals as well as missing ones.
        expected = strict_json(build_case(factors)["payload"]["messages"][1]["content"])
        _require(_same_structure(d, expected))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
        return ["ineligible_inconsistent_or_unknown_material"]
    return []


def build_bank() -> list[dict]:
    return [
        build_case(dict(zip(FACTORS, values, strict=True)))
        for values in itertools.product(*FACTORS.values())
    ]


def contrast_edges(rows: list[dict]) -> list[dict]:
    """All 32 matched edges of the four-factor cube, with exact permitted field changes."""
    if len(rows) != 16 or any(validate_case(r) for r in rows):
        raise ValueError("incomplete, duplicated or ineligible bank")
    by_factors = {tuple(r["factors"][k] for k in FACTORS): r for r in rows}
    if len(by_factors) != 16:
        raise ValueError("incomplete, duplicated or ineligible bank")
    edges = []
    for key, left in by_factors.items():
        for index, (factor, levels) in enumerate(FACTORS.items()):
            if key[index] != levels[0]:
                continue
            target = list(key)
            target[index] = levels[1]
            right = by_factors[tuple(target)]
            changes = changed_paths(
                semantic_payload(left["payload"]), semantic_payload(right["payload"])
            )
            if changes != allowed_paths()[factor]:
                raise ValueError(f"unapproved linked-field changes: {factor}")
            positive, negative = (right, left) if factor == "reference_condition" else (left, right)
            edges.append(
                {
                    "factor": factor,
                    "left_case": left["case_id"],
                    "right_case": right["case_id"],
                    "conditioning": {k: v for k, v in left["factors"].items() if k != factor},
                    "changed_paths": changes,
                    "positive_case": positive["case_id"],
                    "negative_case": negative["case_id"],
                    "outcome": (
                        "I(target_style=fundamental)[positive] "
                        "- I(target_style=fundamental)[negative]"
                    ),
                }
            )
    return edges


def design_specification() -> dict:
    return {
        "status": "offline_design_only_no_observed_answers",
        "material_type": MATERIAL_TYPE,
        "factors": FACTORS,
        "cells": 16,
        "matched_edges": 32,
        "edges_per_factor": 8,
        "linked_field_allowlists": allowed_paths(),
        "primary_outcome": (
            "target_style=fundamental; all four factors use the same named-style outcome"
        ),
        "secondary_outcome": (
            "switch, derived relative to current_style; "
            "not interchangeable across current-style levels"
        ),
        "unknown_handling": (
            "Missing, malformed or decision/target-incoherent answers stay unknown; "
            "never fill stay or drop silently."
        ),
        "analysis": (
            "Show every matched contrast and its conditioning levels before any summary; "
            "shared endpoints are not independent replications. "
            "No significance claim from 16 single answers."
        ),
        "effect_scopes": {
            "current_style": (
                "Joint current/history style-label sensitivity with identical actions and ledger; "
                "not learned habit or personal-history inertia."
            ),
            "reference_condition": (
                "Neutral reference versus above-price reference in this declining path; "
                "not all valuation reasoning, true value or an attainable trading advantage."
            ),
            "benchmark_favored_style": (
                "Sensitivity to supplied named historical benchmark rankings; "
                "not personal counterfactual regret, profit/loss asymmetry "
                "or generated strategy performance."
            ),
            "majority_style": (
                "Sensitivity to supplied lagged majority labels with all other synthetic fields "
                "matched; not real human herding or endogenous market feedback."
            ),
        },
        "prediction_policy": (
            "Report sensitivity in either direction, including no response; "
            "do not tune prompts to force herding or switching."
        ),
        "benchmark_privacy": (
            "Initial exposure and cash/stock construction ledgers stay outside every model "
            "payload; only supplied returns and their derived gap change in the benchmark contrast."
        ),
        "execution_gate": {
            "new_completion_POST_limit": 0,
            "used_authorized_completion_POSTs": 21,
            "authorized_cap": 21,
            "requires": [
                "Independent acceptance of frozen materials and a concrete execution plan.",
                "Explicit new call authorization; 16 design cells are not 16 approved calls.",
                "Resolve how returned-response reuse is handled before claiming fresh samples; "
                "changed prompts or IDs alone do not prove independent generation.",
                "Freeze ordering, repetitions, invalid-answer and stopping rules before execution; "
                "no adaptive outcome filtering or automatic retries.",
            ],
        },
        "human_alignment_established": False,
        "causal_effect_observed": False,
        "new_market_runs": 0,
        "SCIT_resumed": False,
    }
