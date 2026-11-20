"""Prepare four unsubmitted economic cases with matched finite-account strategy returns."""

from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from behavioral_market.agents.policy import rule_order, signal
from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.market import Portfolio
from scripts import audit_finite_strategy_returns as finite

source = finite.source
old = source.materials
METHOD = "finite_rule_order_replay_with_assigned_quotes"
HALF_SPREAD = "0.05"


def make_case(style, condition):
    source.require(style in ("fundamental", "technical"), "unknown style")
    source.require(condition in ("positive", "negative"), "unknown condition")
    base = old.make_case(style, condition)
    agent_id = base["case_id"].split(".")[0]
    tail = [90, 121, 100, 100, 100, 100] if condition == "positive" else [100] * 6
    if style == "technical":
        tail = [100, 110, 100, 100, 100, 100]
    warmup = [100] * 14 + tail
    dates = old.pd.bdate_range(end="2026-01-02", periods=20).strftime("%Y-%m-%d")
    history = old.pd.DataFrame([old.bar(d, p, p) for d, p in zip(dates, warmup, strict=True)])
    portfolio = Portfolio(1000, 2, 100)
    portfolio.mark_to_market(100)
    agent = AgentState(
        AgentPersona(agent_id, 0.5, 0.5, 0.5, 0.5, 0.5, style),
        portfolio,
        style,
        1200,
        1200,
        peak_equity=1200,
    )
    alternative = "technical" if style == "fundamental" else "fundamental"
    anchors = [110, 100, 100, 100, 100] if condition == "positive" else [100, 121, 100, 100, 100]
    trace = []
    for i, (session, close) in enumerate(zip(old.SESSIONS, old.BLOCK_CLOSES, strict=True)):
        previous = float(history.close.iloc[-1])
        observation = {
            **old.features(history),
            "fundamental_value": previous if style == "fundamental" else anchors[i],
        }
        decision = rule_order(agent, observation, old.ZeroNoise())
        source.require(decision.action == "hold", "current strategy must derive HOLD")
        before = float(portfolio.total_equity)
        portfolio.mark_to_market(close)
        equity = float(portfolio.total_equity)
        agent.peak_equity = max(agent.peak_equity, equity)
        agent.memory.append(
            {
                "session": session,
                "owner": agent_id,
                "style": style,
                "action": "hold",
                "filled_quantity": 0,
                "pnl_change": round(equity - before, 4),
                "market_return": round(close / previous - 1, 6),
                "events": [],
                "close": close,
                "cash": float(portfolio.cash),
                "position": portfolio.position,
                "equity": equity,
            }
        )
        trace.append(
            {
                "session_date": session,
                "history_last_session": history.session_date.iloc[-1],
                "start_reference": previous,
                "close": close,
                "fundamental_anchor": observation["fundamental_value"],
                "momentum_5": observation["momentum_5"],
                "current_signal": signal(style, observation),
                "current_decision": decision.model_dump(),
                "current_equity": equity,
                "alternative_signal": signal(alternative, observation),
            }
        )
        history = old.pd.concat(
            [history, old.pd.DataFrame([old.bar(session, previous, close)])], ignore_index=True
        )
    case = {
        "case_id": base["case_id"],
        "original_style": style,
        "economic_condition": condition,
        "construction_not_sent": {"warmup_closes": warmup, "daily_trace": trace},
    }
    actual, counterfactual = (
        finite.replay(case, style, HALF_SPREAD),
        finite.replay(case, alternative, HALF_SPREAD),
    )
    current_return, other_return = Decimal(actual["return"]), Decimal(counterfactual["return"])
    source.require(current_return == 0, "current return must remain zero")
    source.require(
        (other_return > current_return) == (condition == "positive"),
        "finite positive/negative construction differs",
    )
    cf = {
        "method": METHOD,
        "actual_block_return": float(current_return),
        "alternative_block_return": float(other_return),
        "gap": float(other_return - current_return),
    }
    state = agent.context()
    state["counterfactual"] = cf
    payload = deepcopy(base["payload"])
    payload["model"] = "DeepSeek-V4.1-Flash"
    user = source.strict_json(payload["messages"][1]["content"])
    user["agent_state"] = {**state, "persona": {"agent_id": agent_id}}
    observation = user["observation"]
    observation.update(old.features(history))
    observation.update(
        data_origin="constructed_finite_account_controls_not_historical_market_data",
        fundamental_value=trace[-1]["fundamental_anchor"],
        completed_prices=history[["session_date", "close"]].to_dict(orient="records"),
        counterfactual_evidence={
            "method": METHOD,
            "alternative_style": alternative,
            "hypothetical_separate_replay_not_actual_personal_trades": True,
            "execution_assumptions": {
                "initial_cash_each": 1000,
                "initial_shares_each": 2,
                "initial_equity_each": 1200,
                "fee_rate": 0,
                "noise": 0,
                "assigned_half_spread": HALF_SPREAD,
                "counterparty_initial_cash": 10000,
                "counterparty_initial_shares": 100,
                "quotes": "One finite direction-conditioned quote at previous reference "
                "+halfspread for a buy or -halfspread for a sell. Fill only if limits cross. "
                "DAY leftovers expire; closing marks are externally assigned after orders.",
            },
            "original_strategy_replay": actual,
            "alternative_strategy_replay": counterfactual,
            "limits": "Constructed execution and price path,not a real market forecast. "
            "No guaranteed future advantage or required target strategy.",
        },
    )
    user["decision_instruction"] = (
        "The five-session block from 2026-01-05 through 2026-01-09 has ended. Evaluate "
        "whether to stay or switch for subsequent sessions. Stay must target current_style; "
        "switch must target the other style. The counterfactual is a separate finite-account "
        "rule replay under the stated assigned execution assumptions. It is not your personal "
        "trade history,an actual market result or a guaranteed future return. "
        + old.CLARIFICATION.replace("Historical proxy advantage", "Historical simulated advantage")
    )
    payload["messages"][1]["content"] = old.canonical(user)
    case.update(
        treatment="vanilla_llm",
        mean_field_visibility="hidden",
        status="unsubmitted",
        payload=payload,
        planned_payload_sha256=old.digest(payload),
    )
    case["construction_not_sent"].update(
        final_state=state,
        original_replay=actual,
        alternative_replay=counterfactual,
        rule_answer_key_not_defined=True,
    )
    return case


def build_materials():
    return {
        "protocol": "finite-account-economic-materials-v1",
        "status": "materials_only",
        "actual_remote_calls": 0,
        "counterfactual_method": METHOD,
        "cases": [
            make_case(style, condition)
            for style in ("fundamental", "technical")
            for condition in ("positive", "negative")
        ],
        "analysis_scope": "Within each original style,compare the complete economic bundle,"
        "not a gap-only change. Warmup/anchors/derived features and hypothetical replay differ. "
        "No old12/36comparison can isolate this benchmark change. No correct target is defined. "
        "These new materials are unsubmitted and need a separate frozen execution plan/review.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = build_materials()
    args.output.mkdir(parents=True, exist_ok=False)
    source.save(args.output / "materials.json", package)
    print(old.canonical({"cases": 4, "status": "materials_only", "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
