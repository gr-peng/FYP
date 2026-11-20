"""Prepare four coherent hypothetical economic controls, without sending any requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from behavioral_market.agents.policy import rule_order, rule_style, signal
from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.data.features import features
from behavioral_market.llm.prompts import PROMPT_VERSION, prompts
from behavioral_market.market.portfolio import Portfolio

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "clarified-economic-style-controls-v1"
CLARIFICATION = (
    "This answer selects the strategy used in subsequent sessions. It does not itself submit "
    "or cancel an order, execute a trade, change the current portfolio, or automatically incur "
    "a transaction fee. Fundamental/technical style is distinct from BUY/SELL/HOLD. "
    "Historical proxy advantage is not a guaranteed future return or an instruction to switch."
)
SESSIONS = tuple(f"2026-01-{day:02d}" for day in range(5, 10))
BLOCK_CLOSES = (110, 100, 100, 100, 100)
SOURCE_FILES = (
    "scripts/build_style_review_economic_controls.py",
    "src/behavioral_market/llm/prompts.py",
    "src/behavioral_market/llm/schemas.py",
    "src/behavioral_market/agents/state.py",
    "src/behavioral_market/agents/policy.py",
    "src/behavioral_market/data/features.py",
    "src/behavioral_market/market/portfolio.py",
    "src/behavioral_market/market/money.py",
    "src/behavioral_market/simulation/event_runner.py",
)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class ZeroNoise:
    def gauss(self, mean, sigma):
        return 0.0


def bar(session, opening, close):
    return {
        "symbol": "SYNTH",
        "session_date": session,
        "open": opening,
        "high": max(opening, close),
        "low": min(opening, close),
        "close": close,
        "volume": 1000,
    }


def make_case(style, condition):
    if style not in ("fundamental", "technical") or condition not in ("positive", "negative"):
        raise ValueError("Use the fixed two styles and two economic conditions")
    agent_id = "A000" if style == "fundamental" else "A001"
    tail = [100, 110, 100, 100, 100, 100]
    if style == "fundamental":
        tail = [90, 110, 100, 100, 100, 100] if condition == "positive" else [100] * 6
    # Weekday labels belong to a constructed scenario, not an exchange calendar/data archive.
    warmup_dates = pd.bdate_range(end="2026-01-02", periods=20).strftime("%Y-%m-%d")
    warmup = [100] * 14 + tail
    history = pd.DataFrame(
        [bar(day, price, price) for day, price in zip(warmup_dates, warmup, strict=True)]
    )
    portfolio = Portfolio(1000, 2, 100)
    portfolio.mark_to_market(100)
    agent = AgentState(
        AgentPersona(agent_id, 0.5, 0.5, 0.5, 0.5, 0.5, style),
        portfolio,
        style,
        initial_equity=1200,
        block_start_equity=1200,
        peak_equity=1200,
    )
    alternative = "technical" if style == "fundamental" else "fundamental"
    anchors = [110, 110, 100, 100, 100] if condition == "positive" else [100, 121, 100, 100, 100]
    trace = []
    for i, (session, close) in enumerate(zip(SESSIONS, BLOCK_CLOSES, strict=True)):
        previous = float(history.close.iloc[-1])
        anchor = previous if style == "fundamental" else anchors[i]
        observation = {
            **features(history),
            "fundamental_value": anchor,
            "session_date": session,
            "history_last_session": history.session_date.iloc[-1],
        }
        decision = rule_order(agent, observation, ZeroNoise())
        if decision.action != "hold":
            raise ValueError("Economic control must derive five current-style HOLD decisions")
        before = float(portfolio.total_equity)
        alternative_signal = signal(alternative, observation)
        exposure = 1.0 if alternative_signal > 0 else 0.0
        # Same frictionless long/cash update as the frozen production event_runner.
        agent.shadow_growth *= 1 + exposure * (close / previous - 1)
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
                "history_last_session": observation["history_last_session"],
                "start_reference": previous,
                "close": close,
                "fundamental_anchor": anchor,
                "momentum_5": observation["momentum_5"],
                "current_signal": signal(style, observation),
                "current_decision": decision.model_dump(),
                "current_equity": equity,
                "alternative_signal": alternative_signal,
                "alternative_exposure": exposure,
                "alternative_growth": agent.shadow_growth,
            }
        )
        history = pd.concat(
            [history, pd.DataFrame([bar(session, previous, close)])], ignore_index=True
        )
    review = {
        **features(history),
        "fundamental_value": trace[-1]["fundamental_anchor"],
        "fundamental_method": "assigned_fictional_daily_anchor_not_true_firm_value",
        "data_origin": "constructed_economic_control_not_historical_market_data",
        "session_date": SESSIONS[-1],
        "history_last_session": SESSIONS[-1],
        "decision_at_utc": "2026-01-09T21:00:00+00:00",
        "phase": "post_close_style_review",
        "primary_symbol": "SYNTH",
        "visible_events": [],
        "context_returns": {},
        "review_block": {
            "start_session": SESSIONS[0],
            "end_session": SESSIONS[-1],
            "completed_sessions": 5,
        },
        "order_book": {
            "scope": "post_close_after_day_order_expiry",
            "best_bid": None,
            "best_ask": None,
            "last_trade": 100,
        },
        "completed_prices": history[["session_date", "close"]].to_dict(orient="records"),
        "counterfactual_evidence": {
            "method": "existing_signal_positive_long_else_cash_proxy",
            "alternative_style": alternative,
            "not_executed": True,
            "limitations": "Frictionless all-wealth proxy; no execution feasibility, fees, "
            "slippage or guaranteed future advantage is established.",
            "daily_trace": [
                {
                    k: row[k]
                    for k in (
                        "session_date",
                        "start_reference",
                        "close",
                        "fundamental_anchor",
                        "momentum_5",
                        "alternative_signal",
                        "alternative_exposure",
                        "alternative_growth",
                    )
                }
                for row in trace
            ],
        },
    }
    system, user_text = prompts(
        agent, review, "vanilla_llm", "style", 17, mean_field_visibility="hidden"
    )
    user = json.loads(user_text)
    user["decision_instruction"] += " " + CLARIFICATION
    payload = {
        "model": "DeepSeek-V4-Flash",
        "temperature": 0.7,
        "max_tokens": 1200,
        "enable_thinking": False,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        "cache": {"no-cache": True, "no-store": True},
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical(user)},
        ],
    }
    return {
        "case_id": f"{agent_id}.{condition}",
        "original_style": style,
        "economic_condition": condition,
        "treatment": "vanilla_llm",
        "mean_field_visibility": "hidden",
        "status": "unsubmitted",
        "payload": payload,
        "planned_payload_sha256": digest(payload),
        "construction_not_sent": {
            "warmup_closes": warmup,
            "daily_trace": trace,
            "final_state": agent.context(),
            "rule_style_reference": rule_style(agent).model_dump(),
            "rule_reference_is_not_an_LLM_answer_key": True,
        },
    }


def build_materials():
    sources = {}
    for name in SOURCE_FILES:
        data = (REPO / name).read_bytes()
        sources[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {
        "protocol": PROTOCOL,
        "base_prompt_version": PROMPT_VERSION,
        "evidence_kind": "fictional_unsubmitted_economic_controls",
        "status": "materials_only",
        "clarification": CLARIFICATION,
        "source_fingerprints": sources,
        "cases": [
            make_case(style, condition)
            for style in ("fundamental", "technical")
            for condition in ("positive", "negative")
        ],
        "scheduled_opportunities": 0,
        "actual_remote_calls": 0,
        "budget": {"used": 133, "cap": 133, "remaining": 0},
        "analysis_plan": "Within each fixed original style, describe response to the complete "
        "economic-evidence bundle. Warmup or assigned anchors and derived features/trace differ; "
        "this is not a gap-only causal intervention. Rule switch/stay is a deterministic local "
        "sanity reference, never a mandatory correct LLM answer. Historical proxy advantage "
        "does not guarantee future returns. No generation, human or market-validity claim.",
        "historical_study": "The completed18 majority experiment, original prompts and results "
        "remain unchanged. This clarification is only in this distinct future material version.",
    }


def write_materials(destination):
    package = build_materials()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "materials.json").write_text(
        json.dumps(package, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    (destination / "说明.txt").write_text(
        "这是下一版的4份假设材料，尚未发送，不是新的模型结果。\n"
        "所有例子先涨再回到100，本人策略按现有规则得到五次不下单，期末资产仍1200。"
        "替代策略的收益由现有信号和逐期价格计算，分别约+10%或-9.09%，不是手填收益。\n"
        "两种原策略各比较正负经济证据；都隐藏群体汇总，并明确选策略本身不下单、成交或自动收费。\n"
        "为计算这两个方向，先前行情或逐期假设估值也相应改变；不能说只改了收益一个数字。"
        "历史表现较好不保证以后更好，也不规定模型必须切换。\n"
        "原18次结果全部保留，现有生产提示未改。这4份只有材料，没有执行顺序或新增调用额度。\n",
        encoding="utf-8",
    )
    manifest = {}
    for path in sorted(destination.iterdir()):
        data = path.read_bytes()
        manifest[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_materials(args.output)
    print("Prepared four coherent future controls; no requests sent.")


if __name__ == "__main__":
    main()
