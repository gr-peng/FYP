"""Build fictional strategy-review materials locally; this script cannot send requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.llm.prompts import PROMPT_VERSION, prompts
from behavioral_market.llm.schemas import OrderDecision
from behavioral_market.market.portfolio import Portfolio
from behavioral_market.population.mean_field import compute_mean_field

PROTOCOL = "population-information-materials-v1"
REPO = Path(__file__).resolve().parents[1]
SESSIONS = tuple(f"2026-01-{day:02d}" for day in range(5, 10))
CONDITIONS = ("fundamental_majority", "technical_majority", "hidden")
ROUNDS = (CONDITIONS, CONDITIONS[1:] + CONDITIONS[:1], CONDITIONS[2:] + CONDITIONS[:2])
SOURCE_FILES = (
    "scripts/build_population_information_probe.py",
    "scripts/style_cache_control.py",
    "src/behavioral_market/agents/state.py",
    "src/behavioral_market/llm/prompts.py",
    "src/behavioral_market/llm/schemas.py",
    "src/behavioral_market/market/__init__.py",
    "src/behavioral_market/market/portfolio.py",
    "src/behavioral_market/market/money.py",
    "src/behavioral_market/market/exchange.py",
    "src/behavioral_market/market/matching_engine.py",
    "src/behavioral_market/market/order_book.py",
    "src/behavioral_market/market/order.py",
    "src/behavioral_market/market/trade.py",
    "src/behavioral_market/population/mean_field.py",
)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def account(agent_id, style):
    portfolio = Portfolio(1000, 2, 100)
    portfolio.mark_to_market(100)
    memory = [
        {
            "owner": agent_id,
            "session_date": session,
            "submitted_action": "hold",
            "submitted_quantity": 0,
            "executed_quantity": 0,
            "close": 100,
            "cash": 1000,
            "position": 2,
            "equity": 1200,
        }
        for session in SESSIONS
    ]
    return AgentState(
        AgentPersona(agent_id, 0.5, 0.5, 0.5, 0.5, 0.5, style),
        portfolio,
        style,
        initial_equity=1200,
        block_start_equity=1200,
        memory=memory,
        shadow_growth=1.0,
        peak_equity=1200,
    )


def population_snapshot(condition):
    """Ten fictional participants, including both fixed focal accounts, at the prior close."""
    if condition not in CONDITIONS[:2]:
        raise ValueError("Population construction requires one of the two visible conditions")
    peers = ["fundamental"] * (7 if condition == "fundamental_majority" else 1)
    peers += ["technical"] * (8 - len(peers))
    styles = ["fundamental", "technical", *peers]
    agents = [account(f"A{i:03d}", style) for i, style in enumerate(styles)]
    # These are submitted orders at the lagged snapshot, not claimed fills or executions.
    decisions = [
        OrderDecision(
            action="hold" if i < 2 else "buy" if i < 6 else "sell",
            quantity=0 if i < 2 else 1,
            limit_price=None if i < 2 else 99 if i < 6 else 101,
            rationale="Fictional submitted order; execution is not assumed.",
            sentiment="neutral",
        )
        for i in range(10)
    ]
    for agent, decision in zip(agents, decisions, strict=True):
        agent.memory = agent.memory[:-1]  # Snapshot ends Jan8; no Jan9 state or memory.
        agent.memory[-1].update(
            submitted_action=decision.action,
            submitted_quantity=decision.quantity,
            limit_price=decision.limit_price,
        )
    field = compute_mean_field(agents, decisions, as_of_session=SESSIONS[-2])
    ledger = {
        "data_origin": "constructed_fictional_population",
        "as_of_session": SESSIONS[-2],
        "n_participants": 10,
        "includes_focal_accounts": ["A000", "A001"],
        "participants": [
            {
                "agent_id": a.persona.agent_id,
                "current_style": a.current_style,
                "cash": 1000,
                "position": 2,
                "equity": 1200,
                "initial_equity": 1200,
                "memory": deepcopy(a.memory),
                "submitted_order": d.model_dump(),
            }
            for a, d in zip(agents, decisions, strict=True)
        ],
        "mean_field": field,
    }
    return field, ledger


def observation(field):
    return {
        "data_origin": "constructed_fictional_scenario_not_historical_market_data",
        "primary_symbol": "SYNTH",
        "session_date": SESSIONS[-1],
        "decision_at_utc": "2026-01-09T21:00:00+00:00",
        "history_last_session": SESSIONS[-1],
        "completed_closes": [{"session_date": s, "close": 100} for s in SESSIONS],
        "last_close": 100,
        "fundamental_value": 100,
        "fundamental_method": "fictional_scenario_anchor",
        "context_returns": {},
        "visible_events": [],
        "order_book": {
            "scope": "post_close_after_day_order_expiry",
            "best_bid": None,
            "best_ask": None,
            "last_trade": 100,
        },
        "review_block": {
            "start_session": SESSIONS[0],
            "end_session": SESSIONS[-1],
            "completed_sessions": 5,
        },
        "mean_field": deepcopy(field),
    }


def source_fingerprints():
    result = {}
    for name in SOURCE_FILES:
        data = (REPO / name).read_bytes()
        result[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return result


def build_package():
    fields, ledgers = {}, {}
    for condition in CONDITIONS[:2]:
        fields[condition], ledgers[condition] = population_snapshot(condition)
    cases = []
    for agent_id, style in (("A000", "fundamental"), ("A001", "technical")):
        agent = account(agent_id, style)
        for condition in CONDITIONS:
            visibility = "hidden" if condition == "hidden" else "visible"
            field = fields[CONDITIONS[0] if condition == "hidden" else condition]
            system, user = prompts(
                agent,
                observation(field),
                "vanilla_llm",
                "style",
                17,
                mean_field_visibility=visibility,
            )
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
                    {"role": "user", "content": user},
                ],
            }
            cases.append(
                {
                    "case_id": f"{agent_id}.{condition}",
                    "agent_id": agent_id,
                    "from_style": style,
                    "condition": condition,
                    "treatment": "vanilla_llm",
                    "mean_field_visibility": visibility,
                    "payload": payload,
                    "planned_payload_sha256": digest(payload),
                    "status": "unsubmitted",
                }
            )
    by_id = {c["case_id"]: c for c in cases}
    schedule = []
    for round_index, conditions in enumerate(ROUNDS, start=1):
        account_order = ("A001", "A000") if round_index == 2 else ("A000", "A001")
        for local_position, condition in enumerate(conditions, start=1):
            for agent_id in account_order:
                case_id = f"{agent_id}.{condition}"
                schedule.append(
                    {
                        "position": len(schedule) + 1,
                        "slot_id": f"{PROTOCOL}.{agent_id}.r{round_index}.{condition}",
                        "case_id": case_id,
                        "round": round_index,
                        "within_account_round_position": local_position,
                        "planned_payload_sha256": by_id[case_id]["planned_payload_sha256"],
                        "status": "unsubmitted",
                    }
                )
    return {
        "protocol": PROTOCOL,
        "status": "materials_only_not_execution_ready",
        "evidence_kind": "fictional_unsubmitted_materials",
        "endpoint_proposal": "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
        "prompt_version": PROMPT_VERSION,
        "treatment": "vanilla_llm",
        "unique_payload_count": 6,
        "planned_opportunities": 18,
        "source_fingerprints": source_fingerprints(),
        "construction_ledgers_not_sent": ledgers,
        "cases": cases,
        "schedule": schedule,
        "analysis_plan": {
            "primary": "Within each account, fraction targeting fundamental under F-majority "
            "minus fraction targeting fundamental under T-majority.",
            "secondary": "Compare each visible condition with hidden within the same account; "
            "report stay/switch separately from target style.",
            "denominators": "Show planned, dispatched, valid, invalid and unsubmitted counts "
            "per cell. Fractions use valid coherent answers only; zero valid means unknown.",
            "application": "Returned targets are hypothetical choices; this package does not "
            "apply them to accounts or produce orders, fills or market prices.",
            "retain": "Keep constant targets and failures; no outcome-driven rewrites, retries, "
            "repairs, replacements or reordering.",
            "limits": "Three repeats per cell are response observations, not human participants "
            "or proved independent generations. Do not pool account contexts as a causal style "
            "effect. Rationales do not prove a mechanism. Requested mode is not verified mode; "
            "joint no-cache/no-store controls do not prove cache avoidance or fresh generation.",
        },
        "execution_gate": {
            "actual_remote_calls_in_this_package": 0,
            "used_actual_remote_calls": 115,
            "authorized_cap": 115,
            "remaining": 0,
            "executor_validated": False,
            "requires": "Reviewed style-specific executor, frozen full request identities and "
            "fresh authorization for at most18 additional calls (total cap133).",
            "future_policy": "Sequential, no local cache replay, no retries or repairs; retain "
            "all replies and stop on the first execution/provenance/format/mode failure.",
        },
    }


def validate_package(package):
    if canonical(package) != canonical(build_package()):
        raise ValueError("Materials differ from the fixed fictional design or current sources")


def write_package(destination):
    package = build_package()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "materials.json").write_text(
        json.dumps(package, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "说明.txt").write_text(
        "这是尚未发送的假设材料，不是模型实验结果。\n"
        "两个模拟账户原先分别使用基本面、技术策略。账户各有1000现金、2股、总资产1200；"
        "五天价格均为100，全程没有成交，实际和替代策略收益均为0。\n"
        "每个账户比较三种信息：十人中8人选基本面、8人选技术、隐藏群体汇总。"
        "每种预留3次回答，共18次；仅6份不同的请求正文。\n"
        "群体包含这两个账户；两本人不下单，其余固定4份买单价99、4份卖单价101，"
        "不交叉、无成交。"
        "翻转多数策略只改变两个策略占比，不改变本人记忆、收益或买卖方向。\n"
        "主要看每个账户最终想选哪种策略。不能把stay直接当成基本面；"
        "技术账户stay表示继续技术策略。每个账户分别比较，回答全相同也照实保留。\n"
        "当前累计115次额度已用完。本工具没有联网、读取密钥或发送功能；"
        "执行器尚待适配验收。真实发送前须获得新额度。\n",
        encoding="utf-8",
    )
    manifest = {}
    for path in sorted(destination.iterdir()):
        data = path.read_bytes()
        manifest[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_package(args.output)
    print("Prepared 6 fictional payloads and18 unsubmitted opportunities; no requests sent.")


if __name__ == "__main__":
    main()
