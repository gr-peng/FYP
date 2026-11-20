"""Check task/target compatibility from approved interfaces, without reading human rows.

A supported interface is not evidence of model performance or permission to open
labels. This command emits no accuracy score and never executes a data loader.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/research/behavior_target_support_v1.json"
# Pin the reviewed interpretation as well as its source files. A caller cannot
# change the eligible population or source references and keep a passing claim.
# Revisions require a new reviewed contract/checker version, not a new assertion.
CONTRACT_DIGEST = "8e9f51d11ce6722a66a4fa6d10a32fbf1b322ada1f368cc8d98f6bf1f82b504f"
SOURCE_ALLOWLIST = {
    "src/behavioral_market/benchmarks/schemas.py",
    "src/behavioral_market/benchmarks/choices13k.py",
    "experiments/cpc18_data.py",
    "docs/cpc18_external_protocol.md",
    "experiments/replication_quote_states.py",
    "experiments/financial_probability.py",
    "docs/market_replication_data_protocol.md",
    "docs/market_replication_training_audit_20260922.md",
    "experiments/replication_development_data.py",
    "experiments/scopic_states.py",
    "docs/financial_data_protocol.md",
    "docs/financial_training_audit_20260922.md",
    "docs/financial_instructions_audit_20260922.md",
    "docs/rp_progress_update_20261022.md",
    "src/behavioral_market/llm/schemas.py",
    "src/behavioral_market/simulation/event_runner.py",
    "config/frozen/market_replication_partitions_20260922.json",
}
TARGETS = {
    "risk_probability": ("risk_choice_b_rate", ["A", "B"]),
    "active_order_direction": ("active_order_buy", ["BUY", "SELL"]),
    "owned_asset_sale": ("owned_asset_sell_any", ["SELL_SOME", "SELL_NONE"]),
    "human_style_choice": ("fundamental_technical_controller_choice", ["fundamental", "technical"]),
    "daily_trading_action": ("unconditional_daily_action", ["BUY", "SELL", "HOLD"]),
    "full_financial_policy": ("complete_feasible_financial_policy", None),
    "human_market_realism": ("heldout_institution_matched_market_properties", None),
}
SOURCE_BINDINGS = {
    "choices13k_risk": ("human_reference", "risk_choice_b_rate", ["A", "B"]),
    "cpc18_risk": ("human_reference", "risk_choice_b_rate", ["A", "B"]),
    "replication_active_order": ("human_reference", "active_order_buy", ["BUY", "SELL"]),
    "scopic_owned_sell": ("human_reference", "owned_asset_sell_any", ["SELL_SOME", "SELL_NONE"]),
    "hybrid_style_review": (
        "simulator_controller",
        "fundamental_technical_controller_choice",
        ["fundamental", "technical"],
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def validate_contract(contract):
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    require(
        hashlib.sha256(encoded.encode()).hexdigest() == CONTRACT_DIGEST,
        "contract interpretation changed; requires a new reviewed version",
    )
    require(contract["version"] == "behavior-target-support-v1", "unsupported contract version")
    require(
        set(contract["sources"]) == SOURCE_ALLOWLIST,
        "source allowlist differs; no extra data access",
    )
    require(
        contract["evaluation_ready"] is False
        and contract["current_hybrid_human_style_label_support"]
        == "not_verified_in_accepted_interfaces",
        "unsupported human-style or evaluation claim",
    )
    require(
        contract["new_model_POSTs"] == 0
        and contract["human_rows_read"] is False
        and contract["human_rows_sent"] is False
        and contract["final_holdout_outcomes_read"] is False,
        "contract crosses access boundary",
    )
    rows = contract["targets"]
    require(
        len(rows) == len(SOURCE_BINDINGS) and {row["id"] for row in rows} == set(SOURCE_BINDINGS),
        "target identities differ",
    )
    for row in rows:
        role, target, menu = SOURCE_BINDINGS[row["id"]]
        require(
            (row["role"], row["observable_target"], row["menu"]) == (role, target, menu),
            "target binding differs; do not relabel human actions",
        )
        require(
            bool(row["source_refs"]) and set(row["source_refs"]) <= SOURCE_ALLOWLIST,
            "unsupported target source reference",
        )


def load_contract(path=CONTRACT, root=ROOT):
    contract = json.loads(path.read_text())
    validate_contract(contract)
    # Reject unauthorized paths before reading even one source byte. This does
    # not open CSVs/workbooks, invoke loaders, inspect labels or read credentials.
    for name, expected in contract["sources"].items():
        require(fingerprint(root / name) == expected, "source evidence changed: " + name)
    return contract


def assess_claim(contract, request):
    validate_contract(contract)
    require(
        isinstance(request, dict)
        and set(request)
        == {
            "source",
            "target_task",
            "conditioning",
            "input_scope",
            "mapping",
        },
        "claim requires explicit source, task, conditioning, input scope and mapping",
    )
    require(all(isinstance(value, str) for value in request.values()), "claim fields must be text")
    sources = {row["id"]: row for row in contract["targets"]}
    require(
        request["source"] in sources and request["target_task"] in TARGETS, "unknown task/source"
    )
    source = sources[request["source"]]
    target, menu = TARGETS[request["target_task"]]
    reasons = []
    if source["role"] != "human_reference":
        reasons.append("simulator_output_is_not_human_truth")
    if source["observable_target"] != target or source["menu"] != menu:
        reasons.append("no_verified_matching_observed_target")
    if request["mapping"] != "identity":
        reasons.append("action_or_hindsight_relabeling_not_supported")
    if request["conditioning"] != source["conditioning"]:
        reasons.append("eligibility_or_denominator_mismatch")
    if request["input_scope"] != "declared_source_decision_state":
        reasons.append("information_set_or_clock_bridge_not_verified")
    return {
        **request,
        "target_interface_supported": not reasons,
        "reasons": reasons,
        "observed_source_field": source["field"],
        "source_refs": source["source_refs"],
        "evaluation_ready": False,
        "performance_claim_established": False,
        "qualification": "Interface compatibility only. Actual states, choices, timing, leakage, "
        "label access and scoring still require a separately frozen evaluation.",
    }


def default_checks(contract):
    return [
        assess_claim(
            contract,
            {
                "source": source["id"],
                "target_task": task,
                "conditioning": source["conditioning"],
                "input_scope": "declared_source_decision_state",
                "mapping": "identity",
            },
        )
        for source in contract["targets"]
        for task in TARGETS
    ]


def task_card():
    return (
        "我们现有的真人资料，究竟能拿来核对什么？\n\n"
        "Choices13k、CPC18：能核对两种风险选项里，人群更常选哪一个。"
        "它们不能直接回答一个人以后怎样买卖股票，或是否选择基本面、技术策略。\n\n"
        "实验市场订单资料：能核对一个人已经决定提交订单时，是买还是卖。"
        "这批目标没有包含所有不下单的时刻，所以不能拿来计算平时有多少人选择等待。"
        "买入也不等于基本面策略，卖出也不等于技术策略。\n\n"
        "持仓卖出资料：能核对已有某只股票时，是否卖出一部分。"
        "没有卖这只股票，不代表没有交易其他股票；它也不能单独验证整个投资组合。\n\n"
        "当前模拟：模型每五天选一次基本面或技术策略，再由程序下单。"
        "现有已验收接口里，没有已核实的同一问题的真人答案。"
        "因此目前可以检查程序怎样传递策略变化，还不能给它一个像真人的策略准确率。"
        "这不是断言所有原始问卷都没有这类答案；如要改变结论，需要先找到可核对的真实字段和决策时间。\n\n"
        "即使两边都有现金、股票和历史价格，也不代表看见的信息一样。"
        "真人实验有120秒一轮和分红，当前模拟按天且没有分红；真人当时还挂着哪些委托、"
        "占用了多少钱和股票，并不完全可见。真人过去做过的事，也不等于程序自己产生的记忆。\n\n"
        "下一步应先选一个确实看得见的人类动作，固定何时做决定、能看见什么和怎样计分，"
        "再作比较。实验市场仍保留51场训练、16场开发、16场最终检验；开发资料已使用，"
        "再次使用要说明是探索。另一个持仓资料集的2/2/2场划分不能与它混在一起。\n\n"
        "这次只检查现有代码、说明和资料划分，没有打开真人原始记录，没有发送真人资料，"
        "没有打开最终检验结果，也没有增加模型调用。通过字段对应检查，不等于模型已经表现更好。\n"
    )


def generate(output, contract_path=CONTRACT, claims_path=None):
    contract = load_contract(contract_path)
    claims = json.loads(claims_path.read_text()) if claims_path else None
    require(claims is None or isinstance(claims, list), "claims must be a JSON list")
    checks = (
        [assess_claim(contract, row) for row in claims]
        if claims is not None
        else default_checks(contract)
    )
    report = {
        "protocol": contract["version"],
        "new_model_POSTs": 0,
        "contract_file": str(contract_path),
        "contract_fingerprint": fingerprint(contract_path),
        "source_evidence_verified": True,
        "checks": checks,
        "human_rows_read": False,
        "human_rows_sent": False,
        "final_holdout_outcomes_read": False,
        "evaluation_ready": False,
        "numeric_human_accuracy_produced": False,
        "scope": "This is a task-interface checker, not an audit of all existing report text or a "
        "performance evaluator. It does not run any data loader or open label files.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "target_support.json").write_text(
        json.dumps({**contract, "assessment": report}, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "REPORT.txt").write_text(task_card(), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--claims", type=Path)
    args = parser.parse_args()
    result = generate(args.output, args.contract, args.claims)
    print(
        json.dumps(
            {
                "checked_claims": len(result["checks"]),
                "new_model_POSTs": 0,
                "evaluation_ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
