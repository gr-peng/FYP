"""Prepare, evaluate or explicitly authorize six frozen fact-only diagnostics."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from behavioral_market.evaluation.fact_reading_probe import (
    CASE_IDS,
    NAMESPACE,
    SOURCE,
    authorization_template,
    build_materials,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    expected_request,
    make_gold,
    make_package,
    require,
    validate_materials,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import canonical
from behavioral_market.evaluation.style_probe_intake import strict_json

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines()] if path.exists() else []


def readable(result):
    names = {
        "price_reference_relation": "价格相对参考值",
        "benchmark_better_style": "较好的具名基准",
        "majority_style": "群体策略多数",
        "population_as_of_session": "群体快照期数",
        "population_size_excluding_self": "排除自身的人数",
    }
    labels = {"correct": "全对", "wrong": "有事实错误", "unknown": "未知"}
    statuses = {
        "unsubmitted": "未发送",
        "request_record_only": "只有请求记录",
        "awaiting_response": "有发送记录、缺少返回",
        "local_cache_replay": "本地缓存返回、不可评分",
        "transport_failed": "传输失败",
        "valid_schema": "格式合格",
        "invalid_schema": "内容格式不合格",
        "invalid_envelope": "返回结构不合格",
        "invalid_model_or_envelope": "模型或返回结构不合格",
    }
    repeats = {
        "provider_id": "返回编号相同",
        "body_sha256": "完整返回相同",
        "text_sha256": "正文相同",
        "same_id_created_text": "编号、时间与正文均相同",
    }
    text = ["# 六条事实读取检查", ""]
    if result["evidence_kind"] == "synthetic_tool_test":
        text += ["**本地工具测试，不能算模型回答。**", ""]
    text += [
        f"记录到{result['logged_dispatch_attempts']}次发送尝试，"
        f"格式合格{result['schema_valid_returns']}条，"
        f"符合执行约定、可评分{result['protocol_valid_returns']}条。",
        "",
        "| 场景 | 返回状态 | 读取结果 | 回答是否可评分 | 返回重复线索 |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        status = statuses[row["format_status"]]
        if row["format_status"] == "valid_schema" and not row["protocol_compatible"]:
            status += "、返回思考内容或其记录异常"
        reuse = "、".join(repeats[f] for f in row["reuse_flags"]) or "尚未记录到；不证明独立生成"
        text.append(
            f"| {row['case_id']} | {status} | {labels[row['fact_status']]} | "
            f"{'是' if row['protocol_compatible'] else '否'} | {reuse} |"
        )
    text += ["", "| 字段 | 正确 | 答错 | 未知 |", "|---|---|---|---|"]
    for field, counts in result["per_field_counts"].items():
        text.append(
            f"| {names[field]} | {counts['correct']} | {counts['wrong']} | {counts['unknown']} |"
        )
    text += ["", "全部重复组：", ""]
    any_group = False
    for field, groups in result["return_identity_diagnostics"].items():
        for group in groups:
            any_group = True
            text.append(f"- {repeats[field]}：{', '.join(group)}")
    if not any_group:
        text.append("尚无重复组记录；没有回答或没有重复，都不能证明独立生成。")
    text += [
        "",
        "格式合格但事实答错会原样保留并继续；传输、请求身份、本地缓存、格式或模式异常即停，不纠正、不重试。",
        "缺失、无效和答错分别记录。两条新参考值控制是看到原错误后构造的假设材料，不能叫盲测。",
        "期数9和人数100在六条中都相同；正确不证明一般时序或范围理解。新事实任务通过也不能追认旧理由或选择机制正确。",
        "这里只逐字段描述六种指定输入的返回，不计算通用理解准确率，不证明真人相似度或市场反馈。",
        "",
    ]
    return "\n".join(text)


def prepare(args):
    plan = strict_json(args.plan.read_text())
    require(set(plan["input_files"]) == {SOURCE}, "exact declared source required")
    root = args.workspace.resolve()
    source = (root / SOURCE).resolve()
    require(source.is_relative_to(root), "source outside workspace")
    before = fingerprint(source)
    require(before == plan["input_files"][SOURCE], "source fingerprint differs")
    base = read_jsonl(source)
    cases = build_materials(base)
    validate_materials(cases, base)
    gold = make_gold(cases)
    package = make_package(
        cases, gold, plan["frozen_run_id"], plan["fixed_execution_output"], source_rows=base
    )
    validate_package(package, gold)
    baseline = {
        cid: gold["by_case"][cid]["facts"]["price_reference_relation"] == "equal"
        for cid in CASE_IDS
    }
    require(
        sum(baseline.values()) == 4 and not baseline[CASE_IDS[4]] and not baseline[CASE_IDS[5]],
        "constant-equal controls ineffective",
    )
    after = fingerprint(source)
    require(before == after, "source changed while preparing")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "package.json", package)
    (args.output / "inputs.jsonl").write_text("".join(canonical(c) + "\n" for c in cases))
    save(args.output / "hidden_gold.json", gold)
    save(args.output / "authorization_template.json", authorization_template(package))
    (args.output / "expected_requests.jsonl").write_text(
        "".join(canonical(expected_request(c, package)) + "\n" for c in cases)
    )
    save(
        args.output / "input_receipt.json",
        {
            "source": SOURCE,
            "before": before,
            "after": after,
            "unchanged": True,
            "new_model_calls": 0,
        },
    )
    save(
        args.output / "validation.json",
        {
            "fact_cases": 6,
            "base_contexts": 4,
            "new_synthetic_controls": 2,
            "unique_payloads": len({c["payload_sha256"] for c in cases}),
            "constant_equal_relation_baseline": baseline,
            "constant_equal_correct": 4,
            "constant_equal_wrong": 2,
            "gold_digest_bound": package["gold_sha256"],
            "new_model_calls": 0,
        },
    )
    pending = evaluate(package, gold, [], [], [])
    save(args.output / "pending_evaluation.json", pending)
    (args.output / "REPORT.md").write_text(readable(pending))
    (args.output / "PROPOSED_USER_REQUEST.txt").write_text(
        "是否允许将已验收的6条假设事实读取材料发送至学校既有接口 "
        "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions 的 DeepSeek-V4-Flash？"
        "最多新增6次，累计上限从25增至31次，使用既有账户。材料是4个旧假设情境及2个仅改参考值的合成控制，"
        "发送假设账户、记忆、行情、基准收益、群体快照和只回填事实的说明；不含真人记录、密钥或核对答案。"
        "逐条发送，不重试、不修复；传输、身份、格式或模式异常即停。格式合格但事实答错会原样记录并继续剩余场景，"
        "不补发求正确答案。此前25次已用完，因此需要新的具体额度。\n"
    )
    return {"status": "prepared_not_authorized_not_sent", "cases": 6, "new_model_calls": 0}


def load_package(folder):
    package = strict_json((folder / "package.json").read_text())
    gold = strict_json((folder / "hidden_gold.json").read_text())
    validate_package(package, gold)
    require(
        canonical(read_jsonl(folder / "inputs.jsonl")) == canonical(package["cases"]),
        "input file differs",
    )
    require(
        canonical(read_jsonl(folder / "expected_requests.jsonl"))
        == canonical([expected_request(c, package) for c in package["cases"]]),
        "expected request file differs",
    )
    return package, gold


async def execute(args):
    package, gold = load_package(args.package)
    auth = strict_json(args.authorization.read_text())
    check_authorization(package, auth)
    require(
        package["evidence_kind"] == "recorded_model_logs",
        "tool-test package cannot call live service",
    )
    output = Path(package["execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "package.json", package)
    save(output / "authorization_receipt.json", auth)
    from dotenv import dotenv_values

    from behavioral_market.llm.client import OpenAICompatibleClient

    key = dotenv_values(args.workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        save(
            output / "execution_state.json",
            {"status": "stopped", "reason": "AIGC_API_KEY missing", "new_POSTs": 0},
        )
        return {"status": "stopped", "reason": "AIGC_API_KEY missing"}
    client = OpenAICompatibleClient(
        execution_config(package), key, output, output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        await collect(package, auth, client, output)
    finally:
        await client.close()
    result = evaluate(package, gold, *(read_jsonl(output / name) for name in LOGS))
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(readable(result))
    return {
        "execution_complete": result["execution_complete"],
        "logged_dispatch_attempts": result["logged_dispatch_attempts"],
        "fact_status_counts": result["fact_status_counts"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("execute")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "execute":
        result = asyncio.run(execute(args))
    else:
        package, gold = load_package(args.package)
        result = evaluate(package, gold, *(read_jsonl(args.logs / name) for name in LOGS))
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(readable(result))
        result = {"execution_complete": result["execution_complete"]}
    print(canonical(result))


if __name__ == "__main__":
    main()
