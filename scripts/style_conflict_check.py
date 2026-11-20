"""Prepare, evaluate or explicitly authorize the fixed four-case service-return check."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from behavioral_market.evaluation.style_conflict_package import (
    NAMESPACE,
    SELECTED,
    authorization_template,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    expected_request,
    make_package,
    require,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import strict_json

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
SOURCE_PATHS = {
    "iterations/style_mechanism_design_20261022/corrected_materials/inputs.jsonl",
    "iterations/style_mechanism_design_20261022/corrected_materials/contrasts.json",
    "iterations/style_mechanism_design_20261022/corrected_materials/design_specification.json",
    "iterations/style_mechanism_design_20261022/corrected_materials/validation.json",
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines()] if path.exists() else []


def readable(result):
    names = {"fundamental": "基本面", "technical": "技术"}
    statuses = {
        "unsubmitted": "尚未发送",
        "request_record_only": "只有请求记录",
        "awaiting_response": "等待回答",
        "local_cache_replay": "读取了本地旧回答",
        "transport_failed": "传输失败",
        "invalid_response_envelope": "回答结构不完整",
        "invalid_answer_schema": "回答格式不合格",
        "inconsistent_style": "目标策略自相矛盾",
        "model_mismatch": "返回模型不匹配",
    }
    factors = {"majority_style": "群体多数", "benchmark_favored_style": "收益排名"}
    flag_names = {
        "provider_id": "编号相同",
        "body_sha256": "完整JSON相同",
        "text_sha256": "文字相同",
        "same_id_created_text": "编号、时间和文字同时相同",
    }
    text = ["# 四个固定场景：返回结果核对", ""]
    if result["evidence_kind"] == "synthetic_tool_test":
        text += ["**这是本地工具测试，不能算模型回答。**", ""]
    if all(r["status"] == "unsubmitted" for r in result["rows"]):
        text += ["**尚无已记录的新回答；下表保留待执行的未知项。**", ""]
    text += [
        f"预定4个场景，可核对回答{result['valid_returns']}个；日志记录发送尝试{result['logged_dispatch_attempts']}次。",
        "缺失、无效或失败的回答保留未知；配对不完整时差异为空，不能补成保持或零效果。",
        "",
        "| 场景 | 收益较好 | 群体多数 | 状态 | 目标策略 |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        f, c = row["factors"], row["choice"]
        status = (
            "可核对"
            if row["compatible"]
            else (
                "返回了推理内容，按约定停止"
                if row["mode"] == "reasoning_observed"
                else statuses.get(row["status"], "回答或模式不合格")
            )
        )
        text.append(
            f"| {row['case_id']} | {names[f['benchmark_favored_style']]} | "
            f"{names[f['majority_style']]} | {status} | "
            f"{names[c['target_style']] if c else '未知'} |"
        )
    text += [
        "",
        "全部已记录的重复组：",
        "",
    ]
    groups = result["return_identity_diagnostics"]
    found = False
    for key, repeated in groups.items():
        for group in repeated:
            found = True
            text.append(f"- {flag_names[key]}：{', '.join(group)}")
    if not found:
        text.append("未记录到重复组；这仍不证明独立生成。")
    text += [
        "",
        "| 比较因素 | 两端 | 可比 | 选基本面的差异 | 两端直接重复 | 端点在其他场景重复 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["contrasts"]:
        difference = row["target_fundamental_difference"]
        flags = ", ".join(flag_names[k] for k in row["return_identity_flags"])
        elsewhere = ", ".join(flag_names[k] for k in row["endpoint_reused_elsewhere_flags"])
        text.append(
            f"| {factors[row['factor']]} | {row['positive_case']} / {row['negative_case']} | "
            f"{'是' if row['comparable'] else '否'} | "
            f"{difference if difference is not None else '未知'} | "
            f"{flags or '未记录到'} | {elsewhere or '未记录到'} |"
        )
    text += [
        "",
        "四条全选同一种策略也如实保留。仅凭四次指定输入的服务返回，不能证明有或没有追收益、跟风等心理机制。",
        "编号、正文或完整JSON相同会单独标记；文本重复不证明缓存，编号不同不证明独立生成。有复用线索的零差异尤其不能解释成没有敏感性。",
        "策略标签和参考值在这四格中固定，其余12格未安排执行。本结果不代表完整16格、四种机制、真人相似度或市场反馈验证。",
        "日志只提供本地请求和返回的对应关系，不是外部签名或真实生成过程的证明。",
    ]
    return "\n".join(text) + "\n"


def prepare(args):
    plan = strict_json(args.plan.read_text())
    inputs = plan["input_files"]
    require(set(inputs) == SOURCE_PATHS, "exact four declared source paths required")
    root = args.workspace.resolve()
    resolved = {p: (root / p).resolve() for p in inputs}
    require(
        all(p.is_relative_to(root) for p in resolved.values()), "source resolves outside workspace"
    )
    before = {p: fingerprint(path) for p, path in resolved.items()}
    require(before == inputs, "frozen source changed")
    sources = {Path(p).name: path for p, path in resolved.items()}
    bank = read_jsonl(sources["inputs.jsonl"])
    edges = strict_json(sources["contrasts.json"].read_text())
    specification = strict_json(sources["design_specification.json"].read_text())
    validation = strict_json(sources["validation.json"].read_text())
    require(
        specification["cells"] == validation["eligible_cells"] == 16
        and validation["matched_edges"] == 32,
        "source design scope differs",
    )
    package = make_package(bank, edges, plan["frozen_run_id"], plan["execution_output"])
    require(
        [r["case_id"] for r in package["cases"]] == plan["scheduled_case_ids_for_future_package"],
        "preselection changed",
    )
    after = {p: fingerprint(path) for p, path in resolved.items()}
    require(before == after, "source changed while preparing")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "package.json", package)
    (args.output / "scheduled_inputs.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in package["cases"])
    )
    save(args.output / "contrasts.json", package["contrasts"])
    (args.output / "expected_requests.jsonl").write_text(
        "".join(canonical(expected_request(r, package)) + "\n" for r in package["cases"])
    )
    save(args.output / "authorization_template.json", authorization_template(package))
    save(
        args.output / "input_receipt.json",
        {
            "before": before,
            "after": after,
            "all_unchanged": True,
            "package_sha256": package["package_sha256"],
            "selected_row_sha256": {r["case_id"]: digest(r) for r in package["cases"]},
            "new_model_calls": 0,
        },
    )
    save(
        args.output / "analysis_contract.json",
        {
            "primary_outcome": "named target_style; switch is secondary",
            "contrasts": package["contrasts"],
            "unknown_handling": "Never fill stay or zero effects; no favorable selection amon"
            "g duplicates.",
            "stop_failures": [
                "transport",
                "request/identity/payload/hash",
                "local_cache",
                "invalid/incoherent answer",
                "returned reasoning/mode",
            ],
            "provider_reuse": "diagnostic only,no retry; zero differences do not identify a"
            "bsent sensitivity",
            "current_new_authority": 0,
            "future_upper_bound_not_authorized": 4,
            "independent_generation_count": None,
        },
    )
    empty = evaluate(package, [], [], [])
    save(args.output / "pending_evaluation.json", empty)
    (args.output / "REPORT.md").write_text(readable(empty))
    request = (
        "是否允许将已验收的4个全新假设场景发送至学校既有接口 "
        "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions 的 DeepSeek-V4-Flash，"
        "最多新增4次请求，累计上限从21增至25次？发送的是假设账户、交易记忆、行情、"
        "策略基准收益、群体占比和研究说明，不含真人记录或密钥。使用既有账户，"
        "逐条发送，不重试、不修复，首次失败即停；其余12个场景不发送。"
        "四条全选同一种策略也如实报告，不补发求差异。先前21次已用完，因此需要新的具体额度。\n"
    )
    (args.output / "PROPOSED_USER_REQUEST.txt").write_text(request)
    return {
        "status": "prepared_not_authorized_not_sent",
        "selected": list(SELECTED),
        "not_scheduled": len(package["not_scheduled_case_ids"]),
        "new_model_calls": 0,
    }


def load_package(folder):
    package = strict_json((folder / "package.json").read_text())
    validate_package(package)
    require(
        canonical(read_jsonl(folder / "scheduled_inputs.jsonl")) == canonical(package["cases"]),
        "scheduled rows differ",
    )
    require(
        canonical(read_jsonl(folder / "expected_requests.jsonl"))
        == canonical([expected_request(r, package) for r in package["cases"]]),
        "expected requests differ",
    )
    require(
        canonical(strict_json((folder / "contrasts.json").read_text()))
        == canonical(package["contrasts"]),
        "contrast contract differs",
    )
    return package


async def execute(args):
    # All authorization and immutable-file checks precede credentials and network.
    package = load_package(args.package)
    authorization = strict_json(args.authorization.read_text())
    check_authorization(package, authorization)
    require(
        package["evidence_kind"] == "recorded_model_logs",
        "synthetic tool bundle cannot call live service",
    )
    output = Path(package["execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "package.json", package)
    save(output / "authorization_receipt.json", authorization)
    from dotenv import dotenv_values

    from behavioral_market.llm.client import OpenAICompatibleClient

    key = dotenv_values(args.workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        save(
            output / "execution_state.json",
            {"status": "stopped", "stop_reason": "AIGC_API_KEY missing", "new_POSTs": 0},
        )
        return {"status": "stopped", "reason": "AIGC_API_KEY missing"}
    client = OpenAICompatibleClient(
        execution_config(package), key, output, output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        await collect(package, authorization, client, output)
    finally:
        await client.close()
    result = evaluate(package, *(read_jsonl(output / n) for n in LOGS))
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(readable(result))
    return {
        "complete": result["complete"],
        "valid_returns": result["valid_returns"],
        "logged_dispatch_attempts": result["logged_dispatch_attempts"],
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
        result = evaluate(load_package(args.package), *(read_jsonl(args.logs / n) for n in LOGS))
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(readable(result))
        result = {"complete": result["complete"], "valid_returns": result["valid_returns"]}
    print(canonical(result))


if __name__ == "__main__":
    main()
