#!/usr/bin/env python3
"""Audit exactly the twenty predeclared archived evidence files, with no model client."""

import argparse
import json
from pathlib import Path

from behavioral_market.evaluation.archive_response_reuse import audit_archives
from behavioral_market.evaluation.style_probe_intake import strict_json
from scripts.evaluate_style_share_probe import identity, read_jsonl
from scripts.style_probe_mode_v2 import save

BASE = "behavioral-llm-market/outputs/meta_compute_2026/aigc_relay/historical"
MAPPING = "iterations/style_switching_20261022/results/reviews.jsonl"
RECEIPT = "iterations/style_switching_20261022/results/source_receipt.json"


def validate_scope(plan):
    folders = {
        f"{BASE}/{group}/n30_seed{seed}_shock0_e21"
        for group in ("vanilla_llm", "low_herding", "high_herding")
        for seed in (42, 43, 44)
    }
    expected = {
        "response_input_files": {folder + "/llm_responses.jsonl" for folder in folders},
        "auxiliary_request_input_files": {folder + "/llm_requests.jsonl" for folder in folders},
        "auxiliary_matched_decision_mapping_files": {MAPPING, RECEIPT},
    }
    if any(set(plan.get(key, {})) != paths for key, paths in expected.items()):
        raise ValueError(
            "Require exactly nine response logs,nine request logs,two existing mappings"
        )
    return {path: value for key in expected for path, value in plan[key].items()}


def load_inputs(workspace, plan_path):
    plan_fingerprint = identity(plan_path)
    plan = strict_json(plan_path.read_text())
    scope = validate_scope(plan)
    inputs = {str(plan_path.resolve()): plan_fingerprint}
    for relative, expected in scope.items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace) or identity(path) != expected:
            raise ValueError("Declared archive leaves workspace or its fingerprint changed")
        inputs[str(path)] = expected
    batches = []
    for path in sorted(plan["response_input_files"]):
        run_id = str(Path(path).parent)
        batches.append(
            {
                "run_id": run_id,
                "requests": read_jsonl(workspace / run_id / "llm_requests.jsonl"),
                "responses": read_jsonl(workspace / path),
            }
        )
    receipt = strict_json((workspace / RECEIPT).read_text())
    for section in ("response_input_files", "auxiliary_request_input_files"):
        for path, expected in plan[section].items():
            if receipt["inputs"].get(path) != expected:
                raise ValueError(
                    "Mapping provenance differs from the declared response/request logs"
                )
    return batches, read_jsonl(workspace / MAPPING), inputs


def readable_report(result):
    s = result["summary"]
    lines = ["# 旧模拟记录：有没有重复返回\n"]
    lines.append(
        f"检查了 {s['run_count']} 次旧 AI 运行：{s['request_log_rows']:,} 条请求记录、"
        f"{s['response_log_rows']:,} 条响应记录。没有发送新请求。\n"
    )
    lines.append(
        f"可见的响应中有 {s['distinct_present_provider_ids']:,} 个不同响应编号，"
        f"成功响应未带编号 {s['http200_rows_without_provider_id']} 条；"
        f"重复编号组 {s['repeated_provider_id_groups']} 组。\n"
    )
    if s["same_input_opportunity_groups_with_multiple_noncached_successes"] == 0:
        lines.append(
            "**这批记录没有相同提问对应多个不同请求的成功复测机会。"
            "缺少这种对照，不能验证相同输入的独立重复生成，也不能据此排除返回复用。**\n"
        )
    else:
        lines.append(
            f"同一提问对应多个非本地缓存成功请求的组数："
            f"{s['same_input_opportunity_groups_with_multiple_noncached_successes']}。"
            "这只是可比较机会数，不是独立生成的证明。\n"
        )
    lines.append(
        f"可唯一对应的提问内容共有 {s['distinct_uniquely_linked_payloads']:,} 种；"
        f"有 {s['response_rows_with_unknown_payload']} 条响应无法确定对应提问。"
        "请求编号还包含本地记录信息，不能直接拿它当提问内容。\n"
    )
    labels = {"vanilla_llm": "普通提示", "low_herding": "低跟风提示", "high_herding": "高跟风提示"}
    lines.extend(
        ["| 设置 | 请求记录 | 响应记录 | 格式修复请求 | 重试响应 |", "|---|---|---|---|---|"]
    )
    for name, label in labels.items():
        rows = [row for row in s["per_run"] if f"/{name}/" in row["run_id"]]
        values = [
            sum(row[key] for row in rows)
            for key in (
                "request_rows",
                "response_rows",
                "format_repair_request_rows",
                "retry_response_rows",
            )
        ]
        lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    lines.append(
        f"\n其中 {s['format_repair_request_rows']} 条是格式修复请求，"
        f"{s['retry_response_rows']} 条是失败后的重试响应。"
        "这些额外记录不直接算作更多投资者，也不能一概叫作返回复用。\n"
    )
    counts = s["decision_mapping_status_counts"]
    lines.append(
        f"策略选择映射共保留 {s['review_mapping_rows']} 条："
        f"匹配到原回答 {counts.get('matched', 0)} 条，"
        f"另列程序规则 {counts.get('excluded_programmed_rule', 0)} 条。"
        f"匹配选择中涉及重复响应编号 {s['matched_style_choices_with_repeated_provider_id']} 条。\n"
    )
    others = {
        key: count
        for key, count in counts.items()
        if key not in {"matched", "excluded_programmed_rule"}
    }
    if others:
        lines.append("仍无法完整核对的映射：" + json.dumps(others, ensure_ascii=False) + "。\n")
    lines.extend(
        [
            f"完全相同的回答文字出现多次的组数为 {s['repeated_answer_text_groups']}；"
            "文字相同本身不证明缓存，文字或编号不同也不证明生成互相独立。",
            "接口成功返回不代表订单已成交。本轮只核对既有策略选择的映射，不把订单响应当作实际成交或已发表订单。",
            "这只是限定旧记录的回头检查，不改写旧统计，也不推广到未检查的实验；"
            "没有证明跟风规律、真人相似度或市场真实性。",
            "文件指纹只验证本轮读取前后的一致性，不是最初生成以来的外部认证。\n",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior archived audits")
    batches, reviews, inputs = load_inputs(args.workspace.resolve(), args.plan)
    result = audit_archives(batches, reviews)
    for path, expected in inputs.items():
        if identity(Path(path)) != expected:
            raise ValueError("Input changed during archived reuse audit")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", result["summary"])
    save(
        args.output / "repeat_groups.json",
        {key: value for key, value in result.items() if key.startswith("repeated_")},
    )
    save(args.output / "input_receipt.json", {"inputs": inputs, "new_model_calls": 0})
    for name, key in (
        ("response_index.jsonl", "responses"),
        ("style_binding_index.jsonl", "decisions"),
    ):
        (args.output / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result[key]
            )
        )
    (args.output / "REPORT.md").write_text(readable_report(result))
    print(
        json.dumps(
            {key: value for key, value in result["summary"].items() if key != "per_run"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
