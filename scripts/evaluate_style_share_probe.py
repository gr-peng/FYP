#!/usr/bin/env python3
"""Read the fixed probe materials and optional local logs; never call a model."""

import argparse
import hashlib
import json
from pathlib import Path

from behavioral_market.evaluation.style_probe_intake import (
    evaluate_recorded_probe,
    evaluate_synthetic_probe,
    expected_request,
    make_run_manifest,
    strict_json,
)
from behavioral_market.evaluation.style_probe_materials import validate_share_probes

MATERIALS = "iterations/review_driven_20261022/probe_materials"
FROZEN_FILES = {
    "inputs.jsonl": {
        "bytes": 22584,
        "sha256": "b3be6c62b98a2f6461e01a32d913ada43dbef15acb35de2354d4e6bd85b5b3f5",
    },
    "source_receipt.json": {
        "bytes": 12861,
        "sha256": "2b2f79171dc38057f8fd88e9741a700d5165583d053124874fc8bd7bd95965e2",
    },
}


def identity(path):
    body = path.read_bytes()
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines() if line.strip()]


def read_materials(workspace):
    folder = workspace / MATERIALS
    for name, expected in FROZEN_FILES.items():
        if identity(folder / name) != expected:
            raise ValueError("Frozen four-input material set changed")
    cases = read_jsonl(folder / "inputs.jsonl")
    receipt = strict_json((folder / "source_receipt.json").read_text())
    sources = [
        {"identity": origin, "payload": receipt["source_payloads"][origin["agent_id"]]}
        for origin in receipt["source_identities"]
    ]
    validate_share_probes(cases, sources)
    return (
        cases,
        sources,
        {str(folder / name): fingerprint for name, fingerprint in FROZEN_FILES.items()},
    )


def report(summary):
    labels = {
        "unsubmitted": "未提交",
        "request_record_only": "只有请求记录，未见尝试",
        "awaiting_response": "已有尝试，尚缺响应",
        "transport_failed": "调用失败",
        "invalid_response_envelope": "响应格式不合格",
        "model_mismatch": "回答模型不匹配",
        "invalid_answer_schema": "回答内容格式不合格",
        "inconsistent_style": "选择和目标策略矛盾",
        "cache_replay_not_new_execution": "缓存重放，不算本轮新回答",
        "valid_linked_answer": "本地日志关联完整的合格回答",
    }
    lines = ["# 四条策略占比输入：回答核对\n"]
    if summary["evidence_kind"] == "synthetic_tool_test":
        lines.append("**这是合成测试，用来检查工具；不是模型实验结果。**\n")
    elif summary["execution_status"] == "not_started":
        lines.append("**尚未收到执行日志。四条输入都未提交，两对都无法比较。**\n")
    else:
        lines.append(
            "这里只确认本地请求、尝试和响应能否互相对应，不能单凭这些文件证明真实模型调用。\n"
        )
    lines.extend(
        [
            f"共保留 4 条机会、2 对材料。可比较的配对：{summary['comparable_pairs']}/2；"
            f"合格回答：{summary['valid_choice_count']}/4；错误：{summary['error_cases']}；"
            f"缺失：{summary['missing_cases']}；缓存：{summary['cache_replays']}。\n",
            "| 输入 | 当前状态 |\n|---|---|",
        ]
    )
    for row in summary["cases"]:
        lines.append(f"| {row['case_id']} | {labels[row['status']]} |")
    lines.extend(["\n| 配对 | 占比反转后是否改变选择 |", "|---|---|"])
    for pair in summary["pairs"]:
        changed = pair["choice_changed"]
        label = "无法比较" if changed is None else "是" if changed else "否"
        lines.append(f"| {pair['pair_id']} | {label} |")
    lines.append(
        "\n缺回答或回答不合格时，不补成保持策略，也不写成没有作用。"
        "即使收齐这两对，也只能逐对描述，不证明跟风规律、因果作用或更接近真人。\n"
    )
    lines.append(
        "本工具没有启动模型或市场。expected_requests.jsonl 是待发送请求的格式，"
        "不是已经发送的记录。文件指纹和日志对应关系不是外部认证，也不保证文件持续未变。\n"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path)
    parser.add_argument("--run-id", default="style-share-pilot-v1")
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier results; select a new output directory")
    cases, sources, inputs = read_materials(args.workspace.resolve())
    kind = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
    manifest = make_run_manifest(cases, args.run_id, kind)
    requests, attempts, responses = [], [], []
    if args.execution_dir is not None:
        paths = {
            name: args.execution_dir / name
            for name in (
                "run_manifest.json",
                "llm_requests.jsonl",
                "llm_attempts.jsonl",
                "llm_responses.jsonl",
            )
        }
        inputs.update({str(path.resolve()): identity(path) for path in paths.values()})
        manifest = strict_json(paths["run_manifest.json"].read_text())
        if manifest != make_run_manifest(cases, args.run_id, kind):
            raise ValueError("Execution run identifier, evidence mode or manifest differs")
        requests, attempts, responses = (
            read_jsonl(paths[name])
            for name in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
        )
    evaluator = evaluate_synthetic_probe if args.synthetic_test else evaluate_recorded_probe
    summary = evaluator(cases, sources, manifest, requests, attempts, responses)
    for path, expected in inputs.items():
        if identity(Path(path)) != expected:
            raise ValueError("Input changed during the offline evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("summary.json", summary),
        ("run_manifest.json", manifest),
        ("input_receipt.json", {"inputs": inputs, "endpoint_fingerprints_match": True}),
    ):
        (args.output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "expected_requests.jsonl").write_text(
        "".join(
            json.dumps(expected_request(c, manifest), ensure_ascii=False, allow_nan=False) + "\n"
            for c in cases
        )
    )
    (args.output / "REPORT.md").write_text(report(summary))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "execution_status",
                    "opportunities",
                    "valid_choice_count",
                    "comparable_pairs",
                    "error_cases",
                    "missing_cases",
                    "real_model_execution_verified",
                    "model_calls_performed_by_intake",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
