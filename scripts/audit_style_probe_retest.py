#!/usr/bin/env python3
"""Post-hoc, offline response-identity audit; never estimates independent generations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.evaluation.style_probe_retest import evaluate_retest
from scripts.evaluate_style_share_probe import identity, read_jsonl
from scripts.style_probe_mode_v2 import save
from scripts.style_probe_retest import LOGS, check_inputs, load_bundle


def response_reuse_diagnostic(responses):
    """Describe returned identifiers and text; unique IDs never prove independent draws."""
    records = []
    for row in responses:
        body = row.get("response")
        if row.get("http_status") != 200 or not isinstance(body, dict):
            continue
        rid = body.get("id")
        rid = rid if isinstance(rid, str) and rid.strip() else None
        created = body.get("created")
        created = created if type(created) is int and created >= 0 else None
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else {}
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        text = text if isinstance(text, str) else None
        core = {
            "id": rid,
            "created": created,
            "model": body.get("model"),
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "role": message.get("role") if isinstance(message, dict) else None,
            "content": text,
        }
        records.append(
            {
                "planned_id": row["planned_id"],
                "case_id": row["case_id"],
                "response_id": rid,
                "created": created,
                "answer_text_sha256": payload_digest(text) if text is not None else None,
                "core_envelope_sha256": payload_digest(core),
                "full_body_sha256": payload_digest(body),
            }
        )
    id_groups = defaultdict(list)
    cases = defaultdict(list)
    for row in records:
        if row["response_id"] is not None:
            id_groups[row["response_id"]].append(row)
        cases[row["case_id"]].append(row)
    repeated = [
        {
            "response_id": rid,
            "occurrences": len(group),
            "planned_ids": [row["planned_id"] for row in group],
            "case_ids": sorted({row["case_id"] for row in group}),
            "distinct_created_values": sorted(
                {row["created"] for row in group if row["created"] is not None}
            ),
            "distinct_answer_texts": len(
                {
                    row["answer_text_sha256"]
                    for row in group
                    if row["answer_text_sha256"] is not None
                }
            ),
            "distinct_core_envelopes": len({row["core_envelope_sha256"] for row in group}),
            "distinct_full_bodies": len({row["full_body_sha256"] for row in group}),
        }
        for rid, group in sorted(id_groups.items())
        if len(group) > 1
    ]
    return {
        "analysis_status": "posthoc_response_identity_diagnostic",
        "inspected_http_200_response_records": len(records),
        "response_id_present_count": sum(row["response_id"] is not None for row in records),
        "missing_response_id_count": sum(row["response_id"] is None for row in records),
        "distinct_response_ids": len(id_groups),
        "distinct_nonmissing_answer_texts": len(
            {row["answer_text_sha256"] for row in records if row["answer_text_sha256"] is not None}
        ),
        "distinct_core_envelopes": len({row["core_envelope_sha256"] for row in records}),
        "distinct_full_bodies": len({row["full_body_sha256"] for row in records}),
        "repeated_response_identifiers_observed": bool(repeated),
        "independent_generations_established": False,
        "independent_generation_count": None,
        "provider_cache_mechanism_confirmed": False,
        "repeated_identifier_groups": repeated,
        "conditions": [
            {
                "case_id": cid,
                "response_records": len(group),
                "response_id_present_count": sum(row["response_id"] is not None for row in group),
                "distinct_response_ids": len(
                    {row["response_id"] for row in group if row["response_id"] is not None}
                ),
                "distinct_answer_texts": len(
                    {
                        row["answer_text_sha256"]
                        for row in group
                        if row["answer_text_sha256"] is not None
                    }
                ),
                "distinct_core_envelopes": len({row["core_envelope_sha256"] for row in group}),
                "distinct_full_bodies": len({row["full_body_sha256"] for row in group}),
            }
            for cid, group in sorted(cases.items())
        ],
        "records": records,
        "limits": [
            "Repeated identifiers are a returned-response reuse signal; "
            "the cache mechanism is unknown.",
            "Nullable provider metadata can change the full body "
            "while id,created,answer stay equal.",
            "Unique or absent IDs and identical answer text "
            "do not establish independent generation.",
            "No effective sample-size estimate; "
            "post-hoc diagnostic does not rewrite the frozen evaluation.",
        ],
    }


def readable_report(result, evaluation):
    lines = ["# 复测响应编号检查（执行后补充诊断）\n"]
    if evaluation["evidence_kind"] == "synthetic_tool_test":
        lines.append("**这是合成测试，不是实际模型结果。**\n")
    lines.append(
        f"原日志记录了 {evaluation['logged_post_attempts']} 次请求尝试，"
        f"{evaluation['valid_count']} 条格式和模式合格的响应。\n"
    )
    lines.append(
        f"被检查的 {result['inspected_http_200_response_records']} 条成功响应中，"
        f"有 {result['response_id_present_count']} 条带响应编号，"
        f"不同编号 {result['distinct_response_ids']} 个，"
        f"不同正式回答文字 {result['distinct_nonmissing_answer_texts']} 份，"
        f"不同完整响应内容 {result['distinct_full_bodies']} 份。\n"
    )
    if result["repeated_response_identifiers_observed"]:
        lines.append(
            "**观察到响应编号重复，不能把重复接收的记录当成多次独立生成，来证明选择稳定。**\n"
        )
    else:
        lines.append("未发现编号重复，也不能据此证明回答是独立生成的；缺编号时尤其无法判断。\n")
    lines.extend(
        [
            "| 输入 | 成功响应数 | 带编号数 | 不同编号 | 不同回答文字 | 不同完整响应 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in result["conditions"]:
        lines.append(
            f"| {row['case_id']} | {row['response_records']} | "
            f"{row['response_id_present_count']} | {row['distinct_response_ids']} | "
            f"{row['distinct_answer_texts']} | {row['distinct_full_bodies']} |"
        )
    lines.extend(
        [
            "\n编号、创建时间和回答相同，不代表完整响应每个字段都相同；服务附加字段的缺失与空值也会造成区别。",
            "这里观察的是返回复用迹象，尚不能确认发生在哪一层、采用什么缓存机制，也不估算独立生成了几次。",
            "这是看过执行记录后新增的诊断。原先冻结的统计和全部日志保持原样；"
            "本报告限制其解释，不把已收齐记录写成已经证明重复性。",
            "没有发送追加请求，也没有修改原提示来绕过可能的复用。\n",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "materials", "proposal", "bundle", "execution-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", default="style-share-retest-20261022")
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous diagnoses")
    cases, base, sources, planned, manifest, inputs = load_bundle(args)
    paths = [args.execution_dir / name for name in ("run_manifest.json", *LOGS)]
    inputs.update({str(path.resolve()): identity(path) for path in paths})
    if payload_digest(strict_json(paths[0].read_text())) != payload_digest(manifest):
        raise ValueError("Raw execution manifest differs")
    requests, attempts, responses = [read_jsonl(path) for path in paths[1:]]
    evaluation = evaluate_retest(
        cases, base, sources, planned, manifest, requests, attempts, responses
    )
    result = response_reuse_diagnostic(responses)
    check_inputs(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "diagnostic.json", result)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    (args.output / "REPORT.md").write_text(readable_report(result, evaluation))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"conditions", "records", "repeated_identifier_groups", "limits"}
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
