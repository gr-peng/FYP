#!/usr/bin/env python3
"""Recompute the four-case tiny probe from both raw phase packages, including missing phases."""

import argparse
import json
from pathlib import Path

from behavioral_market.evaluation.style_probe_four_case import combine_phases
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import phase_manifest
from scripts.evaluate_style_share_probe import identity, read_jsonl
from scripts.style_probe_mode_v2 import load_variant, save


def readable_report(result):
    lines = ["# 两组策略占比小检查\n"]
    if result["evidence_kind"] == "synthetic_tool_test":
        lines.append("**以下是合成工具测试，不是模型行为结果。**\n")
    complete = result["tiny_four_case_probe_complete"]
    lines.append(
        "四条小检查记录完整。\n" if complete else "**四条小检查尚未完成，缺口不算作没有作用。**\n"
    )
    lines.append(
        f"合格记录：{result['valid_compatible_choices']}/4；"
        f"可比较：{result['comparable_pairs']}/2 对；"
        f"缺失：{result['missing_scheduled_cases']}。\n"
    )
    lines.extend(
        ["| 原策略 | 自己占多数时 | 另一策略占多数时 | 最终选择是否变化 |", "|---|---|---|---|"]
    )
    labels = {"fundamental": "基本面", "technical": "技术"}

    def choice(cell):
        raw = cell["choice"]
        if (
            raw is None
            or cell["answer_status"] != "valid_linked_answer"
            or cell["mode_status"] != "no_returned_reasoning_observed"
        ):
            return "未取得合格选择"
        return (
            ("保持" if raw["decision"] == "stay" else "切换到")
            + labels[raw["target_style"]]
            + "策略"
        )

    for pair in result["pairs"]:
        change = (
            "无法比较"
            if pair["choice_changed"] is None
            else "是"
            if pair["choice_changed"]
            else "否"
        )
        lines.append(
            f"| {labels[pair['from_style']]} | {choice(pair['own_majority'])} | "
            f"{choice(pair['other_majority'])} | {change} |"
        )
    lines.append(
        "\n要看同一组在占比反转后是否改变目标，不能把总切换率或与多数一致的比例直接叫作跟风。"
        "两种原策略的账户和经历也有不同，不能把它们之间的差异单独归因于原策略。\n"
    )
    target = result["all_valid_choices_target_same_style"]
    if target is not None:
        lines.append(
            f"这四条都选择了{labels[target]}策略。这是本批记录的共同目标，不证明群体占比无作用。\n"
        )
    lines.append(
        "这只是两个固定情境、每个条件一次回答的小检查。没有证明真人行为改善、因果规律或市场真实性，"
        "也没有完成 RP 的正式行为研究。日志对应关系不是外部执行认证。\n"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--fundamental-phase", type=Path, required=True)
    parser.add_argument("--technical-phase", type=Path)
    parser.add_argument("--technical-run-id", default="style-share-v2-technical-20261022")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier summaries")
    cases, base, sources, inputs = load_variant(args.workspace.resolve(), args.materials)
    batches = []
    for folder in (args.fundamental_phase, args.technical_phase):
        if folder is None:
            kind = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
            manifest = phase_manifest(cases, args.technical_run_id, kind, pair_id="A000_2026-06-29")
            batches.append({"manifest": manifest, "requests": [], "attempts": [], "responses": []})
            continue
        paths = [
            folder / name
            for name in (
                "phase_manifest.json",
                "llm_requests.jsonl",
                "llm_attempts.jsonl",
                "llm_responses.jsonl",
            )
        ]
        inputs.update({str(p.resolve()): identity(p) for p in paths})
        batches.append(
            {
                "manifest": strict_json(paths[0].read_text()),
                **{
                    key: read_jsonl(p)
                    for key, p in zip(("requests", "attempts", "responses"), paths[1:], strict=True)
                },
            }
        )
    result = combine_phases(cases, base, sources, batches)
    expected = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
    if result["evidence_kind"] != expected:
        raise ValueError("Summary evidence mode differs from requested mode")
    for path, value in inputs.items():
        if identity(Path(path)) != value:
            raise ValueError("Input changed during raw-log summary")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", result)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    (args.output / "REPORT.md").write_text(readable_report(result))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"phases", "pairs", "cases", "limits"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
