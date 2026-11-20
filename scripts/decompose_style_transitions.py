#!/usr/bin/env python3
"""Describe the fixed archived540 AI choices using only four declared artifacts."""

import argparse
import json
from pathlib import Path

from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_transitions import decompose_transitions
from scripts.evaluate_style_share_probe import identity, read_jsonl
from scripts.style_probe_mode_v2 import save

REVIEWS = "iterations/style_switching_20261022/results/reviews.jsonl"
BINDINGS = "iterations/style_archive_reuse_20261022/results/style_binding_index.jsonl"
RECEIPT = "iterations/style_archive_reuse_20261022/results/input_receipt.json"
SUMMARY = "iterations/style_archive_reuse_20261022/results/summary.json"
BASE = "behavioral-llm-market/outputs/meta_compute_2026/aigc_relay/historical"
LABELS = {"vanilla_llm": "普通提示", "low_herding": "低跟风提示", "high_herding": "高跟风提示"}


def validate_plan(plan):
    if (
        set(plan["input_files"]) != {REVIEWS, BINDINGS, RECEIPT, SUMMARY}
        or plan["review_dates"] != ["2026-06-29", "2026-07-14"]
        or plan["groups"] != list(LABELS)
        or plan["seeds"] != [42, 43, 44]
        or plan["expected_AI_runs"] != 9
        or plan["reviews_per_run_per_date"] != 30
        or plan["original_AI_choice_opportunities"] != 540
        or plan["excluded_programmed_rule_opportunities"] != 180
    ):
        raise ValueError("Use only the fixed four artifacts and the original540-choice scope")


def load_inputs(workspace, plan_path):
    fingerprint = identity(plan_path)
    plan = strict_json(plan_path.read_text())
    validate_plan(plan)
    inputs = {str(plan_path.resolve()): fingerprint}
    for relative, expected in plan["input_files"].items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace) or identity(path) != expected:
            raise ValueError("A declared transition source left the workspace or changed")
        inputs[str(path)] = expected
    receipt = strict_json((workspace / RECEIPT).read_text())
    if receipt["inputs"].get(str(workspace / REVIEWS)) != plan["input_files"][REVIEWS]:
        raise ValueError("Prior binding provenance refers to different reviews")
    audit = strict_json((workspace / SUMMARY).read_text())
    if audit["decision_mapping_status_counts"] != {"matched": 540, "excluded_programmed_rule": 180}:
        raise ValueError("Prior audited cohort differs; do not silently replace it")
    return plan, read_jsonl(workspace / REVIEWS), read_jsonl(workspace / BINDINGS), inputs


def cell_text(cell):
    if cell["confirmed_zero_origin"]:
        return "0机会"
    if cell["count"] is None:
        return f"未知（已核实{cell['observed_verified_count']}次）"
    return f"{cell['count']}/{cell['origin_opportunities']}"


def readable_report(result):
    s = result["summary"]
    group_order = {group: i for i, group in enumerate(LABELS)}
    lines = ["# 这批 AI 到底从什么策略改成什么策略\n"]
    lines.append(
        f"保留 {s['scheduled_AI_choices']} 次 AI 选择机会，核实 {s['verified_choices']} 次，"
        f"未核实 {s['unverified_choices']} 次；程序规则 {s['excluded_rule_review_rows']}/"
        f"{s['scheduled_rule_opportunities']} 次另列。"
        "这是旧记录的描述，不是新的模型或真人实验。\n"
    )
    counts = s["complete_direction_counts"]
    if counts is None:
        lines.append("**记录未齐，完整切换数与方向总数仍未知；不补成保持。**\n")
    else:
        lines.append(
            f"这批共 {s['complete_switch_count']} 次策略切换："
            f"{counts['technical_to_fundamental']} 次从技术转向基本面，"
            f"{counts['fundamental_to_technical']} 次从基本面转向技术。"
            "方向明显不对称，但切换本身不能直接叫作跟风。\n"
        )
    lines.append(
        "基本面策略主要看价格与系统参考价值的差距，技术策略主要看价格走势。"
        "下表每格为‘次数/原策略机会’，每种设置各有三次运行；0机会表示原策略不存在，"
        "不是零切换率。未知与0机会分开。\n"
    )
    header = "| 日期 | 设置 | 基本面→基本面 | 基本面→技术 | 技术→基本面 | 技术→技术 |"
    lines.extend([header, "|---|---|---|---|---|---|"])
    for context in sorted(
        result["per_group_date"],
        key=lambda c: (c["review_date"], group_order.get(c["treatment"], len(group_order))),
    ):
        cells = " | ".join(cell_text(cell) for cell in context["cells"])
        label = LABELS.get(context["treatment"], context["treatment"])
        lines.append(f"| {context['review_date']} | {label} | {cells} |")
    lines.append(
        "\n分母是各日期内该原策略的机会总数，不是把各运行比例简单平均。完整逐运行结果在下表保留。\n"
    )
    lines.append(
        f"同一运行、同一代理的两期对应共 {s['continuity_opportunities']} 对："
        f"第二期原策略与第一期目标一致 {s['continuity_consistent']} 对，"
        f"不一致 {s['continuity_mismatched']} 对，无法核对 {s['continuity_unknown']} 对。\n"
    )
    if s["all_continuity_pairs_consistent"] is True:
        lines.append(
            "两期逐代理衔接全部对应，第二期人群构成延续此前选择，不能把它当成新的独立起点。\n"
        )
    else:
        lines.append("两期衔接有不一致或未知，全部单列；不按第一期目标回填第二期原策略。\n")
    lines.extend(
        [
            "即使第一期原策略数量相同，也不代表账户、收益和消息完全相同；"
            "不能直接把组间差异归因于提示，更不能证明跟风机制或真人相似度。",
            "这些是同一批模拟代理的旧选择，不是540位独立真人，也没有证明回答独立生成。",
            "本轮没有新增模型请求、运行市场或读取保留的真人测试数据。\n",
            "| 日期 | 设置/运行编号 | 基本面→基本面 | 基本面→技术 | 技术→基本面 | 技术→技术 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for context in sorted(
        result["per_run_date"],
        key=lambda c: (
            c["review_date"],
            group_order.get(c["treatment"], len(group_order)),
            c["seed"],
        ),
    ):
        cells = " | ".join(cell_text(cell) for cell in context["cells"])
        label = LABELS.get(context["treatment"], context["treatment"])
        lines.append(f"| {context['review_date']} | {label}/{context['seed']} | {cells} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior decompositions")
    plan, reviews, bindings, inputs = load_inputs(args.workspace.resolve(), args.plan)
    runs = {
        f"{BASE}/{group}/n30_seed{seed}_shock0_e21": {"treatment": group, "seed": seed}
        for group in plan["groups"]
        for seed in plan["seeds"]
    }
    rules = [f"{BASE}/rule_abm/n30_seed{seed}_shock0_e21" for seed in plan["seeds"]]
    result = decompose_transitions(
        reviews, bindings, runs, plan["review_dates"], [f"A{i:03d}" for i in range(30)], rules
    )
    for path, expected in inputs.items():
        if identity(Path(path)) != expected:
            raise ValueError("Transition source changed during analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, key in (
        ("summary.json", "summary"),
        ("run_date_contexts.json", "per_run_date"),
        ("group_date_contexts.json", "per_group_date"),
    ):
        save(args.output / name, result[key])
    for name, key in (("choices.jsonl", "choices"), ("continuity.jsonl", "continuity")):
        (args.output / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result[key]
            )
        )
    save(args.output / "input_receipt.json", {"inputs": inputs, "new_model_calls": 0})
    (args.output / "REPORT.md").write_text(readable_report(result))
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
