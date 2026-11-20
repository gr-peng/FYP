"""Run a hand-checkable fictional matched-method comparison, with no model I/O."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from behavioral_market.benchmarks.presentation_comparison import compare_panel, make_panel


def fictional_inputs():
    items = {
        101: (
            "- 100% chance of receiving 10",
            "- 50% chance of receiving 0\n- 50% chance of receiving 20",
        ),
        202: ("- 100% chance of receiving 5", "- fixed but undisclosed probabilities: -5, 15"),
    }
    plan = make_panel(items, ["true_reference", "shuffled_reference"], 2)
    targets, records, counters = {101: 0.8, 202: 0.2}, [], {}
    for opportunity in plan["opportunities"]:
        item, method = opportunity["item_id"], opportunity["method_id"]
        k = item, method
        index = counters.get(k, 0)
        counters[k] = index + 1
        count = (6 if item == 101 else 2) if method == "true_reference" else 4
        original = "B" if index < count else "A"
        choice = next(
            label
            for label, info in opportunity["presentation"]["mapping"].items()
            if info["original_option"] == original
        )
        records.append({**deepcopy(opportunity), "status": "valid", "choice": choice})
    return plan, records, targets


def describe(complete, incomplete):
    a, b = complete["primary"], incomplete["primary"]
    return "\n".join(
        [
            "# 虚构演示：公平比较与缺失回答",
            "",
            "**这是预设的假题、假目标和假回答，不是模型实验。两个方法名只是接口名称。**",
            "",
            "两题各有四种格式、每种两份回答；每方法每题8份，共32个计划机会。",
            "预设目标为80%和20%；左方法的原B比例为75%和25%，右方法均为50%。",
            "",
            "| 演示 | 计划题数 | 完整成对题数 | 左方法平均比例差 | 右方法平均比例差 | 左减右 |",
            "|---|---|---|---|---|---|",
            f"| 完整假回答 | {a['planned_items']} | {a['complete_pair_items']} | "
            f"{a['left_MAE'] * 100:.0f}个百分点 | {a['right_MAE'] * 100:.0f}个百分点 | "
            f"{a['left_minus_right_MAE'] * 100:+.0f}个百分点 |",
            f"| 少1份假回答 | {b['planned_items']} | {b['complete_pair_items']} | "
            "未知 | 未知 | 未知 |",
            "",
            "缺一份回答后，计划仍是两题32个机会，没有删题或把缺失补成零。",
            "完整的局部两端仍可比较；总体方法差保留未知，不把剩余一题当完整结果。",
            "四格和固定另一因素的字母/位置比较都在逐题文件中。",
            "这些预置数字只验证工具计算，不证明真人参考有效，也不证明模型像真人。",
            "本工具没有读取真实数据、访问凭据或调用模型。",
            "",
        ]
    )


def run_demo(output):
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing demonstration")
    plan, records, targets = fictional_inputs()
    kwargs = {
        "left_method": "true_reference",
        "right_method": "shuffled_reference",
        "evidence_kind": "synthetic_tool_test",
    }
    complete = compare_panel(plan, records, targets, **kwargs)
    removed = next(
        r
        for r in records
        if r["item_id"] == 101
        and r["method_id"] == "true_reference"
        and r["content_order"] == r["letter_order"] == "AB"
        and r["repeat_id"] == 0
    )
    remaining = [r for r in records if r is not removed]
    incomplete = compare_panel(plan, remaining, targets, **kwargs)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save("panel.json", plan)
    save("fake_targets.json", [{"item_id": i, "human_b_rate": targets[i]} for i in sorted(targets)])
    for name, rows in (
        ("received_complete.jsonl", records),
        ("received_incomplete.jsonl", remaining),
    ):
        (output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
        )
    save("complete_evaluation.json", complete)
    save("incomplete_evaluation.json", incomplete)
    save(
        "demo_manifest.json",
        {
            "evidence_kind": "synthetic_tool_test",
            "preset_not_observed": True,
            "planned_records": 32,
            "complete_received_records": 32,
            "incomplete_received_records": 31,
            "new_model_POSTs": 0,
            "real_human_data_loaded": False,
            "left_method": kwargs["left_method"],
            "right_method": kwargs["right_method"],
            "difference_direction": "left_minus_right",
            "removed_identity": {
                k: removed[k]
                for k in ("item_id", "method_id", "content_order", "letter_order", "repeat_id")
            },
        },
    )
    (output / "REPORT.md").write_text(describe(complete, incomplete))
    return {
        "evidence_kind": "synthetic_tool_test",
        "new_model_POSTs": 0,
        "complete_primary": complete["primary"],
        "incomplete_primary": incomplete["primary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output), allow_nan=False))


if __name__ == "__main__":
    main()
