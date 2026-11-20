"""Offline sensitivity explanation from exactly three accepted derived files."""

import argparse
import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from behavioral_market.evaluation.choice_format_sensitivity import analyze, require
from behavioral_market.evaluation.style_mechanism_design import canonical
from behavioral_market.evaluation.style_probe_intake import strict_json
from scripts.style_fact_check import fingerprint, save

ITERATION = "iterations/choices13k_format_sensitivity_20261022/"
ORIGIN = {
    "path": "iterations/choices13k_tradeoff_execution_20261022/next_iteration_plan.json",
    "fingerprint": {
        "bytes": 6339,
        "sha256": "442948d5065c7a6b9521559eede651fe5bdf01b20dfe824060b701f1c7d6fe5f",
    },
}
LABELS = {"plain": "无参考", "true_reference": "正确参考", "shuffled_reference": "打乱参考"}


def read_bound(workspace, descriptor):
    path = (workspace / descriptor["path"]).resolve()
    require(path.is_relative_to(workspace.resolve()), "source outside workspace")
    blob = path.read_bytes()
    actual = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    require(canonical(actual) == canonical(descriptor["fingerprint"]), "bound source bytes differ")
    return strict_json(blob.decode())


def checked_plan(workspace):
    plan = strict_json((workspace / ITERATION / "plan.json").read_text())
    require(canonical(plan["origin_plan"]) == canonical(ORIGIN), "fixed accepted origin")
    origin = read_bound(workspace, ORIGIN)
    require(
        set(plan["data_sources"]) == {"packet", "local_target", "evaluation"}
        and canonical(plan["data_sources"]) == canonical(origin["data_sources"]),
        "fixed three data sources",
    )
    require(
        set(plan["metadata_sources"])
        == {"terminal_manifest", "prepared_freeze", "independent_review", "accepted_budget"}
        and canonical(plan["metadata_sources"]) == canonical(origin["metadata_sources"]),
        "fixed metadata sources",
    )
    metadata = {
        name: read_bound(workspace, desc) for name, desc in plan["metadata_sources"].items()
    }
    for name in ("packet", "evaluation"):
        desc = plan["data_sources"][name]
        require(
            metadata["terminal_manifest"]["files"][desc["path"]] == desc["fingerprint"],
            "accepted live entry differs",
        )
    desc = plan["data_sources"]["local_target"]
    require(
        metadata["prepared_freeze"]["files"][desc["path"]] == desc["fingerprint"],
        "accepted local target entry differs",
    )
    require(metadata["independent_review"]["status"] == "accepted", "prior result not accepted")
    budget = metadata["accepted_budget"]
    require(
        type(budget["used_completion_POSTs"]) is int
        and budget["used_completion_POSTs"] == 115
        and type(budget["global_authorized_completion_POST_cap"]) is int
        and budget["global_authorized_completion_POST_cap"] == 115
        and type(budget["reserved_next_completion_POSTs"]) is int
        and budget["reserved_next_completion_POSTs"] == 0
        and budget["active_run_id"] is None
        and budget["status"].endswith("accepted"),
        "accepted terminal budget",
    )
    require(
        canonical(plan["budget"])
        == canonical(
            {"used_POSTs": 115, "cap": 115, "remaining": 0, "new_POSTs": 0, "reserved": 0}
        ),
        "zero-call scope",
    )
    require(
        plan["analysis_kind"]
        == "Post-result sensitivity diagnostic,not a replacement primary metric."
        and plan["no_new_human_format_rates"] is True
        and canonical(plan["serialization_tolerance"])
        == canonical(
            {
                "relative": 0,
                "absolute": 1e-12,
                "purpose": "Compare independently reconstructed rational results "
                "with prior JSON floats only;not a measured error bound.",
            }
        ),
        "fixed diagnostic policy",
    )
    return plan


def load_inputs(workspace):
    plan = checked_plan(workspace)
    # Only the named entries are opened. Other manifest paths are never followed.
    data = {name: read_bound(workspace, desc) for name, desc in plan["data_sources"].items()}
    return plan, data


def _percent(value):
    return f"{100 * value:g}"


def report(result):
    q = _percent(result["overall_human_B_rate"])
    lines = [
        "# 平均结果改善，显示方式下的表现尚不一致",
        "",
        "这是看过结果后做的检查，只重算已验收的24份回答，没有重新问模型，也没有改变原定平均指标。",
        "",
        f"人群总体中{q}%选择原方案B。我们没有四种显示方式各自的人群比例。",
        "",
        "| 方法 | 原B选择次数 | 合并后的原B比例 | 与总体人群的差距 |",
        "|---|---:|---:|---:|",
    ]
    for method in result["pooled_methods"]:
        lines.append(
            f"| {LABELS[method['method_id']]} | {method['original_B_count']}/8 | "
            f"{_percent(method['pooled_original_B_rate'])}% | "
            f"{_percent(method['absolute_error'])}个百分点 |"
        )
    lines.extend(["", "两项原定比较均重算吻合：", ""])
    for pair in result["primary_comparisons"]:
        lines.append(
            f"- {LABELS[pair['left_method']]}减{LABELS[pair['right_method']]}的差距差值为"
            f"{_percent(pair['absolute_error_difference'])}个百分点；负数表示正确参考在平均指标上更接近。"
        )
    lines.extend(
        [
            "",
            f"下表每格仍参照同一个总体{q}%，不是四份格式专属的人类实测结果。"
            "负数表示正确参考更接近该总体参照，正数表示更远。每种方法、每种显示方式只有两份回答。",
            "",
            "| 原方案A位置 | 原方案A字母 | 正确减无参考 | 正确减打乱参考 |",
            "|---|---|---:|---:|",
        ]
    )
    pairs = result["posthoc_format_pairs"]
    for content, letters in (("AB", "AB"), ("AB", "BA"), ("BA", "AB"), ("BA", "BA")):
        group = [r for r in pairs if r["content_order"] == content and r["letter_order"] == letters]
        a_letter = letters[content.index("A")]
        lines.append(
            f"| {'前面' if content[0] == 'A' else '后面'} | {a_letter} | "
            f"{_percent(group[0]['absolute_error_difference'])}个百分点 | "
            f"{_percent(group[1]['absolute_error_difference'])}个百分点 |"
        )
    lines.append("")
    for pair in result["posthoc_format_counts"]:
        lines.append(
            f"相对{LABELS[pair['right_method']]}，正确参考在四种显示方式中："
            f"{pair['closer']}种更接近同一总体参照、{pair['same']}种相同、{pair['farther']}种更远。"
        )
    lines.extend(
        [
            "",
            "这与原定平均指标改善并不矛盾：有的显示方式偏高、有的偏低，先合并比例再比较，偏差可以互相抵消。"
            "不能因此宣称每种显示方式都改善，也不能把这张事后表换成新的主要指标。",
            "",
            f"{result['same_input_cells']}组相同输入重复中，有{result['same_input_disagreement_cells']}组前后选择不同，全部保留。"
            "回答次数不等于独立受试者数；本轮没有重新核验HTTP或后台生成。",
            "",
            "对照RP：保留本题平均结果更接近人群的证据，同时承认显示方式敏感。"
            "仍不能推出总体真人相似度提高、个人交易记忆已学会或市场模拟更真实。",
            "",
            "本轮新增模型请求0次，累计仍为115/115次。",
            "",
        ]
    )
    return "\n".join(lines)


def _csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(workspace, output):
    require(
        output.resolve() == (workspace / ITERATION / "results").resolve(), "fixed new results path"
    )
    require(not output.exists(), "never overwrite sensitivity results")
    plan, data = load_inputs(workspace)
    result = analyze(data["packet"], data["local_target"], data["evaluation"])
    checked_plan(workspace)
    for descriptor in plan["data_sources"].values():
        require(
            fingerprint(workspace / descriptor["path"]) == descriptor["fingerprint"],
            "input changed during analysis",
        )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "format_sensitivity.json", result)
    _csv(
        output / "cells.csv",
        result["format_method_cells"],
        [
            "method_id",
            "content_order",
            "letter_order",
            "planned_records",
            "received_records",
            "original_B_count",
            "original_B_rate",
            "overall_human_reference",
            "format_specific_human_rate",
            "signed_deviation_from_overall",
            "absolute_deviation_from_overall",
            "same_input_disagreement",
        ],
    )
    _csv(
        output / "pairs.csv",
        result["posthoc_format_pairs"],
        [
            "content_order",
            "letter_order",
            "left_method",
            "right_method",
            "left_absolute_error",
            "right_absolute_error",
            "absolute_error_difference",
            "absolute_error_difference_exact",
            "relative_to_same_overall_reference",
            "format_specific_human_rate",
            "description",
        ],
    )
    (output / "REPORT.md").write_text(report(result))
    save(
        output / "source_receipt.json",
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "origin_plan": plan["origin_plan"],
            "data_sources": plan["data_sources"],
            "metadata_sources": plan["metadata_sources"],
            "plan_fingerprint": fingerprint(workspace / ITERATION / "plan.json"),
            "derived_files_decoded": 3,
            "raw_logs_or_parquet_opened": False,
            "credentials_or_HTTP": False,
            "new_model_POSTs": 0,
            "used_POSTs": 115,
            "scope": "Bound derived-source recomputation only,not new remote authentication.",
        },
    )
    return {
        "analysis_completed": True,
        "reconstructed_records": 24,
        "cells": 12,
        "format_pairs": 8,
        "prespecified_primary_reproduced": True,
        "new_POSTs": 0,
        "used_POSTs": 115,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    print(canonical(run(workspace, workspace / ITERATION / "results")))


if __name__ == "__main__":
    main()
