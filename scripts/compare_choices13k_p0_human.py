"""Offline exploratory comparison of received P0 choices and two calibration columns."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import pyarrow.parquet as pq

from behavioral_market.evaluation.style_probe_intake import strict_json

ROOT = "behavioral-llm-market/"
PRIOR = "iterations/choices13k_p0_full_calibration_reuse_20261022/"
PARQUET = ROOT + "data/processed/human_benchmarks/choices13k/calibration_items.parquet"
MANIFEST = ROOT + "data/manifests/choices13k_manifest.json"
PROMPTS = ROOT + "src/behavioral_market/benchmarks/prompts.py"
SELECTION = PRIOR + "selection.json"
EVALUATION = PRIOR + "results/evaluation.json"
PRIOR_PLAN = PRIOR + "next_iteration_plan.json"
REVIEW = PRIOR + "independent_review.json"
SOURCES = {PARQUET, MANIFEST, PROMPTS, SELECTION, EVALUATION, PRIOR_PLAN, REVIEW}
COLUMNS = ["item_id", "human_b_rate"]
SPEC = {
    "items": 200,
    "received_per_item": 10,
    "received_per_orientation": 5,
    "original_B_mapping": {"A_first": "B", "B_first": "A"},
    "retain_all_received_records": True,
    "weighting": "equal_per_item",
    "metrics": [
        "received_MAE",
        "fixed_half_MAE",
        "received_minus_fixed_half_MAE",
        "mean_signed_orientation_difference",
        "mean_absolute_orientation_difference",
        "items_with_orientation_difference",
    ],
    "orientation_difference": "B_first_minus_A_first",
    "evidence_kind": "exploratory_calibration_received_service_choices",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def check_files(workspace, plan):
    require(set(plan["input_files"]) == SOURCES, "exact source allowlist required")
    require(plan["projected_columns"] == COLUMNS, "only two declared columns allowed")
    require(plan["analysis_spec"] == SPEC, "fixed analysis specification differs")
    for relative, expected in plan["input_files"].items():
        path = (workspace / relative).resolve()
        require(path.is_relative_to(workspace.resolve()), "source outside workspace")
        require(fingerprint(path) == expected, "source fingerprint differs: " + relative)


def validate_received(selection, result, plan):
    ids = selection["selected_item_ids"]
    require(
        len(ids) == 200
        and all(type(i) is int and i >= 0 for i in ids)
        and ids == sorted(set(ids)) == plan["selected_item_ids"],
        "exact200 frozen integer IDs required",
    )
    opportunities = selection["opportunities"]
    require(opportunities == plan["planned_opportunities"], "frozen opportunities differ")
    require(len(opportunities) == 2000, "exact2000 opportunities required")
    expected, counts, repeats = {}, defaultdict(Counter), defaultdict(set)
    for op in opportunities:
        item, repeat, order = op["item_id"], op["repeat_id"], op["display_order"]
        require(
            type(item) is int
            and item in ids
            and type(repeat) is int
            and 0 <= repeat < 10
            and order in SPEC["original_B_mapping"],
            "invalid planned identity",
        )
        key = item, repeat
        require(key not in expected, "duplicate planned opportunity")
        expected[key] = order
        counts[item][order] += 1
        repeats[item].add(repeat)
    require(
        set(counts) == set(ids)
        and all(counts[i] == {"A_first": 5, "B_first": 5} for i in ids)
        and all(repeats[i] == set(range(10)) for i in ids),
        "fixed10 and5+5 denominators required",
    )
    for name, value in {
        "items": 200,
        "planned_opportunities": 2000,
        "valid_primary_returns": 2000,
        "unresolved_primary_opportunities": 0,
        "local_replay_opportunities": 0,
        "repair_request_groups": 0,
    }.items():
        require(
            type(result["summary"][name]) is int and result["summary"][name] == value,
            "accepted summary differs",
        )
    require(
        result["human_labels_loaded"] is False
        and result["heldout_opened"] is False
        and type(result["new_model_POSTs"]) is int
        and result["new_model_POSTs"] == 0,
        "accepted audit provenance differs",
    )
    rows, observed = result["rows"], set()
    require(len(rows) == 2000, "missing or extra received row")
    for row in rows:
        item, repeat = row["item_id"], row["repeat_id"]
        require(type(item) is int and type(repeat) is int, "strict received integer identity")
        key = item, repeat
        require(key in expected and key not in observed, "unknown or duplicate received identity")
        observed.add(key)
        require(row["display_order"] == expected[key], "received orientation differs")
        require(
            row["status"] == "valid_primary_return"
            and row["choice"] in ("A", "B")
            and row["issues"] == []
            and row["local_cache"] is False,
            "received choice not an accepted valid primary",
        )
    require(observed == set(expected), "received opportunity coverage differs")
    return ids, rows


def load_inputs(workspace, plan):
    """Validate every non-label input before projecting any human value."""
    check_files(workspace, plan)
    read = lambda path: strict_json((workspace / path).read_text())  # noqa: E731
    selection, result = read(SELECTION), read(EVALUATION)
    prior, review, manifest = read(PRIOR_PLAN), read(REVIEW), read(MANIFEST)
    for key, relative in (
        ("calibration_parquet", PARQUET),
        ("accepted_selection", SELECTION),
        ("accepted_result", EVALUATION),
        ("display_mapping_source", PROMPTS),
    ):
        require(
            prior[key] == {"path": relative, "fingerprint": plan["input_files"][relative]},
            "prior accepted descriptor differs",
        )
    require(prior["projected_columns"] == COLUMNS, "prior two-column scope differs")
    require(
        review["decision"] == "full200_P0_offline_audit_accepted"
        and review["primary_opportunities"] == review["valid_primary_returns"] == 2000,
        "prior audit not accepted",
    )
    require(
        manifest["splits"]["calibration_sha256"] == plan["input_files"][PARQUET]["sha256"]
        and selection["source_files"][PARQUET] == plan["input_files"][PARQUET],
        "frozen calibration split differs",
    )
    ids, rows = validate_received(selection, result, plan)
    table = pq.read_table(workspace / PARQUET, columns=COLUMNS)
    require(table.column_names == COLUMNS, "reader returned undeclared columns")
    targets = validate_targets(ids, table.to_pylist())
    check_files(workspace, plan)
    return ids, rows, targets


def validate_targets(ids, records):
    require(len(records) == 200, "exact200 human target rows required")
    targets = {}
    for row in records:
        require(set(row) == set(COLUMNS), "undeclared target column")
        item, value = row["item_id"], row["human_b_rate"]
        require(
            type(item) is int and item in ids and item not in targets, "human target ID differs"
        )
        require(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "human target must be finite numeric[0,1]",
        )
        targets[item] = value
    require(set(targets) == set(ids), "human target coverage differs")
    return targets


def original_b(row):
    return int(row["choice"] == SPEC["original_B_mapping"][row["display_order"]])


def evaluate(ids, received, targets):
    grouped, rows = defaultdict(list), []
    for row in received:
        grouped[row["item_id"]].append(row)
    for item in ids:
        group = grouped[item]
        rates = {
            order: sum(original_b(r) for r in group if r["display_order"] == order) / 5
            for order in ("A_first", "B_first")
        }
        b_count = sum(original_b(r) for r in group)
        rate, human = b_count / 10, targets[item]
        delta = rates["B_first"] - rates["A_first"]
        rows.append(
            {
                "item_id": item,
                "human_b_rate": human,
                "received_records": 10,
                "received_original_B_count": b_count,
                "received_original_B_rate": rate,
                "received_absolute_error": abs(rate - human),
                "fixed_half_absolute_error": abs(0.5 - human),
                "received_B_rate_A_first": rates["A_first"],
                "received_B_rate_B_first": rates["B_first"],
                "signed_orientation_difference": delta,
                "absolute_orientation_difference": abs(delta),
            }
        )
    model = fmean(r["received_absolute_error"] for r in rows)
    half = fmean(r["fixed_half_absolute_error"] for r in rows)
    return {
        "evidence_kind": SPEC["evidence_kind"],
        "rows": rows,
        "summary": {
            "items": 200,
            "received_records": 2000,
            "records_retained_including_repeats": True,
            "received_MAE": model,
            "fixed_half_MAE": half,
            "received_minus_fixed_half_MAE": model - half,
            "mean_signed_orientation_difference": fmean(
                r["signed_orientation_difference"] for r in rows
            ),
            "mean_absolute_orientation_difference": fmean(
                r["absolute_orientation_difference"] for r in rows
            ),
            "items_with_orientation_difference": sum(
                r["signed_orientation_difference"] != 0 for r in rows
            ),
        },
        "human_columns_projected": COLUMNS,
        "heldout_opened": False,
        "raw_logs_opened": False,
        "new_model_POSTs": 0,
        "independent_generation_count": None,
        "deduplicated_choice_probability": None,
        "limit": "Exploratory calibration received-service frequency, balanced5+5 orientations. "
        "All10 records kept. No independent-sample inference, causal position effect, "
        "human preference mechanism, prompt ranking or heldout confirmation.",
    }


def describe(result):
    s = result["summary"]
    return "\n".join(
        [
            "# 旧回答与真人选择的对照（200道校准题）",
            "",
            "平均比例差越小，表示这批实际收到的选择比例越接近真人。",
            "例如某题真人70%选B、旧返回50%选B，这题差20个百分点，再对200题取平均。",
            "",
            "| 对照内容 | 结果 |",
            "|---|---|",
            f"| 旧返回与真人的平均比例差 | {s['received_MAE'] * 100:.2f}个百分点 |",
            f"| 固定各选一半与真人的平均比例差 | {s['fixed_half_MAE'] * 100:.2f}个百分点 |",
            f"| 前者减后者，负数表示前者较小 | "
            f"{s['received_minus_fixed_half_MAE'] * 100:+.2f}个百分点 |",
            f"| 两种顺序选择原B的比例差，先逐题取绝对值再平均 | "
            f"{s['mean_absolute_orientation_difference'] * 100:.2f}个百分点 |",
            f"| 原B放前面减放后面的比例差，保留正负再平均 | "
            f"{s['mean_signed_orientation_difference'] * 100:+.2f}个百分点 |",
            f"| 两种顺序观测比例不同的题 | {s['items_with_orientation_difference']}/200道 |",
            "",
            "同一200题，每题全部10份返回保留；每种顺序各5份，按原选项还原后比较。",
            "若模型始终选显示A，混合两种顺序后原B比例也为50%，"
            "但两种顺序下为0%与100%。总比例接近真人，不足以证明学到了人的偏好。",
            "",
            "这是看过旧记录后确定的校准题探索对照，重复返回没有删除。",
            "所算频率是实际服务返回的频率，不能解释成独立生成的模型选择分布。",
            "顺序差混合了显示安排、有限返回与服务复用等因素，不能认定为纯位置因果或人的偏好机制。",
            "未重挑提示、拟合参数或做显著性检验；未读取封存测试，旧正式指标保留。新增模型请求0次。",
            "",
        ]
    )


def run(workspace, plan_path, output):
    require(not output.exists(), "output exists; refusing overwrite")
    require(
        output.resolve().is_relative_to((workspace / "iterations").resolve()), "output boundary"
    )
    before = fingerprint(plan_path)
    plan = strict_json(plan_path.read_text())
    ids, rows, targets = load_inputs(workspace, plan)
    result = evaluate(ids, rows, targets)
    require(fingerprint(plan_path) == before, "plan changed during analysis")
    result["input_files"] = plan["input_files"]
    result["plan_fingerprint"] = before
    output.mkdir(parents=True, exist_ok=False)
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(describe(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.workspace, args.plan, args.output)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
