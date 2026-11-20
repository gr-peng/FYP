"""Bounded local exploration of a saved derived probability table, with no API calls."""

import argparse
import json
import os
import shutil
import stat
from pathlib import Path

from experiments import derived_probability_admission as admission
from experiments import probability_recalibration as calibration
from experiments.market_quote_baselines import metrics
from scripts.validate_memory_information import interval, save, sha, strict_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "present-derived-buy-probability-recalibration-v1"
TABLE = "outputs/financial_probability_development_20261004/predictions.json"
MAX_BYTES = 16 * 1024 * 1024
META = {
    "preparation.json": "outputs/financial_probability_development_20261004/preparation.json",
    "analysis.json": "outputs/financial_probability_development_20261004/analysis.json",
    "completion.json": "outputs/financial_probability_development_20261004/main/metadata.json",
    "accounting.json": "outputs/financial_probability_development_20261004/accounting_audit.json",
    "partitions.json": "config/frozen/market_replication_partitions_20260922.json",
}
require = calibration.require


def read(path):
    return strict_json(path.read_bytes())


def sources():
    return sorted(
        {str(p.relative_to(ROOT)) for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py")}
        | {
            "scripts/validate_probability_recalibration.py",
            "scripts/validate_memory_information.py",
            "scripts/analyze_financial_probability.py",
            "tests/test_probability_recalibration.py",
            "tests/test_derived_probability_admission.py",
            "tests/test_probability_recalibration_execution.py",
            "tests/test_conditional_order_memory.py",
            "tests/test_replication_quote_memory.py",
            "tests/test_replication_quote_states.py",
            "pyproject.toml",
        }
    )


def prepare(packet, design_path):
    stage = design_path.parent
    design = read(design_path)
    fingerprint = read(stage / "hash_admission_receipt_v2.json")
    hash_plan = stage / "hash_admission_plan_v2.json"
    hash_review = read(stage / "independent_hash_preflight_v2.json")
    prototype_review = read(stage / "independent_prototype_review_v1.json")
    require(
        design["protocol"] == PROTOCOL
        and design["new_model_POSTs"] == 0
        and design["input_candidate"]["path"] == TABLE
        and design["provenance_admission"]["tolerance_absolute"] == admission.TOLERANCE,
        "unexpected research boundary",
    )
    require(
        fingerprint["table"] == TABLE
        and fingerprint["table_rows_decoded"] is False
        and fingerprint["new_model_POSTs"] == 0
        and 0 < fingerprint["bytes"] <= MAX_BYTES
        and fingerprint["plan_sha256"] == sha(hash_plan.read_bytes())
        and hash_review["plan_sha256"] == fingerprint["plan_sha256"]
        and hash_review["accepted_hash_only"] is True
        and hash_review["mustfix"] == [],
        "hash-only admission not accepted",
    )
    require(
        prototype_review["accepted_synthetic_prototype"] is True
        and prototype_review["mustfix"] == [],
        "prototype not accepted",
    )
    for name, expected in read(hash_plan)["metadata"].items():
        require(sha((ROOT / name).read_bytes()) == expected, "hash-stage metadata changed")
    packet.mkdir(parents=True, exist_ok=False)
    inputs = {name: ROOT / path for name, path in META.items()}
    inputs.update(
        {
            "design.json": design_path,
            "hash_admission.json": stage / "hash_admission_receipt_v2.json",
            "hash_plan.json": hash_plan,
            "hash_review.json": stage / "independent_hash_preflight_v2.json",
            "prototype_review.json": stage / "independent_prototype_review_v1.json",
        }
    )
    for name, path in inputs.items():
        shutil.copyfile(path, packet / name)
    source_hashes = {}
    for name in sources():
        target = packet / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        source_hashes[name] = sha(target.read_bytes())
    plan = {
        "protocol": PROTOCOL,
        "new_model_POSTs": 0,
        "sources": source_hashes,
        "inputs": {name: sha((packet / name).read_bytes()) for name in inputs},
        "table": TABLE,
        "table_sha256": fingerprint["sha256"],
        "table_bytes": fingerprint["bytes"],
        "preparation_decoded_probability_rows": False,
    }
    save(packet / "plan.json", plan)
    (packet / "plan.sha256").write_text(sha((packet / "plan.json").read_bytes()) + "\n")
    return plan


def checked_packet(packet):
    plan = read(packet / "plan.json")
    require(
        sha((packet / "plan.json").read_bytes()) == (packet / "plan.sha256").read_text().strip()
        and plan["protocol"] == PROTOCOL
        and plan["new_model_POSTs"] == 0
        and plan["table"] == TABLE
        and set(plan["sources"]) == set(sources()),
        "execution scope changed",
    )
    for name, expected in plan["sources"].items():
        require(
            sha((ROOT / name).read_bytes()) == expected
            and sha((packet / "sources" / name).read_bytes()) == expected,
            "frozen source changed: " + name,
        )
    for name, expected in plan["inputs"].items():
        require(sha((packet / name).read_bytes()) == expected, "frozen input changed: " + name)
    return plan


def bounded_read(path, expected_bytes):
    require(
        type(expected_bytes) is int and 0 < expected_bytes <= MAX_BYTES, "invalid size boundary"
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == expected_bytes,
            "unexpected file type or size",
        )
        raw = handle.read(expected_bytes)
        after = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == expected_bytes and all(getattr(before, k) == getattr(after, k) for k in fields),
        "source changed during bounded read",
    )
    return raw


def load_records(packet, plan):
    prep, old_analysis, completion = [
        read(packet / name) for name in ("preparation.json", "analysis.json", "completion.json")
    ]
    require(
        old_analysis["preparation_sha256"]
        == completion["preparation_sha256"]
        == sha((packet / "preparation.json").read_bytes())
        and completion["status"] == "complete"
        and completion["valid"] == completion["expected_responses"] == 3200
        and completion["provider_config"]["model"] == "DeepSeek-V4-Flash"
        and old_analysis["analysis_source_sha256"]
        == sha((packet / "sources/scripts/analyze_financial_probability.py").read_bytes()),
        "historical completion/model/source receipt mismatch",
    )
    require(
        prep["files"]["accounting_audit.json"] == sha((packet / "accounting.json").read_bytes()),
        "historical accounting receipt mismatch",
    )
    assignments = read(packet / "partitions.json")["assignments"]
    all_sessions = [s for group in assignments.values() for s in group]
    require(len(all_sessions) == len(set(all_sessions)), "overlapping original partitions")
    require(len(assignments["development"]) == 16, "unexpected development cohort")
    raw = bounded_read(ROOT / TABLE, plan["table_bytes"])
    return admission.admit(
        raw, plan["table_sha256"], prep, old_analysis, assignments["development"]
    )


def score(records, result):
    rows = calibration.scoring_records(records)
    predictions = result["predictions"]
    overall, per_session = metrics(rows, predictions), {}
    for session in sorted({r["session"] for r in records}):
        ids = [i for i, r in enumerate(records) if r["session"] == session]
        per_session[session] = metrics(
            [rows[i] for i in ids], {k: p[ids] for k, p in predictions.items()}
        )
    contrasts = {}
    for reference in ("raw", "training_constant"):
        differences = {
            s: value["methods"]["monotone"]["brier_session_person_balanced"]
            - value["methods"][reference]["brier_session_person_balanced"]
            for s, value in per_session.items()
        }
        contrasts["monotone-minus-" + reference] = {
            **interval(differences, result["components"]),
            "per_session": differences,
        }
    return {"overall": overall, "per_session": per_session, "contrasts": contrasts}


def describe(summary):
    names = {
        "raw": "旧表里原有的模型概率",
        "intercept": "只调整整体偏向买或卖的程度",
        "monotone": "同时调整整体偏向和概率幅度",
        "training_constant": "按其他场次的固定买入比例",
    }
    lines = [
        "旧概率表能否通过调整报出的概率来降低误差？（本地探索，零新增模型调用）",
        "",
        "使用现存400条派生预测，来自16场以前已使用过的开发实验。"
        "每次用其他整场关联组学习如何调整，再对没有参与调整的那组评分。",
        "只预测已经下单时买还是卖；数字越小表示概率预测误差越低，不是正确率。",
        "",
    ]
    for name, values in summary["overall"]["methods"].items():
        lines.append(f"{names[name]}：{values['brier_session_person_balanced']:.6f}")
    lines += ["", "调整后的误差减去参照误差：负数表示降低，正数表示变差。"]
    for name, value in summary["contrasts"].items():
        reference = name.removeprefix("monotone-minus-")
        lo, hi = value["percentile_95"]
        lines.append(
            f"相对{names[reference]}：{value['difference']:+.6f}；"
            f"描述性95%区间[{lo:+.6f}, {hi:+.6f}]。"
        )
    lines += [
        "",
        "现存表与原先保存的名单和全部汇总成绩一致。本轮没有重新核对原始3200条模型回答、"
        "每条真人标签，或两种选项顺序的平均过程。新文件指纹只能固定当前材料。",
        "原记录标注的是DeepSeek-V4-Flash，不是当前4.1版本。这些开发资料已使用过，"
        "因此不是全新验证，最终16场未打开。",
        "描述性区间只对固定的组外预测重抽，没有重新学习；同一人使用不同代码仍无法排除。"
        "调整在同一评分组内不反转顺序，不保证不同评分组之间的整体排序不变。",
        "结果仅限这份现存材料和这两种固定调整方式。改善不证明真人心理或记忆机制；"
        "没有改善也不证明完全没有有用信息。不能据此把修正应用于新模型或声称模拟市场更真实。",
        "",
    ]
    return "\n".join(lines)


def execute(packet, output):
    plan = checked_packet(packet)
    review = read(packet / "independent_preflight.json")
    require(
        review.get("accepted_for_execution") is True
        and review.get("mustfix") == []
        and review.get("plan_sha256") == sha((packet / "plan.json").read_bytes()),
        "independent review does not accept this executable packet",
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "status.json", {"status": "started", "new_model_POSTs": 0})
    try:
        records, receipt = load_records(packet, plan)
        groups = calibration.components(records)
        save(output / "components.json", groups)
        receipt.update(
            components_sha256_before_fit=sha((output / "components.json").read_bytes()),
            component_count=len(groups),
            new_model_POSTs=0,
        )
        save(output / "admission.json", receipt)
        result = calibration.cross_validate(records)
        require(result["components"] == groups, "grouping changed during fitting")
        save(output / "models.json", result["models"])
        saved = [
            {**row, "predictions": {k: float(p[i]) for k, p in result["predictions"].items()}}
            for i, row in enumerate(records)
        ]
        save(output / "predictions.json", saved)
        summary = score(records, result)
        summary.update(
            protocol=PROTOCOL,
            new_model_POSTs=0,
            present_derived_table_exploration=True,
            plan_sha256=sha((packet / "plan.json").read_bytes()),
        )
        checked_packet(packet)
        require(
            sha(bounded_read(ROOT / TABLE, plan["table_bytes"])) == plan["table_sha256"],
            "table changed",
        )
        save(output / "summary.json", summary)
        (output / "REPORT.txt").write_text(describe(summary), encoding="utf-8")
        save(
            output / "status.json",
            {"status": "complete_pending_independent_audit", "new_model_POSTs": 0},
        )
        return summary
    except BaseException as exc:
        for name in ("summary.json", "REPORT.txt"):
            if (output / name).exists():
                (output / name).rename(output / ("incomplete_" + name))
        save(
            output / "status.json",
            {
                "status": "incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "new_model_POSTs": 0,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--prepare-design", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_design:
        require(args.output is None, "preparation and execution are separate")
        prepare(args.packet, args.prepare_design)
        print(
            json.dumps({"prepared": True, "probability_rows_decoded": False, "new_model_POSTs": 0})
        )
    else:
        require(args.output is not None, "new output required")
        summary = execute(args.packet, args.output)
        print(json.dumps({"records": summary["overall"]["records"], "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
