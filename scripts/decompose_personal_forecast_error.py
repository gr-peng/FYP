"""Read only sealed E forecasts and decompose their errors; no refitting or API."""

import argparse
import io
import json
import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import personal_error_decomposition as model
from experiments.market_quote_baselines import metrics
from scripts.validate_memory_information import save, sha, strict_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "iterations/memory_training_validation_20261107"
PROTOCOL = "personal-mean-and-within-person-error-decomposition-v2"
PREDICTION = "results_v1/predictions.jsonl"
SOURCE_SEAL = "fec73187f993abdf6e349e5d87c6d4664139be8cc7ba935c8c7315ded750c520"
EXPECTED = {"records": 63143, "sessions": 51, "recorded_codes": 948}
METADATA = {
    "old_summary.json": "results_v1/summary.json",
    "old_folds.json": "results_v1/folds.json",
    "accepted_E.json": "accepted_summary.json",
    "E_review.json": "independent_actual_review_v1.json",
}
require = model.require


def read(path):
    return strict_json(path.read_bytes())


def sources():
    return sorted(
        {str(p.relative_to(ROOT)) for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py")}
        | {
            "scripts/decompose_personal_forecast_error.py",
            "scripts/validate_memory_information.py",
            "tests/test_personal_error_decomposition.py",
            "tests/test_personal_error_execution.py",
            "tests/test_conditional_order_memory.py",
            "tests/test_replication_quote_memory.py",
            "tests/test_replication_quote_states.py",
            "pyproject.toml",
        }
    )


def prepare(packet, design_path):
    design, review = (
        read(design_path),
        read(design_path.parent / "independent_design_review_v2.json"),
    )
    require(
        design["protocol"] == PROTOCOL
        and design["new_model_POSTs"] == 0
        and review["accepted_design"] is True
        and review["mustfix"] == []
        and review["design_sha256"] == sha(design_path.read_bytes()),
        "design not accepted",
    )
    require(
        sha((SOURCE / "final_freeze.json").read_bytes()) == SOURCE_SEAL
        and design["source_seal_sha256"] == SOURCE_SEAL,
        "sealed E stage differs",
    )
    seal = read(SOURCE / "final_freeze.json")
    require(set(design["inputs"]) == {PREDICTION, *METADATA.values()}, "unexpected source scope")
    for name, expected in design["inputs"].items():
        require(seal["files"][name] == expected, "design input differs from E seal")
    inputs = {name: SOURCE / value for name, value in METADATA.items()}
    for name, relative in METADATA.items():
        require(
            sha(inputs[name].read_bytes()) == seal["files"][relative]["sha256"],
            "E metadata changed",
        )
    require(
        read(inputs["E_review.json"])["accepted_actual_results"] is True, "E results not accepted"
    )
    inputs.update(
        {
            "source_seal.json": SOURCE / "final_freeze.json",
            "design.json": design_path,
            "design_review.json": design_path.parent / "independent_design_review_v2.json",
        }
    )
    packet.mkdir(parents=True, exist_ok=False)
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
        "prediction_relative_path": PREDICTION,
        "prediction": seal["files"][PREDICTION],
        "source_seal_sha256": SOURCE_SEAL,
        "preparation_read_prediction_bytes": False,
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
        and plan["prediction_relative_path"] == PREDICTION
        and plan["source_seal_sha256"] == SOURCE_SEAL
        and set(plan["sources"]) == set(sources()),
        "execution scope changed",
    )
    for name, expected in plan["sources"].items():
        require(
            sha((ROOT / name).read_bytes()) == expected
            and sha((packet / "sources" / name).read_bytes()) == expected,
            "source changed: " + name,
        )
    for name, expected in plan["inputs"].items():
        require(sha((packet / name).read_bytes()) == expected, "frozen input changed: " + name)
    return plan


def verified_bytes(path, expected):
    size = expected["bytes"]
    require(type(size) is int and 0 < size <= 64 * 1024 * 1024, "invalid fixed file size")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == size, "unexpected source type/size"
        )
        raw = handle.read(size)
        after = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == size and all(getattr(before, k) == getattr(after, k) for k in fields),
        "source changed during read",
    )
    require(sha(raw) == expected["sha256"], "prediction hash mismatch before decoding")
    return raw


def compare(actual, expected, path="metrics"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(), path + " schema mismatch"
        )
        return sum(compare(actual[k], v, path + "." + k) for k, v in expected.items())
    require(
        type(actual) in (int, float)
        and type(expected) in (int, float)
        and np.isfinite(actual)
        and np.isfinite(expected),
        path + " invalid number",
    )
    if type(expected) is int:
        require(type(actual) is int and actual == expected, path + " count mismatch")
    else:
        require(abs(actual - expected) <= model.TOLERANCE, path + " score mismatch")
    return 1


def load_records(packet, plan):
    raw = verified_bytes(SOURCE / PREDICTION, plan["prediction"])
    records = [strict_json(line) for line in io.BytesIO(raw)]
    model.validate(records)
    folds, old = read(packet / "old_folds.json"), read(packet / "old_summary.json")
    sessions = {r["session"] for r in records}
    require(
        len(records) == EXPECTED["records"]
        and len(sessions) == EXPECTED["sessions"]
        and len({r["participant"] for r in records}) == EXPECTED["recorded_codes"]
        and sessions == set(folds["session_to_fold"])
        and all(r["fold"] == folds["session_to_fold"][r["session"]] for r in records),
        "original forecast coverage/identity/fold mismatch",
    )
    require(set(old["per_session"]) == sessions, "original score session coverage mismatch")
    components = folds["components"]
    require(
        isinstance(components, list)
        and bool(components)
        and all(isinstance(group, list) and group for group in components),
        "invalid original components",
    )
    flat = [s for group in components for s in group]
    require(
        len(flat) == len(set(flat)) and set(flat) == sessions,
        "original component coverage mismatch",
    )
    component_for = {s: i for i, group in enumerate(components) for s in group}
    person_sessions = defaultdict(set)
    for r in records:
        person_sessions[r["participant"]].add(r["session"])
    require(
        len(person_sessions) == folds["recorded_participants"]
        and sum(len(ss) > 1 for ss in person_sessions.values())
        == folds["cross_session_recorded_participant_codes"]
        and all(len({component_for[s] for s in ss}) == 1 for ss in person_sessions.values())
        and all(len({folds["session_to_fold"][s] for s in group}) == 1 for group in components),
        "recorded-person/component/fold mismatch",
    )
    scoring = [
        {
            "provenance": {k: r[k] for k in ("session", "participant", "order_id")},
            "target": {"buy": r["observed_buy"]},
        }
        for r in records
    ]
    predictions = {m: np.array([r["probabilities"][m] for r in records]) for m in model.METHODS}
    numbers = compare(metrics(scoring, predictions), old["overall"], "overall")
    for session in sorted(sessions):
        ids = [i for i, r in enumerate(records) if r["session"] == session]
        numbers += compare(
            metrics([scoring[i] for i in ids], {k: p[ids] for k, p in predictions.items()}),
            old["per_session"][session],
            "session." + session,
        )
    return (
        records,
        components,
        {
            "records": len(records),
            "original_metric_numbers": numbers,
            "prediction_sha256": plan["prediction"]["sha256"],
            "original_training_cache_read": False,
            "new_model_POSTs": 0,
        },
    )


def describe(result):
    names = dict(
        zip(
            model.METHODS,
            [
                "账户和公开信息",
                "加本轮本人历史",
                "再加更早本人历史",
                "学习资料固定比例",
                "固定各一半",
                "本轮个人历史简单比例",
                "全部历史简单比例",
            ],
            strict=True,
        )
    )
    lines = [
        "个人历史带来的误差改善，具体发生在哪里？（已保存预测的算术分解）",
        "",
        "只分析此前51场训练实验的63,143条已保存预测；没有重新训练，也没有打开原始训练记录。",
        "总误差分成两部分：预测均值与每个人平均买卖倾向的差距平方，以及个人内部逐次变动的误差。",
        "变动误差降低也可能只是少报没有依据的起伏，因此同时列出预测起伏、答案起伏和共同变化。",
        "",
    ]
    for method, v in result["overall"].items():
        lines.append(
            f"{names[method]}：总误差{v['total_error']:.6f} = "
            f"平均倾向部分{v['personal_mean_bias']:.6f} + "
            f"个人内变动部分{v['within_person_error']:.6f}。"
        )
        lines.append(
            f"  预测起伏{v['prediction_variance']:.6f}，答案起伏{v['label_variance']:.6f}，"
            f"共同变化{v['prediction_label_covariance']:.6f}。"
        )
    lines += ["", "加入历史后的差值减去加入前；负值表示该部分误差降低。"]
    for pair, value in result["contrasts"].items():
        first, second = pair.split("-minus-")
        d = value["differences"]
        lines.append(
            f"{names[first]}相对{names[second]}：总差{d['total_error']:+.6f} = "
            f"平均倾向差{d['personal_mean_bias']:+.6f} + "
            f"个人内变动差{d['within_person_error']:+.6f}。"
        )
        lines.append(
            f"  个人内变动差 = 预测起伏差{d['prediction_variance']:+.6f} − "
            f"2×共同变化差{d['prediction_label_covariance']:+.6f}；答案起伏保持相同。"
        )
        for term, label in zip(
            model.TERMS[:3], ("总差", "平均倾向差", "个人内变动差"), strict=True
        ):
            lo, hi = value["percentile_95"][term]
            lines.append(f"  {label}的描述性95%区间[{lo:+.6f}, {hi:+.6f}]。")
    p = result["population"]
    one = p["single_target"]
    constant = p["constant_labels_including_single_target"]
    multiple = p["constant_labels_multiple_targets"]
    lines += [
        "",
        f"记录中共有{p['session_person_units']}个场次内参与者单位；"
        f"仅一个时点的有{one['people']}个，占平衡评分权重{one['balanced_weight']:.2%}。",
        f"答案始终一样的有{constant['people']}个（包含单时点者），权重{constant['balanced_weight']:.2%}。"
        "仅一个时点的中心变化项为零；答案不变时，程序乱报起伏仍会有变动误差。",
        f"其中有多个时点但答案不变的有{multiple['people']}个，"
        f"占平衡评分权重{multiple['balanced_weight']:.2%}。",
        "事后观察到的个人平均答案只用于拆账，不是预测输入或可提前使用的答案。"
        "这些是旧训练范围内预测的探索性描述，不能解释心理机制或因果关系。",
        "六个区间分别描述固定预测的场次重抽，不重新训练，不是联合确认检验；区间端点不必相加。"
        "同一人使用不同代码仍无法排除。这里也不验证语言模型、是否下单、报价或整个市场真实性。",
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
        "packet not accepted",
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "status.json", {"status": "started", "new_model_POSTs": 0})
    try:
        records, components, receipt = load_records(packet, plan)
        save(output / "admission.json", receipt)
        result = model.decompose(records)
        result["contrasts"] = model.contrasts(result, components)
        old_overall = read(packet / "old_summary.json")["overall"]["methods"]
        for method, value in result["overall"].items():
            expected = old_overall[method]["brier_session_person_balanced"]
            require(
                abs(value["total_error"] - expected) <= model.TOLERANCE,
                "decomposition differs from E score",
            )
        save(output / "person_components.json", result.pop("people"))
        checked_packet(packet)
        verified_bytes(SOURCE / PREDICTION, plan["prediction"])
        result.update(
            protocol=PROTOCOL,
            new_model_POSTs=0,
            no_fitting=True,
            plan_sha256=sha((packet / "plan.json").read_bytes()),
        )
        save(output / "summary.json", result)
        (output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
        save(
            output / "status.json",
            {"status": "complete_pending_independent_audit", "new_model_POSTs": 0},
        )
        return result
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
        require(args.output is None, "separate preparation and execution")
        prepare(args.packet, args.prepare_design)
        print(json.dumps({"prepared": True, "prediction_rows_read": False, "new_model_POSTs": 0}))
    else:
        require(args.output is not None, "new output required")
        result = execute(args.packet, args.output)
        print(json.dumps({"records": result["population"]["records"], "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
