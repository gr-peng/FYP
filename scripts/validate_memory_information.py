"""Training-cache-only exploratory conditional-order memory cross-validation.

Preparation never opens human states. Execution needs a frozen, independently
accepted packet and reads one already approved training cache locally. No model
transport, credentials, raw archive, development or final-test loaders are used.
"""

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import conditional_order_memory as model
from experiments.market_quote_baselines import metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "conditional-order-memory-training-cv-v1"
CACHE = "outputs/market_quote_memory_training_20261004/states.jsonl"
MANIFEST = "outputs/market_quote_memory_training_20261004/manifest.json"
PARTITIONS = "config/frozen/market_replication_partitions_20260922.json"
CACHE_SHA = "5e5c0a228d59f0e5f9a28a3a11d727d764ce5795bc6ba9bc6a39142d148be2db"
ENGINEERING_REVIEW = (
    ROOT.parent
    / "iterations/memory_information_engineering_20261107/independent_implementation_review_v2.json"
)
ENGINEERING_SHA = "7b2ac5e00d13b24b42d28f549902b1b793d77d676c7e81d234de924cf27aea77"
SALT = "memory-information-20261107:"
require = model.require


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def strict_json(data):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(
        data,
        object_pairs_hook=unique_object,
        parse_float=finite_float,
        parse_constant=invalid_constant,
    )


def source_names():
    return sorted(
        {
            str(p.relative_to(ROOT))
            for folder in ("src", "experiments")
            for p in (ROOT / folder).rglob("*.py")
        }
        | {
            "scripts/validate_memory_information.py",
            "scripts/build_market_quote_memory.py",
            "tests/test_memory_information_validation.py",
            "tests/test_conditional_order_memory.py",
            "tests/test_replication_quote_memory.py",
            "tests/test_replication_quote_states.py",
            "pyproject.toml",
        }
    )


def prepare(packet, design_path):
    design = read(design_path)
    require(
        design["protocol"] == PROTOCOL
        and design["new_model_POSTs"] == 0
        and design["input"]["only_human_state_file"] == CACHE
        and design["input"]["sha256"] == CACHE_SHA
        and design["input"]["records"] == 63143
        and design["input"]["sessions"] == 51,
        "unexpected research/data boundary",
    )
    require(
        sha(ENGINEERING_REVIEW.read_bytes()) == ENGINEERING_SHA
        and read(ENGINEERING_REVIEW)["accepted_synthetic_engineering"] is True,
        "memory adapter not accepted",
    )
    manifest, partitions = read(ROOT / MANIFEST), read(ROOT / PARTITIONS)
    require(
        manifest["states_sha256"] == CACHE_SHA
        and manifest["partition_sha256"] == sha((ROOT / PARTITIONS).read_bytes())
        and len(partitions["assignments"]["train"]) == 51,
        "training provenance differs",
    )
    for name, expected in manifest["code_hashes"].items():
        require(sha((ROOT / name).read_bytes()) == expected, "historical exporter source changed")
    require(
        sha((ROOT / "scripts/build_market_quote_memory.py").read_bytes())
        == manifest["builder_sha256"],
        "historical builder changed",
    )
    packet.mkdir(parents=True, exist_ok=False)
    for name, path in {
        "design.json": design_path,
        "training_manifest.json": ROOT / MANIFEST,
        "partitions.json": ROOT / PARTITIONS,
        "engineering_review.json": ENGINEERING_REVIEW,
        "engineering_design.json": ENGINEERING_REVIEW.parent / "design_v1.json",
    }.items():
        shutil.copyfile(path, packet / name)
    sources = {}
    for name in source_names():
        target = packet / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        sources[name] = sha(target.read_bytes())
    inputs = {
        name: sha((packet / name).read_bytes())
        for name in (
            "design.json",
            "training_manifest.json",
            "partitions.json",
            "engineering_review.json",
            "engineering_design.json",
        )
    }
    plan = {
        "protocol": PROTOCOL,
        "new_model_POSTs": 0,
        "sources": sources,
        "inputs": inputs,
        "human_state_path": CACHE,
        "human_state_sha256": CACHE_SHA,
        "preparation_opened_human_states": False,
    }
    save(packet / "plan.json", plan)
    (packet / "plan.sha256").write_text(sha((packet / "plan.json").read_bytes()) + "\n")
    return plan


def checked_packet(packet):
    require(
        sha((packet / "plan.json").read_bytes()) == (packet / "plan.sha256").read_text().strip(),
        "plan hash mismatch",
    )
    plan = read(packet / "plan.json")
    require(
        plan["protocol"] == PROTOCOL
        and plan["new_model_POSTs"] == 0
        and plan["human_state_path"] == CACHE
        and plan["human_state_sha256"] == CACHE_SHA
        and set(plan["sources"]) == set(source_names()),
        "packet scope changed",
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


def validate_clock(row):
    p, s = row["provenance"], row["state"]
    for pk, sk, low, high in [
        ("repeat", "repetition", 1, 2),
        ("period", "period", 1, 10),
        ("second", "elapsed_seconds", 0, 120),
    ]:
        require(
            type(p[pk]) is int and low <= p[pk] <= high and s[sk] == p[pk],
            "provenance/state decision clock mismatch",
        )
    periods = [r["period"] for r in s["earlier_periods"]]
    require(
        periods == sorted(set(periods)) and all(1 <= t < p["period"] for t in periods),
        "current/future public period in input",
    )
    last_trade = s["earlier_current_period_trades"]["last_completed_trade_second"]
    require(last_trade is None or 0 <= last_trade < p["second"], "nonpast trade in input")
    m = s[model.COMPLETED]
    if p["repeat"] == 1:
        require(sum(m["previous_repetitions"].values()) == 0, "history before first repetition")
    if p["period"] == 1:
        require(
            sum(m["earlier_periods_this_repetition"].values()) == 0, "history before first period"
        )
    last = m["last_completed_period_with_own_orders"]
    if last is not None:
        require(
            (last["repetition"], last["period"]) < (p["repeat"], p["period"])
            and 1 <= last["repetition"] <= 2
            and 1 <= last["period"] <= 10,
            "nonpast personal completed period",
        )
        recent_is_current = sum(m["earlier_periods_this_repetition"].values()) > 0
        require(
            (last["repetition"] == p["repeat"]) == recent_is_current,
            "last active period came from wrong repetition",
        )


def load_cache(path, expected_sha, train_sessions, expected_count):
    # Parse the very same immutable bytes that passed the source hash. The input
    # path is the sole approved training cache in the actual execution entrypoint.
    raw = path.read_bytes()
    require(sha(raw) == expected_sha, "training cache hash mismatch; no rows decoded")
    records = []
    for line in io.BytesIO(raw):
        row = strict_json(line)
        require(row["provenance"]["session"] in train_sessions, "record outside training partition")
        model.features(row["state"], "plus_completed_history_23")
        validate_clock(row)
        records.append(row)
    model.record_keys(records, labels=True)
    require(
        len(records) == expected_count
        and {r["provenance"]["session"] for r in records} == set(train_sessions),
        "training coverage differs",
    )
    return records


def grouped_folds(records):
    sessions = sorted({r["provenance"]["session"] for r in records})
    parent = {s: s for s in sessions}
    people = defaultdict(set)
    for r in records:
        people[r["provenance"]["participant"]].add(r["provenance"]["session"])

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for linked in people.values():
        first, *rest = sorted(linked)
        for s in rest:
            parent[find(s)] = find(first)
    components = defaultdict(list)
    for s in sessions:
        components[find(s)].append(s)

    def key(group):
        return group[0] if len(group) == 1 else "linked:" + json.dumps(group, separators=(",", ":"))

    groups = sorted(components.values(), key=lambda g: sha((SALT + key(g)).encode()))
    require(len(groups) >= 5, "fewer than five independent recorded-identity components")
    mapping = {s: i % 5 for i, group in enumerate(groups) for s in group}
    require(set(mapping.values()) == set(range(5)), "empty fold")
    require(
        all(len({mapping[s] for s in linked}) == 1 for linked in people.values()),
        "recorded participant crossed folds",
    )
    return {
        "session_to_fold": mapping,
        "components": groups,
        "recorded_participants": len(people),
        "cross_session_recorded_participant_codes": sum(len(s) > 1 for s in people.values()),
        "identity_limit": "Different codes may still denote one real person; "
        "no real-world identity certification.",
    }


def interval(session_differences, components, draws=10000):
    sums = np.array([sum(session_differences[s] for s in group) for group in components])
    counts = np.array([len(group) for group in components])
    rng = np.random.default_rng(20261107)
    selected = rng.integers(len(components), size=(draws, len(components)))
    values = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "difference": float(sums.sum() / counts.sum()),
        "percentile_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "component_clusters": len(components),
        "draws": draws,
        "scope": "Descriptive conditional-on-fixed-OOF-predictions interval; "
        "no refitting or fit uncertainty; not simultaneous.",
    }


def cross_validate(records, folds, output):
    keys = model.record_keys(records, labels=True)
    require(set(folds["session_to_fold"]) == {s for s, _ in keys}, "fold coverage differs")
    predictions, coverage = {}, np.zeros(len(records), dtype=int)
    for fold in range(5):
        indices = [
            i
            for i, r in enumerate(records)
            if folds["session_to_fold"][r["provenance"]["session"]] == fold
        ]
        held = set(indices)
        training = [r for i, r in enumerate(records) if i not in held]
        evaluation = [records[i] for i in indices]
        print(
            json.dumps(
                {"fold": fold, "train_records": len(training), "score_records": len(evaluation)}
            ),
            flush=True,
        )
        fitted, values = model.split_predictions(training, evaluation)
        require(not predictions or set(predictions) == set(values), "method coverage changed")
        for name, p in values.items():
            predictions.setdefault(name, np.full(len(records), np.nan))[indices] = p
        for arm, fitted_model in fitted.items():
            save(output / "models" / f"fold{fold}_{arm}.json", fitted_model)
        coverage[indices] += 1
    require(
        np.all(coverage == 1)
        and len(predictions) == 7
        and all(np.isfinite(p).all() for p in predictions.values()),
        "incomplete common OOF coverage",
    )
    overall = metrics(records, predictions)
    by_session = {}
    for session in sorted(folds["session_to_fold"]):
        ids = [i for i, r in enumerate(records) if r["provenance"]["session"] == session]
        by_session[session] = metrics(
            [records[i] for i in ids], {k: p[ids] for k, p in predictions.items()}
        )
    contrasts = {}
    for first, second in [
        ("plus_current_period_17", "account_public_task_15"),
        ("plus_completed_history_23", "plus_current_period_17"),
    ]:
        differences = {
            s: v["methods"][first]["brier_session_person_balanced"]
            - v["methods"][second]["brier_session_person_balanced"]
            for s, v in by_session.items()
        }
        contrasts[first + "-minus-" + second] = {
            **interval(differences, folds["components"]),
            "per_session": differences,
        }
    with (output / "predictions.jsonl").open("x") as handle:
        for i, r in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "session": keys[i][0],
                        "order_id": keys[i][1],
                        "participant": r["provenance"]["participant"],
                        "fold": folds["session_to_fold"][keys[i][0]],
                        "observed_buy": r["target"]["buy"],
                        "probabilities": {k: float(p[i]) for k, p in predictions.items()},
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    return {
        "overall": overall,
        "per_session": by_session,
        "contrasts": contrasts,
        "common_records": len(records),
        "methods": len(predictions),
    }


def describe(result):
    names = {
        "account_public_task_15": "账户和公开信息",
        "plus_current_period_17": "再加本轮本人买卖历史",
        "plus_completed_history_23": "再加更早回合本人历史",
        "fold_training_constant": "学习资料的固定买入比例",
        "constant_half": "一半买、一半卖",
        "current_history_prior4": "本轮个人历史的简单比例",
        "combined_history_prior4": "全部已知个人历史的简单比例",
    }
    lines = [
        "个人买卖历史是否增加预测信息？（训练资料内部的探索，零模型调用）",
        "",
        "只预测已经决定下单时买还是卖。51场原训练实验按整场及相同参与者代码分组，"
        "每次用其中四组学习，再预测另一组。没有使用开发或最终检验资料。",
        "这些资料以前已使用，不能把本次称为全新独立验证。评分越小，概率预测误差越低；不是正确率。",
        "",
    ]
    for name, values in result["overall"]["methods"].items():
        lines.append(f"{names[name]}：{values['brier_session_person_balanced']:.6f}")
    lines += ["", "加入历史后的误差减去加入前：负数表示改善，正数表示变差。"]
    for name, contrast in result["contrasts"].items():
        first, second = name.split("-minus-")
        lo, hi = contrast["percentile_95"]
        lines.append(
            f"{names[first]} 相对 {names[second]}：{contrast['difference']:+.6f}；"
            f"描述性95%区间[{lo:+.6f}, {hi:+.6f}]。"
        )
    lines += [
        "",
        "区间按完整场次关联组重抽，只针对这次固定预测，没有重复训练，未覆盖训练参数的不确定性。"
        "同一人若使用不同代码，资料无法排除；已知相同代码和同一场次都保持在一组。",
        "第一组移除的是明确的个人动作统计，账户和公开历史仍受过去影响。"
        "缓存中被保留的下单时点不包含全部历史订单，本轮没有重新核查原始完整历史。",
        "结果只讨论这一类统计方法能否利用已经发生的本人历史；不证明人的心理机制、"
        "基本面／技术策略选择、是否下单、怎样报价，或由AI自主交易形成的市场真实。",
        "",
    ]
    return "\n".join(lines)


def execute(packet, output):
    checked_packet(packet)
    preflight = read(packet / "independent_preflight.json")
    require(
        preflight.get("accepted_for_execution") is True
        and preflight.get("mustfix") == []
        and preflight.get("plan_sha256") == sha((packet / "plan.json").read_bytes()),
        "independent preflight does not approve this packet",
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "status.json", {"status": "started", "new_model_POSTs": 0})
    try:
        partitions = read(packet / "partitions.json")["assignments"]
        all_sessions = [s for group in partitions.values() for s in group]
        require(len(all_sessions) == len(set(all_sessions)), "overlapping original partitions")
        records = load_cache(ROOT / CACHE, CACHE_SHA, partitions["train"], 63143)
        folds = grouped_folds(records)
        if all(len(group) == 1 for group in folds["components"]):
            require(
                folds["session_to_fold"]
                == read(packet / "engineering_design.json")["candidate_folds"],
                "singleton mapping differs from the predeclared candidate",
            )
        save(output / "folds.json", folds)
        save(
            output / "admission.json",
            {
                "rows": len(records),
                "sessions": len(folds["session_to_fold"]),
                "source_sha256": CACHE_SHA,
                "new_model_POSTs": 0,
                "training_human_rows_read_locally": True,
                "human_rows_sent": False,
                "development_or_final_rows_read": False,
                "folds_sha256_before_first_fit": sha((output / "folds.json").read_bytes()),
            },
        )
        result = cross_validate(records, folds, output)
        checked_packet(packet)
        require(
            sha((ROOT / CACHE).read_bytes()) == CACHE_SHA, "training cache changed during execution"
        )
        result.update(
            protocol=PROTOCOL,
            new_model_POSTs=0,
            exploratory_reused_training=True,
            plan_sha256=sha((packet / "plan.json").read_bytes()),
        )
        save(output / "summary.json", result)
        (output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
        save(
            output / "status.json",
            {
                "status": "complete_pending_independent_audit",
                "new_model_POSTs": 0,
                "common_records": result["common_records"],
            },
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
        require(args.output is None, "separate preparation from execution")
        prepare(args.packet, args.prepare_design)
        print(json.dumps({"prepared": True, "human_states_opened": False, "new_model_POSTs": 0}))
    else:
        require(args.output is not None, "new output required")
        result = execute(args.packet, args.output)
        print(json.dumps({"common_records": result["common_records"], "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
