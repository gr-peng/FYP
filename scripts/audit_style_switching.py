#!/usr/bin/env python3
"""Audit a fixed archived strategy-review cohort; never start model/simulation jobs."""

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from behavioral_market.evaluation.style_diagnostics import (
    bind_archived_answer,
    normalize_review,
    summarize_reviews,
    validate_review_status,
)
from behavioral_market.llm.schemas import StyleDecision


def validate_plan(plan):
    base = "behavioral-llm-market/outputs/meta_compute_2026/aigc_relay/historical"
    expected = {
        (group, seed, f"{base}/{group}/n30_seed{seed}_shock0_e21")
        for group in ("rule_abm", "vanilla_llm", "low_herding", "high_herding")
        for seed in (42, 43, 44)
    }
    actual = {(r["treatment"], r["seed"], r["relative_folder"]) for r in plan["runs"]}
    if len(plan["runs"]) != 12 or actual != expected:
        raise ValueError("Plan must contain all12 fixed primary runs without replacement")
    files = set()
    for group, _seed, folder in expected:
        names = ["run_metadata.json", "style_switches.parquet", "config_snapshot.json"]
        if group != "rule_abm":
            names += ["llm_requests.jsonl", "llm_responses.jsonl"]
        files.update(folder + "/" + name for name in names)
    if set(plan["source_snapshot"]) != files:
        raise ValueError("Read only the fixed54 archived input files")
    if plan["review_dates"] != ["2026-06-29", "2026-07-14"]:
        raise ValueError("Review dates differ from the fixed original cohort")


def identity(path):
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def checked_path(workspace, relative):
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("Input leaves workspace")
    return path


def load_contexts(folder, metadata, expected_keys):
    """Link the exact archived prompt and raw answer to each published review."""
    requests, responses = {}, {}
    request_count = response_count = 0
    with (folder / "llm_requests.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("kind") != "style":
                continue
            request_count += 1
            for name in ("seed", "treatment", "mode", "shock", "include_secondary"):
                if item[name] != metadata[name]:
                    raise ValueError("Archived request belongs to another run")
            original_identity = {
                k: v for k, v in item.items() if k not in {"request_hash", "payload"}
            }
            calculated = hashlib.sha256(
                json.dumps(
                    [item["payload"], original_identity, metadata["prompt_version"]],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if calculated != item["request_hash"]:
                raise ValueError("Archived request hash does not match its payload/identity")
            if calculated in requests:
                raise ValueError("Duplicated archived style request")
            key = item["agent_id"], item["session_date"]
            if key not in expected_keys:
                raise ValueError("Extra review request outside fixed identities")
            if type(item["repair"]) is not int or item["repair"] not in (0, 1):
                raise ValueError("Unexpected repair identity")
            users = [m["content"] for m in item["payload"]["messages"] if m["role"] == "user"]
            context = json.loads(users[0])
            if context["simulation_seed"] != metadata["seed"]:
                raise ValueError("Prompt seed differs from run identity")
            requests[calculated] = {
                "key": key,
                "repair": item["repair"],
                "context": context,
                "payload": item["payload"],
            }
    with (folder / "llm_responses.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("kind") != "style":
                continue
            response_count += 1
            request_hash = item["request_hash"]
            if request_hash not in requests:
                raise ValueError("Style response has no matching request")
            key = item["agent_id"], item["session_date"]
            if key != requests[request_hash]["key"]:
                raise ValueError("Response belongs to another agent/date")
            for name in ("seed", "treatment", "mode", "shock", "include_secondary"):
                if item[name] != metadata[name]:
                    raise ValueError("Response belongs to another run")
            if item["repair"] != requests[request_hash]["repair"]:
                raise ValueError("Response repair identity differs from request")
            body = item.get("response")
            if not isinstance(body, dict) or not body.get("choices"):
                continue
            try:
                raw = json.loads(body["choices"][0]["message"]["content"])
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if request_hash in responses:
                raise ValueError("Multiple successful answers for a single style request")
            responses[request_hash] = raw
    initial = {}
    for value in requests.values():
        if value["repair"] == 0:
            if value["key"] in initial:
                raise ValueError("Multiple initial prompts for one review")
            initial[value["key"]] = value["context"]
    if set(initial) != expected_keys:
        raise ValueError("Missing initial prompt for a published review")
    for value in requests.values():
        if value["context"] != initial[value["key"]]:
            raise ValueError("Repair changed decision context instead of format only")
    return (
        requests,
        responses,
        initial,
        {"style_requests": request_count, "style_response_records": response_count},
    )


def verify_answer(row, requests, responses):
    """Accept only an actual archived answer, not a repaired/default HOLD record."""
    validate_review_status(row["validation_status"], "vanilla_llm")
    key = row["agent_id"], row["session_date"]
    if row["validation_status"] not in {"ok", "inconsistent_style"}:
        return None
    matches = []
    for request_hash, request in requests.items():
        if request["key"] != key or request_hash not in responses:
            continue
        try:
            raw = StyleDecision.model_validate(responses[request_hash]).model_dump()
        except ValidationError:
            continue
        coherent = (raw["decision"] == "stay") == (raw["target_style"] == row["from_style"])
        expected = {
            "decision": raw["decision"] if coherent else "stay",
            "target_style": raw["target_style"] if coherent else row["from_style"],
            "rationale": raw["rationale"],
            "validation_status": "ok" if coherent else "inconsistent_style",
        }
        if all(row[name] == value for name, value in expected.items()):
            matches.append(request_hash)
    if len(matches) != 1:
        raise ValueError("Published review cannot be uniquely reconstructed from its raw answer")
    return matches[0]


def percentage(value):
    return "无法比较" if value is None else f"{100 * value:.1f}%"


def report(summary):
    labels = {
        "rule_abm": "写死规则的程序",
        "vanilla_llm": "普通 AI",
        "low_herding": "低跟风提示 AI",
        "high_herding": "高跟风提示 AI",
    }
    overall, conditional = [], []
    for group in summary["per_treatment"]:
        overall.append(
            f"| {labels[group['treatment']]} | {group['runs']} | "
            f"{group['opportunities']} | {group['switches']} | "
            f"{percentage(group['equal_run_switch_rate'])} |"
        )
        for key, label in [
            ("loss", "这段时间亏损"),
            ("proxy_better", "简化替代方案算起来更好"),
            ("other_style_majority", "另一种策略在群体中占多数"),
        ]:
            cell = group["conditions"][key]
            a, b = cell["condition_true"], cell["condition_false"]
            conditional.append(
                f"| {labels[group['treatment']]} | {label} | "
                f"{a['switches']}/{a['n']}（{percentage(a['switch_rate'])}） | "
                f"{b['switches']}/{b['n']}（{percentage(b['switch_rate'])}） | "
                f"{cell['condition_unavailable_or_tied']} | {cell['runs_with_both_conditions']}/"
                f"{cell['expected_runs']} |"
            )
    valuation = "\n".join(
        f"- {row['review_date']}：同一日可见的价格与价值基准组合共 {row['distinct_values']} 种。"
        for row in summary["valuation_by_review_date"]
    )
    date_rows = "\n".join(
        f"| {labels[r['treatment']]} | {r['review_date']} | {r['loss_count']}/"
        f"{r['valid_choices']} | {r['switches']} |"
        for r in summary["by_review_date"]
        if r["treatment"] != "rule_abm"
    )
    support_rows = "\n".join(
        f"| {labels[g['treatment']]} | "
        f"{g['conditions']['loss']['same_run_date_style_blocks_with_both_conditions']}/"
        f"{g['conditions']['loss']['same_run_date_style_blocks']} |"
        for g in summary["per_treatment"]
        if g["treatment"] != "rule_abm"
    )
    return (
        """# 策略切换：这次实际补了什么

本轮没有再跑市场，也没有向模型发新请求，而是补上了直接检查旧 AI 决策的分析工具。
检查对象预先限定为原历史回放主批次：四组设置、每组42/43/44三个种子，30名代理、
23个交易日、每名代理两次策略复盘，共720次机会。其中540次是AI回答，180次是程序规则。
这是一项回头分析已有结果的检查，不是新的预注册实验，也不是真人测试。

## 第一件事：确认是 AI 真正做的选择

逐次核对原请求里给出的账户收益、替代方案、当前策略和滞后群体信息，再把原回答
与保存的切换表核对。默认保持、格式错误和被程序纠正的回答不会冒充有效选择；缺记录
或文件不一致会阻止发布，不能只挑成功的几组。规则组单独标明，不能把写死规则的
反应当成 AI 学到了行为规律。

## 总体结果

| 组别 | 完整运行 | 复盘机会 | 切换次数 | 每次运行等权的切换率 |
|---|---:|---:|---:|---:|
"""
        + "\n".join(overall)
        + """

这只是代理在当前提示和环境里的表现。高跟风组的总切换率与普通组不同，也不能直接
解释为“更像真人”或“更跟风”：还要看它当时看到了什么、有多少可比情形。

## 最重要的发现：亏损与复盘日期混在了一起

| 组别 | 复盘日期 | 亏损复盘/有效机会 | 切换次数 |
|---|---|---:|---:|
"""
        + date_rows
        + """

普通AI和高跟风AI的亏损集中在第一次复盘，第二次复盘全都没有亏损。第一次和第二次
的价格信号、新闻与已经选择的策略也不同。因此“亏损时切换更多”只是这批记录的现象，
不能单独证明亏损引发了切换。

进一步按同一次运行、同一复盘日期、同一种原有策略划分，真正同时有亏损与未亏损
样本的小组数量如下。它显示是否存在可比情形，不是显著性检验。
小组只计当时实际存在的原有策略，全部720次复盘机会仍然保留。

| 组别 | 同时有两边样本的小组/全部小组 |
|---|---:|
"""
        + support_rows
        + """

## 哪些因素能比较，哪些比较不了

下表的“切换/机会”始终保留分母。缺少一边样本时写“无法比较”，不会写成0%的效果。
最后一列表示三个种子中，有几个同时包含条件成立与不成立的情况。

| 组别 | 检查条件 | 条件成立：切换/机会 | 条件不成立：切换/机会 | 持平或未记录 | 有两边样本的运行 |
|---|---|---:|---:|---:|---:|
"""
        + "\n".join(conditional)
        + """

同一个代理的两次复盘并不是两个独立的人，三个种子也不支持强结论。这里不重新计算
显著性或宣布某因素导致了切换；只是找出已有记录里有哪些反应、哪些缺对照。

## 价格偏离与替代收益的限制

"""
        + valuation
        + """

如果同一日期只有一种价格和价值基准组合，就无法把估值因素和那个日期的其他信息
分开。这里的价值基准还是人为指定的参考价，并不是真实公司价值估计。

“另一种策略可能赚更多”使用的是不计交易摩擦的简化计算，不是另一个完整投资者
实际能赚到的钱，也不是与其他人同期收益的公平比较。它能检查代理是否参考了输入
数字，但不能代替 RP 中所有“相对表现”问题。

## 对 RP 的实际推进

之前只有整体切换率，现在能追到每次复盘前看到的条件，并明确哪些比较没有足够样本。
下一轮应保持其余输入一致、分别改变一个可观察因素，再检查选择是否跟着变化；之后
仍需找匹配的真人行为依据。不能为了得到更高切换率就宣布更真实。

所有结论限于已有历史回放：没有证明自主市场更真实、真人行为改善或某个因素的因果
作用。旧实验和暂停的 SCIT 分支保持不变。输入文件及实际核对次数见summary.json、
source_receipt.json，逐次结果见reviews.jsonl；没有打开保留的真人测试场次。
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 2**30, 8 * 2**30))
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    for relative, expected in plan["source_snapshot"].items():
        if identity(checked_path(workspace, relative)) != expected:
            raise ValueError("Input differs from the fixed plan snapshot: " + relative)
    if args.output.exists():
        raise FileExistsError("Preserve old reports; choose a new output path")
    records, expected_runs, receipts, source_versions = [], {}, [], set()
    for spec in plan["runs"]:
        folder = checked_path(workspace, spec["relative_folder"])
        meta = json.loads((folder / "run_metadata.json").read_text())
        if any(
            meta[name] != value
            for name, value in {
                "status": "complete",
                "mode": "historical",
                "n_agents": 30,
                "sessions": 23,
                "seed": spec["seed"],
                "treatment": spec["treatment"],
                "shock": 0,
                "include_secondary": True,
            }.items()
        ):
            raise ValueError("Run differs from the fixed primary cohort")
        source_versions.add(
            (meta["source_hash"], meta["data_manifest_hash"], meta["prompt_version"])
        )
        table = folder / "style_switches.parquet"
        if identity(table)["sha256"] != meta["artifact_hashes"][table.name]:
            raise ValueError("Review table differs from its original publication hash")
        rows = pd.read_parquet(table, use_threads=False).to_dict("records")
        keys = {(r["agent_id"], r["session_date"]) for r in rows}
        fixed_keys = {(f"A{i:03}", day) for i in range(30) for day in plan["review_dates"]}
        if len(rows) != len(keys) or keys != fixed_keys:
            raise ValueError("Missing/duplicated agent-date opportunities")
        run_id = spec["relative_folder"]
        expected_runs[run_id] = spec["expected_reviews"]
        receipt = {"run_id": run_id, "review_rows": len(rows), "raw_answers_matched": 0}
        if spec["treatment"] != "rule_abm":
            requests, responses, initial, counts = load_contexts(folder, meta, keys)
            receipt.update(counts)
        for row in rows:
            if spec["treatment"] == "rule_abm":
                expected_choice = (
                    "switch"
                    if (row["counterfactual_block_return"] - row["actual_block_return"] > 0.01)
                    else "stay"
                )
                if row["decision"] != expected_choice or row["validation_status"] != "rule":
                    raise ValueError("Rule baseline differs from its declared threshold")
                context, matched = None, None
            else:
                matched = verify_answer(row, requests, responses)
                receipt["raw_answers_matched"] += matched is not None
                context = initial[(row["agent_id"], row["session_date"])]
            normalized = normalize_review(
                row,
                run_id,
                spec["seed"],
                spec["treatment"],
                context,
            )
            records.append(bind_archived_answer(normalized, matched))
        receipts.append(receipt)
    if len(source_versions) != 1:
        raise ValueError("Do not combine providers/source/data/prompt versions")
    summary = summarize_reviews(records, expected_runs)
    if len(records) != plan["expected_total_reviews"]:
        raise ValueError("Incorrect full cohort denominator")
    for relative, expected in plan["source_snapshot"].items():
        if identity(checked_path(workspace, relative)) != expected:
            raise ValueError("Archived input changed during analysis")
    summary.update(
        scope="fixed historical archived batch; retrospective diagnostics only",
        new_model_calls=0,
        new_market_runs=0,
        per_run_source_checks=receipts,
        raw_llm_answers_matched=sum(r["raw_answers_matched"] for r in receipts),
        source_versions=[list(v) for v in source_versions],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (args.output / "reviews.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    )
    (args.output / "source_receipt.json").write_text(
        json.dumps(
            {
                "plan": identity(args.plan),
                "inputs": plan["source_snapshot"],
                "endpoint_hashes_match": True,
                "raw_logs_originally_manifest_bound": False,
                "raw_log_limit": (
                    "Logs are fingerprinted now and request hashes recomputed; "
                    "no claim of an original external signature or continuous immutability."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (args.output / "REPORT.md").write_text(report(summary))
    print(
        json.dumps(
            {
                "review_rows": len(records),
                "raw_llm_answers_matched": summary["raw_llm_answers_matched"],
                "runs": len(expected_runs),
                "new_model_calls": 0,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
