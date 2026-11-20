#!/usr/bin/env python3
"""Prepare, collect or evaluate the fixed sixteen-slot repeated-input check."""

import argparse
import asyncio
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from behavioral_market.evaluation.style_probe_intake import parse_choice, strict_json
from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode
from behavioral_market.evaluation.style_probe_retest import (
    NAMESPACE,
    evaluate_retest,
    expected_request,
    request_identity,
    run_manifest,
    validate_schedule,
)
from behavioral_market.llm.client import FatalAPIError, OpenAICompatibleClient
from scripts.evaluate_style_share_probe import identity, read_jsonl
from scripts.style_probe_mode_v2 import load_variant, save, v2_config

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")


def load_inputs(root, materials, proposal):
    cases, base, sources, inputs = load_variant(root, materials)
    path = proposal / "scheduled_payloads.jsonl"
    inputs[str(path.resolve())] = identity(path)
    planned = read_jsonl(path)
    validate_schedule(cases, planned)
    return cases, base, sources, planned, inputs


def check_inputs(inputs):
    for path, fingerprint in inputs.items():
        if identity(Path(path)) != fingerprint:
            raise ValueError("Retest input changed during the operation")


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior bundles")
    cases, _base, _sources, planned, inputs = load_inputs(
        args.workspace.resolve(), args.materials, args.proposal
    )
    kind = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
    manifest = run_manifest(cases, planned, args.run_id, kind)
    requests = [expected_request(slot, manifest) for slot in planned]
    check_inputs(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "run_manifest.json", manifest)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    (args.output / "expected_requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in requests)
    )
    return {"status": "prepared_not_sent", "scheduled_count": 16, "model_calls": 0}


def load_bundle(args):
    cases, base, sources, planned, inputs = load_inputs(
        args.workspace.resolve(), args.materials, args.proposal
    )
    kind = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
    expected = run_manifest(cases, planned, args.run_id, kind)
    paths = [args.bundle / name for name in ("run_manifest.json", "expected_requests.jsonl")]
    inputs.update({str(path.resolve()): identity(path) for path in paths})
    manifest = strict_json(paths[0].read_text())
    if payload_digest(manifest) != payload_digest(expected) or payload_digest(
        read_jsonl(paths[1])
    ) != payload_digest([expected_request(slot, manifest) for slot in planned]):
        raise ValueError("Prepared retest bundle differs from the fixed inputs and run identity")
    return cases, base, sources, planned, manifest, inputs


def check_client(cases, client):
    config = client.config
    if (
        client.cache_namespace != NAMESPACE
        or type(config.get("transport_retries")) is not int
        or config["transport_retries"] != 0
        or type(config.get("max_concurrency")) is not int
        or config["max_concurrency"] != 1
        or any(
            k in config
            for k in ("thinking", "reasoning_effort", "extra_body", "allowed_openai_params")
        )
    ):
        raise ValueError("Retest requires its fresh namespace, sequential calls and no retries")
    for case in cases:
        payload = {
            key: copy.deepcopy(config[key])
            for key in (
                "model",
                "temperature",
                "max_tokens",
                "enable_thinking",
                "chat_template_kwargs",
            )
        }
        payload.update(
            messages=case["payload"]["messages"], response_format={"type": "json_object"}
        )
        if payload_digest(payload) != case["payload_sha256"]:
            raise ValueError("Client would send a different retest payload")


async def collect_retest(cases, planned, manifest, client, output):
    if payload_digest(manifest) != payload_digest(
        run_manifest(cases, planned, manifest["run_id"], manifest["evidence_kind"])
    ):
        raise ValueError("Retest manifest mismatch")
    # No resume or existing logs, including cache-only replay of a previous attempt.
    if any((output / name).exists() and (output / name).stat().st_size for name in LOGS):
        raise FileExistsError("Do not resume or append to previous retest logs")
    if any(client.cache.iterdir()):
        raise FileExistsError("Retest requires an empty fresh cache")
    check_client(cases, client)
    for name in LOGS:
        (output / name).touch(exist_ok=True)
    by_id = {case["case_id"]: case for case in cases}
    state = {
        "status": "running",
        "stop_reason": None,
        "evidence_kind": manifest["evidence_kind"],
        "scheduled_count": 16,
        "max_completion_posts": 16,
        "retest_complete": False,
        "real_model_execution_verified": False,
        "started_utc": datetime.now(UTC).isoformat(),
        "slots": [{"planned_id": slot["planned_id"], "status": "unsubmitted"} for slot in planned],
    }

    def publish():
        save(output / "execution_state.json", state)

    publish()
    active_index = None
    try:
        await client.verify_model()
        for i, slot in enumerate(planned):
            active_index = i
            state["slots"][i]["status"] = "request_in_progress"
            publish()
            body, meta = await client.completion(
                slot["payload"]["messages"], request_identity(slot, manifest)
            )
            if meta.get("request_hash") != expected_request(slot, manifest)["request_hash"]:
                raise ValueError("Actual retest request differs from frozen expectation")
            answer, _choice = parse_choice(body, by_id[slot["case_id"]])
            mode = returned_mode(body)
            cached = bool(meta.get("cache_hit"))
            compatible = (
                answer == "valid_linked_answer"
                and mode == "no_returned_reasoning_observed"
                and not cached
            )
            state["slots"][i].update(
                status="valid" if compatible else "failed",
                answer_status=answer,
                mode_status=mode,
                cache_replay=cached,
            )
            if not compatible:
                state.update(
                    status="stopped",
                    stop_reason="cache_replay"
                    if cached
                    else mode
                    if mode != "no_returned_reasoning_observed"
                    else answer,
                )
                break
            publish()
        else:
            state.update(status="complete", retest_complete=True)
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="cancelled; in-flight billing unknown")
        raise
    except (FatalAPIError, ValueError) as exc:
        state.update(status="stopped", stop_reason=str(exc))
    except Exception as exc:
        state.update(status="stopped", stop_reason="unexpected_error:" + type(exc).__name__)
    finally:
        if (
            active_index is not None
            and state["slots"][active_index]["status"] == "request_in_progress"
        ):
            state["slots"][active_index]["status"] = (
                "cancelled_in_progress" if state["status"] == "cancelled" else "call_error"
            )
        state["finished_utc"] = datetime.now(UTC).isoformat()
        publish()
    return state


async def execute(args):
    if args.output.exists():
        raise FileExistsError("No automatic resume; preserve earlier attempts")
    root = args.workspace.resolve()
    cases, base, _sources, planned, manifest, inputs = load_bundle(args)
    config_path = root / "behavioral-llm-market/config/meta_compute_2026.json"
    inputs[str(config_path)] = identity(config_path)
    config = v2_config(strict_json(config_path.read_text()), base, cases)
    check_inputs(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in LOGS:
        (args.output / name).touch(exist_ok=False)
    save(args.output / "run_manifest.json", manifest)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    key = dotenv_values(root / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        state = {
            "status": "stopped",
            "stop_reason": "AIGC_API_KEY missing",
            "retest_complete": False,
        }
        save(args.output / "execution_state.json", state)
        return state
    client = OpenAICompatibleClient(
        config, key, args.output, args.output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        state = await collect_retest(cases, planned, manifest, client, args.output)
    finally:
        await client.close()
    check_inputs(inputs)
    return {key: state[key] for key in ("status", "stop_reason", "retest_complete")}


def readable_report(result):
    lines = ["# 同样问题重复四次：结果核对\n"]
    if result["evidence_kind"] == "synthetic_tool_test":
        lines.append("**这是合成工具测试，不是实际模型结果。**\n")
    lines.append(
        f"预定 16 次，合格 {result['valid_count']}，缺失 {result['missing_count']}，"
        f"错误 {result['error_count']}。这仍是两个原情境；旧四条发现记录未计入。\n"
    )
    if not result["retest_complete"]:
        lines.append("**本次复测未完成；不把缺失补成保持，也不将不完整情境算成没有差异。**\n")
    labels = {"fundamental": "基本面", "technical": "技术"}
    lines.extend(
        [
            "| 原策略 | 自己的策略占比 | 保持 | 切换 | 合格/预定 | 缺失 | 错误 | "
            "相同输入选过不同策略 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in result["conditions"]:
        variation = (
            "出现过"
            if row["variation_observed"] is True
            else "这四次相同"
            if row["variation_observed"] is False
            else "尚不完整"
        )
        share = "80%" if row["condition"] == "own_style_majority" else "20%"
        lines.append(
            f"| {labels[row['from_style']]} | {share} | "
            f"{row['decision_counts']['stay']} | {row['decision_counts']['switch']} | "
            f"{row['valid_count']}/4 | "
            f"{row['missing_count']} | {row['error_count']} | {variation} |"
        )
    lines.extend(
        [
            "\n| 原策略 | 轮次 | 自己占多数时选择 | 另一策略占多数时选择 | 是否改变 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in result["rounds"]:
        own = labels.get(row["own_majority_target"], "缺合格回答")
        other = labels.get(row["other_majority_target"], "缺合格回答")
        changed = (
            "无法比较" if row["choice_changed"] is None else "是" if row["choice_changed"] else "否"
        )
        lines.append(
            f"| {labels[row['from_style']]} | {row['repeat']} | {own} | {other} | {changed} |"
        )
    for pair in result["pairs"]:
        if pair["all_rounds_comparable"]:
            lines.append(
                f"\n{labels[pair['from_style']]}情境：八条均合格；"
                f"四轮比较中有 {pair['changed_rounds']} 轮的两次选择不同。"
            )
        else:
            lines.append(
                f"\n{labels[pair['from_style']]}情境：只收齐 {pair['comparable_rounds']}/4 轮，"
                "本情境的主要比较尚无法完成。"
            )
    lines.extend(
        [
            "\n主要比较要求同一情境的八条都合格。逐轮表保留已有回答，不用局部成功替代完整比较。",
            "四次都相同不能证明没有随机变化；理由不同不等于选择不同。本地无缓存不保证服务后台无缓存，或回答互相独立。",
            "这些是两个情境的重复，不是16名投资者，也没有证明跟风规律、因果作用、真人相似度或市场真实性。",
            "这里只核对本地日志；离线评价不认证外部实际执行。\n",
        ]
    )
    return "\n".join(lines)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError("Preserve old retest evaluations")
    cases, base, sources, planned, manifest, inputs = load_bundle(args)
    logs = [[], [], []]
    if args.execution_dir:
        paths = [args.execution_dir / name for name in ("run_manifest.json", *LOGS)]
        inputs.update({str(path.resolve()): identity(path) for path in paths})
        if payload_digest(strict_json(paths[0].read_text())) != payload_digest(manifest):
            raise ValueError("Stored retest run identity or evidence mode differs")
        logs = [read_jsonl(path) for path in paths[1:]]
    result = evaluate_retest(cases, base, sources, planned, manifest, *logs)
    check_inputs(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", result)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    (args.output / "REPORT.md").write_text(readable_report(result))
    return {
        k: v
        for k, v in result.items()
        if k not in {"conditions", "pairs", "rounds", "slots", "limits"}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--run-id", default="style-share-retest-20261022")
    parser.add_argument("--execution-dir", type=Path)
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.action != "prepare" and args.bundle is None:
        parser.error("run/evaluate requires the independently reviewable --bundle")
    if args.action == "run" and args.synthetic_test:
        parser.error("Synthetic tests are offline only")
    result = (
        prepare(args)
        if args.action == "prepare"
        else asyncio.run(execute(args))
        if args.action == "run"
        else evaluate(args)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
