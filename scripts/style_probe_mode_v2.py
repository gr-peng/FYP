#!/usr/bin/env python3
"""Prepare, run or evaluate a separate, at-most-two-POST v2 compatibility phase."""

import argparse
import asyncio
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from behavioral_market.evaluation.style_probe_intake import parse_choice, strict_json
from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.evaluation.style_probe_mode_v2 import (
    MODE_CONTROL,
    NAMESPACE,
    PAIR_ID,
    PAIR_ROLES,
    build_variant,
    evaluate_phase,
    expected_request,
    phase_manifest,
    request_identity,
    returned_mode,
    validate_variant,
)
from behavioral_market.llm.client import FatalAPIError, OpenAICompatibleClient
from scripts.evaluate_style_share_probe import identity, read_jsonl, read_materials
from scripts.run_style_share_probe import execution_config


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_variant(root, folder):
    base, sources, inputs = read_materials(root)
    path = folder / "candidate_inputs.jsonl"
    fingerprint = identity(path)
    cases = read_jsonl(path)
    validate_variant(cases, base, sources)
    inputs[str(path.resolve())] = fingerprint
    return cases, base, sources, inputs


def prepare(root, output):
    if output.exists():
        raise FileExistsError("Preserve previous material sets")
    base, sources, inputs = read_materials(root)
    cases = build_variant(base, sources)
    checks = validate_variant(cases, base, sources)
    for path, fingerprint in inputs.items():
        if identity(Path(path)) != fingerprint:
            raise ValueError("Base materials changed during preparation")
    receipt = {
        "base_inputs": inputs,
        "checks": checks,
        "delta_by_case": [
            {
                "case_id": a["case_id"],
                "base_payload_sha256": a["payload_sha256"],
                "variant_payload_sha256": b["payload_sha256"],
                "added_field": {"chat_template_kwargs": MODE_CONTROL},
            }
            for a, b in zip(base, cases, strict=True)
        ],
        "execution_status": "materials_ready_not_run",
        "new_model_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "candidate_inputs.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in cases)
    )
    save(output / "variant_receipt.json", receipt)
    return checks


def v2_config(base_config, base_cases, cases):
    config = execution_config(base_config, base_cases)
    config["chat_template_kwargs"] = copy.deepcopy(MODE_CONTROL)
    for case in cases:
        rebuilt = {
            k: config[k]
            for k in (
                "model",
                "temperature",
                "max_tokens",
                "enable_thinking",
                "chat_template_kwargs",
            )
        }
        rebuilt.update(
            messages=case["payload"]["messages"], response_format={"type": "json_object"}
        )
        if payload_digest(rebuilt) != case["payload_sha256"]:
            raise ValueError("V2 request construction differs")
    return config


async def collect_phase(cases, manifest, client, output):
    expected_manifest = phase_manifest(
        cases,
        manifest["run_id"],
        manifest["evidence_kind"],
        pair_id=manifest.get("extension_pair_id", PAIR_ID),
    )
    if payload_digest(manifest) != payload_digest(expected_manifest):
        raise ValueError("Phase manifest mismatch")
    by_id = {c["case_id"]: c for c in cases}
    state = {
        "status": "running",
        "phase_status": "not_yet_checked",
        "case_results": [],
        "candidate_count": 4,
        "scheduled_count": 2,
        "max_completion_posts": 2,
        "not_scheduled_case_ids": [
            c["case_id"] for c in cases if c["case_id"] not in manifest["scheduled_case_ids"]
        ],
        "stop_reason": None,
        "four_case_behavior_study_complete": False,
        "evidence_kind": manifest["evidence_kind"],
        "real_model_execution_verified": False,
        "started_utc": datetime.now(UTC).isoformat(),
    }

    def publish():
        save(output / "phase_state.json", state)

    publish()
    try:
        if client.cache_namespace != NAMESPACE or client.config["transport_retries"] != 0:
            raise ValueError("Require v2 namespace and no retries")
        await client.verify_model()
        for cid in manifest["scheduled_case_ids"]:
            case = by_id[cid]
            body, meta = await client.completion(
                case["payload"]["messages"], request_identity(case, manifest)
            )
            if meta.get("request_hash") != expected_request(case, manifest)["request_hash"]:
                raise ValueError("Actual v2 request differs from frozen material")
            answer, _choice = parse_choice(body, case)
            mode = returned_mode(body)
            cached = bool(meta.get("cache_hit"))
            state["case_results"].append(
                {
                    "case_id": cid,
                    "answer_contract_status": answer,
                    "mode_contract_status": mode,
                    "cache_replay": cached,
                }
            )
            if (
                answer != "valid_linked_answer"
                or mode != "no_returned_reasoning_observed"
                or cached
            ):
                state.update(
                    status="stopped",
                    phase_status="failed",
                    stop_reason="cache_replay"
                    if cached
                    else mode
                    if mode != "no_returned_reasoning_observed"
                    else answer,
                )
                break
            publish()
        else:
            state.update(
                status="complete",
                phase_status="synthetic_compatibility_pair_passed"
                if manifest["evidence_kind"] == "synthetic_tool_test"
                else "compatibility_pair_passed",
            )
    except asyncio.CancelledError:
        state.update(
            status="cancelled",
            phase_status="incomplete",
            stop_reason="cancelled; in-flight billing unknown",
        )
        raise
    except (FatalAPIError, ValueError) as exc:
        state.update(status="stopped", phase_status="failed", stop_reason=str(exc))
    except Exception as exc:
        state.update(
            status="stopped",
            phase_status="failed",
            stop_reason="unexpected_error:" + type(exc).__name__,
        )
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        publish()
    return state


async def execute(args):
    root = args.workspace.resolve()
    cases, base, _sources, inputs = load_variant(root, args.materials)
    config_path = root / "behavioral-llm-market/config/meta_compute_2026.json"
    inputs[str(config_path)] = identity(config_path)
    config = v2_config(strict_json(config_path.read_text()), base, cases)
    if args.output.exists():
        raise FileExistsError("No automatic resume; preserve prior phase")
    manifest = phase_manifest(cases, args.run_id, "recorded_model_logs", pair_id=args.pair_id)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl"):
        (args.output / name).touch(exist_ok=False)
    save(args.output / "phase_manifest.json", manifest)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    key = dotenv_values(root / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        state = {
            "status": "stopped",
            "phase_status": "failed",
            "stop_reason": "AIGC_API_KEY missing",
            "case_results": [],
            "four_case_behavior_study_complete": False,
        }
        save(args.output / "phase_state.json", state)
        return state
    client = OpenAICompatibleClient(
        config, key, args.output, args.output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        state = await collect_phase(cases, manifest, client, args.output)
    finally:
        await client.close()
    for path, fingerprint in inputs.items():
        if identity(Path(path)) != fingerprint:
            raise ValueError("Frozen phase input changed during execution")
    return {
        k: state[k]
        for k in ("status", "phase_status", "stop_reason", "four_case_behavior_study_complete")
    }


def evaluate(args):
    if args.output.exists():
        raise FileExistsError("Preserve old phase evaluations")
    cases, base, sources, inputs = load_variant(args.workspace.resolve(), args.materials)
    kind = "synthetic_tool_test" if args.synthetic_test else "recorded_model_logs"
    manifest = phase_manifest(cases, args.run_id, kind, pair_id=args.pair_id)
    logs = [[], [], []]
    if args.execution_dir:
        paths = [
            args.execution_dir / name
            for name in (
                "phase_manifest.json",
                "llm_requests.jsonl",
                "llm_attempts.jsonl",
                "llm_responses.jsonl",
            )
        ]
        inputs.update({str(path.resolve()): identity(path) for path in paths})
        stored = strict_json(paths[0].read_text())
        if payload_digest(stored) != payload_digest(manifest):
            raise ValueError("Stored phase differs")
        logs = [read_jsonl(path) for path in paths[1:]]
    result = evaluate_phase(cases, base, sources, manifest, *logs)
    for path, fingerprint in inputs.items():
        if identity(Path(path)) != fingerprint:
            raise ValueError("Phase input changed during evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", result)
    save(args.output / "input_receipt.json", {"inputs": inputs})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--run-id", default="style-share-mode-v2-20261022")
    parser.add_argument("--pair-id", choices=tuple(PAIR_ROLES), default=PAIR_ID)
    parser.add_argument("--execution-dir", type=Path)
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.workspace.resolve(), args.output)
    elif args.materials is None:
        parser.error("run/evaluate requires --materials")
    elif args.action == "run":
        if args.synthetic_test:
            parser.error("Synthetic testing is offline only; do not use run")
        result = asyncio.run(execute(args))
    else:
        result = evaluate(args)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"cases", "delta_by_case"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
