#!/usr/bin/env python3
"""Attempt at most four frozen inputs on the existing school route, without repair or resume."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from behavioral_market.evaluation.style_probe_intake import (
    NAMESPACE,
    expected_request,
    make_run_manifest,
    parse_choice,
    request_identity,
)
from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.llm.client import FatalAPIError, OpenAICompatibleClient
from scripts.evaluate_style_share_probe import identity, read_materials


def execution_config(base, cases):
    llm = base["llm"]
    provider = llm["providers"]["aigc_relay"]
    if (
        provider["allowed_host"] != "aigc-api.hkust-gz.edu.cn"
        or provider["chat_completions_url"]
        != "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
        or provider["models_url"] != "https://aigc-api.hkust-gz.edu.cn/api/v2/models"
        or provider["api_key_env"] != "AIGC_API_KEY"
        or provider["model"] != "DeepSeek-V4-Flash"
        or provider["trust_env"] is not False
    ):
        raise ValueError("The existing authorized school route differs; do not substitute it")
    config = {**{k: v for k, v in llm.items() if k != "providers"}, **provider}
    if set(config) - {
        "temperature",
        "max_tokens",
        "enable_thinking",
        "max_concurrency",
        "transport_retries",
        "timeout_seconds",
        "model",
        "api_key_env",
        "chat_completions_url",
        "models_url",
        "allowed_host",
        "cache_dir",
        "trust_env",
    }:
        raise ValueError("Unexpected configuration could change the frozen request")
    config.update(transport_retries=0, max_concurrency=1)
    for case in cases:
        rebuilt = {
            key: config[key] for key in ("model", "temperature", "max_tokens", "enable_thinking")
        }
        rebuilt.update(
            messages=case["payload"]["messages"], response_format={"type": "json_object"}
        )
        if payload_digest(rebuilt) != case["payload_sha256"]:
            raise ValueError("Configured completion would change the frozen payload")
    return config


async def collect(cases, manifest, client, output):
    state = {
        "status": "running",
        "compatibility_status": "not_yet_checked",
        "stop_reason": None,
        "reasoning_observed": False,
        "case_results": [],
        "started_utc": datetime.now(UTC).isoformat(),
        "max_completion_posts": 4,
        "transport_retries": 0,
        "new_market_runs": 0,
    }

    def save():
        (output / "execution_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        )

    save()
    try:
        if len(cases) != 4 or manifest != make_run_manifest(
            cases, manifest["run_id"], manifest["evidence_kind"]
        ):
            raise ValueError("Require the complete fixed four-input manifest")
        if client.cache_namespace != NAMESPACE or client.config["transport_retries"] != 0:
            raise ValueError("Require the fixed namespace and exactly one POST per input")
        await client.verify_model()
        for case in cases:
            body, meta = await client.completion(
                case["payload"]["messages"], request_identity(case, manifest)
            )
            if meta.get("request_hash") != expected_request(case, manifest)["request_hash"]:
                raise ValueError("Dispatched request hash differs from frozen expectation")
            status, _choice = parse_choice(body, case)
            reasoning = (
                any(c.get("message", {}).get("reasoning_content") for c in body.get("choices", []))
                if isinstance(body, dict)
                else False
            )
            state["reasoning_observed"] |= bool(reasoning)
            state["case_results"].append(
                {
                    "case_id": case["case_id"],
                    "answer_status": status,
                    "reasoning_observed": bool(reasoning),
                    "cache_hit": bool(meta.get("cache_hit")),
                }
            )
            if reasoning or status != "valid_linked_answer" or meta.get("cache_hit"):
                state.update(
                    status="stopped",
                    compatibility_status="failed",
                    stop_reason="reasoning_observed"
                    if reasoning
                    else status
                    if status != "valid_linked_answer"
                    else "unexpected_cache_replay",
                )
                break
            save()
        else:
            state.update(status="complete", compatibility_status="passed")
    except asyncio.CancelledError:
        state.update(
            status="cancelled",
            compatibility_status="incomplete",
            stop_reason="cancelled; in-flight billing unknown",
        )
        raise
    except (FatalAPIError, ValueError) as exc:
        state.update(status="stopped", compatibility_status="failed", stop_reason=str(exc))
    except Exception as exc:
        state.update(
            status="stopped",
            compatibility_status="failed",
            stop_reason="unexpected_error:" + type(exc).__name__,
        )
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save()
    return state


async def run(args):
    root = args.workspace.resolve()
    cases, _sources, inputs = read_materials(root)
    route_path = root / "behavioral-llm-market/config/meta_compute_2026.json"
    inputs[str(route_path)] = identity(route_path)
    config = execution_config(json.loads(route_path.read_text()), cases)
    if args.output.exists():
        raise FileExistsError(
            "No automatic resume: preserve previous attempts and select a new protocol"
        )
    manifest = make_run_manifest(cases, args.run_id, "recorded_model_logs")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl"):
        (args.output / name).touch(exist_ok=False)
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "input_receipt.json").write_text(json.dumps({"inputs": inputs}, indent=2) + "\n")
    key = dotenv_values(root / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        (args.output / "execution_state.json").write_text(
            json.dumps(
                {
                    "status": "stopped",
                    "compatibility_status": "failed",
                    "stop_reason": "AIGC_API_KEY missing",
                    "case_results": [],
                }
            )
            + "\n"
        )
        return {"status": "stopped", "stop_reason": "AIGC_API_KEY missing"}
    client = OpenAICompatibleClient(
        config, key, args.output, args.output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        state = await collect(cases, manifest, client, args.output)
    finally:
        await client.close()
    for path, expected in inputs.items():
        if identity(Path(path)) != expected:
            raise ValueError("Frozen source changed during the bounded execution")
    return {
        key: state[key]
        for key in ("status", "compatibility_status", "stop_reason", "reasoning_observed")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
