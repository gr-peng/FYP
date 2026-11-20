import asyncio
import json
from argparse import Namespace

import httpx
import pytest

from behavioral_market.evaluation.style_probe_intake import (
    NAMESPACE,
    evaluate_synthetic_probe,
    make_run_manifest,
)
from behavioral_market.evaluation.style_probe_materials import build_share_probes, payload_digest
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.evaluate_style_share_probe import read_jsonl
from scripts.run_style_share_probe import collect, execution_config
from scripts.run_style_share_probe import run as run_probe


@pytest.fixture
def setup():
    provider = {
        "model": "DeepSeek-V4-Flash",
        "api_key_env": "AIGC_API_KEY",
        "chat_completions_url": "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
        "models_url": "https://aigc-api.hkust-gz.edu.cn/api/v2/models",
        "allowed_host": "aigc-api.hkust-gz.edu.cn",
        "cache_dir": "unused",
        "trust_env": False,
    }
    base = {
        "llm": {
            "temperature": 0.2,
            "max_tokens": 512,
            "enable_thinking": False,
            "max_concurrency": 32,
            "transport_retries": 5,
            "timeout_seconds": 120,
            "providers": {"aigc_relay": provider},
        }
    }
    sources = []
    for i, style in enumerate(("fundamental", "technical")):
        context = {
            "agent_state": {"persona": {"agent_id": f"A{i:03}"}, "current_style": style},
            "observation": {
                "session_date": "2026-06-29",
                "phase": "post_close_style_review",
                "mean_field": {
                    "as_of_session": "2026-06-26",
                    "fundamental_share": 0.5,
                    "technical_share": 0.5,
                },
            },
        }
        payload = {
            "model": "DeepSeek-V4-Flash",
            "temperature": 0.2,
            "max_tokens": 512,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "synthetic test"},
                {"role": "user", "content": json.dumps(context)},
            ],
        }
        h = payload_digest(payload)
        sources.append(
            {
                "identity": {
                    "run_id": "source",
                    "review_date": "2026-06-29",
                    "agent_id": f"A{i:03}",
                    "from_style": style,
                    "treatment": "vanilla_llm",
                    "original_request_hash": h,
                    "matched_answer_request_hash": h,
                },
                "payload": payload,
            }
        )
    cases = build_share_probes(sources)
    return cases, sources, base


def exercise(tmp_path, setup, mode="ok", reasoning_at=0):
    cases, sources, base = setup
    config = execution_config(base, cases)
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                503 if mode == "registry_failed" else 200,
                json={"data": [{"id": "DeepSeek-V4-Flash"}]},
            )
        payload = json.loads(request.content)
        posts.append(payload)
        context = json.loads(payload["messages"][1]["content"])
        current = context["agent_state"]["current_style"]
        raw = {"decision": "stay", "target_style": current, "rationale": "synthetic transport test"}
        if mode == "contradiction":
            raw["target_style"] = "technical" if current == "fundamental" else "fundamental"
        message = {"role": "assistant", "content": json.dumps(raw)}
        if mode == "reasoning" and len(posts) == reasoning_at + 1:
            message["reasoning_content"] = "synthetic thought"
        if mode == "invalid_json":
            message["content"] = "not json"
        body = {
            "model": "other" if mode == "wrong_model" else "DeepSeek-V4-Flash",
            "choices": [
                {"finish_reason": "length" if mode == "truncated" else "stop", "message": message}
            ],
        }
        if mode == "outer_invalid":
            return httpx.Response(200, text="broken outer JSON")
        if mode == "http_failed":
            return httpx.Response(503, text="failed")
        return httpx.Response(200, json=body)

    async def run():
        client = OpenAICompatibleClient(
            config, "synthetic-test-key", tmp_path, tmp_path / "cache", cache_namespace=NAMESPACE
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manifest = make_run_manifest(cases, "transport-unit", "synthetic_tool_test")
        try:
            return await collect(cases, manifest, client, tmp_path), manifest
        finally:
            await client.close()

    state, manifest = asyncio.run(run())
    return state, posts, manifest


def test_four_post_requests_keep_frozen_payloads_and_are_usable_by_intake(setup, tmp_path):
    state, posts, manifest = exercise(tmp_path, setup)
    assert state["status"] == "complete" and state["compatibility_status"] == "passed"
    assert len(posts) == 4
    assert posts == [c["payload"] for c in setup[0]]
    result = evaluate_synthetic_probe(
        setup[0],
        setup[1],
        manifest,
        read_jsonl(tmp_path / "llm_requests.jsonl"),
        read_jsonl(tmp_path / "llm_attempts.jsonl"),
        read_jsonl(tmp_path / "llm_responses.jsonl"),
    )
    assert result["valid_choice_count"] == 4 and result["changed_pairs"] == 0
    assert not result["research_eligible"]


@pytest.mark.parametrize(
    "mode",
    [
        "reasoning",
        "truncated",
        "invalid_json",
        "contradiction",
        "outer_invalid",
        "http_failed",
        "wrong_model",
    ],
)
def test_first_failure_stops_without_a_second_post_or_format_repair(setup, tmp_path, mode):
    state, posts, _ = exercise(tmp_path, setup, mode)
    assert len(posts) == 1
    assert state["status"] == "stopped" and state["compatibility_status"] == "failed"
    assert len(read_jsonl(tmp_path / "llm_attempts.jsonl")) == 1


def test_registry_failure_leaves_all_cases_unsubmitted(setup, tmp_path):
    state, posts, _ = exercise(tmp_path, setup, "registry_failed")
    assert not posts and state["status"] == "stopped"
    assert state["case_results"] == []


def test_last_case_reasoning_failure_does_not_become_compatibility_success(setup, tmp_path):
    state, posts, _ = exercise(tmp_path, setup, "reasoning", reasoning_at=3)
    assert len(posts) == 4 and state["reasoning_observed"]
    assert state["status"] == "stopped" and state["compatibility_status"] == "failed"


@pytest.mark.parametrize("change", ["model", "endpoint", "thinking"])
def test_route_model_or_payload_changes_are_rejected(setup, change):
    cases, _, base = setup
    if change == "model":
        base["llm"]["providers"]["aigc_relay"]["model"] = "different"
    elif change == "endpoint":
        base["llm"]["providers"]["aigc_relay"]["chat_completions_url"] = "https://other.example/"
    else:
        base["llm"]["thinking"] = {"type": "disabled"}
    with pytest.raises(ValueError):
        execution_config(base, cases)


def test_missing_credentials_are_terminal_and_existing_execution_cannot_be_restarted(
    setup, tmp_path, monkeypatch
):
    cases, sources, base = setup
    path = tmp_path / "behavioral-llm-market/config/meta_compute_2026.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(base))
    monkeypatch.setattr(
        "scripts.run_style_share_probe.read_materials", lambda root: (cases, sources, {})
    )
    monkeypatch.setattr("scripts.run_style_share_probe.dotenv_values", lambda path: {})
    args = Namespace(workspace=tmp_path, output=tmp_path / "execution", run_id="missing-key-test")
    result = asyncio.run(run_probe(args))
    assert result["status"] == "stopped"
    assert not read_jsonl(args.output / "llm_requests.jsonl")
    with pytest.raises(FileExistsError):
        asyncio.run(run_probe(args))


def test_cancellation_is_recorded_as_terminal_not_running(setup, tmp_path):
    cases, _, _ = setup

    class CancelledClient:
        cache_namespace = NAMESPACE
        config = {"transport_retries": 0}

        async def verify_model(self):
            return "DeepSeek-V4-Flash"

        async def completion(self, messages, identity):
            raise asyncio.CancelledError

    manifest = make_run_manifest(cases, "cancel-test", "synthetic_tool_test")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect(cases, manifest, CancelledClient(), tmp_path))
    state = json.loads((tmp_path / "execution_state.json").read_text())
    assert state["status"] == "cancelled" and state["compatibility_status"] == "incomplete"
    assert "billing unknown" in state["stop_reason"]
