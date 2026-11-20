import asyncio
import copy
import json

import httpx
import pytest
from test_style_probe_execution import setup as legacy_setup  # noqa: F401

from behavioral_market.evaluation.style_probe_materials import build_share_probes, payload_digest
from behavioral_market.evaluation.style_probe_mode_v2 import (
    MODE_CONTROL,
    NAMESPACE,
    build_variant,
    evaluate_phase,
    expected_request,
    phase_manifest,
    returned_mode,
    validate_variant,
)
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.evaluate_style_share_probe import read_jsonl
from scripts.style_probe_mode_v2 import collect_phase, v2_config


@pytest.fixture
def materials(request):
    _cases, sources, base_config = request.getfixturevalue("legacy_setup")
    # Match the independently selected real-source role mapping, using synthetic data only.
    for source in sources:
        agent = "A001" if source["identity"]["from_style"] == "fundamental" else "A000"
        source["identity"]["agent_id"] = agent
        context = json.loads(source["payload"]["messages"][1]["content"])
        context["agent_state"]["persona"]["agent_id"] = agent
        source["payload"]["messages"][1]["content"] = json.dumps(context)
        h = payload_digest(source["payload"])
        source["identity"].update(original_request_hash=h, matched_answer_request_hash=h)
    base = build_share_probes(sources)
    return build_variant(base, sources), base, sources, base_config


def logs(materials):
    cases = materials[0]
    m = phase_manifest(cases, "synthetic-v2", "synthetic_tool_test")
    by_id = {c["case_id"]: c for c in cases}
    requests = []
    attempts = []
    responses = []
    for i, cid in enumerate(m["scheduled_case_ids"]):
        case = by_id[cid]
        r = expected_request(case, m)
        identity = {k: v for k, v in r.items() if k != "payload"}
        start = f"2026-10-22T00:00:0{2 * i}+00:00"
        end = f"2026-10-22T00:00:0{2 * i + 1}+00:00"
        body = {
            "model": "DeepSeek-V4-Flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "decision": "stay",
                                "target_style": "fundamental",
                                "rationale": "SYNTHETIC TEST",
                            }
                        ),
                    },
                }
            ],
        }
        requests.append(r)
        attempts.append({**identity, "retry_count": 0, "dispatched_at": start})
        responses.append(
            {
                **identity,
                "retry_count": 0,
                "cached": False,
                "http_status": 200,
                "requested_at": start,
                "response_received_at": end,
                "latency_seconds": 10,
                "response": body,
            }
        )
    return m, requests, attempts, responses


def evaluate(materials, records):
    return evaluate_phase(*materials[:3], *records)


def test_only_mode_controls_added_and_old_inputs_remain_unchanged(materials):
    cases, base, sources, _ = materials
    snapshot = copy.deepcopy(base)
    assert validate_variant(cases, base, sources)["all_original_messages_verbatim"]
    for old, new in zip(base, cases, strict=True):
        stripped = copy.deepcopy(new["payload"])
        assert stripped.pop("chat_template_kwargs") == {"thinking": False, "enable_thinking": False}
        assert stripped == old["payload"]
    assert base == snapshot


@pytest.mark.parametrize(
    "alter", ["message", "tokens", "model", "extra", "boolean_type", "case_id"]
)
def test_any_change_beyond_the_fixed_mode_addition_fails(materials, alter):
    cases, base, sources, _ = copy.deepcopy(materials)
    if alter == "message":
        cases[0]["payload"]["messages"][0]["content"] += " changed"
    elif alter == "tokens":
        cases[0]["payload"]["max_tokens"] = 1024
    elif alter == "model":
        cases[0]["payload"]["model"] = "different"
    elif alter == "extra":
        cases[0]["payload"]["thinking"] = {"type": "disabled"}
    elif alter == "boolean_type":
        cases[0]["payload"]["chat_template_kwargs"]["thinking"] = 0
    else:
        cases[0]["case_id"] = "different"
    cases[0]["payload_sha256"] = payload_digest(cases[0]["payload"])
    with pytest.raises(ValueError):
        validate_variant(cases, base, sources)


def test_empty_phase_separates_two_unsubmitted_from_two_not_scheduled(materials):
    m, *_ = logs(materials)
    r = evaluate(materials, (m, [], [], []))
    assert r["scheduled_missing"] == 2 and r["not_scheduled_count"] == 2
    assert r["submitted_count"] == 0 and r["choice_changed"] is None
    assert not r["four_case_behavior_study_complete"]
    assert [row["answer_contract_status"] for row in r["cases"]].count("not_scheduled") == 2


def test_two_answers_only_complete_the_compatibility_phase(materials):
    r = evaluate(materials, logs(materials))
    assert r["phase_status"] == "synthetic_compatibility_pair_passed"
    assert r["compatible_count"] == 2 and r["choice_changed"] is False
    assert not r["four_case_behavior_study_complete"] and not r["internal_thinking_disabled_proven"]


@pytest.mark.parametrize("signal", ["content", "tokens"])
def test_last_answer_reasoning_rejects_mode_even_with_valid_json(materials, signal):
    records = logs(materials)
    body = records[3][-1]["response"]
    if signal == "content":
        body["choices"][0]["message"]["reasoning_content"] = "synthetic thought"
    else:
        body["usage"] = {"completion_tokens_details": {"reasoning_tokens": 3}}
    r = evaluate(materials, records)
    assert r["phase_status"] == "incomplete" and r["choice_changed"] is None
    failed = next(row for row in r["cases"] if row["case_id"] == records[3][-1]["case_id"])
    assert failed["answer_contract_status"] == "valid_linked_answer"
    assert failed["mode_contract_status"] == "reasoning_observed"


@pytest.mark.parametrize("alter", ["unscheduled", "duplicate", "retry", "hash", "payload"])
def test_foreign_duplicate_or_changed_phase_logs_fail(materials, alter):
    records = logs(materials)
    if alter == "unscheduled":
        records[1][0]["case_id"] = next(
            c["case_id"] for c in materials[0] if c["pair_id"] != "A001_2026-06-29"
        )
    elif alter == "duplicate":
        records[3].append(copy.deepcopy(records[3][0]))
    elif alter == "retry":
        records[2][0]["retry_count"] = 1
    elif alter == "hash":
        records[3][0]["request_hash"] = "f" * 64
    else:
        records[1][0]["payload"]["max_tokens"] = 1024
    with pytest.raises(ValueError):
        evaluate(materials, records)


def test_second_request_after_a_failed_first_answer_is_not_accepted(materials):
    records = logs(materials)
    records[3][0]["response"]["choices"][0]["message"]["content"] = "bad json"
    with pytest.raises(ValueError, match="stop-on-first"):
        evaluate(materials, records)


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_invalid_mode_usage_is_not_treated_as_zero_reasoning(value):
    assert (
        returned_mode({"usage": {"completion_tokens_details": {"reasoning_tokens": value}}})
        == "invalid_mode_evidence"
    )


@pytest.mark.parametrize("reasoning_at", [None, 0, 1])
def test_actual_client_uses_at_most_two_posts_and_stops_on_either_reasoning_signal(
    materials, tmp_path, reasoning_at
):
    cases, base, _sources, base_config = materials
    config = v2_config(base_config, base, cases)
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
        posts.append(json.loads(request.content))
        body = copy.deepcopy(logs(materials)[3][0]["response"])
        if reasoning_at is not None and len(posts) == reasoning_at + 1:
            body["usage"] = {"completion_tokens_details": {"reasoning_tokens": 5}}
        return httpx.Response(200, json=body)

    async def run():
        client = OpenAICompatibleClient(
            config, "synthetic-test-key", tmp_path, tmp_path / "cache", cache_namespace=NAMESPACE
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        m = phase_manifest(cases, "mock-v2", "synthetic_tool_test")
        try:
            return await collect_phase(cases, m, client, tmp_path)
        finally:
            await client.close()

    state = asyncio.run(run())
    assert len(posts) == (2 if reasoning_at is None else reasoning_at + 1)
    assert all(p["chat_template_kwargs"] == MODE_CONTROL for p in posts)
    assert not state["four_case_behavior_study_complete"]
    assert state["phase_status"] == (
        "synthetic_compatibility_pair_passed" if reasoning_at is None else "failed"
    )
    result = evaluate_phase(
        cases,
        base,
        materials[2],
        phase_manifest(cases, "mock-v2", "synthetic_tool_test"),
        read_jsonl(tmp_path / "llm_requests.jsonl"),
        read_jsonl(tmp_path / "llm_attempts.jsonl"),
        read_jsonl(tmp_path / "llm_responses.jsonl"),
    )
    assert result["compatible_count"] == (2 if reasoning_at is None else reasoning_at)
    assert not result["four_case_behavior_study_complete"]
