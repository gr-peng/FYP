import asyncio
import copy
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_style_probe_execution import setup as legacy_setup  # noqa: F401
from test_style_probe_mode_v2 import materials as v2_materials  # noqa: F401

from behavioral_market.evaluation.style_probe_retest import (
    NAMESPACE,
    evaluate_retest,
    expected_request,
    planned_schedule,
    run_manifest,
    validate_schedule,
)
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.evaluate_style_share_probe import read_jsonl
from scripts.style_probe_mode_v2 import v2_config
from scripts.style_probe_retest import collect_retest, readable_report


@pytest.fixture
def probe(request):
    return request.getfixturevalue("v2_materials")


def envelope(style, target=None):
    target = target or style
    return {
        "model": "DeepSeek-V4-Flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": "stay" if style == target else "switch",
                            "target_style": target,
                            "rationale": "SYNTHETIC TOOL TEST",
                        }
                    ),
                },
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    }


def logs(probe, count=16, targets=None):
    planned = planned_schedule(probe[0])
    manifest = run_manifest(probe[0], planned, "unit-retest", "synthetic_tool_test")
    by_id = {c["case_id"]: c for c in probe[0]}
    requests, attempts, responses = [], [], []
    for i, slot in enumerate(planned[:count]):
        request = expected_request(slot, manifest)
        origin = {k: v for k, v in request.items() if k != "payload"}
        start = datetime(2026, 10, 22, tzinfo=UTC) + timedelta(seconds=2 * i)
        finish = start + timedelta(seconds=1)
        style = by_id[slot["case_id"]]["source"]["from_style"]
        requests.append(request)
        attempts.append({**origin, "retry_count": 0, "dispatched_at": start.isoformat()})
        responses.append(
            {
                **origin,
                "retry_count": 0,
                "cached": False,
                "http_status": 200,
                "requested_at": start.isoformat(),
                "response_received_at": finish.isoformat(),
                "response": envelope(style, (targets or {}).get(slot["planned_id"])),
            }
        )
    return manifest, requests, attempts, responses


def evaluate(probe, records):
    return evaluate_retest(*probe[:3], planned_schedule(probe[0]), *records)


def test_repeats_have_distinct_local_hashes_and_balanced_unchanged_payloads(probe):
    planned = planned_schedule(probe[0])
    manifest = run_manifest(probe[0], planned, "unit-retest", "synthetic_tool_test")
    originals = {case["case_id"]: case for case in probe[0]}
    assert Counter(slot["case_id"] for slot in planned) == Counter(dict.fromkeys(originals, 4))
    assert len({expected_request(s, manifest)["request_hash"] for s in planned}) == 16
    for cid, original in originals.items():
        group = [s for s in planned if s["case_id"] == cid]
        assert sorted(s["position"] for s in group) == [1, 2, 3, 4]
        assert all(s["payload"] == original["payload"] for s in group)
    sequences = [
        "".join(s["code"] for s in planned if s["repeat"] == repeat) for repeat in range(1, 5)
    ]
    assert sequences == ["ABDC", "BCAD", "CDBA", "DACB"]
    for own, other in [("A", "B"), ("C", "D")]:
        assert sum(s.index(own) < s.index(other) for s in sequences) == 2


@pytest.mark.parametrize("alter", ["order", "prompt", "missing", "repeat_in_prompt"])
def test_any_schedule_or_payload_change_rejected(probe, alter):
    planned = planned_schedule(probe[0])
    if alter == "order":
        planned[0], planned[1] = planned[1], planned[0]
    elif alter == "missing":
        planned.pop()
    else:
        planned[0]["payload"]["messages"][1]["content"] += (
            " repeat 1" if alter == "repeat_in_prompt" else " changed"
        )
    with pytest.raises(ValueError):
        validate_schedule(probe[0], planned)


def test_empty_result_retains_all_sixteen_missing_and_null_context_comparisons(probe):
    result = evaluate(probe, logs(probe, 0))
    assert result["scheduled_count"] == result["missing_count"] == 16
    assert result["valid_count"] == result["error_count"] == result["logged_post_attempts"] == 0
    assert all(
        c["missing_count"] == 4 and c["variation_observed"] is None for c in result["conditions"]
    )
    assert all(p["changed_rounds"] is None for p in result["pairs"])
    assert not result["retest_complete"]


def test_complete_fixed_choices_and_changed_reasons_do_not_claim_variability_or_herding(probe):
    records = logs(probe)
    for i, response in enumerate(records[3]):
        raw = json.loads(response["response"]["choices"][0]["message"]["content"])
        raw["rationale"] = f"DIFFERENT SYNTHETIC REASON {i}"
        response["response"]["choices"][0]["message"]["content"] = json.dumps(raw)
    result = evaluate(probe, records)
    assert result["retest_complete"] and result["valid_count"] == 16
    assert all(c["variation_observed"] is False for c in result["conditions"])
    assert all(p["changed_rounds"] == 0 for p in result["pairs"])
    assert result["discovery_answers_in_primary_counts"] == 0
    assert not any(
        result[k]
        for k in (
            "human_alignment_established",
            "causal_effect_established",
            "independent_sampling_established",
            "research_eligible",
        )
    )
    assert "herding_rate" not in result
    report = readable_report(result)
    assert "合成工具测试" in report and "不是16名投资者" in report and "这四次相同" in report


def test_same_input_variation_distinguished_from_condition_and_incomplete_pair(probe):
    records = logs(probe, targets={"r02-p3-A": "technical"})
    result = evaluate(probe, records)
    a = next(
        c
        for c in result["conditions"]
        if c["case_id"].endswith("own_style_majority") and c["from_style"] == "fundamental"
    )
    assert a["decision_counts"] == {"stay": 3, "switch": 1}
    assert a["variation_observed"] is True
    # The final B slot is the last fundamental answer; the technical pair is complete.
    incomplete = evaluate(probe, logs(probe, 15))
    fundamental = next(p for p in incomplete["pairs"] if p["from_style"] == "fundamental")
    technical = next(p for p in incomplete["pairs"] if p["from_style"] == "technical")
    assert fundamental["changed_rounds"] is None and not incomplete["retest_complete"]
    assert technical["all_rounds_comparable"]


@pytest.mark.parametrize("stream", [1, 2, 3])
@pytest.mark.parametrize("alter", ["duplicate", "skip", "swap", "hash", "boolean_repeat"])
def test_duplicate_skipped_reordered_or_foreign_identity_rejected(probe, stream, alter):
    records = logs(probe)
    rows = records[stream]
    if alter == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alter == "skip":
        rows.pop(3)
    elif alter == "swap":
        rows[0], rows[1] = rows[1], rows[0]
    elif alter == "hash":
        rows[0]["request_hash"] = "f" * 64
    else:
        rows[0]["repeat"] = True
    with pytest.raises(ValueError):
        evaluate(probe, records)


@pytest.mark.parametrize("failure", ["json", "reasoning", "http", "missing", "overlap"])
def test_continuing_after_failure_or_missing_answer_rejected(probe, failure):
    records = logs(probe)
    body = records[3][0]["response"]
    if failure == "json":
        body["choices"][0]["message"]["content"] = "bad json"
    elif failure == "reasoning":
        body["usage"]["completion_tokens_details"]["reasoning_tokens"] = 7
    elif failure == "http":
        records[3][0].update(http_status=503, response=None)
    elif failure == "missing":
        records[3].pop(0)
    else:
        start = records[2][0]["dispatched_at"]
        records[2][1]["dispatched_at"] = records[3][1]["requested_at"] = start
    with pytest.raises(ValueError):
        evaluate(probe, records)


def test_last_failure_is_error_not_missing_and_prevents_complete_context_result(probe):
    records = logs(probe)
    records[3][-1]["response"]["choices"][0]["finish_reason"] = "length"
    result = evaluate(probe, records)
    assert (
        result["valid_count"] == 15 and result["error_count"] == 1 and result["missing_count"] == 0
    )
    assert any(p["changed_rounds"] is None for p in result["pairs"])
    assert not result["retest_complete"]


def test_last_inflight_missing_response_remains_missing_not_a_valid_choice(probe):
    records = logs(probe)
    records[3].pop()
    result = evaluate(probe, records)
    assert result["valid_count"] == 15 and result["missing_count"] == 1
    assert result["error_count"] == 0 and result["logged_post_attempts"] == 16
    assert result["slots"][-1]["answer_status"] == "awaiting_response"
    assert not result["retest_complete"]


def test_discovery_logs_or_mixed_evidence_are_not_retest_answers(probe):
    from test_style_probe_mode_v2 import logs as discovery_logs

    manifest = logs(probe, 0)[0]
    with pytest.raises(ValueError):
        evaluate(probe, (manifest, *discovery_logs(probe)[1:]))
    records = logs(probe)
    records[0]["evidence_kind"] = "recorded_model_logs"
    with pytest.raises(ValueError):
        evaluate(probe, records)


def test_cache_replay_is_not_a_new_answer_or_completed_retest(probe):
    records = logs(probe, 1)
    records[2].clear()
    records[3][0]["cached"] = True
    result = evaluate(probe, records)
    assert result["valid_count"] == result["logged_post_attempts"] == 0
    assert result["cache_replays"] == result["error_count"] == 1
    assert result["missing_count"] == 15


@pytest.mark.parametrize(
    "at,kind",
    [
        (None, "ok"),
        (0, "json"),
        (7, "reasoning"),
        (15, "reasoning"),
        (3, "http"),
        (2, "wrong_model"),
    ],
)
def test_actual_client_global_order_distinct_keys_and_first_failure_stop(probe, tmp_path, at, kind):
    cases, base, sources, base_config = probe
    planned = planned_schedule(cases)
    manifest = run_manifest(cases, planned, "mock-retetest", "synthetic_tool_test")
    by_id = {c["case_id"]: c for c in cases}
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
        index = len(posts)
        posts.append(json.loads(request.content))
        slot = planned[index]
        body = envelope(by_id[slot["case_id"]]["source"]["from_style"])
        if index == at:
            if kind == "json":
                body["choices"][0]["message"]["content"] = "bad json"
            elif kind == "reasoning":
                body["usage"]["completion_tokens_details"]["reasoning_tokens"] = 5
            elif kind == "wrong_model":
                body["model"] = "other"
            else:
                return httpx.Response(503, text="SYNTHETIC HTTP FAILURE")
        return httpx.Response(200, json=body)

    async def run():
        client = OpenAICompatibleClient(
            v2_config(base_config, base, cases),
            "synthetic-test-key",
            tmp_path,
            tmp_path / "cache",
            cache_namespace=NAMESPACE,
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await collect_retest(cases, planned, manifest, client, tmp_path)
        finally:
            await client.close()

    state = asyncio.run(run())
    count = 16 if at is None else at + 1
    assert len(posts) == count
    assert posts == [s["payload"] for s in planned[:count]]
    requests = read_jsonl(tmp_path / "llm_requests.jsonl")
    assert len({r["request_hash"] for r in requests}) == count
    assert requests == [expected_request(s, manifest) for s in planned[:count]]
    result = evaluate_retest(
        cases,
        base,
        sources,
        planned,
        manifest,
        requests,
        read_jsonl(tmp_path / "llm_attempts.jsonl"),
        read_jsonl(tmp_path / "llm_responses.jsonl"),
    )
    assert result["valid_count"] == (16 if at is None else at)
    assert result["missing_count"] == 16 - count
    assert state["retest_complete"] is (at is None)
    assert len(state["slots"]) == 16


def test_collector_refuses_existing_log_or_cache_before_contacting_provider(probe, tmp_path):
    cases, base, _sources, config = probe
    planned = planned_schedule(cases)
    manifest = run_manifest(cases, planned, "no-resume", "synthetic_tool_test")
    (tmp_path / "llm_requests.jsonl").write_text("prior request\n")

    async def run():
        client = OpenAICompatibleClient(
            v2_config(config, base, cases),
            "synthetic-test-key",
            tmp_path,
            tmp_path / "cache",
            cache_namespace=NAMESPACE,
        )
        try:
            with pytest.raises(FileExistsError):
                await collect_retest(cases, planned, manifest, client, tmp_path)
            (tmp_path / "llm_requests.jsonl").unlink()
            (client.cache / "old.json").write_text("{}")
            with pytest.raises(FileExistsError):
                await collect_retest(cases, planned, manifest, client, tmp_path)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["registry", "cancelled"])
def test_preflight_failure_or_cancellation_stops_without_resume(probe, tmp_path, failure):
    cases, base, sources, config = probe
    planned = planned_schedule(cases)
    manifest = run_manifest(cases, planned, "failure-unit", "synthetic_tool_test")
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                503 if failure == "registry" else 200, json={"data": [{"id": "DeepSeek-V4-Flash"}]}
            )
        posts.append(request)
        raise asyncio.CancelledError()

    async def run():
        client = OpenAICompatibleClient(
            v2_config(config, base, cases),
            "synthetic-test-key",
            tmp_path,
            tmp_path / "cache",
            cache_namespace=NAMESPACE,
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            if failure == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await collect_retest(cases, planned, manifest, client, tmp_path)
                return json.loads((tmp_path / "execution_state.json").read_text())
            return await collect_retest(cases, planned, manifest, client, tmp_path)
        finally:
            await client.close()

    state = asyncio.run(run())
    assert len(posts) == (1 if failure == "cancelled" else 0)
    assert state["status"] == ("cancelled" if failure == "cancelled" else "stopped")
    result = evaluate_retest(
        cases,
        base,
        sources,
        planned,
        manifest,
        read_jsonl(tmp_path / "llm_requests.jsonl"),
        read_jsonl(tmp_path / "llm_attempts.jsonl"),
        read_jsonl(tmp_path / "llm_responses.jsonl"),
    )
    assert result["valid_count"] == 0 and not result["retest_complete"]
    assert result["error_count"] == (1 if failure == "cancelled" else 0)
    assert result["missing_count"] == (15 if failure == "cancelled" else 16)
