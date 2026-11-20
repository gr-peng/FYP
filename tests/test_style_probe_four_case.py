import asyncio
import copy
import json

import httpx
import pytest
from test_style_probe_execution import setup as legacy_setup  # noqa: F401
from test_style_probe_mode_v2 import logs as fundamental_logs
from test_style_probe_mode_v2 import materials as v2_materials  # noqa: F401

from behavioral_market.evaluation.style_probe_four_case import combine_phases
from behavioral_market.evaluation.style_probe_mode_v2 import (
    NAMESPACE,
    expected_request,
    phase_manifest,
)
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.evaluate_style_share_probe import read_jsonl
from scripts.style_probe_mode_v2 import collect_phase, v2_config
from scripts.summarize_style_probe_v2 import readable_report

TECHNICAL_PAIR = "A000_2026-06-29"


@pytest.fixture
def probe(request):
    return request.getfixturevalue("v2_materials")


def technical_logs(probe, kind="synthetic_tool_test", run_id="technical-unit"):
    cases = probe[0]
    by_id = {c["case_id"]: c for c in cases}
    m = phase_manifest(cases, run_id, kind, pair_id=TECHNICAL_PAIR)
    requests = []
    attempts = []
    responses = []
    for i, cid in enumerate(m["scheduled_case_ids"]):
        request = expected_request(by_id[cid], m)
        origin = {k: v for k, v in request.items() if k != "payload"}
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
                                "decision": "switch",
                                "target_style": "fundamental",
                                "rationale": "SYNTHETIC TEST",
                            }
                        ),
                    },
                }
            ],
        }
        requests.append(request)
        attempts.append({**origin, "retry_count": 0, "dispatched_at": start})
        responses.append(
            {
                **origin,
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


def batch(records):
    return dict(zip(("manifest", "requests", "attempts", "responses"), records, strict=True))


def test_only_fixed_extension_pair_is_allowed_and_default_manifest_shape_preserved(probe):
    cases = probe[0]
    default = phase_manifest(cases, "fundamental-unit", "synthetic_tool_test")
    assert "extension_pair_id" not in default
    assert all(cid.startswith("A001_") for cid in default["scheduled_case_ids"])
    technical = technical_logs(probe)[0]
    assert technical["extension_pair_id"] == TECHNICAL_PAIR
    assert all(cid.startswith("A000_") for cid in technical["scheduled_case_ids"])
    with pytest.raises(ValueError):
        phase_manifest(cases, "bad", "synthetic_tool_test", pair_id="A002_2026-06-29")


def test_four_records_with_one_fixed_target_are_not_mislabelled_as_herding(probe):
    result = combine_phases(
        *probe[:3], [batch(fundamental_logs(probe)), batch(technical_logs(probe))]
    )
    assert result["tiny_four_case_probe_complete"] and result["valid_compatible_choices"] == 4
    assert result["changed_pairs"] == 0
    assert result["all_valid_choices_target_same_style"] == "fundamental"
    assert (
        not result["formal_behavior_study_complete"] and not result["human_alignment_established"]
    )
    assert "herding_rate" not in result and "majority_follow_rate" not in result
    technical = next(p for p in result["pairs"] if p["from_style"] == "technical")
    assert technical["own_majority"]["choice"]["decision"] == "switch"
    assert technical["other_majority"]["choice"]["decision"] == "switch"
    report = readable_report(result)
    assert "四条都选择了基本面策略" in report and "合成工具测试" in report


def test_missing_technical_phase_preserves_two_missing_and_null_global_result(probe):
    m, *_ = technical_logs(probe)
    result = combine_phases(*probe[:3], [batch(fundamental_logs(probe)), batch((m, [], [], []))])
    assert result["valid_compatible_choices"] == 2 and result["missing_scheduled_cases"] == 2
    assert result["comparable_pairs"] == 1 and result["changed_pairs"] is None
    assert not result["tiny_four_case_probe_complete"]
    assert "无法比较" in readable_report(result)


def test_missing_last_response_does_not_turn_into_zero_effect(probe):
    technical = technical_logs(probe)
    technical[3].pop()
    result = combine_phases(*probe[:3], [batch(fundamental_logs(probe)), batch(technical)])
    assert result["valid_compatible_choices"] == 3 and result["missing_scheduled_cases"] == 1
    assert result["changed_pairs"] is None


@pytest.mark.parametrize(
    "problem",
    ["duplicate_phase", "duplicate_answer", "wrong_materials", "mixed_evidence", "claimed_summary"],
)
def test_invalid_joint_sources_cannot_be_averaged_into_a_complete_probe(probe, problem):
    first = batch(fundamental_logs(probe))
    second = batch(technical_logs(probe))
    if problem == "duplicate_phase":
        second = copy.deepcopy(first)
    elif problem == "duplicate_answer":
        second["responses"].append(copy.deepcopy(second["responses"][0]))
    elif problem == "wrong_materials":
        second["manifest"]["candidates_sha256"] = "f" * 64
    elif problem == "mixed_evidence":
        second = batch(technical_logs(probe, kind="recorded_model_logs"))
    else:
        second["summary"] = {"complete": True}
    with pytest.raises(ValueError):
        combine_phases(*probe[:3], [first, second])


def test_technical_live_adapter_is_bounded_and_matches_independent_log_evaluation(probe, tmp_path):
    cases, base, sources, base_config = probe
    config = v2_config(base_config, base, cases)
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
        posts.append(json.loads(request.content))
        return httpx.Response(200, json=technical_logs(probe)[3][0]["response"])

    async def run():
        client = OpenAICompatibleClient(
            config, "synthetic-test-key", tmp_path, tmp_path / "cache", cache_namespace=NAMESPACE
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        m = technical_logs(probe)[0]
        try:
            return await collect_phase(cases, m, client, tmp_path)
        finally:
            await client.close()

    state = asyncio.run(run())
    assert len(posts) == 2 and state["phase_status"] == "synthetic_compatibility_pair_passed"
    technical = batch(
        (
            technical_logs(probe)[0],
            read_jsonl(tmp_path / "llm_requests.jsonl"),
            read_jsonl(tmp_path / "llm_attempts.jsonl"),
            read_jsonl(tmp_path / "llm_responses.jsonl"),
        )
    )
    result = combine_phases(cases, base, sources, [batch(fundamental_logs(probe)), technical])
    assert result["tiny_four_case_probe_complete"]
