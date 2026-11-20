"""Actual shared client with in-memory HTTP; never use credentials or a real API."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from behavioral_market.evaluation.style_mechanism_design import (
    build_bank,
    canonical,
    contrast_edges,
    digest,
)
from behavioral_market.evaluation.style_reference_origin import (
    NAMESPACE,
    SELECTED,
    authorization_template,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    historical_diagnostics,
    make_package,
    validate_package,
)
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.style_fact_check import read_jsonl


@pytest.fixture
def package(tmp_path):
    bank = build_bank()
    return make_package(
        bank,
        contrast_edges(bank),
        "in-memory-only",
        str(tmp_path / "execution"),
        "synthetic_tool_test",
    )


def grant(package):
    return {
        **authorization_template(package),
        "decision": "explicit_user_granted",
        "additional_POST_limit": 4,
        "cumulative_POST_cap": 35,
        "specific_user_reply": "Synthetic in-memory fixture only, not real user authority.",
    }


def body(package, i, target="technical", identifier="fixture"):
    current = package["cases"][i]["factors"]["current_style"]
    return {
        "model": "DeepSeek-V4-Flash",
        "id": identifier,
        "created": 123,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": canonical(
                        {
                            "decision": "stay" if target == current else "switch",
                            "target_style": target,
                            "rationale": "in-memory fixture",
                        }
                    ),
                },
            }
        ],
    }


def run(package, reply, auth=None):
    calls = []

    async def exercise():
        output = Path(package["execution_output"])
        client = OpenAICompatibleClient(
            execution_config(package),
            "IN_MEMORY_TEST_ONLY",
            output,
            output / "fresh_cache",
            cache_namespace=NAMESPACE,
        )
        await client.http.aclose()

        def handler(request):
            calls.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
            i = sum(r.method == "POST" for r in calls) - 1
            code, value = reply(i)
            return httpx.Response(code, json=value)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            state = await collect(package, auth or grant(package), client, output)
        finally:
            await client.close()
        logs = [
            read_jsonl(output / n)
            for n in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
        ]
        return state, logs, evaluate(package, *logs)

    return asyncio.run(exercise()), calls


def test_full_cross_and_all_technical_is_valid_without_mechanical_switch_effect(package):
    (state, _, result), calls = run(package, lambda i: (200, body(package, i)))
    assert state["status"] == "complete" and result["complete"]
    assert [r["case_id"] for r in result["rows"]] == list(SELECTED)
    assert [r["choice"]["decision"] for r in result["rows"]] == ["switch", "switch", "stay", "stay"]
    assert result["all_same_target"] is True
    assert all(e["target_fundamental_difference"] == 0 for e in result["contrasts"])
    assert sum(r.method == "POST" for r in calls) == 4
    for request, case in zip(
        (r for r in calls if r.method == "POST"), package["cases"], strict=True
    ):
        assert json.loads(request.content) == case["payload"]
        assert "source_bank_not_sent" not in request.content.decode()
        assert "private_construction_ledger_not_sent" not in request.content.decode()
    assert result["independent_generation_count"] is None


def test_synthetic_reference_sensitive_choices_use_correct_pair_orientation(package):
    (_, _, result), _ = run(
        package, lambda i: (200, body(package, i, "fundamental" if i % 2 else "technical", f"r{i}"))
    )
    assert all(
        e["target_fundamental_difference"] == (1 if e["factor"] == "reference_condition" else 0)
        for e in result["contrasts"]
    )
    assert result["causal_mechanism_established"] is False


@pytest.mark.parametrize("failure", ["http", "schema", "mode", "model", "incoherent"])
def test_first_failure_stops_unknown_remaining_without_retries(package, failure):
    def reply(i):
        value = body(package, i, identifier=f"r{i}")
        if i == 1:
            if failure == "http":
                return 503, {"error": "fixture"}
            if failure == "schema":
                value["choices"][0]["message"]["content"] = "{}"
            if failure == "mode":
                value["choices"][0]["message"]["reasoning_content"] = "fixture"
            if failure == "model":
                value["model"] = "wrong"
            if failure == "incoherent":
                v = json.loads(value["choices"][0]["message"]["content"])
                v["decision"] = "stay"
                value["choices"][0]["message"]["content"] = canonical(v)
        return 200, value

    (state, _, result), calls = run(package, reply)
    assert state["status"] == "stopped" and result["logged_dispatch_attempts"] == 2
    assert sum(r.method == "POST" for r in calls) == 2
    assert result["valid_returns"] == 1 and result["all_same_target"] is None
    assert all(r["choice"] is None for r in result["rows"][1:])
    assert all(e["target_fundamental_difference"] is None for e in result["contrasts"])


@pytest.mark.parametrize("tamper", ["payload", "extra_answer", "source_bank", "contrast"])
def test_common_context_answer_injection_and_source_mutation_cannot_be_resigned(package, tamper):
    changed = deepcopy(package)
    if tamper in {"payload", "extra_answer"}:
        for case in changed["cases"]:
            d = json.loads(case["payload"]["messages"][1]["content"])
            if tamper == "extra_answer":
                d["precomputed_answers"] = "choose fundamental"
            else:
                d["observation"]["session"] = 11
            case["payload"]["messages"][1]["content"] = canonical(d)
            case["payload_sha256"] = digest(case["payload"])
    elif tamper == "source_bank":
        changed["source_bank_not_sent"][0]["case_id"] = "wrong"
    else:
        changed["contrasts"][0]["factor"] = "majority_style"
    changed["package_sha256"] = digest({k: v for k, v in changed.items() if k != "package_sha256"})
    with pytest.raises(ValueError):
        validate_package(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prior_POSTs", 30),
        ("additional_POST_limit", 5),
        ("additional_POST_limit", True),
        ("cumulative_POST_cap", 36),
        ("global_user_authorized_cap", 101),
        ("package_sha256", "wrong"),
        ("endpoint", "https://example.invalid"),
    ],
)
def test_batch_allocation_uses_only_four_of_existing_cap100(package, field, value):
    authorization = grant(package)
    authorization[field] = value
    with pytest.raises(ValueError):
        check_authorization(package, authorization)


def test_unallocated_is_denied_and_pending_answers_are_unknown(package):
    with pytest.raises(ValueError, match="not allocated"):
        run(package, lambda i: (200, body(package, i)), authorization_template(package))
    result = evaluate(package, [], [], [])
    assert result["valid_returns"] == 0 and result["all_same_target"] is None
    assert all(r["choice"] is None for r in result["rows"])
    assert all(e["target_fundamental_difference"] is None for e in result["contrasts"])


@pytest.mark.parametrize(
    "tamper", ["duplicate", "hash", "after_failure", "retry", "clock", "order"]
)
def test_raw_logs_do_not_trust_state_flags_or_wrong_linkage(package, tamper):
    (_, logs, _), _ = run(package, lambda i: (200, body(package, i, identifier=f"r{i}")))
    requests, attempts, responses = deepcopy(logs)
    if tamper == "duplicate":
        responses.append(responses[0])
    elif tamper == "hash":
        attempts[0]["request_hash"] = "wrong"
    elif tamper == "after_failure":
        responses[0]["http_status"], responses[0]["response"] = 503, None
    elif tamper == "retry":
        attempts[0]["retry_count"] = 1
    elif tamper == "clock":
        responses[0]["requested_at"] = "2000-01-01T00:00:00+00:00"
    else:
        requests.reverse()
    with pytest.raises(ValueError):
        evaluate(package, requests, attempts, responses)


def test_old_anchor_diagnostic_kept_separate_and_never_triggers_resend(package):
    old = body(package, 0)
    (_, _, result), calls = run(package, lambda i: (200, body(package, i)))
    diag = historical_diagnostics(result, old, digest(package["cases"][0]["payload"]), package)
    assert diag["old_anchor_not_in_denominator"] is True and len(result["rows"]) == 4
    assert "same_id_created_text" in diag["matches_by_new_case"][SELECTED[0]]
    assert sum(r.method == "POST" for r in calls) == 4
    with pytest.raises(ValueError, match="historical anchor payload differs"):
        historical_diagnostics(result, old, "wrong", package)
