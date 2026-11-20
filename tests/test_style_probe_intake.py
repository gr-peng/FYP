import copy
import hashlib
import json

import pytest

from behavioral_market.evaluation.style_probe_intake import (
    NAMESPACE,
    evaluate_recorded_probe,
    evaluate_synthetic_probe,
    expected_request,
    make_run_manifest,
)
from behavioral_market.evaluation.style_probe_materials import build_share_probes, payload_digest
from scripts.evaluate_style_share_probe import strict_json


@pytest.fixture
def materials():
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
            "model": "unit-test-only",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "unit test"},
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
    return build_share_probes(sources), sources


def fixture_logs(cases, kind="synthetic_tool_test"):
    """Construct local test data, including tests of the recorded-log shape; never a model call."""
    manifest = make_run_manifest(cases, "unit-only", kind)
    requests, attempts, responses = [], [], []
    for case in cases:
        request = expected_request(case, manifest)
        identity = {k: v for k, v in request.items() if k != "payload"}
        requests.append(request)
        attempts.append(
            {**identity, "retry_count": 0, "dispatched_at": "2026-10-22T00:00:00+00:00"}
        )
        switched = (
            case["source"]["from_style"] == "fundamental"
            and case["condition"] == "other_style_majority"
        )
        target = "technical" if switched else case["source"]["from_style"]
        body = {
            "model": case["payload"]["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "decision": "switch" if switched else "stay",
                                "target_style": target,
                                "rationale": "SYNTHETIC UNIT TEST, not a model answer",
                            }
                        ),
                    },
                }
            ],
        }
        responses.append(
            {
                **identity,
                "retry_count": 0,
                "requested_at": attempts[-1]["dispatched_at"],
                "response_received_at": "2026-10-22T00:00:01+00:00",
                "cached": False,
                "http_status": 200,
                "latency_seconds": 100,
                "response": body,
            }
        )
    return manifest, requests, attempts, responses


def run(materials, logs):
    return evaluate_synthetic_probe(*materials, *logs)


def test_empty_logs_preserve_four_missing_opportunities_and_null_effects(materials):
    cases, sources = materials
    manifest = make_run_manifest(cases, "not-started", "recorded_model_logs")
    result = evaluate_recorded_probe(cases, sources, manifest, [], [], [])
    assert result["execution_status"] == "not_started"
    assert result["opportunities"] == result["missing_cases"] == 4
    assert result["pair_opportunities"] == 2 and result["comparable_pairs"] == 0
    assert result["changed_pairs"] is None
    assert all(r["choice"] is None for r in result["cases"])
    assert all(p["choice_changed"] is None for p in result["pairs"])


def test_complete_synthetic_logs_compare_pairs_without_becoming_research_results(materials):
    result = run(materials, fixture_logs(materials[0]))
    assert result["execution_status"] == "synthetic_test_complete"
    assert result["valid_choice_count"] == 4 and result["comparable_pairs"] == 2
    assert result["changed_pairs"] == 1
    assert result["evidence_scope"] == "synthetic_tool_test_no_model_execution"
    assert not result["research_eligible"] and not result["real_model_execution_verified"]


def test_client_request_hash_is_distinct_from_payload_only_hash(materials):
    cases, _ = materials
    manifest, requests, _, _ = fixture_logs(cases)
    identity = {k: v for k, v in requests[0].items() if k not in {"payload", "request_hash"}}
    digest = hashlib.sha256(
        json.dumps([cases[0]["payload"], identity, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    assert requests[0]["request_hash"] == digest
    assert digest != cases[0]["payload_sha256"]
    assert manifest["materials_sha256"] == payload_digest(cases)


@pytest.mark.parametrize(
    "phase,expected", [("request", "request_record_only"), ("attempt", "awaiting_response")]
)
def test_partial_execution_does_not_become_default_stay(materials, phase, expected):
    m, requests, attempts, responses = fixture_logs(materials[0])
    result = run(materials, (m, requests[:1], attempts[:1] if phase == "attempt" else [], []))
    assert result["cases"][0]["status"] == expected
    assert result["cases"][0]["choice"] is None
    assert result["valid_choice_count"] == 0 and result["changed_pairs"] is None


def test_single_sided_pair_is_not_reported_as_no_population_effect(materials):
    m, requests, attempts, responses = fixture_logs(materials[0])
    result = run(materials, (m, requests[:1], attempts[:1], responses[:1]))
    assert result["valid_choice_count"] == 1
    assert result["missing_cases"] == 3 and result["comparable_pairs"] == 0
    assert all(p["choice_changed"] is None for p in result["pairs"])


@pytest.mark.parametrize(
    "alter", ["case", "payload", "hash", "repair", "duplicate", "model_settings"]
)
def test_wrong_requests_cannot_be_joined_to_frozen_cases(materials, alter):
    logs = fixture_logs(materials[0])
    request = logs[1][0]
    if alter == "case":
        request["case_id"] = logs[1][1]["case_id"]
    elif alter == "payload":
        request["payload"]["messages"][1]["content"] += " "
    elif alter == "hash":
        request["request_hash"] = materials[0][0]["payload_sha256"]
    elif alter == "repair":
        request["repair"] = 1
    elif alter == "model_settings":
        request["payload"]["temperature"] = 0.9
    else:
        logs[1].append(copy.deepcopy(request))
    with pytest.raises(ValueError):
        run(materials, logs)


@pytest.mark.parametrize("alter", ["hash", "case", "retry", "repair", "clock", "duplicate"])
def test_response_must_match_its_request_and_attempt(materials, alter):
    logs = fixture_logs(materials[0])
    response = logs[3][0]
    if alter == "hash":
        response["request_hash"] = "f" * 64
    elif alter == "case":
        response["case_id"] = logs[1][1]["case_id"]
    elif alter == "retry":
        response["retry_count"] = 1
    elif alter == "repair":
        response["repair"] = 1
    elif alter == "clock":
        response["requested_at"] = "2026-10-22T00:00:01+00:00"
    else:
        logs[3].append(copy.deepcopy(response))
    with pytest.raises(ValueError):
        run(materials, logs)


@pytest.mark.parametrize(
    "alter,expected",
    [
        ("schema", "invalid_answer_schema"),
        ("coherence", "inconsistent_style"),
        ("model", "model_mismatch"),
        ("truncated", "invalid_response_envelope"),
    ],
)
def test_response_errors_remain_explicit_and_do_not_produce_pair_effects(
    materials, alter, expected
):
    logs = fixture_logs(materials[0])
    body = logs[3][0]["response"]
    if alter == "schema":
        body["choices"][0]["message"]["content"] = '{"decision":"stay"}'
    elif alter == "coherence":
        body["choices"][0]["message"]["content"] = json.dumps(
            {"decision": "stay", "target_style": "technical", "rationale": "test"}
        )
    elif alter == "model":
        body["model"] = "another-model"
    else:
        body["choices"][0]["finish_reason"] = "length"
    result = run(materials, logs)
    assert result["cases"][0]["status"] == expected
    assert not result["cases"][0]["valid_choice"]
    assert result["error_cases"] == 1 and result["comparable_pairs"] == 1
    assert result["changed_pairs"] is None


def test_transport_failure_keeps_attempts_without_inventing_a_choice(materials):
    logs = fixture_logs(materials[0])
    logs[3][0].update(http_status=503, response=None)
    result = run(materials, logs)
    assert result["cases"][0]["status"] == "transport_failed"
    assert result["cases"][0]["choice"] is None
    assert result["failed_transport_responses"] == 1


def test_transport_retry_keeps_failure_and_accepts_only_the_single_success(materials):
    logs = fixture_logs(materials[0])
    _, _, attempts, responses = logs
    retry_attempt = {**attempts[0], "retry_count": 1, "dispatched_at": "2026-10-22T00:00:02+00:00"}
    retry_response = {
        **copy.deepcopy(responses[0]),
        "retry_count": 1,
        "requested_at": retry_attempt["dispatched_at"],
        "response_received_at": "2026-10-22T00:00:03+00:00",
    }
    attempts.append(retry_attempt)
    responses[0].update(http_status=429, response=None)
    responses.append(retry_response)
    result = run(materials, logs)
    assert result["valid_choice_count"] == 4
    assert result["logged_transport_attempts"] == 5 and result["failed_transport_responses"] == 1
    responses[0].update(http_status=200, response=copy.deepcopy(retry_response["response"]))
    with pytest.raises(ValueError, match="Multiple successful"):
        run(materials, logs)


def test_cached_answer_is_not_a_new_executed_choice(materials):
    m, requests, _, responses = fixture_logs(materials[0])
    identity = {k: v for k, v in requests[0].items() if k != "payload"}
    cached = {**identity, "cached": True, "status": "ok", "response": responses[0]["response"]}
    result = run(materials, (m, requests[:1], [], [cached]))
    assert result["cache_replays"] == 1 and result["valid_choice_count"] == 0
    assert result["cases"][0]["status"] == "cache_replay_not_new_execution"
    assert result["changed_pairs"] is None


def test_changing_only_the_test_label_cannot_promote_fixture_logs(materials):
    logs = fixture_logs(materials[0])
    logs[0]["evidence_kind"] = "recorded_model_logs"
    with pytest.raises(ValueError):
        evaluate_recorded_probe(*materials, *logs)


def test_self_declared_real_log_shape_does_not_authenticate_external_execution(materials):
    logs = fixture_logs(materials[0], kind="recorded_model_logs")
    result = evaluate_recorded_probe(*materials, *logs)
    assert result["execution_status"] == "local_logs_complete"
    assert result["evidence_scope"] == "local_execution_logs_linked_not_externally_authenticated"
    assert not result["real_model_execution_verified"] and not result["research_eligible"]


@pytest.mark.parametrize(
    "field,value",
    [("repair", False), ("max_transport_attempts", 3.0), ("protocol_version", "legacy")],
)
def test_manifest_contract_is_strict_even_for_equal_looking_json_numbers(materials, field, value):
    logs = fixture_logs(materials[0])
    logs[0][field] = value
    with pytest.raises(ValueError):
        run(materials, logs)


@pytest.mark.parametrize(
    "text", ['{"case_id":"a","case_id":"b"}', '{"latency_seconds":NaN}', '{"x":Infinity}']
)
def test_ambiguous_or_nonfinite_json_logs_are_rejected(text):
    with pytest.raises(ValueError):
        strict_json(text)


@pytest.mark.parametrize("field", ["decision", "target_style", "rationale"])
def test_duplicate_keys_inside_the_actual_answer_are_not_resolved_by_picking_the_last(
    materials, field
):
    logs = fixture_logs(materials[0])
    raw = '{"decision":"stay","target_style":"fundamental","rationale":"synthetic"}'
    insertion = {
        "decision": '"decision":"switch",',
        "target_style": '"target_style":"technical",',
        "rationale": '"rationale":"another explanation",',
    }[field]
    logs[3][0]["response"]["choices"][0]["message"]["content"] = "{" + insertion + raw[1:]
    result = run(materials, logs)
    assert result["cases"][0]["status"] == "invalid_answer_schema"
    assert result["cases"][0]["choice"] is None
    assert result["changed_pairs"] is None


def test_overflowed_json_number_is_nonfinite_even_without_a_nan_constant():
    with pytest.raises(ValueError, match="Nonfinite"):
        strict_json('{"latency_seconds":1e9999}')
