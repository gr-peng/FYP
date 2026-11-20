import copy
import json

import pytest

from behavioral_market.evaluation.archive_response_reuse import audit_archives
from behavioral_market.evaluation.style_probe_materials import payload_digest
from scripts.audit_archived_response_reuse import (
    BASE,
    MAPPING,
    RECEIPT,
    readable_report,
    validate_scope,
)


def request(local_id="one", payload=None, kind="order", repair=0):
    context = {
        "agent_state": {"current_style": "fundamental", "persona": {"agent_id": "A001"}},
        "observation": {"session_date": "2026-06-29"},
    }
    return {
        "request_hash": payload_digest(local_id),
        "kind": kind,
        "agent_id": "A001",
        "session_date": "2026-06-29",
        "treatment": "vanilla_llm",
        "repair": repair,
        "payload": payload
        if payload is not None
        else {
            "model": "synthetic-model",
            "messages": [{"role": "user", "content": json.dumps(context)}],
        },
    }


def response(req, rid="provider-one", cached=False, retry=0, text=None):
    raw = {"decision": "stay", "target_style": "fundamental", "rationale": "SYNTHETIC TEST"}
    return {
        **{key: value for key, value in req.items() if key != "payload"},
        "cached": cached,
        "retry_count": retry,
        "http_status": 200,
        "response": {
            "id": rid,
            "created": 123,
            "model": "synthetic-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text or json.dumps(raw)},
                }
            ],
        },
    }


def batch(requests, responses, run_id="run-one"):
    return {"run_id": run_id, "requests": requests, "responses": responses}


def review(req, run_id="run-one"):
    return {
        "run_id": run_id,
        "raw_answer_request_hash": req["request_hash"],
        "authority": "archived_llm",
        "valid_choice": True,
        "treatment": "vanilla_llm",
        "agent_id": req["agent_id"],
        "review_date": req["session_date"],
        "from_style": "fundamental",
        "decision": "stay",
        "target_style": "fundamental",
    }


def test_new_logical_requests_same_payload_and_reused_id_are_explicit_not_independent_draws():
    a, b = request("a"), request("b")
    result = audit_archives([batch([a, b], [response(a), response(b)])], [])
    s = result["summary"]
    assert s["same_input_opportunity_groups_with_multiple_noncached_successes"] == 1
    assert s["repeated_provider_ids_across_noncached_logical_requests"] == 1
    group = result["repeated_provider_ids"][0]
    assert group["payload_relationship"] == "same" and group["logical_requests"] == 2
    assert s["independent_generation_count"] is None and not s["independent_sampling_established"]
    assert not s["provider_cache_mechanism_confirmed"]


def test_local_cache_replay_without_http_metadata_is_kept_separate_from_new_response():
    a, b = request("a"), request("b")
    replay = response(b, cached=True)
    replay.pop("http_status")
    replay.pop("retry_count")
    result = audit_archives([batch([a, b], [response(a), replay])], [])
    assert result["summary"]["cache_state_counts"]["local_cache"] == 1
    assert result["summary"]["repeated_provider_id_groups"] == 1
    assert result["summary"]["repeated_provider_ids_across_noncached_logical_requests"] == 0
    assert result["summary"]["same_input_opportunity_groups_with_multiple_noncached_successes"] == 0


def test_transport_failure_then_retry_not_counted_as_a_second_logical_request():
    req = request()
    failed = response(req)
    failed.update(http_status=502, response=None)
    result = audit_archives([batch([req], [failed, response(req, retry=1)])], [])
    s = result["summary"]
    assert s["request_log_rows"] == 1 and s["response_log_rows"] == 2
    assert (
        s["http200_response_rows"]
        == s["http_non200_or_unknown_rows"]
        == s["retry_response_rows"]
        == 1
    )
    assert s["http200_rows_without_provider_id"] == 0
    assert s["same_input_opportunity_groups_with_multiple_noncached_successes"] == 0


def test_two_successes_of_same_retry_lineage_not_two_independent_local_requests():
    req = request()
    result = audit_archives([batch([req], [response(req), response(req, retry=1)])], [])
    assert result["summary"]["repeated_provider_id_groups"] == 1
    assert result["summary"]["repeated_provider_ids_across_noncached_logical_requests"] == 0
    assert result["repeated_provider_ids"][0]["logical_requests"] == 1


def test_format_repair_appends_message_and_retains_its_separate_payload():
    a = request()
    payload = copy.deepcopy(a["payload"])
    payload["messages"].append({"role": "user", "content": "fix the format"})
    b = request("repair", payload, repair=1)
    result = audit_archives([batch([a, b], [response(a, "a"), response(b, "b")])], [])
    assert result["summary"]["format_repair_request_rows"] == 1
    assert result["summary"]["distinct_uniquely_linked_payloads"] == 2
    assert result["summary"]["repeated_payload_groups_across_logical_requests"] == 0


def test_distinct_ids_and_payloads_do_not_prove_the_service_never_reuses_answers():
    a, b = request("a"), request("b", {"model": "synthetic-model", "messages": []})
    result = audit_archives([batch([a, b], [response(a, "a"), response(b, "b")])], [])
    s = result["summary"]
    assert (
        s["repeated_provider_id_groups"]
        == s["same_input_opportunity_groups_with_multiple_noncached_successes"]
        == 0
    )
    assert s["repeated_answer_text_groups"] == 1
    assert not s["independent_sampling_established"] and s["independent_generation_count"] is None


@pytest.mark.parametrize(
    "alter", ["missing_id", "missing_cache", "unknown_payload", "origin_mismatch"]
)
def test_unknown_evidence_never_defaults_to_a_known_independent_response(alter):
    req = request()
    a, b = response(req), response(req)
    requests = [req]
    if alter == "missing_id":
        a["response"].pop("id")
        b["response"].pop("id")
    elif alter == "missing_cache":
        a.pop("cached")
        b.pop("cached")
    elif alter == "unknown_payload":
        requests = []
    else:
        a["agent_id"] = b["agent_id"] = "other-agent"
    result = audit_archives([batch(requests, [a, b])], [])
    assert not result["summary"]["independent_sampling_established"]
    if alter in {"unknown_payload", "origin_mismatch"}:
        assert result["summary"]["response_rows_with_unknown_payload"] == 2
        assert result["repeated_provider_ids"][0]["payload_relationship"] == "unknown"
    if alter == "missing_id":
        assert result["summary"]["http200_rows_without_provider_id"] == 2
    if alter == "missing_cache":
        assert result["summary"]["cache_state_counts"]["unknown"] == 2


def test_identical_request_log_duplicates_keep_unique_payload_but_not_extra_opportunities():
    req = request(kind="style")
    result = audit_archives([batch([req, copy.deepcopy(req)], [response(req)])], [review(req)])
    assert result["summary"]["duplicate_request_log_rows"] == 1
    assert result["summary"]["distinct_uniquely_linked_payloads"] == 1
    assert result["summary"]["same_input_opportunity_groups_with_multiple_noncached_successes"] == 0
    assert result["decisions"][0]["status"] == "matched"


def test_conflicting_payloads_under_same_hash_remain_unknown_and_unmatched():
    req = request(kind="style")
    conflict = copy.deepcopy(req)
    conflict["payload"]["messages"] = []
    result = audit_archives([batch([req, conflict], [response(req)])], [review(req)])
    assert result["summary"]["response_rows_with_unknown_payload"] == 1
    assert result["decisions"][0]["status"] == "missing_or_ambiguous_request"


def test_cross_run_reuse_linked_to_style_choice_and_rules_are_explicitly_excluded():
    req = request(kind="style")
    rows = [review(req), review(req, "run-two"), {"authority": "programmed_rule"}]
    result = audit_archives(
        [batch([req], [response(req)]), batch([req], [response(req)], "run-two")], rows
    )
    assert result["summary"]["decision_mapping_status_counts"] == {
        "matched": 2,
        "excluded_programmed_rule": 1,
    }
    assert (
        result["summary"]["matched_style_choices_with_id_repeated_across_noncached_requests"] == 2
    )
    assert result["repeated_provider_ids"][0]["logical_requests"] == 2


def test_opposite_valid_answers_under_same_identity_are_ambiguous_before_review_comparison():
    req = request(kind="style")
    other = json.dumps(
        {"decision": "switch", "target_style": "technical", "rationale": "SYNTHETIC"}
    )
    responses = [response(req, "one"), response(req, "two", text=other)]
    result = audit_archives([batch([req], responses)], [review(req)])
    assert result["decisions"][0]["status"] == "missing_or_ambiguous_matching_response"
    assert result["decisions"][0]["response_line"] is None


def test_cached_style_answer_can_match_a_published_choice_without_claiming_new_generation():
    req = request(kind="style")
    cached = response(req, cached=True)
    cached.pop("http_status")
    cached.pop("retry_count")
    result = audit_archives([batch([req], [cached])], [review(req)])
    assert result["decisions"][0]["status"] == "matched"
    assert result["decisions"][0]["cache_state"] == "local_cache"
    assert result["summary"]["http200_response_rows"] == 0
    assert not result["summary"]["independent_sampling_established"]


@pytest.mark.parametrize(
    "alter",
    ["wrong_context", "wrong_choice", "duplicate_success", "duplicate_mapping", "wrong_kind"],
)
def test_bad_or_ambiguous_style_binding_never_selects_a_favorite(alter):
    req = request(kind="style")
    resp = response(req)
    reviews = [review(req)]
    responses = [resp]
    if alter == "wrong_context":
        context = json.loads(req["payload"]["messages"][0]["content"])
        context["agent_state"]["current_style"] = "technical"
        req["payload"]["messages"][0]["content"] = json.dumps(context)
    elif alter == "wrong_choice":
        resp["response"]["choices"][0]["message"]["content"] = json.dumps(
            {"decision": "switch", "target_style": "technical", "rationale": "SYNTHETIC"}
        )
    elif alter == "duplicate_success":
        responses.append(copy.deepcopy(resp))
    elif alter == "duplicate_mapping":
        reviews.append(copy.deepcopy(reviews[0]))
    else:
        req["kind"] = resp["kind"] = "order"
    result = audit_archives([batch([req], responses)], reviews)
    assert all(row["status"] != "matched" for row in result["decisions"])
    assert len(result["decisions"]) == len(reviews)


def test_repeated_id_changed_answer_not_reported_as_identical_body_or_cache_mechanism():
    a, b = request("a"), request("b")
    result = audit_archives([batch([a, b], [response(a, text="one"), response(b, text="two")])], [])
    assert result["repeated_provider_ids"][0]["distinct_nonmissing_answer_texts"] == 2
    assert not result["summary"]["provider_cache_mechanism_confirmed"]


def test_report_does_not_deny_repeated_ids_when_payloads_differ():
    a, b = request("a"), request("b", {"model": "synthetic-model", "messages": []})
    result = audit_archives([batch([a, b], [response(a), response(b)])], [])
    assert result["summary"]["same_input_opportunity_groups_with_multiple_noncached_successes"] == 0
    report = readable_report(result)
    assert "重复编号组 1 组" in report
    assert "没有看到重复编号" not in report


def test_scope_includes_only_explicit_twenty_files_and_rejects_hidden_expansion():
    folders = {
        f"{BASE}/{group}/n30_seed{seed}_shock0_e21"
        for group in ("vanilla_llm", "low_herding", "high_herding")
        for seed in (42, 43, 44)
    }
    plan = {
        "response_input_files": {p + "/llm_responses.jsonl": {} for p in folders},
        "auxiliary_request_input_files": {p + "/llm_requests.jsonl": {} for p in folders},
        "auxiliary_matched_decision_mapping_files": {MAPPING: {}, RECEIPT: {}},
    }
    assert len(validate_scope(plan)) == 20
    plan["auxiliary_request_input_files"]["unapproved.jsonl"] = {}
    with pytest.raises(ValueError):
        validate_scope(plan)
