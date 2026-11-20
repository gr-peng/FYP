import copy

from scripts.audit_style_probe_retest import response_reuse_diagnostic


def response(index, rid, text="same answer", metadata=None):
    return {
        "planned_id": f"synthetic-slot-{index}",
        "case_id": "synthetic-condition",
        "cached": False,
        "http_status": 200,
        "response": {
            "id": rid,
            "created": 123,
            "model": "synthetic-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                    "provider_specific_fields": metadata or {},
                }
            ],
        },
    }


def test_repeated_ids_and_text_but_nullable_metadata_differ_do_not_claim_full_body_identical():
    rows = [response(1, "reused"), response(2, "reused", metadata={"stop_reason": None})]
    before = copy.deepcopy(rows)
    result = response_reuse_diagnostic(rows)
    assert result["repeated_response_identifiers_observed"]
    assert result["distinct_response_ids"] == result["distinct_core_envelopes"] == 1
    assert result["distinct_full_bodies"] == 2
    assert result["independent_generation_count"] is None
    assert not result["provider_cache_mechanism_confirmed"]
    assert rows == before


def test_unique_ids_with_identical_text_do_not_prove_independent_sampling():
    result = response_reuse_diagnostic([response(1, "a"), response(2, "b")])
    assert result["distinct_response_ids"] == 2
    assert result["distinct_nonmissing_answer_texts"] == 1
    assert not result["repeated_response_identifiers_observed"]
    assert not result["independent_generations_established"]
    assert result["independent_generation_count"] is None


def test_reused_id_with_changed_text_is_explicit_not_assumed_same_answer():
    result = response_reuse_diagnostic([response(1, "a", "one"), response(2, "a", "two")])
    assert result["repeated_identifier_groups"][0]["distinct_answer_texts"] == 2
    assert result["distinct_core_envelopes"] == 2
    assert not result["provider_cache_mechanism_confirmed"]


def test_missing_text_is_not_an_additional_answer_within_repeated_id_group():
    missing = response(2, "same-id")
    missing["response"]["choices"][0]["message"].pop("content")
    result = response_reuse_diagnostic([response(1, "same-id", "one actual answer"), missing])
    assert result["distinct_nonmissing_answer_texts"] == 1
    assert result["conditions"][0]["distinct_answer_texts"] == 1
    assert result["repeated_identifier_groups"][0]["distinct_answer_texts"] == 1


def test_missing_ids_or_empty_logs_are_unknown_not_independent_samples():
    rows = [response(1, None), response(2, " ")]
    for source in (rows, []):
        result = response_reuse_diagnostic(source)
        assert result["response_id_present_count"] == result["distinct_response_ids"] == 0
        assert result["missing_response_id_count"] == len(source)
        assert not result["independent_generations_established"]
        assert result["independent_generation_count"] is None


def test_failed_transport_not_counted_as_inspected_success_and_missing_text_not_a_unique_answer():
    failed = {
        "planned_id": "failed",
        "case_id": "synthetic-condition",
        "http_status": 503,
        "response": None,
    }
    no_text = response(1, "one")
    no_text["response"]["choices"][0]["message"]["content"] = None
    result = response_reuse_diagnostic([failed, no_text])
    assert result["inspected_http_200_response_records"] == 1
    assert result["distinct_nonmissing_answer_texts"] == 0
