"""In-memory checks of fact scoring and execution limits; no real model calls."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from behavioral_market.evaluation.fact_reading_probe import (
    BASE_IDS,
    CASE_IDS,
    FIELDS,
    INSTRUCTION,
    NAMESPACE,
    SCHEMA,
    SYSTEM,
    authorization_template,
    build_materials,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    expected_request,
    gold_for_case,
    make_gold,
    make_package,
    parse_answer,
    validate_materials,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import (
    build_bank,
    canonical,
    changed_paths,
    digest,
    semantic_payload,
)
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.style_fact_check import execute, load_package, prepare, readable, save


@pytest.fixture
def base():
    return [c for c in build_bank() if c["case_id"] in BASE_IDS]


@pytest.fixture
def package(base, tmp_path):
    cases = build_materials(base)
    return make_package(
        cases,
        make_gold(cases),
        "in-memory-only",
        str(tmp_path / "execution"),
        source_rows=base,
        evidence_kind="synthetic_tool_test",
    )


def grant(package):
    return {
        **authorization_template(package),
        "decision": "explicit_user_granted",
        "additional_POST_limit": 6,
        "cumulative_POST_cap": 31,
        "specific_user_reply": "In-memory fixture only, not real user permission.",
    }


def answer(facts, identifier="fixture"):
    return {
        "model": "DeepSeek-V4-Flash",
        "id": identifier,
        "created": 123,
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": canonical(facts)}}
        ],
    }


def facts(package, i):
    return gold_for_case(package["cases"][i])["facts"]


def run_fixture(package, reply, authorization=None, bad_config=False):
    calls = []

    async def run():
        output = Path(package["execution_output"])
        config = execution_config(package)
        if bad_config:
            config["transport_retries"] = 1
        client = OpenAICompatibleClient(
            config, "IN_MEMORY_TEST_ONLY", output, output / "fresh_cache", cache_namespace=NAMESPACE
        )
        await client.http.aclose()

        def handler(request):
            calls.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
            i = sum(r.method == "POST" for r in calls) - 1
            status, body = reply(i)
            return httpx.Response(status, json=body)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            state = await collect(package, authorization or grant(package), client, output)
        finally:
            await client.close()
        logs = [
            [strict_json(line) for line in (output / name).read_text().splitlines()]
            for name in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
        ]
        return state, logs, evaluate(package, make_gold(package["cases"]), *logs)

    return asyncio.run(run()), calls


def test_common_fact_task_preserves_context_and_only_control_reference_changes(base):
    cases = build_materials(base)
    validate_materials(cases, base)
    for i, c in enumerate(cases):
        source = base[i] if i < 4 else base[0]
        original = strict_json(source["payload"]["messages"][1]["content"])
        visible = strict_json(c["payload"]["messages"][1]["content"])
        assert c["payload"]["messages"][0]["content"] == SYSTEM
        assert visible["decision_instruction"] == INSTRUCTION
        assert visible["output_schema"] == SCHEMA
        original["decision_instruction"], original["output_schema"] = INSTRUCTION, SCHEMA
        if i >= 4:
            original["observation"]["reference"]["value"] = 90 if i == 4 else 105
        assert visible == original
        assert c["source_row_sha256"] == digest(source)
        assert "private_construction_ledger_not_sent" not in canonical(c["payload"])
    for c in cases[4:]:
        assert changed_paths(
            semantic_payload(cases[0]["payload"]), semantic_payload(c["payload"])
        ) == ["messages.1.content.observation.reference.value"]


def test_gold_uses_visible_numbers_and_constant_equal_fails_two_controls(package):
    expected = [
        ("equal", "fundamental", "fundamental"),
        ("equal", "fundamental", "technical"),
        ("equal", "technical", "fundamental"),
        ("equal", "technical", "technical"),
        ("above", "fundamental", "fundamental"),
        ("below", "fundamental", "fundamental"),
    ]
    for i, values in enumerate(expected):
        key = facts(package, i)
        assert tuple(key[f] for f in list(FIELDS)[:3]) == values
        assert key["population_as_of_session"] == 9
        assert key["population_size_excluding_self"] == 100
    changed = deepcopy(package["cases"][0])
    changed["control_reference_value"] = 1000000
    assert gold_for_case(changed)["facts"] == facts(package, 0)
    visible = strict_json(changed["payload"]["messages"][1]["content"])
    visible["observation"]["reference"]["value"] = 91
    changed["payload"]["messages"][1]["content"] = canonical(visible)
    assert gold_for_case(changed)["facts"]["price_reference_relation"] == "above"


def test_unknown_is_distinct_from_wrong_and_no_general_score(package):
    result = evaluate(package, make_gold(package["cases"]), [], [], [])
    assert not result["execution_complete"] and result["all_fields_correct"] is None
    assert result["fact_status_counts"] == {"unknown": 6}
    assert all(
        c == {"correct": 0, "wrong": 0, "unknown": 6} for c in result["per_field_counts"].values()
    )
    assert result["general_understanding_score"] is None
    assert result["old_rationale_correctness_established"] is False


def test_schema_valid_wrong_facts_continue_all_six_and_keep_exact_posted_payloads(package):
    def reply(i):
        value = facts(package, i)
        value["price_reference_relation"] = "equal"
        value["population_as_of_session"] = 10
        return 200, answer(value, f"r{i}")

    (state, logs, result), calls = run_fixture(package, reply)
    assert state["status"] == "complete" and result["execution_complete"]
    assert len(logs[0]) == len(logs[1]) == len(logs[2]) == 6
    assert result["all_fields_correct"] is False
    assert result["per_field_counts"]["price_reference_relation"] == {
        "correct": 4,
        "wrong": 2,
        "unknown": 0,
    }
    assert result["per_field_counts"]["population_as_of_session"] == {
        "correct": 0,
        "wrong": 6,
        "unknown": 0,
    }
    for request, case in zip(
        (r for r in calls if r.method == "POST"), package["cases"], strict=True
    ):
        assert json.loads(request.content) == case["payload"]
        assert "by_case" not in request.content.decode()
        assert "source_row_sha256" not in request.content.decode()
        assert "control_reference_value" not in request.content.decode()
        assert "gold_sha256" not in request.content.decode()


@pytest.mark.parametrize("value", [True, "9", 9.0, -1, None])
def test_integer_contract_rejects_coercion_but_not_a_wrong_nonnegative_integer(package, value):
    key = facts(package, 0)
    key["population_as_of_session"] = value
    assert parse_answer(answer(key)) == ("invalid_schema", None)
    key["population_as_of_session"] = 10
    assert parse_answer(answer(key)) == ("valid_schema", key)


@pytest.mark.parametrize(
    "bad_content",
    [
        "{}",
        "[]",
        '{"price_reference_relation":"equal","price_reference_relation":"above"}',
        '{"population_as_of_session":NaN}',
        '{"population_as_of_session":1e999}',
    ],
)
def test_invalid_json_and_schema_are_not_scored_as_wrong_facts(package, bad_content):
    body = answer(facts(package, 0))
    body["choices"][0]["message"]["content"] = bad_content
    assert parse_answer(body) == ("invalid_schema", None)


@pytest.mark.parametrize("failure", ["http", "schema", "reasoning", "model"])
def test_protocol_failure_stops_after_first_failure_without_retries(package, failure):
    def reply(i):
        body = answer(facts(package, i), f"r{i}")
        if i == 1:
            if failure == "http":
                return 503, {"error": "in-memory fixture"}
            if failure == "schema":
                body["choices"][0]["message"]["content"] = "{}"
            if failure == "reasoning":
                body["choices"][0]["message"]["reasoning_content"] = "fixture"
            if failure == "model":
                body["model"] = "wrong-model"
        return 200, body

    (state, _, result), calls = run_fixture(package, reply)
    assert state["status"] == "stopped"
    assert sum(r.method == "POST" for r in calls) == result["logged_dispatch_attempts"] == 2
    assert result["protocol_valid_returns"] == 1
    assert result["all_fields_correct"] is None
    assert result["fact_status_counts"] == {"correct": 1, "unknown": 5}
    assert all(
        c == {"correct": 1, "wrong": 0, "unknown": 5} for c in result["per_field_counts"].values()
    )


def test_duplicate_provider_identity_is_diagnostic_and_never_causes_resending(package):
    (_, _, result), calls = run_fixture(package, lambda i: (200, answer(facts(package, 0))))
    assert result["execution_complete"] and sum(r.method == "POST" for r in calls) == 6
    assert all("same_id_created_text" in r["reuse_flags"] for r in result["rows"])
    assert result["independent_generation_count"] is None
    assert result["per_field_counts"]["price_reference_relation"]["wrong"] == 2


def test_zero_authority_blocks_even_registry_get_and_changed_client_is_rejected(package):
    with pytest.raises(ValueError, match="not authorized"):
        run_fixture(
            package, lambda i: (200, answer(facts(package, i))), authorization_template(package)
        )
    with pytest.raises(ValueError, match="client differs"):
        run_fixture(package, lambda i: (200, answer(facts(package, i))), bad_config=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("additional_POST_limit", 5),
        ("additional_POST_limit", True),
        ("cumulative_POST_cap", 32),
        ("prior_POSTs", 24),
        ("package_sha256", "changed"),
        ("endpoint", "https://example.invalid"),
        ("model", "wrong"),
        ("run_id", "wrong"),
        ("specific_user_reply", None),
    ],
)
def test_grant_bound_to_exact_six_materials_and_route(package, field, value):
    authorization = grant(package)
    authorization[field] = value
    with pytest.raises(ValueError):
        check_authorization(package, authorization)


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate",
        "after_failure",
        "retry",
        "foreign",
        "order",
        "payload",
        "clock",
        "overlap",
        "hash",
        "extra",
        "cache",
    ],
)
def test_raw_log_audit_rejects_wrong_linkage_and_protocol_violation(package, tamper):
    (_, logs, _), _ = run_fixture(package, lambda i: (200, answer(facts(package, i), f"r{i}")))
    requests, attempts, responses = deepcopy(logs)
    if tamper == "duplicate":
        responses.append(deepcopy(responses[0]))
    elif tamper == "after_failure":
        responses[0]["http_status"], responses[0]["response"] = 503, None
    elif tamper == "retry":
        attempts[0]["retry_count"] = 1
    elif tamper == "foreign":
        requests[0]["case_id"] = "unexpected"
    elif tamper == "order":
        requests.reverse()
    elif tamper == "payload":
        requests[0]["payload"]["temperature"] = 1
    elif tamper == "clock":
        responses[0]["requested_at"] = "2000-01-01T00:00:00+00:00"
    elif tamper == "overlap":
        attempts[1]["dispatched_at"] = attempts[0]["dispatched_at"]
    elif tamper == "hash":
        attempts[0]["request_hash"] = "wrong"
    elif tamper == "extra":
        responses[0]["invented"] = True
    else:
        responses[0]["cached"] = True
    with pytest.raises(ValueError):
        evaluate(package, make_gold(package["cases"]), requests, attempts, responses)


def test_partial_dispatch_and_local_cache_are_unknown(package):
    (_, logs, _), _ = run_fixture(package, lambda i: (200, answer(facts(package, i), f"r{i}")))
    for attempts in ([], logs[1][:1]):
        result = evaluate(package, make_gold(package["cases"]), logs[0][:1], attempts, [])
        assert result["fact_status_counts"] == {"unknown": 6}
        assert result["all_fields_correct"] is None
    cached = {
        k: v
        for k, v in logs[2][0].items()
        if k in set(expected_request(package["cases"][0], package)) - {"payload"}
    }
    cached.update(cached=True, status="ok", response=logs[2][0]["response"])
    result = evaluate(package, make_gold(package["cases"]), logs[0][:1], [], [cached])
    assert result["rows"][0]["format_status"] == "local_cache_replay"
    assert result["fact_status_counts"] == {"unknown": 6}


def test_prepare_and_cli_deny_before_reading_credentials_or_creating_execution(base, tmp_path):
    root = tmp_path / "workspace"
    relative = "iterations/style_conflict_package_20261022/package/scheduled_inputs.jsonl"
    source = root / relative
    source.parent.mkdir(parents=True)
    raw = "".join(canonical(c) + "\n" for c in base).encode()
    source.write_bytes(raw)
    from hashlib import sha256

    plan = tmp_path / "plan.json"
    save(
        plan,
        {
            "input_files": {relative: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}},
            "frozen_run_id": "fixture-prepare",
            "fixed_execution_output": str(tmp_path / "live"),
        },
    )
    folder = tmp_path / "package"
    assert (
        prepare(SimpleNamespace(workspace=root, plan=plan, output=folder))["new_model_calls"] == 0
    )
    package, gold = load_package(folder)
    assert gold == make_gold(package["cases"])
    assert "未知" in readable(evaluate(package, gold, [], [], []))
    assert not Path(package["execution_output"]).exists()
    with pytest.raises(ValueError, match="not authorized"):
        asyncio.run(
            execute(
                SimpleNamespace(
                    workspace=root,
                    package=folder,
                    authorization=folder / "authorization_template.json",
                )
            )
        )
    assert not Path(package["execution_output"]).exists()
    gold["by_case"][CASE_IDS[0]]["facts"]["price_reference_relation"] = "below"
    save(folder / "hidden_gold.json", gold)
    with pytest.raises(ValueError, match="hidden key altered"):
        load_package(folder)


def test_changed_materials_and_control_cannot_be_resigned_as_valid(package, base):
    changed = deepcopy(package)
    c = changed["cases"][4]
    visible = strict_json(c["payload"]["messages"][1]["content"])
    visible["observation"]["last_close"] = 96
    c["payload"]["messages"][1]["content"] = canonical(visible)
    c["payload_sha256"] = digest(c["payload"])
    changed["gold_sha256"] = digest(make_gold(changed["cases"]))
    changed["package_sha256"] = digest({k: v for k, v in changed.items() if k != "package_sha256"})
    with pytest.raises(ValueError, match="materials or source linkage"):
        validate_package(changed)
    with pytest.raises(ValueError, match="materials or source linkage"):
        validate_materials(changed["cases"], base)


@pytest.mark.parametrize(
    "tamper",
    [
        "common_answer_leak",
        "common_context_change",
        "source_snapshot",
        "false_source_hash",
    ],
)
def test_resigning_common_changes_cannot_bypass_source_binding_or_cli(package, tmp_path, tamper):
    changed = deepcopy(package)
    gold = make_gold(changed["cases"])
    if tamper == "source_snapshot":
        changed["source_snapshot_not_sent"]["canonical_jsonl_rows"][0]["case_id"] = "fake"
    else:
        for case in changed["cases"]:
            visible = strict_json(case["payload"]["messages"][1]["content"])
            if tamper == "common_answer_leak":
                visible["precomputed_answers_for_all_six"] = gold["by_case"]
            elif tamper == "common_context_change":
                visible["observation"]["session"] = 11
            else:
                case["source_row_sha256"] = "0" * 64
            case["payload"]["messages"][1]["content"] = canonical(visible)
            case["payload_sha256"] = digest(case["payload"])
    gold = make_gold(changed["cases"])
    changed["gold_sha256"] = digest(gold)
    changed["package_sha256"] = digest({k: v for k, v in changed.items() if k != "package_sha256"})
    expected = (
        "frozen source content" if tamper == "source_snapshot" else "materials or source linkage"
    )
    with pytest.raises(ValueError, match=expected):
        validate_package(changed, gold)
    folder = tmp_path / "resigned_bad_package"
    folder.mkdir()
    save(folder / "package.json", changed)
    save(folder / "hidden_gold.json", gold)
    with pytest.raises(ValueError, match=expected):
        load_package(folder)
