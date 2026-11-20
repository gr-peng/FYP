"""In-memory transport checks; none of these fixtures contact a model service."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from behavioral_market.evaluation.style_conflict_package import (
    NAMESPACE,
    SELECTED,
    authorization_template,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    make_package,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import (
    build_bank,
    canonical,
    contrast_edges,
    digest,
)
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.style_conflict_check import (
    SOURCE_PATHS,
    execute,
    load_package,
    prepare,
    readable,
    save,
)


@pytest.fixture
def package(tmp_path):
    bank = build_bank()
    return make_package(
        bank,
        contrast_edges(bank),
        "local-tool-test",
        str(tmp_path / "execution"),
        "synthetic_tool_test",
    )


def grant(package):
    return {
        **authorization_template(package),
        "decision": "explicit_user_granted",
        "additional_POST_limit": 4,
        "cumulative_POST_cap": 25,
        "specific_user_reply": "Synthetic in-memory transport fixture only; no actual user a"
        "uthorization.",
    }


def answer(style="fundamental", identifier="response", reasoning=False):
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
                            "decision": "stay" if style == "fundamental" else "switch",
                            "target_style": style,
                            "rationale": "local tool fixture",
                        }
                    ),
                    **({"reasoning_content": "fixture"} if reasoning else {}),
                },
            }
        ],
    }


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
            index = sum(r.method == "POST" for r in calls) - 1
            status, body = reply(index)
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
        return state, logs, evaluate(package, *logs)

    return asyncio.run(run()), calls


def test_missing_all_answers_preserves_unknowns_and_twelve_unscheduled(package):
    result = evaluate(package, [], [], [])
    assert len(result["rows"]) == len(result["contrasts"]) == 4
    assert result["valid_returns"] == result["logged_dispatch_attempts"] == 0
    assert all(r["choice"] is None for r in result["rows"])
    assert all(r["target_fundamental_difference"] is None for r in result["contrasts"])
    assert result["all_same_target"] is None and len(result["not_scheduled_case_ids"]) == 12


def test_four_identical_targets_are_valid_even_with_identical_provider_identity(package):
    (state, logs, result), calls = run_fixture(package, lambda i: (200, answer()))
    assert state["status"] == "complete" and result["complete"]
    assert sum(r.method == "POST" for r in calls) == 4
    assert result["all_same_target"] is True
    assert all(r["target_fundamental_difference"] == 0 for r in result["contrasts"])
    assert all("same_id_created_text" in r["return_identity_flags"] for r in result["contrasts"])
    assert result["independent_generation_count"] is None
    assert result["real_external_execution_authenticated"] is False
    for request, case in zip(
        (r for r in calls if r.method == "POST"), package["cases"], strict=True
    ):
        assert json.loads(request.content) == case["payload"]
        assert "private_construction_ledger_not_sent" not in request.content.decode()
    assert [r["case_id"] for r in logs[0]] == list(SELECTED)


def test_named_target_contrasts_recover_given_majority_following_fixture(package):
    (_, _, result), _ = run_fixture(
        package, lambda i: (200, answer("fundamental" if i % 2 == 0 else "technical", f"r{i}"))
    )
    assert result["complete"] and not result["all_same_target"]
    for row in result["contrasts"]:
        assert row["target_fundamental_difference"] == (
            1 if row["factor"] == "majority_style" else 0
        )
    assert result["human_alignment_established"] is False
    assert result["causal_mechanism_established"] is False


@pytest.mark.parametrize("failure", ["http", "schema", "reasoning", "incoherent", "model"])
def test_first_failure_stops_without_retries_or_filling_remaining_answers(package, failure):
    def reply(i):
        if i != 1:
            return 200, answer(identifier=f"r{i}")
        body = answer(identifier="bad", reasoning=failure == "reasoning")
        if failure == "http":
            return 503, {"error": "fixture"}
        if failure == "schema":
            body["choices"][0]["message"]["content"] = "{}"
        if failure == "incoherent":
            body["choices"][0]["message"]["content"] = canonical(
                {"decision": "stay", "target_style": "technical", "rationale": "fixture"}
            )
        if failure == "model":
            body["model"] = "different-model"
        return 200, body

    (state, _, result), calls = run_fixture(package, reply)
    assert state["status"] == "stopped"
    assert sum(r.method == "POST" for r in calls) == result["logged_dispatch_attempts"] == 2
    assert result["valid_returns"] == 1
    assert result["all_same_target"] is None
    assert all(r["choice"] is None for r in result["rows"][1:])
    assert all(r["target_fundamental_difference"] is None for r in result["contrasts"])


def test_default_zero_authority_prevents_even_registry_get(package):
    with pytest.raises(ValueError, match="not authorized"):
        run_fixture(package, lambda i: (200, answer()), authorization_template(package))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("additional_POST_limit", 5),
        ("additional_POST_limit", True),
        ("cumulative_POST_cap", 26),
        ("prior_POSTs", 20),
        ("package_sha256", "different"),
        ("model", "different"),
        ("endpoint", "https://example.invalid"),
        ("specific_user_reply", None),
    ],
)
def test_specific_grant_must_match_exact_package_route_and_budget(package, field, value):
    auth = grant(package)
    auth[field] = value
    with pytest.raises(ValueError):
        check_authorization(package, auth)


def test_changed_client_retry_setting_is_rejected_before_get(package):
    with pytest.raises(ValueError, match="configuration"):
        run_fixture(package, lambda i: (200, answer()), bad_config=True)


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate",
        "after_failure",
        "retry",
        "foreign",
        "reverse_order",
        "payload",
        "clock",
        "cache",
    ],
)
def test_raw_log_evaluation_rejects_wrong_linkage_and_stop_contract(package, tamper):
    (_, logs, _), _ = run_fixture(package, lambda i: (200, answer(identifier=f"r{i}")))
    requests, attempts, responses = deepcopy(logs)
    if tamper == "duplicate":
        responses.append(deepcopy(responses[0]))
    elif tamper == "after_failure":
        responses[0]["http_status"], responses[0]["response"] = 503, None
    elif tamper == "retry":
        attempts[0]["retry_count"] = 1
    elif tamper == "foreign":
        requests[0]["case_id"] = "synthetic_1111"
    elif tamper == "reverse_order":
        requests.reverse()
    elif tamper == "payload":
        requests[0]["payload"]["temperature"] = 1
    elif tamper == "clock":
        responses[0]["requested_at"] = "2000-01-01T00:00:00+00:00"
    else:
        responses[0]["cached"] = True
    with pytest.raises(ValueError):
        evaluate(package, requests, attempts, responses)


def test_request_or_dispatch_without_response_is_unknown_not_zero(package):
    (_, logs, _), _ = run_fixture(package, lambda i: (200, answer(identifier=f"r{i}")))
    for attempts in ([], logs[1][:1]):
        result = evaluate(package, logs[0][:1], attempts, [])
        assert result["valid_returns"] == 0
        assert result["rows"][0]["status"] == (
            "awaiting_response" if attempts else "request_record_only"
        )
        assert all(r["target_fundamental_difference"] is None for r in result["contrasts"])


@pytest.mark.parametrize(
    "field", ["currently_authorized_new_POSTs", "transport_retries", "max_concurrency"]
)
def test_rehashed_limit_changes_do_not_redefine_protocol(package, field):
    package[field] = 8
    package["package_sha256"] = digest({k: v for k, v in package.items() if k != "package_sha256"})
    with pytest.raises(ValueError, match="limits"):
        validate_package(package)


def test_cli_rejects_pending_grant_before_output_credentials_or_network(package, tmp_path):
    from behavioral_market.evaluation.style_conflict_package import expected_request

    folder = tmp_path / "bundle"
    folder.mkdir()
    save(folder / "package.json", package)
    save(folder / "contrasts.json", package["contrasts"])
    (folder / "scheduled_inputs.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in package["cases"])
    )
    (folder / "expected_requests.jsonl").write_text(
        "".join(canonical(expected_request(r, package)) + "\n" for r in package["cases"])
    )
    save(tmp_path / "no-authorization.json", authorization_template(package))
    assert load_package(folder) == package
    with pytest.raises(ValueError, match="not authorized"):
        asyncio.run(
            execute(
                SimpleNamespace(
                    package=folder,
                    authorization=tmp_path / "no-authorization.json",
                    workspace=tmp_path,
                )
            )
        )
    assert not Path(package["execution_output"]).exists()


def test_cancelled_dispatch_is_unknown_and_never_retried(package):
    def cancel(index):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        run_fixture(package, cancel)
    output = Path(package["execution_output"])
    logs = [
        [strict_json(x) for x in (output / name).read_text().splitlines()]
        for name in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
    ]
    result = evaluate(package, *logs)
    assert result["logged_dispatch_attempts"] == 1
    assert result["valid_returns"] == 0 and not result["complete"]
    assert strict_json((output / "execution_state.json").read_text())["status"] == "cancelled"
    assert all(r["target_fundamental_difference"] is None for r in result["contrasts"])


def test_existing_state_blocks_collection_even_with_empty_logs(package):
    output = Path(package["execution_output"])
    output.mkdir()
    (output / "execution_state.json").write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match="never resume"):
        run_fixture(package, lambda i: (200, answer()))
    assert not (output / "llm_attempts.jsonl").exists()


@pytest.mark.parametrize(
    "tamper", ["attempt_extra", "response_extra", "negative_latency", "missing_latency"]
)
def test_unexpected_log_fields_or_impossible_latency_are_rejected(package, tamper):
    (_, logs, _), _ = run_fixture(package, lambda i: (200, answer(identifier=f"r{i}")))
    requests, attempts, responses = deepcopy(logs)
    if tamper == "attempt_extra":
        attempts[0]["payload"] = {"different": True}
    elif tamper == "response_extra":
        responses[0]["payload"] = {"different": True}
    elif tamper == "negative_latency":
        responses[0]["latency_seconds"] = -1
    else:
        responses[0].pop("latency_seconds")
    with pytest.raises(ValueError):
        evaluate(package, requests, attempts, responses)


def test_diagonal_response_reuse_is_visible_globally_and_on_incident_contrasts(package):
    def reply(index):
        style = "fundamental" if index in (0, 3) else "technical"
        return 200, answer(style, "shared_" + style)

    (_, _, result), _ = run_fixture(package, reply)
    assert len(result["return_identity_diagnostics"]["same_id_created_text"]) == 2
    for edge in result["contrasts"]:
        assert not edge["return_identity_flags"]
        assert "same_id_created_text" in edge["endpoint_reused_elsewhere_flags"]
    report = readable(result)
    assert "全部已记录的重复组" in report and "端点在其他场景重复" in report
    assert "synthetic_0000, synthetic_0011" in report


def test_same_basename_files_cannot_replace_declared_scope(tmp_path):
    plan = tmp_path / "plan.json"
    save(plan, {"input_files": {"elsewhere/" + Path(p).name: {} for p in SOURCE_PATHS}})
    with pytest.raises(ValueError, match="source paths"):
        prepare(SimpleNamespace(plan=plan, workspace=tmp_path, output=tmp_path / "out"))


def test_declared_source_symlink_outside_workspace_rejected_before_read(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"tool_fixture":true}')
    path = root / sorted(SOURCE_PATHS)[0]
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    plan = tmp_path / "plan.json"
    save(plan, {"input_files": dict.fromkeys(SOURCE_PATHS, {})})
    with pytest.raises(ValueError, match="outside workspace"):
        prepare(SimpleNamespace(plan=plan, workspace=root, output=root / "out"))
