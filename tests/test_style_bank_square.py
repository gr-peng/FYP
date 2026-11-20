"""Frozen-bank squares, recorded-source guards and in-memory execution only."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from behavioral_market.evaluation.style_bank_square import (
    NAMESPACE,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    historical_diagnostics,
    make_package,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import (
    build_bank,
    canonical,
    contrast_edges,
    digest,
)
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts.style_bank_square import (
    allocate_authorization,
    execute,
    load_history,
    load_package,
    prepare,
    verify_budget_sources,
)
from scripts.style_fact_check import fingerprint, read_jsonl, save

IDS = ["synthetic_0100", "synthetic_0101", "synthetic_0110", "synthetic_0111"]


def answer(current="fundamental", target="fundamental", identifier="fixture"):
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
                            "decision": "stay" if current == target else "switch",
                            "target_style": target,
                            "rationale": "in-memory only",
                        }
                    ),
                },
            }
        ],
    }


@pytest.fixture
def context(tmp_path):
    bank = build_bank()
    grant_path, ledger_path = (
        tmp_path / "iterations/grant.json",
        tmp_path / "iterations/ledger.json",
    )
    grant_path.parent.mkdir()
    save(
        grant_path,
        {
            "endpoint": "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
            "model": "DeepSeek-V4-Flash",
            "global_cumulative_completion_POST_cap": 100,
            "prior_used_completion_POSTs": 25,
            "specific_user_reply": "In-memory fixture only, no real user authority.",
        },
    )
    save(
        ledger_path,
        {
            "global_authorized_completion_POST_cap": 100,
            "used_completion_POSTs": 35,
            "reserved_next_completion_POSTs": 0,
            "active_run_id": None,
            "global_authorization_source": str(grant_path),
            "status": "test_completed_accepted",
        },
    )
    history = tmp_path / "iterations/history"
    history.mkdir()
    anchor = next(c for c in bank if c["case_id"] == IDS[0])
    for name, extra in [
        ("llm_requests.jsonl", {"payload": anchor["payload"]}),
        ("llm_attempts.jsonl", {}),
        ("llm_responses.jsonl", {"response": answer(), "http_status": 200, "cached": False}),
    ]:
        (history / name).write_text(
            canonical({"case_id": IDS[0], "request_hash": "fixture", **extra}) + "\n"
        )

    def descriptor(p):
        return {"path": str(p.relative_to(tmp_path)), "fingerprint": fingerprint(p)}

    plan = {
        "selected_case_ids": IDS,
        "frozen_run_id": "in-memory-only",
        "fixed_execution_output": str(tmp_path / "execution"),
        "evidence_kind": "synthetic_tool_test",
        "prior_used_completion_POSTs": 35,
        "budget_sources": {
            "grant": descriptor(grant_path),
            "prior_ledger": descriptor(ledger_path),
        },
        "historical_anchor": {
            "case_id": IDS[0],
            "sources": {str(p.relative_to(tmp_path)): fingerprint(p) for p in history.iterdir()},
        },
    }
    return SimpleNamespace(
        bank=bank,
        plan=plan,
        workspace=tmp_path,
        package=make_package(bank, plan),
        ledger=ledger_path,
        grant=grant_path,
    )


def run(context, reply, authorization=None):
    package = context.package
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
            status, value = reply(i)
            return httpx.Response(status, json=value)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        auth = authorization or allocate_authorization(
            package, verify_budget_sources(context.workspace, package)
        )
        try:
            state = await collect(package, auth, client, output)
        finally:
            await client.close()
        logs = [
            read_jsonl(output / n)
            for n in ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
        ]
        return state, logs, evaluate(package, *logs)

    return asyncio.run(exercise()), calls


def test_same_target_valid_and_private_metadata_not_posted(context):
    (state, _, result), calls = run(context, lambda i: (200, answer(identifier=f"r{i}")))
    assert state["status"] == "complete" and result["all_same_target"] is True
    assert all(e["target_fundamental_difference"] == 0 for e in result["contrasts"])
    for request, case in zip(
        (r for r in calls if r.method == "POST"), context.package["cases"], strict=True
    ):
        assert json.loads(request.content) == case["payload"]
        for forbidden in [
            "source_bank_not_sent",
            "private_construction_ledger_not_sent",
            "budget_sources",
        ]:
            assert forbidden not in request.content.decode()


def test_majority_following_fixture_has_two_majority_differences_not_benchmark(context):
    (_, _, result), _ = run(
        context,
        lambda i: (
            200,
            answer(target="fundamental" if i % 2 == 0 else "technical", identifier=f"r{i}"),
        ),
    )
    assert all(
        e["target_fundamental_difference"] == (1 if e["factor"] == "majority_style" else 0)
        for e in result["contrasts"]
    )
    assert result["causal_mechanism_established"] is False


def test_runner_supports_other_bank_square_and_last_four_calls(context):
    plan = deepcopy(context.plan)
    plan["selected_case_ids"] = [
        "synthetic_0000",
        "synthetic_0100",
        "synthetic_1000",
        "synthetic_1100",
    ]
    plan["historical_anchor"]["case_id"] = "synthetic_0100"
    plan["prior_used_completion_POSTs"] = 96
    package = make_package(context.bank, plan)
    validate_package(package)
    assert package["batch_cumulative_stop"] == 100
    assert {e["factor"] for e in package["contrasts"]} == {"current_style", "reference_condition"}
    for prior in (97, True):
        plan["prior_used_completion_POSTs"] = prior
        with pytest.raises(ValueError):
            make_package(context.bank, plan)


@pytest.mark.parametrize(
    "ids",
    [
        ["synthetic_0100"] * 4,
        ["synthetic_0000", "synthetic_0001", "synthetic_0010", "synthetic_1000"],
    ],
)
def test_selection_must_be_complete_unique_square(context, ids):
    plan = deepcopy(context.plan)
    plan["selected_case_ids"] = ids
    with pytest.raises(ValueError):
        make_package(context.bank, plan)


@pytest.mark.parametrize("mutation", ["answers", "context", "source", "edges"])
def test_resigned_common_changes_rejected(context, mutation):
    package = deepcopy(context.package)
    if mutation in {"answers", "context"}:
        for case in package["cases"]:
            d = json.loads(case["payload"]["messages"][1]["content"])
            if mutation == "answers":
                d["expected_answers"] = "fundamental"
            else:
                d["observation"]["session"] = 11
            case["payload"]["messages"][1]["content"] = canonical(d)
            case["payload_sha256"] = digest(case["payload"])
    elif mutation == "source":
        package["source_bank_not_sent"][0]["case_id"] = "wrong"
    else:
        package["contrasts"].pop()
    package["package_sha256"] = digest({k: v for k, v in package.items() if k != "package_sha256"})
    with pytest.raises(ValueError):
        validate_package(package)


@pytest.mark.parametrize("failure", ["http", "schema", "mode", "model"])
def test_first_failure_stops_without_unknown_becoming_zero(context, failure):
    def reply(i):
        value = answer(identifier=f"r{i}")
        if i == 1:
            if failure == "http":
                return 503, {"error": "fixture"}
            if failure == "schema":
                value["choices"][0]["message"]["content"] = "{}"
            if failure == "mode":
                value["choices"][0]["message"]["reasoning_content"] = "fixture"
            if failure == "model":
                value["model"] = "wrong"
        return 200, value

    (state, _, result), calls = run(context, reply)
    assert state["status"] == "stopped" and result["valid_returns"] == 1
    assert sum(r.method == "POST" for r in calls) == 2
    assert all(e["target_fundamental_difference"] is None for e in result["contrasts"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prior_POSTs", 34),
        ("additional_POST_limit", True),
        ("additional_POST_limit", 5),
        ("cumulative_POST_cap", 40),
        ("global_user_authorized_cap", 101),
        ("budget_sources", None),
    ],
)
def test_allocation_bound_to_current_package_and_prior_usage(context, field, value):
    package = context.package
    auth = allocate_authorization(package, verify_budget_sources(context.workspace, package))
    auth[field] = value
    with pytest.raises(ValueError):
        check_authorization(package, auth)


@pytest.mark.parametrize("mutation", ["used", "reserved", "cap", "grant_route", "changed_bytes"])
def test_actual_budget_sources_rechecked_before_credentials(context, mutation):
    p = context.grant if mutation == "grant_route" else context.ledger
    d = json.loads(p.read_text())
    if mutation == "used":
        d["used_completion_POSTs"] = 36
    elif mutation == "reserved":
        d["reserved_next_completion_POSTs"] = 1
    elif mutation == "cap":
        d["global_authorized_completion_POST_cap"] = 101
    elif mutation == "grant_route":
        d["endpoint"] = "https://example.invalid"
    else:
        d["comment"] = "tampered"
    save(p, d)
    if mutation != "changed_bytes":
        key = "grant" if mutation == "grant_route" else "prior_ledger"
        context.package["budget_sources"][key]["fingerprint"] = fingerprint(p)
    with pytest.raises(ValueError):
        verify_budget_sources(context.workspace, context.package)


@pytest.mark.parametrize("tamper", ["duplicate", "after_failure", "hash", "retry", "order"])
def test_raw_linkage_and_stop_contract_checked(context, tamper):
    (_, logs, _), _ = run(context, lambda i: (200, answer(identifier=f"r{i}")))
    requests, attempts, responses = deepcopy(logs)
    if tamper == "duplicate":
        responses.append(responses[0])
    elif tamper == "after_failure":
        responses[0]["http_status"], responses[0]["response"] = 503, None
    elif tamper == "hash":
        attempts[0]["request_hash"] = "wrong"
    elif tamper == "retry":
        attempts[0]["retry_count"] = 1
    else:
        requests.reverse()
    with pytest.raises(ValueError):
        evaluate(context.package, requests, attempts, responses)


def test_anchor_is_0100_not_old0000_and_never_pooled(context):
    (_, _, result), calls = run(context, lambda i: (200, answer()))
    old = load_history(context.workspace, context.package)
    diag = historical_diagnostics(result, old["body"], old["payload_sha256"], context.package)
    assert diag["old_anchor_case_id"] == "synthetic_0100"
    assert diag["old_anchor_not_in_denominator"] and len(result["rows"]) == 4
    assert "same_id_created_text" in diag["matches_by_new_case"][IDS[0]]
    assert sum(r.method == "POST" for r in calls) == 4
    wrong = next(c for c in context.bank if c["case_id"] == "synthetic_0000")
    with pytest.raises(ValueError, match="historical anchor payload differs"):
        historical_diagnostics(result, old["body"], digest(wrong["payload"]), context.package)


def test_prepare_roundtrip_and_unallocated_cli_denied_before_output_or_credentials(context):
    workspace = context.workspace
    bank = workspace / BANK_PATH
    bank.parent.mkdir(parents=True)
    bank.write_text("".join(canonical(c) + "\n" for c in context.bank))
    edges = bank.parent / "contrasts.json"
    save(edges, contrast_edges(context.bank))
    plan = {
        **context.plan,
        "source_files": {str(p.relative_to(workspace)): fingerprint(p) for p in [bank, edges]},
    }
    plan_path = workspace / "plan.json"
    save(plan_path, plan)
    folder = workspace / "package"
    assert (
        prepare(SimpleNamespace(workspace=workspace, plan=plan_path, output=folder))["cases"] == 4
    )
    package = load_package(folder)
    assert evaluate(package, [], [], [])["all_same_target"] is None
    with pytest.raises(ValueError, match="not allocated"):
        asyncio.run(
            execute(
                SimpleNamespace(
                    workspace=workspace,
                    package=folder,
                    authorization=folder / "authorization_template.json",
                )
            )
        )
    assert not Path(package["execution_output"]).exists()
