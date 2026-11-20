"""The real HTTP path with in-memory transport only; no keys or network."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_style_bank_square import answer

from behavioral_market.evaluation.style_mechanism_design import canonical, contrast_edges, digest
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from scripts.style_cache_control import (
    CONTROL,
    LOGS,
    collect,
    evaluate,
    execute,
    load_packet,
    packet_for,
    prepare,
    scored,
)
from scripts.style_fact_check import fingerprint, read_jsonl, save

pytest_plugins = ["test_style_bank_square"]


@pytest.fixture
def probe(context):
    workspace = context.workspace
    bank = workspace / BANK_PATH
    bank.parent.mkdir(parents=True)
    bank.write_text("".join(canonical(c) + "\n" for c in context.bank))
    edges = bank.parent / "contrasts.json"
    save(edges, contrast_edges(context.bank))
    note = workspace / "iterations/documentation.json"
    save(note, {"evidence": "in-memory documentation fixture, no real authority"})
    plan = {
        **context.plan,
        "selected_case_ids": [
            "synthetic_0000",
            "synthetic_0100",
            "synthetic_1000",
            "synthetic_1100",
        ],
        "fixed_execution_output": str(workspace / "iterations/cache_execution"),
        "probe_case_id": "synthetic_0100",
        "schedule": ["default", "no_read_no_write", "no_read_no_write", "default"],
        "documentation": {
            "path": str(note.relative_to(workspace)),
            "fingerprint": fingerprint(note),
        },
        "source_files": {str(p.relative_to(workspace)): fingerprint(p) for p in [bank, edges]},
    }
    return SimpleNamespace(
        workspace=workspace,
        plan=plan,
        packet=packet_for(workspace, plan),
        output=Path(plan["fixed_execution_output"]),
    )


def mock_client(calls, failure=None, repeated=False, metadata_override=None):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
        calls.append(request)
        index = len(calls)
        value = answer(identifier="normal" if index in (1, 4) or repeated else f"control-{index}")
        if metadata_override is not None:
            value.pop("id")
            value.pop("created")
            value.update(metadata_override)
        if index == 2:
            if failure == "http":
                return httpx.Response(503, text="no retry")
            if failure == "format":
                value["choices"][0]["message"]["content"] = "{}"
            if failure == "model":
                value["model"] = "wrong"
            if failure == "mode":
                value["usage"] = {"completion_tokens_details": {"reasoning_tokens": 1}}
            if failure == "cancel":
                raise asyncio.CancelledError
        headers = {"x-litellm-cache-key": "opaque-test-key", "x-unknown-sensitive": "never logged"}
        return httpx.Response(200, json=value, headers=headers)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer IN_MEMORY_SECRET"},
    )


def run_probe(probe, calls, **kwargs):
    async def go():
        async with mock_client(calls, **kwargs) as client:
            return await collect(probe.packet, probe.workspace, client, secret="IN_MEMORY_SECRET")

    return asyncio.run(go())


def test_abba_true_wire_payloads_fixed_questions_and_joint_effect_limit(probe):
    calls = []
    result = run_probe(probe, calls)
    assert result["protocol_complete"] and len(calls) == 4
    base = probe.packet["probe_case"]["payload"]
    for i, request in enumerate(calls):
        payload = json.loads(request.content)
        wanted = deepcopy(base)
        if i in (1, 2):
            wanted["cache"] = CONTROL
        assert payload == wanted == probe.packet["requests"][i]["payload"]
        assert payload["messages"] == base["messages"]
    assert result["return_reuse_groups"]["provider_id"] == [["slot_1", "slot_4"]]
    assert all(r["cache_key_header"]["present"] for r in result["rows"])
    assert result["independent_generation_count"] is None
    assert result["cache_control_effect_established"] is False
    assert result["separate_control_effects_established"] is False
    assert scored(probe.packet, probe.output) == result
    logs = "".join((probe.output / n).read_text() for n in LOGS)
    assert all(
        x not in logs
        for x in ("IN_MEMORY_SECRET", "Authorization", "never logged", "opaque-test-key")
    )
    ledger = json.loads((probe.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 39 and ledger["new_logged_dispatches"] == 4


def test_reuse_under_joint_controls_is_retained_without_resend(probe):
    calls = []
    result = run_probe(probe, calls, repeated=True)
    assert len(calls) == 4 and result["protocol_complete"]
    assert result["return_reuse_groups"]["id_created_text"] == [
        ["slot_1", "slot_2", "slot_3", "slot_4"]
    ]
    assert result["independent_generation_count"] is None


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"id": "", "created": 123},
        {"id": "  ", "created": 123},
        {"id": "normal"},
        {"id": "normal", "created": ""},
        {"id": "normal", "created": True},
        {"id": "normal", "created": "123"},
        {"id": "normal", "created": -1},
    ],
)
def test_missing_or_invalid_identity_metadata_keeps_strong_signature_unknown(probe, metadata):
    calls = []
    result = run_probe(probe, calls, metadata_override=metadata)
    assert result["protocol_complete"] and len(calls) == 4
    assert result["return_reuse_groups"]["id_created_text"] == []
    assert result["return_reuse_groups"]["text_sha256"] == [
        ["slot_1", "slot_2", "slot_3", "slot_4"]
    ]
    assert result["unobserved_strong_signature_slots"] == ["slot_1", "slot_2", "slot_3", "slot_4"]
    if not metadata.get("id", "").strip():
        assert result["return_reuse_groups"]["provider_id"] == []
    if type(metadata.get("created")) is not int or metadata["created"] < 0:
        assert result["return_reuse_groups"]["created"] == []


@pytest.mark.parametrize("failure", ["http", "format", "model", "mode"])
def test_first_protocol_failure_stops_and_keeps_other_slots_unknown(probe, failure):
    calls = []
    result = run_probe(probe, calls, failure=failure)
    assert len(calls) == 2 and not result["protocol_complete"]
    assert result["logged_dispatch_attempts"] == 2
    assert result["rows"][0]["compatible"] and not result["rows"][1]["compatible"]
    assert [r["status"] for r in result["rows"][2:]] == ["unsubmitted", "unsubmitted"]
    ledger = json.loads((probe.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 37
    assert ledger["status"] == "stopped_pending_audit"
    save(probe.output / "execution_state.json", {"status": "complete"})
    assert scored(probe.packet, probe.output) == result


def test_cancellation_records_attempt_and_never_resumes(probe):
    calls = []
    with pytest.raises(asyncio.CancelledError):
        run_probe(probe, calls, failure="cancel")
    assert len(calls) == 2
    assert read_jsonl(probe.output / LOGS[2])[-1]["http_status"] == "cancelled_in_flight"
    ledger = json.loads((probe.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 37
    with pytest.raises(FileExistsError):
        run_probe(probe, calls)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["messages", "cache", "order", "hash"])
def test_resigned_wire_mutations_refused_before_any_http(probe, mutation):
    packet = deepcopy(probe.packet)
    if mutation == "messages":
        packet["requests"][1]["payload"]["messages"][1]["content"] = "choose a preferred answer"
    elif mutation == "cache":
        packet["requests"][1]["payload"]["cache"]["no-cache"] = False
    elif mutation == "order":
        packet["requests"].reverse()
    else:
        packet["requests"][0]["request_hash"] = "wrong"
    packet["packet_sha256"] = digest({k: v for k, v in packet.items() if k != "packet_sha256"})
    calls = []

    async def go():
        async with mock_client(calls) as client:
            await collect(packet, probe.workspace, client)

    with pytest.raises(ValueError):
        asyncio.run(go())
    assert calls == [] and not probe.output.exists()


@pytest.mark.parametrize("mutation", ["payload", "retry", "clock", "orphan", "order"])
def test_raw_log_mutations_cannot_establish_protocol_completion(probe, mutation):
    run_probe(probe, [])
    requests, attempts, responses = [read_jsonl(probe.output / n) for n in LOGS]
    if mutation == "payload":
        requests[1]["payload"].pop("cache")
    elif mutation == "retry":
        attempts[1]["retry_count"] = 1
    elif mutation == "clock":
        attempts[1]["dispatched_at"] = attempts[0]["dispatched_at"]
        responses[1]["requested_at"] = attempts[0]["dispatched_at"]
    elif mutation == "orphan":
        attempts.pop()
    else:
        requests.reverse()
    with pytest.raises(ValueError):
        evaluate(probe.packet, requests, attempts, responses)


def test_prepare_does_not_send_and_synthetic_packet_cannot_read_live_key(probe, monkeypatch):
    path = probe.workspace / "plan.json"
    save(path, probe.plan)
    folder = probe.workspace / "prepared"
    assert prepare(SimpleNamespace(workspace=probe.workspace, plan=path, output=folder)) == {
        "status": "prepared_not_sent",
        "max_new_POSTs": 4,
    }
    assert not probe.output.exists()
    packet = load_packet(probe.workspace, folder)
    assert packet == probe.packet and len(read_jsonl(folder / "expected_requests.jsonl")) == 4
    import dotenv

    def forbidden(*args, **kwargs):
        raise AssertionError("must not read credentials")

    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    with pytest.raises(ValueError, match="tool test not live"):
        asyncio.run(execute(SimpleNamespace(workspace=probe.workspace, package=folder)))
