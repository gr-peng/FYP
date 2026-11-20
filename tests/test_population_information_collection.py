"""Bounded synthetic transport tests. No real credentials, HTTP, data or model evidence."""

import asyncio
import json
import socket
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import dotenv
import httpx
import pytest

from scripts import collect_population_information_probe as probe


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This synthetic test must not use real networking or credentials")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


@pytest.fixture
def packet():
    return probe.make_packet("synthetic_tool_test")


def body_for(request, number, target="fundamental"):
    return {
        "id": f"fictional-response-{number}",
        "created": 1768000000 + number,
        "model": request["payload"]["model"],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": json.dumps(
                        {
                            "decision": "stay" if target == request["from_style"] else "switch",
                            "target_style": target,
                            "rationale": "Fictional scripted response.",
                        }
                    ),
                },
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    }


def prefix(packet, count=1):
    requests, attempts, responses = [], [], []
    beginning = datetime(2026, 1, 10, tzinfo=UTC)
    for i, planned in enumerate(packet["requests"][:count]):
        request = deepcopy(planned)
        origin = probe.identity(request)
        start = beginning + timedelta(seconds=2 * i)
        requests.append(request)
        attempts.append({**origin, "dispatched_at": start.isoformat(), "retry_count": 0})
        responses.append(
            {
                **origin,
                "requested_at": start.isoformat(),
                "response_received_at": (start + timedelta(seconds=1)).isoformat(),
                "retry_count": 0,
                "cached": False,
                "http_status": 200,
                "latency_seconds": 1,
                "response": body_for(request, i),
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    return requests, attempts, responses


async def mock_collection(packet, output, *, failure=None, failure_index=0, constant_id=False):
    calls = {"get": 0, "post": 0}

    async def respond(request):
        if request.method == "GET":
            calls["get"] += 1
            if failure == "registry":
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": packet["model"]}]})
        number = calls["post"]
        planned = packet["requests"][number]
        assert json.loads(request.content) == planned["payload"]
        calls["post"] += 1
        body = body_for(planned, 0 if constant_id else number)
        if failure == "variability":
            body = body_for(
                planned, number, "technical" if planned["round"] == 2 else "fundamental"
            )
        if number == failure_index:
            if failure == "transport":
                raise httpx.ReadError("Fictional transport failure")
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "http":
                return httpx.Response(429)
            if failure == "outer_duplicate":
                return httpx.Response(200, text='{"model":"x","model":"x"}')
            if failure == "outer_nonfinite":
                return httpx.Response(200, text='{"model":NaN}')
            choice = body["choices"][0]
            if failure == "model":
                body["model"] = "not-the-requested-model"
            elif failure == "role":
                choice["message"]["role"] = "user"
            elif failure == "finish":
                choice["finish_reason"] = "length"
            elif failure == "two_choices":
                body["choices"].append(deepcopy(choice))
            elif failure == "inner_duplicate":
                choice["message"]["content"] = (
                    '{"decision":"stay","decision":"switch","target_style":"technical","rationale":"fake"}'
                )
            elif failure == "inner_nonfinite":
                choice["message"]["content"] = (
                    '{"decision":"stay","target_style":NaN,"rationale":"fake"}'
                )
            elif failure == "inconsistent":
                choice["message"]["content"] = json.dumps(
                    {
                        "decision": "switch",
                        "target_style": planned["from_style"],
                        "rationale": "fake",
                    }
                )
            elif failure == "mode":
                choice["message"]["reasoning_content"] = "Fictional returned reasoning."
            elif failure == "mode_bool":
                body["usage"]["completion_tokens_details"]["reasoning_tokens"] = False
            elif failure == "unknown_signature":
                body.pop("id")
                body.pop("created")
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as client:
        result = await probe.collect(
            packet,
            output,
            client,
            None,
            probe.evaluate,
            {
                "prior_POSTs": 115,
                "global_cap": 133,
                "grant_source": "FICTIONAL_TEST_NOT_AUTHORIZATION",
            },
        )
    return result, calls


def test_complete_eighteen_constant_targets_are_valid_and_do_not_mean_eighteen_people(
    packet, tmp_path
):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / "fictional"))
    assert calls == {"get": 1, "post": 18}
    assert result["protocol_complete"] is True
    assert result["logged_dispatch_attempts"] == result["planned"] == 18
    assert result["evidence_kind"] == "synthetic_tool_test"
    assert result["independent_generation_count"] is None
    assert result["external_model_provenance_authenticated"] is False
    assert all(c["valid"] == c["planned"] == 3 for c in result["cells"])
    assert all(c["target_counts"] == {"fundamental": 3, "technical": 0} for c in result["cells"])
    assert all(c["stay_count"] == 3 for c in result["cells"] if c["agent_id"] == "A000")
    assert all(c["switch_count"] == 3 for c in result["cells"] if c["agent_id"] == "A001")
    assert all(
        c["complete"] and c["fundamental_fraction_difference"] == 0 for c in result["contrasts"]
    )
    ledger = json.loads((tmp_path / "fictional/budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 133  # Fictional ledger in this fresh temp only.
    assert ledger["reserved_next_completion_POSTs"] == 0
    assert ledger["no_resume"] is True


@pytest.mark.parametrize(
    "failure",
    [
        "http",
        "transport",
        "outer_duplicate",
        "outer_nonfinite",
        "model",
        "role",
        "finish",
        "two_choices",
        "inner_duplicate",
        "inner_nonfinite",
        "inconsistent",
        "mode",
        "mode_bool",
    ],
)
def test_first_bad_return_stops_without_repair_or_retry(packet, tmp_path, failure):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / failure, failure=failure))
    assert calls["post"] == 1
    assert not result["protocol_complete"]
    assert result["rows"][0]["choice"] is None
    assert result["rows"][0]["compatible"] is False
    assert sum(r["status"] == "unsubmitted" for r in result["rows"]) == 17
    assert all(c["fundamental_fraction_difference"] is None for c in result["contrasts"])
    if failure == "inconsistent":
        assert result["rows"][0]["parsed_choice"] is not None
        assert result["rows"][0]["answer_status"] == "inconsistent_style"
        assert result["cells"][0]["valid"] == 0


def test_last_bad_return_cannot_claim_complete(packet, tmp_path):
    result, calls = asyncio.run(
        mock_collection(packet, tmp_path / "last", failure="inconsistent", failure_index=17)
    )
    assert calls["post"] == 18
    assert not result["protocol_complete"]
    last_case = packet["requests"][-1]
    cell = next(
        c
        for c in result["cells"]
        if (c["agent_id"], c["condition"]) == (last_case["agent_id"], last_case["condition"])
    )
    assert cell["valid"] == 2 and cell["invalid"] == 1
    assert all(
        c["fundamental_fraction_difference"] is None
        for c in result["contrasts"]
        if c["agent_id"] == cell["agent_id"] and cell["condition"] in (c["left"], c["right"])
    )


def test_cancellation_counts_once_preserves_unsent_and_cannot_resume(packet, tmp_path):
    output = tmp_path / "cancelled"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mock_collection(packet, output, failure="cancel"))
    logs = [probe.read_jsonl(output / n) for n in probe.LOGS]
    result = probe.evaluate(packet, *logs)
    assert result["logged_dispatch_attempts"] == 1
    assert result["rows"][0]["status"] == "transport_failed"
    assert sum(r["status"] == "unsubmitted" for r in result["rows"]) == 17
    assert json.loads((output / "execution_state.json").read_text())["status"] == "cancelled"
    assert json.loads((output / "budget_ledger.json").read_text())["used_completion_POSTs"] == 116
    with pytest.raises(FileExistsError):
        asyncio.run(mock_collection(packet, output))


def test_registry_failure_uses_no_post(packet, tmp_path):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / "registry", failure="registry"))
    assert calls == {"get": 1, "post": 0}
    assert result["logged_dispatch_attempts"] == 0
    assert all(r["status"] == "unsubmitted" for r in result["rows"])


def test_partial_observations_do_not_create_a_paired_effect_or_missing_stays(packet):
    result = probe.evaluate(packet, *prefix(packet, 2))
    assert sum(c["valid"] for c in result["cells"]) == 2
    assert sum(c["unsubmitted"] for c in result["cells"]) == 16
    assert sum(c["stay_count"] + c["switch_count"] for c in result["cells"]) == 2
    assert all(c["fundamental_fraction_difference"] is None for c in result["contrasts"])
    hidden = [c for c in result["cells"] if c["condition"] == "hidden"]
    assert all(c["fundamental_fraction_descriptive"] is None for c in hidden)
    requests, attempts, responses = prefix(packet)
    assert probe.evaluate(packet, requests, [], [])["rows"][0]["status"] == "request_record_only"
    assert (
        probe.evaluate(packet, requests, attempts, [])["rows"][0]["status"] == "awaiting_response"
    )


def test_repeated_return_and_unknown_metadata_are_diagnostics_not_retries(packet, tmp_path):
    repeated, calls = asyncio.run(mock_collection(packet, tmp_path / "reuse", constant_id=True))
    assert calls["post"] == 18 and repeated["protocol_complete"]
    assert repeated["repeated_strong_signature_groups"]
    assert repeated["independent_generation_count"] is None
    unknown, calls = asyncio.run(
        mock_collection(packet, tmp_path / "unknown", failure="unknown_signature")
    )
    assert calls["post"] == 18 and unknown["protocol_complete"]
    assert unknown["unknown_signature_slots"] == [packet["requests"][0]["slot_id"]]
    variable, calls = asyncio.run(
        mock_collection(packet, tmp_path / "variable", failure="variability")
    )
    assert calls["post"] == 18 and variable["protocol_complete"]
    assert all(c["different_targets_on_same_input"] for c in variable["cells"])


@pytest.mark.parametrize(
    "tamper",
    [
        "wire",
        "resign",
        "order",
        "orphan",
        "duplicate",
        "round_bool",
        "retry",
        "cached",
        "clock",
        "overlap",
        "latency",
        "header",
        "extra",
        "after_failure",
    ],
)
def test_raw_log_tampering_is_rejected(packet, tamper):
    requests, attempts, responses = prefix(packet, 2)
    if tamper == "wire":
        requests[0]["payload"]["temperature"] = 0
    elif tamper == "resign":
        requests[0]["payload"]["model"] = "another-model"
        requests[0]["request_hash"] = probe.materials.digest(requests[0]["payload"])
    elif tamper == "order":
        requests.reverse()
    elif tamper == "orphan":
        requests = []
    elif tamper == "duplicate":
        responses.append(deepcopy(responses[0]))
    elif tamper == "round_bool":
        attempts[0]["round"] = True
    elif tamper == "retry":
        attempts[0]["retry_count"] = 1
    elif tamper == "cached":
        responses[0]["cached"] = True
    elif tamper == "clock":
        responses[0]["response_received_at"] = "2026-01-09T00:00:00+00:00"
    elif tamper == "overlap":
        attempts[1]["dispatched_at"] = attempts[0]["dispatched_at"]
        responses[1]["requested_at"] = attempts[0]["dispatched_at"]
    elif tamper == "latency":
        responses[0]["latency_seconds"] = -1
    elif tamper == "header":
        responses[0]["cache_key_header"] = {"present": True, "value_sha256": None}
    elif tamper == "extra":
        attempts[0]["preferred_answer"] = "fundamental"
    else:
        responses[0]["response"]["model"] = "wrong-model"
    with pytest.raises(ValueError):
        probe.evaluate(packet, requests, attempts, responses)


def test_resigned_packet_or_source_receipt_cannot_change_the_fixed_materials(packet, monkeypatch):
    for field in ("requests", "source_receipt"):
        changed = deepcopy(packet)
        if field == "requests":
            changed["requests"][0]["payload"]["temperature"] = 0.123
        else:
            name = next(iter(changed["source_receipt"]))
            changed["source_receipt"][name]["sha256"] = "0" * 64
        core = {k: v for k, v in changed.items() if k != "packet_sha256"}
        changed["packet_sha256"] = probe.materials.digest(core)
        with pytest.raises(ValueError, match="packet differs"):
            probe.check_packet(changed)
    original = probe.fingerprint

    def altered_source(path):
        result = original(path)
        if path.name == "prompts.py":
            result["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(probe, "fingerprint", altered_source)
    with pytest.raises(ValueError, match="source bytes differ"):
        probe.make_packet()


def grant_for(packet):
    return {
        "decision": "approved_by_user",
        "scope": probe.grant_scope(packet),
        "evidence": {
            "kind": "explicit_user_reply",
            "question_item_id": "FICTIONAL_TEST_ONLY",
            "question": probe.APPROVAL_QUESTION,
            "verbatim_reply": probe.APPROVAL_REPLY,
            "received_at_utc": "2026-01-10T00:00:00+00:00",
        },
    }


@pytest.mark.parametrize(
    "failure",
    [
        "numeric_only",
        "short_cap",
        "wrong_packet",
        "no_evidence",
        "pending",
        "synthetic",
        "negative_reply",
        "wrong_question",
    ],
)
def test_invalid_authorization_fails_before_credentials_client_or_registry(
    tmp_path, monkeypatch, failure
):
    folder = tmp_path / "prepared"
    real = probe.prepare(folder)
    grant = grant_for(real)
    if failure == "numeric_only":
        grant = {"prior_POSTs": 115, "global_cap": 133}
    elif failure == "short_cap":
        grant["scope"]["global_cap"] = 132
    elif failure == "wrong_packet":
        grant["scope"]["packet_sha256"] = "0" * 64
    elif failure == "no_evidence":
        grant["evidence"] = {}
    elif failure == "pending":
        grant["decision"] = "pending"
    elif failure == "negative_reply":
        grant["evidence"]["verbatim_reply"] = "不允许"
    elif failure == "wrong_question":
        grant["evidence"]["question"] = "An unrelated permission question."
    else:
        fake = probe.make_packet("synthetic_tool_test")
        probe.save(folder / "packet.json", fake)
        (folder / "expected_requests.jsonl").write_text(
            "".join(probe.materials.canonical(r) + "\n" for r in fake["requests"])
        )
        grant = grant_for(fake)
    receipt = tmp_path / "fictional_authorization.json"
    probe.save(receipt, grant)
    monkeypatch.setattr(probe, "GRANT_PATH", receipt)

    def no_client(*args, **kwargs):
        pytest.fail("Client created before the authorization gate")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_client)
    with pytest.raises(ValueError):
        asyncio.run(probe.execute(folder, receipt))


def test_prepare_load_and_expected_records_cannot_be_mistaken_for_responses(tmp_path):
    folder = tmp_path / "planned"
    packet = probe.prepare(folder)
    assert probe.load_packet(folder) == packet
    pending = json.loads((folder / "pending_evaluation.json").read_text())
    assert pending["logged_dispatch_attempts"] == 0 and pending["protocol_complete"] is False
    assert all(c["valid"] == c["dispatched"] == 0 for c in pending["cells"])
    assert all(c["unsubmitted"] == 3 for c in pending["cells"])
    assert all(c["fundamental_fraction_difference"] is None for c in pending["contrasts"])
    with pytest.raises(FileExistsError):
        probe.prepare(folder)
    (folder / "expected_requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="expected wire records differ"):
        probe.load_packet(folder)


def hour_grant_for(packet):
    return {
        "decision": "approved_by_user",
        "scope": probe.grant_scope(packet),
        "evidence": {
            "kind": "explicit_user_message",
            "verbatim_reply": probe.HOUR_APPROVAL_REPLY,
            "received_at_utc": "2026-10-23T16:00:53+00:00",
            "expires_at_utc": "2026-10-23T17:00:53+00:00",
        },
    }


@pytest.mark.parametrize(
    "offset,allowed", [(-1, False), (0, True), (3599, True), (3600, False), (3601, False)]
)
def test_actual_hour_message_is_bounded_without_rewriting_the_reply(monkeypatch, offset, allowed):
    packet = probe.make_packet()
    grant = hour_grant_for(packet)
    start = datetime(2026, 10, 23, 16, 0, 53, tzinfo=UTC)
    monkeypatch.setattr(probe, "now_utc", lambda: start + timedelta(seconds=offset))
    if allowed:
        result = probe.verify_authorization(packet, grant)
        assert result["global_cap"] == 133 and result["prior_POSTs"] == 115
        assert result["expires_at_utc"] == grant["evidence"]["expires_at_utc"]
        assert grant["evidence"]["verbatim_reply"] == "同意之后一个小时你的所有操作"
    else:
        with pytest.raises(ValueError, match="future or expired"):
            probe.verify_authorization(packet, grant)


@pytest.mark.parametrize("tamper", ["different_reply", "two_hours", "naive_time"])
def test_hour_message_cannot_be_extended_or_fabricated(monkeypatch, tamper):
    packet = probe.make_packet()
    grant = hour_grant_for(packet)
    monkeypatch.setattr(probe, "now_utc", lambda: datetime(2026, 10, 23, 16, 1, tzinfo=UTC))
    if tamper == "different_reply":
        grant["evidence"]["verbatim_reply"] = "不允许"
    elif tamper == "two_hours":
        grant["evidence"]["expires_at_utc"] = "2026-10-23T18:00:53+00:00"
    else:
        grant["evidence"]["received_at_utc"] = "2026-10-23T16:00:53"
    with pytest.raises(ValueError):
        probe.verify_authorization(packet, grant)


def test_expired_hour_exits_before_credentials_or_client(tmp_path, monkeypatch):
    folder = tmp_path / "prepared"
    packet = probe.prepare(folder)
    receipt = tmp_path / "fictional_hour_receipt.json"
    probe.save(receipt, hour_grant_for(packet))
    monkeypatch.setattr(probe, "GRANT_PATH", receipt)
    monkeypatch.setattr(probe, "now_utc", lambda: datetime(2026, 10, 23, 17, 1, tzinfo=UTC))

    def no_client(*args, **kwargs):
        pytest.fail("Expired approval reached client construction")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_client)
    with pytest.raises(ValueError, match="future or expired"):
        asyncio.run(probe.execute(folder, receipt))


@pytest.mark.parametrize("expired", [False, True])
def test_deadline_hook_blocks_transport_before_dispatch(monkeypatch, expired):
    calls = []
    current = datetime(2026, 10, 23, 17 if expired else 16, 1, tzinfo=UTC)
    monkeypatch.setattr(probe, "now_utc", lambda: current)

    async def attempt():
        async def fictional_transport(request):
            calls.append(request.method)
            return httpx.Response(200, json={"fictional": True})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(fictional_transport),
            trust_env=False,
            event_hooks={"request": [probe.deadline_hook("2026-10-23T17:00:53+00:00")]},
        ) as client:
            await client.post("https://example.invalid/fictional", json={"fictional": True})

    if expired:
        with pytest.raises(ValueError, match="expired"):
            asyncio.run(attempt())
        assert calls == []
    else:
        asyncio.run(attempt())
        assert calls == ["POST"]
