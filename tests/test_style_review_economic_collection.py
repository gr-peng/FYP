"""Synthetic transport and arithmetic checks; never actual model or human evidence."""

import asyncio
import inspect
import json
import socket
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import dotenv
import httpx
import pytest

from scripts import collect_style_review_economic_controls as probe


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic checks must not access credentials or a real network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


@pytest.fixture
def packet():
    return probe.make_packet("synthetic_tool_test")


def target_for(request, policy):
    alternate = "technical" if request["from_style"] == "fundamental" else "fundamental"
    if policy == "economic":
        return alternate if request["condition"] == "positive" else request["from_style"]
    if policy == "reverse":
        return request["from_style"] if request["condition"] == "positive" else alternate
    if policy == "variable":
        return "technical" if request["round"] == 2 else "fundamental"
    return policy


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


def prefix(packet, count=1, policy="fundamental"):
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
                "response": body_for(request, i, target_for(request, policy)),
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    return requests, attempts, responses


async def mock_collection(
    packet,
    output,
    *,
    failure=None,
    failure_index=0,
    constant_id=False,
    policy="fundamental",
    cap=145,
):
    calls = {"get": 0, "post": 0}

    async def respond(request):
        if request.method == "GET":
            calls["get"] += 1
            if failure == "registry":
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": packet["model"]}]})
        number = calls["post"]
        calls["post"] += 1
        planned = packet["requests"][number]
        assert json.loads(request.content) == planned["payload"]
        body = body_for(planned, 0 if constant_id else number, target_for(planned, policy))
        if number == failure_index:
            if failure == "transport":
                raise httpx.ConnectError("Fictional failure")
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
                body["model"] = "wrong-model"
            elif failure == "role":
                choice["message"]["role"] = "user"
            elif failure == "finish":
                choice["finish_reason"] = "length"
            elif failure == "two_choices":
                body["choices"].append(deepcopy(choice))
            elif failure == "inner_duplicate":
                choice["message"]["content"] = (
                    '{"decision":"stay","decision":"switch","target_style":"technical",'
                    '"rationale":"fictional"}'
                )
            elif failure == "inner_nonfinite":
                choice["message"]["content"] = '{"decision":"stay","target_style":NaN}'
            elif failure == "inconsistent":
                choice["message"]["content"] = json.dumps(
                    {
                        "decision": "switch",
                        "target_style": planned["from_style"],
                        "rationale": "Fictional inconsistent response.",
                    }
                )
            elif failure == "mode":
                choice["message"]["reasoning_content"] = "Fictional reasoning"
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
                "prior_POSTs": 133,
                "global_cap": cap,
                "grant_source": "FICTIONAL_TEST_ONLY_NOT_AUTHORIZATION",
            },
        )
    return result, calls


def test_schedule_repeats_exact_four_payloads_without_local_rule_answer(packet):
    materials, _, _ = probe.checked_materials()
    accepted = {case["case_id"]: case["payload"] for case in materials["cases"]}
    requests = packet["requests"]
    assert len(requests) == len({r["slot_id"] for r in requests}) == 12
    assert len({r["request_hash"] for r in requests}) == 12
    assert Counter(r["case_id"] for r in requests) == {key: 3 for key in accepted}
    for row in requests:
        assert row["payload"] == accepted[row["case_id"]]
        assert "rule_style_reference" not in probe.materials.canonical(row)
        assert "construction_not_sent" not in probe.materials.canonical(row)
    for agent in ("A000", "A001"):
        grouped = [r["condition"] for r in requests if r["agent_id"] == agent]
        assert grouped == (
            ["positive", "negative", "negative", "positive", "positive", "negative"]
            if agent == "A000"
            else ["negative", "positive", "positive", "negative", "negative", "positive"]
        )
    assert Counter((r["condition"], r["within_account_round_position"]) for r in requests) == {
        (condition, position): 3 for condition in ("positive", "negative") for position in (1, 2)
    }


@pytest.mark.parametrize(
    "policy,primary,raw_f",
    [
        ("economic", [1, 1], [-1, 1]),
        ("reverse", [-1, -1], [1, -1]),
        ("fundamental", [0, 0], [0, 0]),
        ("technical", [0, 0], [0, 0]),
    ],
)
def test_nonzero_direction_and_constant_choices_are_calculated_per_original_style(
    packet, tmp_path, policy, primary, raw_f
):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / policy, policy=policy))
    assert calls == {"get": 1, "post": 12}
    assert result["protocol_complete"] and result["logged_dispatch_attempts"] == 12
    assert [
        r["alternative_fraction_positive_minus_negative"] for r in result["contrasts"]
    ] == primary
    assert [r["fundamental_fraction_positive_minus_negative"] for r in result["contrasts"]] == raw_f
    assert all(c["valid"] == c["planned"] == 3 for c in result["cells"])
    assert result["independent_generation_count"] is None
    assert result["external_model_provenance_authenticated"] is False
    assert "本地假回答" in probe.describe(result)
    ledger = json.loads((tmp_path / policy / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 145  # Fictional ledger only in this test's temp.
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
def test_first_invalid_return_stops_without_repair_or_counting_as_stay(packet, tmp_path, failure):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / failure, failure=failure))
    assert calls["post"] == 1 and not result["protocol_complete"]
    assert result["rows"][0]["choice"] is None
    assert sum(r["status"] == "unsubmitted" for r in result["rows"]) == 11
    assert all(
        c["alternative_fraction_positive_minus_negative"] is None for c in result["contrasts"]
    )
    assert sum(c["stay_count"] + c["switch_count"] for c in result["cells"]) == 0
    if failure == "inconsistent":
        assert result["rows"][0]["parsed_choice"] is not None
        assert result["rows"][0]["answer_status"] == "inconsistent_style"


def test_final_invalid_return_keeps_partial_cell_out_of_main_contrast(packet, tmp_path):
    result, calls = asyncio.run(
        mock_collection(packet, tmp_path / "last", failure="inconsistent", failure_index=11)
    )
    assert calls["post"] == 12 and not result["protocol_complete"]
    failed = next(c for c in result["cells"] if c["invalid"])
    assert failed["valid"] == 2 and failed["invalid"] == 1
    contrast = next(c for c in result["contrasts"] if c["agent_id"] == failed["agent_id"])
    assert contrast["alternative_fraction_positive_minus_negative"] is None
    assert "未知" in probe.describe(result)


def test_cancelled_request_counted_once_no_resume(packet, tmp_path):
    output = tmp_path / "cancelled"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mock_collection(packet, output, failure="cancel"))
    result = probe.evaluate(packet, *(probe.read_jsonl(output / name) for name in probe.LOGS))
    assert result["logged_dispatch_attempts"] == 1
    assert result["rows"][0]["status"] == "transport_failed"
    assert sum(c["unsubmitted"] for c in result["cells"]) == 11
    assert json.loads((output / "budget_ledger.json").read_text())["used_completion_POSTs"] == 134
    assert json.loads((output / "execution_state.json").read_text())["status"] == "cancelled"
    with pytest.raises(FileExistsError):
        asyncio.run(mock_collection(packet, output))


def test_registry_failure_and_short_quota_do_not_send_post(packet, tmp_path):
    result, calls = asyncio.run(mock_collection(packet, tmp_path / "registry", failure="registry"))
    assert calls == {"get": 1, "post": 0}
    assert result["logged_dispatch_attempts"] == 0
    with pytest.raises(ValueError, match="exceeds authorized cap"):
        asyncio.run(mock_collection(packet, tmp_path / "short", cap=144))
    assert not (tmp_path / "short").exists()


def test_expected_and_pending_records_never_become_responses_or_missing_stays(packet):
    requests, attempts, _ = prefix(packet, 1)
    for logs, status in (
        ([requests, [], []], "request_record_only"),
        ([requests, attempts, []], "awaiting_response"),
    ):
        result = probe.evaluate(packet, *logs)
        assert result["rows"][0]["status"] == status
        assert all(c["stay_count"] == c["switch_count"] == c["valid"] == 0 for c in result["cells"])
        assert all(
            c["alternative_fraction_positive_minus_negative"] is None for c in result["contrasts"]
        )


def test_reuse_unknown_metadata_and_variability_are_retained_without_replacements(packet, tmp_path):
    repeated, calls = asyncio.run(mock_collection(packet, tmp_path / "reuse", constant_id=True))
    assert calls["post"] == 12 and repeated["protocol_complete"]
    assert repeated["repeated_strong_signature_groups"]
    unknown, calls = asyncio.run(
        mock_collection(packet, tmp_path / "unknown", failure="unknown_signature")
    )
    assert calls["post"] == 12 and unknown["protocol_complete"]
    assert unknown["unknown_signature_slots"] == [packet["requests"][0]["slot_id"]]
    variable, calls = asyncio.run(
        mock_collection(packet, tmp_path / "variation", policy="variable")
    )
    assert calls["post"] == 12 and variable["protocol_complete"]
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
        "http_bool",
    ],
)
def test_log_mutations_rejected(packet, tamper):
    requests, attempts, responses = prefix(packet, 2)
    if tamper == "wire":
        requests[0]["payload"]["temperature"] = 0
    elif tamper == "resign":
        requests[0]["payload"]["model"] = "another"
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
        attempts[0]["answer_key"] = "fundamental"
    elif tamper == "http_bool":
        responses[0]["http_status"] = True
    else:
        responses[0]["response"]["model"] = "wrong-model"
    with pytest.raises(ValueError):
        probe.evaluate(packet, requests, attempts, responses)


def test_resigning_changed_packet_cannot_override_accepted_material_or_source(packet, monkeypatch):
    for field in ("requests", "source_receipt", "adapter_sources"):
        changed = deepcopy(packet)
        if field == "requests":
            changed[field][0]["payload"]["temperature"] = 0.123
        else:
            changed[field][next(iter(changed[field]))]["sha256"] = "0" * 64
        changed["packet_sha256"] = probe.materials.digest(
            {k: v for k, v in changed.items() if k != "packet_sha256"}
        )
        with pytest.raises(ValueError, match="packet differs"):
            probe.check_packet(changed)
    original = probe.fingerprint

    def altered(path):
        result = original(path)
        if path.name == "prompts.py":
            result["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(probe, "fingerprint", altered)
    with pytest.raises(ValueError, match="accepted source differs"):
        probe.make_packet()


def grant_for(packet):
    return {
        "decision": "approved_by_user",
        "scope": probe.grant_scope(packet),
        "evidence": {
            "kind": "explicit_user_reply",
            "source_message_ref": "FICTIONAL_TEST_ONLY",
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
        "no_message_ref",
        "expired_hour",
    ],
)
def test_invalid_grant_fails_before_credentials_or_client(tmp_path, monkeypatch, failure):
    folder = tmp_path / "prepared"
    real = probe.prepare(folder)
    grant = grant_for(real)
    if failure == "numeric_only":
        grant = {"prior_POSTs": 133, "global_cap": 145}
    elif failure == "short_cap":
        grant["scope"]["global_cap"] = 144
    elif failure == "wrong_packet":
        grant["scope"]["packet_sha256"] = "0" * 64
    elif failure == "no_evidence":
        grant["evidence"] = {}
    elif failure == "pending":
        grant["decision"] = "pending"
    elif failure == "negative_reply":
        grant["evidence"]["verbatim_reply"] = "不允许"
    elif failure == "wrong_question":
        grant["evidence"]["question"] = "An unrelated question."
    elif failure == "no_message_ref":
        grant["evidence"]["source_message_ref"] = ""
    elif failure == "expired_hour":
        grant["evidence"] = {
            "kind": "explicit_user_hour",
            "source_message_ref": "OLD_MESSAGE",
            "verbatim_reply": probe.HOUR_APPROVAL_REPLY,
            "received_at_utc": "2026-10-23T16:00:53+00:00",
            "expires_at_utc": "2026-10-23T17:00:53+00:00",
        }
    else:
        fake = probe.make_packet("synthetic_tool_test")
        probe.save(folder / "packet.json", fake)
        (folder / "expected_requests.jsonl").write_text(
            "".join(probe.materials.canonical(r) + "\n" for r in fake["requests"])
        )
        grant = grant_for(fake)
    receipt = tmp_path / "fictional_grant.json"
    probe.save(receipt, grant)
    monkeypatch.setattr(probe, "GRANT_PATH", receipt)

    def no_client(*args, **kwargs):
        pytest.fail("Client created before authorization passed")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_client)
    with pytest.raises(ValueError):
        asyncio.run(probe.execute(folder, receipt))


def test_specific_affirmative_reply_is_bound_to_complete_scope():
    real = probe.make_packet()
    grant = grant_for(real)
    grant["evidence"]["verbatim_reply"] = "同意"
    result = probe.verify_authorization(real, grant)
    assert result["prior_POSTs"] == 133 and result["global_cap"] == 145
    assert result["expires_at_utc"] is None


@pytest.mark.parametrize("offset,allowed", [(-1, False), (0, True), (3599, True), (3600, False)])
def test_hour_window_never_extends_and_deadline_blocks_transport(monkeypatch, offset, allowed):
    real = probe.make_packet()
    grant = grant_for(real)
    start = datetime(2026, 1, 10, tzinfo=UTC)
    end = start + timedelta(hours=1)
    grant["evidence"] = {
        "kind": "explicit_user_hour",
        "source_message_ref": "FICTIONAL_HOUR",
        "verbatim_reply": probe.HOUR_APPROVAL_REPLY,
        "received_at_utc": start.isoformat(),
        "expires_at_utc": end.isoformat(),
    }
    monkeypatch.setattr(probe, "now_utc", lambda: start + timedelta(seconds=offset))
    if allowed:
        assert probe.verify_authorization(real, grant)["expires_at_utc"] == end.isoformat()
    else:
        with pytest.raises(ValueError, match="future or expired"):
            probe.verify_authorization(real, grant)
    monkeypatch.setattr(probe, "now_utc", lambda: end)
    with pytest.raises(ValueError, match="expired"):
        asyncio.run(probe.deadline_hook(end.isoformat())(httpx.Request("GET", probe.ENDPOINT)))
    grant["evidence"]["expires_at_utc"] = (end + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="cannot be extended"):
        probe.verify_authorization(real, grant)


def test_prepared_expected_records_remain_unsubmitted_and_cannot_overwrite(tmp_path):
    folder = tmp_path / "planned"
    packet = probe.prepare(folder)
    assert probe.load_packet(folder) == packet
    pending = json.loads((folder / "pending_evaluation.json").read_text())
    assert pending["logged_dispatch_attempts"] == 0 and not pending["protocol_complete"]
    assert all(c["unsubmitted"] == 3 and c["valid"] == 0 for c in pending["cells"])
    assert all(
        c["alternative_fraction_positive_minus_negative"] is None for c in pending["contrasts"]
    )
    with pytest.raises(FileExistsError):
        probe.prepare(folder)
    (folder / "expected_requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="expected requests differ"):
        probe.load_packet(folder)


def test_live_client_preserves_environment_proxy_tls_and_zero_retry_defaults(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    assert inspect.signature(httpx.AsyncHTTPTransport).parameters["retries"].default == 0

    async def inspect_client_without_sending():
        async with probe.authorized_http_client("FICTIONAL_NOT_A_REAL_KEY", None) as client:
            assert client._trust_env is True
            assert client.follow_redirects is False
            transport = client._transport_for_url(httpx.URL(probe.ENDPOINT))
            assert transport is not client._transport
            assert transport._pool._retries == 0
            assert transport._pool._ssl_context.verify_mode != 0
            assert transport._pool._ssl_context.check_hostname is True

    asyncio.run(inspect_client_without_sending())
