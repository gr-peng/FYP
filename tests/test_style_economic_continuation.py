"""Synthetic checks for frozen batches, cumulative quota and diagnostic integration."""

import asyncio
import json
import shutil
import socket
from copy import deepcopy

import dotenv
import httpx
import pytest

from behavioral_market.llm.transport_diagnostics import DiagnosticClient
from scripts import collect_style_review_economic_continuation as tool


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic checks cannot read real credentials or use real networking")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


@pytest.fixture
def plan():
    return {
        "run_id": "economic-auto-fictional-test01",
        "prior_POSTs": 133,
        "route": "environment_proxy",
        "model": "DeepSeek-V4.1-Flash",
    }


@pytest.fixture
def ledger():
    return {
        "global_authorized_completion_POST_cap": 500,
        "used_completion_POSTs": 133,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 367,
        "active_run_id": None,
        "version": 1,
    }


def test_exact_payloads_and_new_identity_stay_bound_to_frozen_plan(plan, tmp_path):
    folder = tmp_path / "planned"
    packet = tool.prepare(folder, plan, "synthetic_tool_test")
    assert tool.load_packet(folder) == (plan, packet)
    old = tool.original.make_packet("synthetic_tool_test")
    expected = [deepcopy(r["payload"]) for r in old["requests"]]
    for row in expected:
        row["model"] = plan["model"]
    assert expected == [r["payload"] for r in packet["requests"]]
    assert not (
        {r["slot_id"] for r in old["requests"]} & {r["slot_id"] for r in packet["requests"]}
    )
    assert packet["budget_basis"] == {"prior_POSTs": 133, "global_cap": 500}
    assert len(packet["source_receipt"]) + len(packet["adapter_sources"]) == 53
    pending = tool.evaluator(plan)(packet, [], [], [])
    assert pending["logged_dispatch_attempts"] == 0
    assert all(c["unsubmitted"] == 3 for c in pending["cells"])
    assert pending["requested_model"] == "DeepSeek-V4.1-Flash"
    assert pending["model_version_changed_from_old18"] is True
    assert "DeepSeek-V4.1-Flash" in tool.describe(pending)
    with pytest.raises(FileExistsError):
        tool.prepare(folder, plan)


@pytest.mark.parametrize("change", ["wire", "plan", "old_run", "source"])
def test_resigning_cannot_override_external_frozen_plan_or_source(plan, change):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    if change == "wire":
        packet["requests"][0]["payload"]["temperature"] = 0
    elif change == "plan":
        packet["study_plan"]["prior_POSTs"] += 1
    elif change == "old_run":
        packet = tool.original.make_packet("synthetic_tool_test")
    else:
        packet["adapter_sources"][next(iter(packet["adapter_sources"]))]["sha256"] = "0" * 64
    packet["packet_sha256"] = tool.digest({k: v for k, v in packet.items() if k != "packet_sha256"})
    with pytest.raises(ValueError, match="packet differs"):
        tool.evaluator(plan)(packet, [], [], [])


def test_original_default_validator_remains_strict():
    old = tool.original.make_packet("synthetic_tool_test")
    old["requests"][0]["payload"]["temperature"] = 0
    old["packet_sha256"] = tool.digest({k: v for k, v in old.items() if k != "packet_sha256"})
    with pytest.raises(ValueError, match="packet differs"):
        tool.original.evaluate(old, [], [], [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_authorized_completion_POST_cap", 501),
        ("used_completion_POSTs", 132),
        ("used_completion_POSTs", True),
        ("reserved_next_completion_POSTs", 1),
        ("available_unreserved_POSTs", 368),
        ("active_run_id", "another-run"),
    ],
)
def test_budget_refuses_changed_cap_stale_count_or_active_run(plan, ledger, field, value):
    ledger[field] = value
    with pytest.raises(ValueError):
        tool.check_budget(ledger, plan)


def test_whole_batch_must_fit_and_lock_cannot_be_acquired_twice(plan, ledger, tmp_path):
    valid = {**plan, "prior_POSTs": 488}
    tool.validate_plan(valid)
    tool.check_budget(
        {**ledger, "used_completion_POSTs": 488, "available_unreserved_POSTs": 12}, valid
    )
    with pytest.raises(ValueError):
        tool.validate_plan({**plan, "prior_POSTs": 489})
    with pytest.raises(ValueError):
        tool.validate_plan({**plan, "prior_POSTs": True})
    path = tmp_path / "budget.json"
    with tool.exclusive_budget(path):
        with pytest.raises(BlockingIOError):
            with tool.exclusive_budget(path):
                pytest.fail("A second simultaneous batch acquired the budget")


def body(request, index):
    target = "technical" if request["from_style"] == "fundamental" else "fundamental"
    if request["condition"] == "negative":
        target = request["from_style"]
    return {
        "id": f"fictional-{index}",
        "created": 1768000000 + index,
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
                            "rationale": "Fictional scripted output.",
                        }
                    ),
                },
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    }


async def collect_fictional(plan, output, diagnostic_dir, failure):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    calls = []

    async def respond(request):
        calls.append(request.method)
        if request.method == "GET":
            if failure == "registry":
                raise httpx.ConnectError("FAKE_SECRET_UNSAFE_EXCEPTION_TEXT")
            return httpx.Response(200, json={"data": [{"id": packet["model"]}]})
        index = calls.count("POST") - 1
        if failure == "post":
            raise httpx.ConnectError("FAKE_SECRET_UNSAFE_EXCEPTION_TEXT")
        assert json.loads(request.content) == packet["requests"][index]["payload"]
        return httpx.Response(200, json=body(packet["requests"][index], index))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as client:
        wrapped = DiagnosticClient(client, diagnostic_dir, route="synthetic_transport")
        result = await tool.original.collect(
            packet,
            output,
            wrapped,
            None,
            tool.evaluator(plan),
            {"prior_POSTs": 133, "global_cap": 500, "grant_source": "FICTIONAL_NOT_AUTHORIZATION"},
        )
    return result, calls


@pytest.mark.parametrize("failure,posts", [(None, 12), ("registry", 0), ("post", 1)])
def test_integrated_diagnostics_stop_and_conservative_budget(
    plan, ledger, tmp_path, failure, posts
):
    output = tmp_path / "output"
    diag = tmp_path / "diagnostics"
    result, calls = asyncio.run(collect_fictional(plan, output, diag, failure))
    assert calls.count("GET") == 1 and calls.count("POST") == posts
    assert result["logged_dispatch_attempts"] == posts
    terminal = tool.terminal_budget(ledger, plan, output)
    assert terminal["used_completion_POSTs"] == 133 + posts
    assert terminal["available_unreserved_POSTs"] == 367 - posts
    assert terminal["reserved_next_completion_POSTs"] == 0 and terminal["active_run_id"] is None
    assert "FAKE_SECRET" not in "".join(p.read_text() for p in diag.iterdir())
    if failure is None:
        assert result["protocol_complete"]
        assert [c["alternative_fraction_positive_minus_negative"] for c in result["contrasts"]] == [
            1,
            1,
        ]
        assert len(list(diag.iterdir())) == 26
    else:
        assert not result["protocol_complete"]
        assert all(
            c["alternative_fraction_positive_minus_negative"] is None for c in result["contrasts"]
        )
    with pytest.raises(FileExistsError):
        asyncio.run(collect_fictional(plan, output, tmp_path / "other-diag", None))


def test_budget_counts_raw_attempts_not_summary_flags(plan, ledger, tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    tool.save(output / "execution_state.json", {"status": "complete", "attempts": 12})
    assert tool.terminal_budget(ledger, plan, output)["used_completion_POSTs"] == 133
    path = tmp_path / "ledger.json"
    tool.save_budget(path, ledger)
    assert tool.read(path) == ledger
    assert not path.with_name(path.name + ".pending").exists()


def test_synthetic_or_unreviewed_execution_stops_before_credentials(plan, tmp_path, monkeypatch):
    folder = tmp_path / plan["run_id"]
    tool.prepare(folder, plan, "synthetic_tool_test")
    monkeypatch.setattr(tool, "AUTO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="synthetic packet"):
        asyncio.run(tool.execute(folder))
    assert not (folder / "live_execution").exists()
    assert not (folder / "transport_diagnostics").exists()


def test_invalid_user_grant_cannot_be_replaced_by_numeric_quota(plan, tmp_path, monkeypatch):
    changed = deepcopy(tool.read(tool.GRANT_PATH))
    changed["verbatim_reply"] = "Not a user authorization"
    path = tmp_path / "fake-receipt.json"
    tool.save(path, changed)
    monkeypatch.setattr(tool, "GRANT_PATH", path)
    with pytest.raises(ValueError, match="actual cumulative500"):
        tool.make_packet(plan, "synthetic_tool_test")


@pytest.mark.parametrize("failure", ["missing_key", "wrong_route"])
def test_pre_http_failure_is_terminal_and_releases_temp_budget(
    plan, ledger, tmp_path, monkeypatch, failure
):
    folder = tmp_path / plan["run_id"]
    packet = tool.prepare(folder, plan)
    # Fictional preflight is confined to this temp harness; no real client may send.
    tool.save(
        folder / "independent_preflight.json",
        {"accepted_for_execution": True, "mustfix": [], "packet_sha256": packet["packet_sha256"]},
    )
    budget = tmp_path / "budget.json"
    tool.save_budget(budget, ledger)
    monkeypatch.setattr(tool, "BUDGET_PATH", budget)
    monkeypatch.setattr(tool, "AUTO_ROOT", tmp_path)
    calls = {"credentials": 0, "factory": 0, "http": 0}

    def fake_credentials(*args, **kwargs):
        calls["credentials"] += 1
        return {} if failure == "missing_key" else {"AIGC_API_KEY": "FICTIONAL_TEST_ONLY"}

    async def no_http(request):
        calls["http"] += 1
        pytest.fail("No HTTP before the route gate")

    def fictional_factory(*args, **kwargs):
        calls["factory"] += 1
        return httpx.AsyncClient(transport=httpx.MockTransport(no_http), trust_env=False)

    monkeypatch.setattr(dotenv, "dotenv_values", fake_credentials)
    monkeypatch.setattr(tool.original, "authorized_http_client", fictional_factory)
    with pytest.raises(
        ValueError, match="missing" if failure == "missing_key" else "route differs"
    ):
        asyncio.run(tool.execute(folder))
    assert calls == {"credentials": 1, "factory": int(failure == "wrong_route"), "http": 0}
    assert (folder / "execution_started.json").exists()
    terminal = tool.read(budget)
    assert terminal["used_completion_POSTs"] == 133
    assert terminal["reserved_next_completion_POSTs"] == 0
    assert terminal["available_unreserved_POSTs"] == 367
    assert terminal["active_run_id"] is None
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    assert calls == {"credentials": 1, "factory": int(failure == "wrong_route"), "http": 0}


def test_copied_packet_cannot_execute_same_identity_from_another_folder(
    plan, tmp_path, monkeypatch
):
    original_folder = tmp_path / plan["run_id"]
    tool.prepare(original_folder, plan)
    copied = tmp_path / "copied"
    shutil.copytree(original_folder, copied)
    monkeypatch.setattr(tool, "AUTO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="canonical folder"):
        asyncio.run(tool.execute(copied))


def test_user_confirmed_direct_school_route_keeps_tls_and_zero_retries(plan):
    direct = {**plan, "route": "school_direct"}
    packet = tool.make_packet(direct, "synthetic_tool_test")
    assert packet["network_route_confirmation"] == tool.original.fingerprint(
        tool.ROUTE_CONFIRMATION
    )

    async def inspect_without_sending():
        async with tool.school_client("FICTIONAL_NOT_A_REAL_KEY", "school_direct") as client:
            assert client._trust_env is True
            assert client.follow_redirects is False
            transport = client._transport_for_url(httpx.URL(tool.original.ENDPOINT))
            assert transport is client._transport
            assert transport._pool._retries == 0
            assert transport._pool._ssl_context.verify_mode != 0
            assert transport._pool._ssl_context.check_hostname is True

    asyncio.run(inspect_without_sending())


def test_direct_route_cannot_use_unrelated_confirmation(plan, tmp_path, monkeypatch):
    evidence = deepcopy(tool.read(tool.ROUTE_CONFIRMATION))
    evidence["verbatim_reply"] = "An unrelated answer"
    path = tmp_path / "unrelated-route.json"
    tool.save(path, evidence)
    monkeypatch.setattr(tool, "ROUTE_CONFIRMATION", path)
    with pytest.raises(ValueError, match="confirmed school network path"):
        tool.make_packet({**plan, "route": "school_direct"}, "synthetic_tool_test")


def test_new_model_is_explicit_not_an_automatic_alias_or_fallback(plan, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="available Flash version"):
        tool.validate_plan({**plan, "model": "DeepSeek-V4-Flash"})
    missing = deepcopy(tool.read(tool.REGISTRY_EVIDENCE))
    missing["model_ids"].remove("DeepSeek-V4.1-Flash")
    path = tmp_path / "missing-model.json"
    tool.save(path, missing)
    monkeypatch.setattr(tool, "REGISTRY_EVIDENCE", path)
    with pytest.raises(ValueError, match="model registry evidence"):
        tool.make_packet(plan, "synthetic_tool_test")
