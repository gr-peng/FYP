"""The new finite12 retains strict raw evidence and the cumulative500 execution guards."""

import asyncio
import json
import runpy
import shutil
import socket
from copy import deepcopy

import dotenv
import httpx
import pytest

from scripts import collect_finite_economic_probe as tool

helpers = runpy.run_path(str(tool.shared.REPO / "tests/test_style_review_economic_collection.py"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic checks cannot use real networking or credentials")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


@pytest.fixture
def plan():
    return {
        "run_id": "economic-auto-finite-fictional",
        "prior_POSTs": 181,
        "route": "school_direct",
        "model": tool.shared.MODEL,
    }


@pytest.fixture
def ledger():
    return {
        "global_authorized_completion_POST_cap": 500,
        "used_completion_POSTs": 181,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 319,
        "active_run_id": None,
        "version": 4,
    }


def test_exact_new_materials_old_schedule_and_no_reused_answers(plan, tmp_path):
    packet = tool.prepare(tmp_path / "new", plan, "synthetic_tool_test")
    assert tool.load_packet(tmp_path / "new") == (plan, packet)
    assert packet["accepted_materials_sha256"] == tool.original.fingerprint(
        tool.MATERIALS / "materials_v1/materials.json"
    )["sha256"]
    assert packet["schedule_base_materials_sha256"] != packet["accepted_materials_sha256"]
    cases = {c["case_id"]: c["payload"] for c in tool.checked_materials()["cases"]}
    assert len(packet["requests"]) == len({r["slot_id"] for r in packet["requests"]}) == 12
    assert all(r["payload"] == cases[r["case_id"]] for r in packet["requests"])
    assert all(
        "long_cash_signal_proxy" not in tool.canonical(r["payload"]) for r in packet["requests"]
    )
    assert all(r["kind"] == tool.PROTOCOL for r in packet["requests"])
    pending = tool.evaluator(plan)(packet, [], [], [])
    assert pending["planned"] == 12 and len(pending["cells"]) == 4
    assert all(c["unsubmitted"] == 3 and c["valid"] == 0 for c in pending["cells"])
    assert all(
        c["alternative_fraction_positive_minus_negative"] is None for c in pending["contrasts"]
    )


@pytest.mark.parametrize(
    "policy,expected", [("economic", 1), ("reverse", -1), ("fundamental", 0), ("technical", 0)]
)
def test_economic_contrasts_are_not_forced_to_follow_winner(plan, policy, expected):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    result = tool.evaluator(plan)(packet, *helpers["prefix"](packet, count=12, policy=policy))
    assert result["protocol_complete"]
    assert [c["alternative_fraction_positive_minus_negative"] for c in result["contrasts"]] == [
        expected,
        expected,
    ]
    assert result["independent_generation_count"] is None
    tampered = deepcopy(packet)
    tampered["requests"][0]["payload"]["temperature"] = 0
    tampered["packet_sha256"] = tool.digest(
        {k: v for k, v in tampered.items() if k != "packet_sha256"}
    )
    with pytest.raises(ValueError, match="packet differs"):
        tool.evaluator(plan)(tampered, [], [], [])


@pytest.mark.parametrize(
    "failure,posts", [(None, 12), ("registry", 0), ("invalid", 1), ("cancel", 1), ("key", 0)]
)
def test_mock_actual_entry_budget_start_marker_and_first_failure(
    plan, ledger, tmp_path, monkeypatch, failure, posts
):
    monkeypatch.setattr(tool, "AUTO_ROOT", tmp_path)
    monkeypatch.setattr(tool.shared, "BUDGET_PATH", tmp_path / "budget.json")
    tool.shared.save_budget(tool.shared.BUDGET_PATH, ledger)
    folder = tmp_path / plan["run_id"]
    packet = tool.prepare(folder, plan)
    tool.save(
        folder / "independent_preflight.json",
        {"accepted_for_execution": True, "mustfix": [], "packet_sha256": packet["packet_sha256"]},
    )
    monkeypatch.setattr(
        dotenv,
        "dotenv_values",
        lambda *a: {"AIGC_API_KEY": None if failure == "key" else "FICTIONAL_NONCREDENTIAL"},
    )
    calls = []

    async def respond(request):
        calls.append(request.method)
        budget = tool.read(tool.shared.BUDGET_PATH)
        assert budget["reserved_next_completion_POSTs"] == 12
        assert budget["available_unreserved_POSTs"] == 307
        if request.method == "GET":
            return httpx.Response(
                503 if failure == "registry" else 200, json={"data": [{"id": packet["model"]}]}
            )
        index = calls.count("POST") - 1
        planned = packet["requests"][index]
        assert json.loads(request.content) == planned["payload"]
        if failure == "cancel":
            raise asyncio.CancelledError
        body = helpers["body_for"](planned, index, helpers["target_for"](planned, "economic"))
        if failure == "invalid":
            body["model"] = "wrong-model"
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        tool.shared,
        "school_client",
        lambda *a: httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False),
    )
    if failure == "key":
        with pytest.raises(ValueError, match="missing"):
            asyncio.run(tool.execute(folder))
    elif failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(tool.execute(folder))
    else:
        result = asyncio.run(tool.execute(folder))
        assert result["logged_dispatch_attempts"] == posts
        assert result["protocol_complete"] is (failure is None)
    assert calls.count("POST") == posts
    budget = tool.read(tool.shared.BUDGET_PATH)
    assert budget["used_completion_POSTs"] == 181 + posts
    assert budget["available_unreserved_POSTs"] == 319 - posts
    assert budget["active_run_id"] is None and budget["reserved_next_completion_POSTs"] == 0
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    with pytest.raises(ValueError, match="canonical folder"):
        asyncio.run(tool.execute(copy))
