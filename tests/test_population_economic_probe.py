"""Offline checks of the36 schedule, matched information, raw outcomes and dispatch gates."""

import asyncio
import json
import shutil
import socket
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import dotenv
import httpx
import pytest

from scripts import collect_population_economic_probe as tool


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic checks cannot use real credentials or networking")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


@pytest.fixture
def plan():
    return {
        "run_id": "economic-auto-population-fictional",
        "prior_POSTs": 145,
        "route": "school_direct",
        "model": tool.shared.MODEL,
    }


@pytest.fixture
def ledger():
    return {
        "global_authorized_completion_POST_cap": 500,
        "used_completion_POSTs": 145,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 355,
        "active_run_id": None,
        "version": 3,
    }


def body(request, number, target):
    return {
        "id": f"fictional-{number}",
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


def target(request, policy):
    alternative = "technical" if request["from_style"] == "fundamental" else "fundamental"
    if policy == "economic":
        return alternative if request["condition"] == "positive" else request["from_style"]
    if policy in ("majority", "reverse"):
        if request["population_condition"] == "hidden":
            return request["from_style"]
        choice = "fundamental" if request["population_condition"] == tool.GROUPS[0] else "technical"
        return (
            choice
            if policy == "majority"
            else ("technical" if choice == "fundamental" else "fundamental")
        )
    return policy


def logs(packet, policy, count=36):
    requests, attempts, responses = [], [], []
    for i, planned in enumerate(packet["requests"][:count]):
        row = deepcopy(planned)
        origin = tool.original.identity(row)
        start = datetime(2026, 1, 10, tzinfo=UTC) + timedelta(seconds=2 * i)
        requests.append(row)
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
                "response": body(row, i, target(row, policy)),
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    return requests, attempts, responses


def test_matched_payloads_and_local_cohort_are_coherent(plan, tmp_path):
    packet = tool.prepare(tmp_path / "new", plan, "synthetic_tool_test")
    assert tool.load_packet(tmp_path / "new") == (plan, packet)
    requests = packet["requests"]
    assert len(requests) == len({r["slot_id"] for r in requests}) == 36
    assert [r["position"] for r in requests] == list(range(1, 37))
    assert set(Counter((r["case_id"], r["population_condition"]) for r in requests).values()) == {3}
    assert set(
        Counter(
            (r["case_id"], r["population_condition"], r["within_group_position"]) for r in requests
        ).values()
    ) == {1}
    material = tool.original.checked_materials()[0]
    for case in material["cases"]:
        matching = [r for r in requests if r["case_id"] == case["case_id"]]
        bodies = {
            group: [r["payload"] for r in matching if r["population_condition"] == group]
            for group in tool.GROUPS
        }
        assert all(len({tool.canonical(p) for p in values}) == 1 for values in bodies.values())
        expected_hidden = deepcopy(case["payload"])
        expected_hidden["model"] = plan["model"]
        assert bodies["hidden"][0] == expected_hidden
        users = [json.loads(bodies[g][0]["messages"][1]["content"]) for g in tool.GROUPS[:2]]
        fields = [u["observation"].pop("mean_field") for u in users]
        assert users[0] == users[1]
        assert fields[0].pop("fundamental_share") == 0.8
        assert fields[1].pop("fundamental_share") == 0.2
        assert fields[0].pop("technical_share") == 0.2
        assert fields[1].pop("technical_share") == 0.8
        assert (
            fields[0]
            == fields[1]
            == {
                "as_of_session": "2026-01-08",
                "positive_sentiment_share": 0,
                "negative_sentiment_share": 0,
                "buy_sell_imbalance": 0,
                "majority_action": None,
                "majority_strength": 0,
                "active_order_quantity": 8,
                "average_pnl": 0,
            }
        )
        for group in tool.GROUPS[:2]:
            cohort = packet["construction_not_sent"][f"{case['case_id']}.{group}"]
            participants = cohort["participants"]
            assert len(participants) == 10
            assert [p["current_style"] for p in participants[:2]] == ["fundamental", "technical"]
            assert sum(p["pending_order"]["action"] == "buy" for p in participants) == 4
            assert sum(p["pending_order"]["action"] == "sell" for p in participants) == 4
            for p in participants:
                assert p["initial_equity"] == p["equity"] == 1200
                assert p["memory"][0]["equity"] == 1220
                assert [m["session"] for m in p["memory"]] == [
                    f"2026-01-{d:02}" for d in range(5, 9)
                ]
                assert all(
                    m["cash"] == 1000
                    and m["position"] == 2
                    and m["equity"] == 1000 + 2 * m["close"]
                    for m in p["memory"]
                )
    assert all("construction_not_sent" not in tool.canonical(r["payload"]) for r in requests)
    assert len(packet["source_receipt"]) + len(packet["adapter_sources"]) == 55
    pending = tool.evaluator(plan)(packet, [], [], [])
    assert pending["planned"] == 36 and len(pending["cells"]) == 12
    assert all(c["unsubmitted"] == 3 and c["stay_count"] == 0 for c in pending["cells"])


@pytest.mark.parametrize(
    "policy,population,economic",
    [
        ("majority", 1, 0),
        ("reverse", -1, 0),
        ("economic", 0, 1),
        ("fundamental", 0, 0),
        ("technical", 0, 0),
    ],
)
def test_primary_raw_f_and_secondary_alternative_have_distinct_directions(
    plan, policy, population, economic
):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    result = tool.evaluator(plan)(packet, *logs(packet, policy))
    assert result["planned"] == result["logged_dispatch_attempts"] == 36
    assert result["protocol_complete"]
    assert [c["difference"] for c in result["primary_population_contrasts"]] == [population] * 4
    assert [c["difference"] for c in result["secondary_economic_contrasts"]] == [economic] * 6
    assert all(c["valid"] == 3 for c in result["cells"])
    assert "contrasts" not in result  # The old12 economic summaries must not leak through.
    assert result["independent_generation_count"] is None


def test_missing_and_invalid_are_not_counted_as_stay_or_zero_effect(plan):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    prefix = logs(packet, "fundamental", 1)
    result = tool.evaluator(plan)(packet, *prefix)
    assert sum(c["valid"] for c in result["cells"]) == 1
    assert sum(c["unsubmitted"] for c in result["cells"]) == 35
    assert all(c["difference"] is None for c in result["primary_population_contrasts"])
    prefix[2][0]["response"]["model"] = "wrong-model"
    result = tool.evaluator(plan)(packet, *prefix)
    assert sum(c["valid"] for c in result["cells"]) == 0
    assert sum(c["invalid"] for c in result["cells"]) == 1
    assert sum(c["stay_count"] for c in result["cells"]) == 0
    complete = logs(packet, "fundamental")
    complete[2][0]["response"]["model"] = "wrong-model"
    with pytest.raises(ValueError, match="request after"):
        tool.evaluator(plan)(packet, *complete)


def test_resigning_tampered_material_is_rejected_and_repeats_are_retained(plan):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    packet["requests"][0]["payload"]["temperature"] = 0
    packet["packet_sha256"] = tool.digest({k: v for k, v in packet.items() if k != "packet_sha256"})
    with pytest.raises(ValueError, match="packet differs"):
        tool.evaluator(plan)(packet, [], [], [])
    packet = tool.make_packet(plan, "synthetic_tool_test")
    raw = logs(packet, "fundamental")
    for response in raw[2]:
        response["response"]["id"] = "same-fictional-signature"
        response["response"]["created"] = 1768000000
    result = tool.evaluator(plan)(packet, *raw)
    assert result["protocol_complete"]
    # Response content contains stay for A000 and switch for A001, so their full
    # signatures differ even though the model id, response id and timestamp agree.
    assert len(result["repeated_strong_signature_groups"]) == 2
    assert sorted(map(len, result["repeated_strong_signature_groups"])) == [18, 18]


def test_whole36_budget_boundary_and_raw_attempt_count(plan, ledger, tmp_path):
    last = {**plan, "prior_POSTs": 464}
    tool.validate_plan(last)
    tool.check_budget(
        {**ledger, "used_completion_POSTs": 464, "available_unreserved_POSTs": 36}, last
    )
    with pytest.raises(ValueError):
        tool.validate_plan({**plan, "prior_POSTs": 465})
    with pytest.raises(ValueError):
        tool.check_budget(
            {**ledger, "used_completion_POSTs": 465, "available_unreserved_POSTs": 35},
            {**plan, "prior_POSTs": 465},
        )
    output = tmp_path / "raw"
    output.mkdir()
    (output / tool.original.LOGS[1]).write_text("{}\n" * 36)
    terminal = tool.terminal_budget(ledger, plan, output)
    assert terminal["used_completion_POSTs"] == 181
    assert terminal["available_unreserved_POSTs"] == 319
    assert terminal["reserved_next_completion_POSTs"] == 0
    (output / tool.original.LOGS[1]).write_text("{}\n" * 37)
    with pytest.raises(ValueError):
        tool.terminal_budget(ledger, plan, output)


@pytest.mark.parametrize(
    "failure,expected_posts",
    [(None, 36), ("registry", 0), ("transport", 3), ("invalid", 3), ("cancel", 3), ("key", 0)],
)
def test_actual_execute_with_mock_transport_reserves36_stops_and_cannot_resume(
    plan, ledger, tmp_path, monkeypatch, failure, expected_posts
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
        assert budget["reserved_next_completion_POSTs"] == 36
        assert budget["available_unreserved_POSTs"] == 319
        assert budget["active_run_id"] == plan["run_id"]
        if request.method == "GET":
            return httpx.Response(
                503 if failure == "registry" else 200, json={"data": [{"id": packet["model"]}]}
            )
        index = calls.count("POST") - 1
        expected = packet["requests"][index]
        assert json.loads(request.content) == expected["payload"]
        if index == 2:
            if failure == "transport":
                raise httpx.ConnectError("FICTIONAL_UNSAFE_EXCEPTION_TEXT")
            if failure == "cancel":
                raise asyncio.CancelledError
        answer = body(expected, index, target(expected, "majority"))
        if failure == "invalid" and index == 2:
            answer["model"] = "wrong-model"
        return httpx.Response(200, json=answer)

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
        assert result["logged_dispatch_attempts"] == expected_posts
        assert result["protocol_complete"] is (failure is None)
    assert calls.count("POST") == expected_posts
    budget = tool.read(tool.shared.BUDGET_PATH)
    assert budget["used_completion_POSTs"] == 145 + expected_posts
    assert budget["available_unreserved_POSTs"] == 355 - expected_posts
    assert budget["active_run_id"] is None and budget["reserved_next_completion_POSTs"] == 0
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    with pytest.raises(ValueError, match="canonical folder"):
        asyncio.run(tool.execute(copy))
    diagnostic = folder / "transport_diagnostics"
    assert not diagnostic.exists() or "FICTIONAL_UNSAFE" not in "".join(
        p.read_text() for p in diagnostic.iterdir()
    )
