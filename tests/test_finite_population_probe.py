"""Archived finite inputs, controlled displays, raw outcomes and bounded dispatch."""

import asyncio
import json
import shutil
from collections import Counter
from copy import deepcopy

import dotenv
import httpx
import pytest
from test_population_economic_probe import body, logs, target
from test_population_economic_probe import offline as offline

from scripts import collect_finite_population_probe as tool


@pytest.fixture
def plan():
    return {
        "run_id": "economic-auto-finite-population-fictional",
        "prior_POSTs": 193,
        "route": "school_direct",
        "model": tool.shared.MODEL,
    }


def test_old_evaluator_still_enforces_its_default_packet_validator(plan, monkeypatch):
    def reject(_packet, _plan):
        raise ValueError("original validator still required")

    monkeypatch.setattr(tool.population, "check_packet", reject)
    with pytest.raises(ValueError, match="original validator still required"):
        tool.population.evaluator(plan)({}, [], [], [])


def test_preserve_finite_bytes_and_change_only_displayed_shares(plan, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not rebuild old material with current production code")

    monkeypatch.setattr(tool.shared, "make_packet", forbidden)
    monkeypatch.setattr(tool.original, "checked_materials", forbidden)
    packet = tool.make_packet(plan, "synthetic_tool_test")
    rows = packet["requests"]
    assert len(rows) == len({r["slot_id"] for r in rows}) == 36
    assert [r["position"] for r in rows] == list(range(1, 37))
    assert set(Counter((r["case_id"], r["population_condition"]) for r in rows).values()) == {3}
    assert set(
        Counter(
            (r["case_id"], r["population_condition"], r["within_group_position"]) for r in rows
        ).values()
    ) == {1}
    for case in tool.checked_materials()["cases"]:
        requests = [r for r in rows if r["case_id"] == case["case_id"]]
        payloads = {
            group: [r["payload"] for r in requests if r["population_condition"] == group]
            for group in tool.GROUPS
        }
        assert all(len({tool.canonical(p) for p in group}) == 1 for group in payloads.values())
        assert payloads["hidden"][0] == case["payload"]
        users = [json.loads(payloads[group][0]["messages"][1]["content"]) for group in tool.GROUPS]
        fields = [u["observation"].pop("mean_field") for u in users[:2]]
        assert {k for k in fields[0] if fields[0][k] != fields[1][k]} == {
            "fundamental_share",
            "technical_share",
        }
        assert (fields[0]["fundamental_share"], fields[1]["fundamental_share"]) == (0.8, 0.2)
        assert fields[0]["as_of_session"] == fields[1]["as_of_session"] == "2026-01-08"
        users[2]["observation"].pop("mean_field_availability")
        assert users[0] == users[1] == users[2]
        assert "construction_not_sent" not in tool.canonical(payloads)


def test_historical_archive_tampering_and_new_input_archive_tampering_fail(
    plan, tmp_path, monkeypatch
):
    folder = tmp_path / "packet"
    packet = tool.prepare(folder, plan, "synthetic_tool_test")
    assert tool.load_packet(folder) == (plan, packet)
    (folder / "execution_inputs/design.json").write_text("{}")
    with pytest.raises(ValueError, match="input archive differs"):
        tool.load_packet(folder)
    historical = tmp_path / "historical"
    shutil.copytree(tool.MATERIALS, historical)
    monkeypatch.setattr(tool, "MATERIALS", historical)
    (
        historical / "sources_v1/behavioral-llm-market/src/behavioral_market/agents/state.py"
    ).write_text("# changed")
    with pytest.raises(ValueError, match="historical evidence differs"):
        tool.make_packet(plan, "synthetic_tool_test")


@pytest.mark.parametrize(
    "policy,primary,secondary",
    [("majority", 1, 0), ("reverse", -1, 0), ("fundamental", 0, 0), ("economic", 0, 1)],
)
def test_all_directions_and_zero_effect_are_reported(plan, policy, primary, secondary):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    result = tool.evaluator(plan)(packet, *logs(packet, policy))
    assert result["protocol_complete"] and result["logged_dispatch_attempts"] == 36
    assert [c["difference"] for c in result["primary_population_contrasts"]] == [primary] * 4
    assert [c["difference"] for c in result["secondary_economic_contrasts"]] == [secondary] * 6
    assert result["counterfactual_method"] == tool.METHOD
    assert result["independent_generation_count"] is None


def test_invalid_missing_and_resigned_payload_never_become_success(plan):
    packet = tool.make_packet(plan, "synthetic_tool_test")
    raw = logs(packet, "fundamental", 1)
    raw[2][0]["response"]["model"] = "wrong-model"
    result = tool.evaluator(plan)(packet, *raw)
    assert sum(c["invalid"] for c in result["cells"]) == 1
    assert sum(c["unsubmitted"] for c in result["cells"]) == 35
    assert sum(c["stay_count"] for c in result["cells"]) == 0
    assert all(c["difference"] is None for c in result["primary_population_contrasts"])
    changed = deepcopy(packet)
    changed["requests"][0]["payload"]["temperature"] = 0
    changed["packet_sha256"] = tool.digest(
        {k: v for k, v in changed.items() if k != "packet_sha256"}
    )
    with pytest.raises(ValueError, match="packet differs"):
        tool.evaluator(plan)(changed, [], [], [])


def test_whole36_cap_and_repeated_signatures(plan):
    tool.validate_plan({**plan, "prior_POSTs": 464})
    with pytest.raises(ValueError, match="whole36"):
        tool.validate_plan({**plan, "prior_POSTs": 465})
    packet = tool.make_packet(plan, "synthetic_tool_test")
    raw = logs(packet, "fundamental")
    for r in raw[2]:
        r["response"]["id"] = "same-fictional-response"
        r["response"]["created"] = 1768000000
    result = tool.evaluator(plan)(packet, *raw)
    assert result["protocol_complete"]
    assert sorted(map(len, result["repeated_strong_signature_groups"])) == [18, 18]


@pytest.mark.parametrize(
    "failure,expected_posts",
    [(None, 36), ("registry", 0), ("transport", 3), ("invalid", 3), ("cancel", 3), ("key", 0)],
)
def test_mock_dispatch_reserves_whole_batch_and_never_retries_or_resumes(
    plan, tmp_path, monkeypatch, failure, expected_posts
):
    monkeypatch.setattr(tool, "AUTO_ROOT", tmp_path)
    monkeypatch.setattr(tool.shared, "BUDGET_PATH", tmp_path / "budget.json")
    ledger = {
        "global_authorized_completion_POST_cap": 500,
        "used_completion_POSTs": 193,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 307,
        "active_run_id": None,
        "version": 6,
    }
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
        assert budget["available_unreserved_POSTs"] == 271
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
    assert budget["used_completion_POSTs"] == 193 + expected_posts
    assert budget["available_unreserved_POSTs"] == 307 - expected_posts
    assert budget["active_run_id"] is None and budget["reserved_next_completion_POSTs"] == 0
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    copied = tmp_path / "copy"
    shutil.copytree(folder, copied)
    with pytest.raises(ValueError, match="canonical folder"):
        asyncio.run(tool.execute(copied))
    diagnostics = folder / "transport_diagnostics"
    assert not diagnostics.exists() or "FICTIONAL_UNSAFE" not in "".join(
        p.read_text() for p in diagnostics.iterdir()
    )
