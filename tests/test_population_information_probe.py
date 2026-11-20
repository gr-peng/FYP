"""Fictional accounting, information contrasts and planned opportunity checks; no model calls."""

import importlib.util
import json
import socket
from collections import Counter
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_population_information_probe.py"
spec = importlib.util.spec_from_file_location("population_information_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This material-only test must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def package():
    return builder.build_package()


def user(case):
    return json.loads(case["payload"]["messages"][1]["content"])


def cases_for(package, agent_id):
    return {c["condition"]: c for c in package["cases"] if c["agent_id"] == agent_id}


@pytest.mark.parametrize("agent_id", ["A000", "A001"])
def test_majority_changes_only_two_share_values(package, agent_id):
    cases = cases_for(package, agent_id)
    first, second = (deepcopy(cases[c]["payload"]) for c in builder.CONDITIONS[:2])
    first_user, second_user = user(cases[builder.CONDITIONS[0]]), user(cases[builder.CONDITIONS[1]])
    first_field = first_user["observation"].pop("mean_field")
    second_field = second_user["observation"].pop("mean_field")
    assert first_field.pop("fundamental_share") == 0.8
    assert first_field.pop("technical_share") == 0.2
    assert second_field.pop("fundamental_share") == 0.2
    assert second_field.pop("technical_share") == 0.8
    assert first_field == second_field
    assert first_field["majority_action"] is None
    assert first_field["active_order_quantity"] == 8
    assert first_user == second_user
    first["messages"][1]["content"] = second["messages"][1]["content"] = "<compared above>"
    assert first == second


@pytest.mark.parametrize("agent_id", ["A000", "A001"])
def test_hidden_removes_only_aggregate_via_production_switch(package, agent_id):
    cases = cases_for(package, agent_id)
    hidden = user(cases["hidden"])
    visible = user(cases["fundamental_majority"])
    assert "mean_field" not in hidden["observation"]
    assert hidden["observation"].pop("mean_field_availability") == "withheld_by_experiment"
    visible["observation"].pop("mean_field")
    assert hidden == visible
    agent = builder.account(agent_id, cases["hidden"]["from_style"])
    for condition in builder.CONDITIONS[:2]:
        field, _ = builder.population_snapshot(condition)
        observation = builder.observation(field)
        original = deepcopy(observation)
        system, prompt = builder.prompts(
            agent, observation, "vanilla_llm", "style", 17, mean_field_visibility="hidden"
        )
        assert prompt == cases["hidden"]["payload"]["messages"][1]["content"]
        assert system == cases["hidden"]["payload"]["messages"][0]["content"]
        assert observation == original


def test_population_accounting_counts_and_lagged_order_intents(package):
    ledgers = package["construction_ledgers_not_sent"]
    for condition, expected_f in zip(builder.CONDITIONS[:2], (8, 2), strict=True):
        ledger = ledgers[condition]
        people = ledger["participants"]
        assert len(people) == ledger["n_participants"] == 10
        assert len({p["agent_id"] for p in people}) == 10
        assert sum(p["current_style"] == "fundamental" for p in people) == expected_f
        assert people[0]["current_style"] == "fundamental"
        assert people[1]["current_style"] == "technical"
        assert Counter(p["submitted_order"]["action"] for p in people) == {
            "hold": 2,
            "buy": 4,
            "sell": 4,
        }
        for person in people:
            assert person["cash"] + person["position"] * 100 == person["equity"] == 1200
            assert person["initial_equity"] == 1200
            assert len(person["memory"]) == 4
            assert person["memory"][-1]["session_date"] == ledger["as_of_session"]
            assert all(m["executed_quantity"] == 0 for m in person["memory"])
            last = person["memory"][-1]
            order = person["submitted_order"]
            assert last["submitted_action"] == order["action"]
            assert last["submitted_quantity"] == order["quantity"]
            if order["action"] == "buy":
                assert order["limit_price"] == 99
                assert order["quantity"] * order["limit_price"] <= person["cash"]
            elif order["action"] == "sell":
                assert order["limit_price"] == 101
                assert order["quantity"] <= person["position"]
        assert ledger["mean_field"]["average_pnl"] == 0
        assert ledger["mean_field"]["buy_sell_imbalance"] == 0
        assert ledger["mean_field"]["majority_action"] is None
        assert ledger["mean_field"]["positive_sentiment_share"] == 0
        assert ledger["mean_field"]["negative_sentiment_share"] == 0
    first, second = (deepcopy(ledgers[c]) for c in builder.CONDITIONS[:2])
    for a, b in zip(first["participants"], second["participants"], strict=True):
        a.pop("current_style")
        b.pop("current_style")
    for key in ("fundamental_share", "technical_share"):
        first["mean_field"].pop(key)
        second["mean_field"].pop(key)
    assert first == second


@pytest.mark.parametrize("agent_id,style", [("A000", "fundamental"), ("A001", "technical")])
def test_private_memory_current_style_and_flat_proxy_are_preserved(package, agent_id, style):
    for case in cases_for(package, agent_id).values():
        prompt = user(case)
        state, obs = prompt["agent_state"], prompt["observation"]
        assert state["persona"] == {"agent_id": agent_id}
        assert state["current_style"] == style
        assert {m["owner"] for m in state["memory"]} == {agent_id}
        assert [m["session_date"] for m in state["memory"]] == list(builder.SESSIONS)
        assert all(m["submitted_action"] == "hold" for m in state["memory"])
        assert all(m["equity"] == m["cash"] + 100 * m["position"] for m in state["memory"])
        assert state["portfolio"]["cash"] == 1000
        assert state["portfolio"]["position"] == 2
        assert state["portfolio"]["total_equity"] == 1200
        assert state["counterfactual"] == {
            "method": "long_cash_signal_proxy",
            "actual_block_return": 0,
            "alternative_block_return": 0,
            "gap": 0,
        }
        assert state["drawdown"] == 0
        assert obs["last_close"] == obs["fundamental_value"] == 100
        assert len(obs["completed_closes"]) == obs["review_block"]["completed_sessions"] == 5
        assert obs["history_last_session"] == obs["review_block"]["end_session"]
        if "mean_field" in obs:
            assert obs["mean_field"]["as_of_session"] < obs["session_date"]
        assert "review" in case["payload"]["messages"][0]["content"].lower()
        assert "do not submit an order" in case["payload"]["messages"][0]["content"]
        assert "5-session block" in prompt["decision_instruction"]


def test_six_payloads_eighteen_opportunities_and_balanced_round_positions(package):
    cases, schedule = package["cases"], package["schedule"]
    assert len(cases) == package["unique_payload_count"] == 6
    assert len({c["planned_payload_sha256"] for c in cases}) == 6
    assert len(schedule) == package["planned_opportunities"] == 18
    assert [s["position"] for s in schedule] == list(range(1, 19))
    assert len({s["slot_id"] for s in schedule}) == 18
    assert Counter(s["case_id"] for s in schedule) == {c["case_id"]: 3 for c in cases}
    lookup = {c["case_id"]: c for c in cases}
    for agent_id in ("A000", "A001"):
        for condition in builder.CONDITIONS:
            slots = [s for s in schedule if s["case_id"] == f"{agent_id}.{condition}"]
            assert {s["round"] for s in slots} == {1, 2, 3}
            assert {s["within_account_round_position"] for s in slots} == {1, 2, 3}
    for slot in schedule:
        assert slot["planned_payload_sha256"] == lookup[slot["case_id"]]["planned_payload_sha256"]
        assert slot["status"] == "unsubmitted"
    for case in cases:
        assert case["planned_payload_sha256"] == builder.digest(case["payload"])
        assert case["treatment"] == "vanilla_llm"
        assert case["payload"]["cache"] == {"no-cache": True, "no-store": True}


def test_materials_do_not_claim_execution_or_new_allowance(package):
    assert package["status"] == "materials_only_not_execution_ready"
    assert package["evidence_kind"] == "fictional_unsubmitted_materials"
    assert package["execution_gate"]["remaining"] == 0
    assert package["execution_gate"]["actual_remote_calls_in_this_package"] == 0
    assert package["execution_gate"]["executor_validated"] is False
    for row in package["cases"] + package["schedule"]:
        assert row["status"] == "unsubmitted"
        assert not {"request_hash", "response", "dispatched_at", "cached"}.intersection(row)
    assert "not human participants" in package["analysis_plan"]["limits"]
    assert "zero valid means unknown" in package["analysis_plan"]["denominators"]


@pytest.mark.parametrize("change", ["cash", "payload_hash", "schedule", "quota"])
def test_fixed_material_validation_rejects_content_or_scope_changes(package, change):
    changed = deepcopy(package)
    if change == "cash":
        prompt = user(changed["cases"][0])
        prompt["agent_state"]["portfolio"]["cash"] += 1
        changed["cases"][0]["payload"]["messages"][1]["content"] = json.dumps(prompt)
        changed["cases"][0]["planned_payload_sha256"] = builder.digest(
            changed["cases"][0]["payload"]
        )
    elif change == "payload_hash":
        changed["cases"][0]["planned_payload_sha256"] = "0" * 64
    elif change == "schedule":
        changed["schedule"][0], changed["schedule"][1] = (
            changed["schedule"][1],
            changed["schedule"][0],
        )
    else:
        changed["execution_gate"]["remaining"] = 18
    with pytest.raises(ValueError, match="fixed fictional design"):
        builder.validate_package(changed)


def test_writer_is_deterministic_and_refuses_existing_output(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    package = builder.write_package(first)
    builder.write_package(second)
    builder.validate_package(package)
    assert sorted(p.name for p in first.iterdir()) == [
        "artifact_manifest.json",
        "materials.json",
        "说明.txt",
    ]
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "artifact_manifest.json").read_text())
    for name, fingerprint in manifest.items():
        data = (first / name).read_bytes()
        assert len(data) == fingerprint["bytes"]
        assert builder.hashlib.sha256(data).hexdigest() == fingerprint["sha256"]
    before = (first / "materials.json").read_bytes()
    with pytest.raises(FileExistsError):
        builder.write_package(first)
    assert (first / "materials.json").read_bytes() == before
