"""Four artificial economic controls and a real-runner ledger boundary; zero external calls."""

import asyncio
import json
import math
import socket
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import pytest
import test_style_path_integration as integration
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.llm.prompts import prompts
from scripts import build_style_review_economic_controls as builder


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Economic-control tests must not open a network socket")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def materials():
    return builder.build_materials()


@pytest.mark.parametrize("style", ["fundamental", "technical"])
@pytest.mark.parametrize(
    "condition,expected", [("positive", Fraction(1, 10)), ("negative", Fraction(-1, 11))]
)
def test_derived_finances_and_exposure_match_hand_calculation(style, condition, expected):
    case = builder.make_case(style, condition)
    trace = case["construction_not_sent"]["daily_trace"]
    state = case["construction_not_sent"]["final_state"]
    assert [r["current_decision"]["action"] for r in trace] == ["hold"] * 5
    assert [r["current_decision"]["quantity"] for r in trace] == [0] * 5
    assert [r["alternative_exposure"] for r in trace] == (
        [1, 0, 0, 0, 0] if condition == "positive" else [0, 1, 0, 0, 0]
    )
    growth = Fraction(1)
    completed = case["construction_not_sent"]["warmup_closes"][:]
    for row in trace:
        assert row["history_last_session"] < row["session_date"]
        assert row["start_reference"] == completed[-1]
        momentum = round(completed[-1] / completed[-6] - 1, 8)
        assert row["momentum_5"] == pytest.approx(momentum)
        assert row["current_signal"] == 0
        if style == "fundamental":
            assert row["fundamental_anchor"] == row["start_reference"]
            assert row["alternative_signal"] == pytest.approx(momentum)
        else:
            assert row["alternative_signal"] == pytest.approx(
                row["fundamental_anchor"] / row["start_reference"] - 1
            )
        assert row["current_equity"] == 1000 + 2 * row["close"]
        if row["alternative_exposure"]:
            growth *= Fraction(row["close"], int(row["start_reference"]))
        assert row["alternative_growth"] == pytest.approx(float(growth))
        completed.append(row["close"])
    assert growth - 1 == expected
    assert state["portfolio"] == {
        "cash": 1000,
        "position": 2,
        "avg_cost": 100,
        "realized_pnl": 0,
        "unrealized_pnl": 0,
        "total_equity": 1200,
    }
    assert state["counterfactual"]["actual_block_return"] == 0
    assert state["counterfactual"]["alternative_block_return"] == pytest.approx(float(expected))
    assert state["counterfactual"]["gap"] == pytest.approx(float(expected))
    assert state["drawdown"] == pytest.approx(1 - 1200 / 1220)
    assert [r["pnl_change"] for r in state["memory"]] == [20, -20, 0, 0, 0]
    assert sum(r["filled_quantity"] for r in state["memory"]) == 0
    reference = case["construction_not_sent"]["rule_style_reference"]
    assert reference["decision"] == ("switch" if condition == "positive" else "stay")


def test_pairs_keep_actual_history_but_do_not_pretend_only_gap_changed(materials):
    for style in ("fundamental", "technical"):
        pair = [c for c in materials["cases"] if c["original_style"] == style]
        a, b = [c["construction_not_sent"] for c in pair]
        assert a["final_state"]["portfolio"] == b["final_state"]["portfolio"]
        assert a["final_state"]["memory"] == b["final_state"]["memory"]
        if style == "fundamental":
            assert a["warmup_closes"] != b["warmup_closes"]
        else:
            assert [r["fundamental_anchor"] for r in a["daily_trace"]] != [
                r["fundamental_anchor"] for r in b["daily_trace"]
            ]
    assert "not a gap-only causal intervention" in materials["analysis_plan"]
    assert materials["scheduled_opportunities"] == materials["actual_remote_calls"] == 0
    assert materials["budget"] == {"used": 133, "cap": 133, "remaining": 0}


def test_new_version_adds_only_the_clarification_to_the_base_review_task(materials):
    for case in materials["cases"]:
        payload = case["payload"]
        user = json.loads(payload["messages"][1]["content"])
        state = deepcopy(case["construction_not_sent"]["final_state"])
        agent = SimpleNamespace(
            context=lambda state=state: deepcopy(state),
            persona=SimpleNamespace(
                agent_id=state["persona"]["agent_id"],
            ),
        )
        system, original_text = prompts(
            agent, user["observation"], "vanilla_llm", "style", 17, mean_field_visibility="hidden"
        )
        original = json.loads(original_text)
        assert (
            original["decision_instruction"] + " " + builder.CLARIFICATION
            == user["decision_instruction"]
        )
        original.pop("decision_instruction")
        user.pop("decision_instruction")
        assert original == user
        assert system == payload["messages"][0]["content"]
        assert "mean_field" not in user["observation"]
        assert user["observation"]["mean_field_availability"] == "withheld_by_experiment"
        assert user["agent_state"]["persona"] == {"agent_id": state["persona"]["agent_id"]}
        assert "rule_style_reference" not in builder.canonical(payload)
        assert "construction_not_sent" not in builder.canonical(payload)
        assert case["planned_payload_sha256"] == builder.digest(payload)
        assert all(
            math.isfinite(v)
            for v in user["agent_state"]["counterfactual"].values()
            if isinstance(v, float)
        )


@pytest.mark.parametrize("reply", ["stay", "switch"])
def test_production_style_application_does_not_transact_or_change_portfolio(
    tmp_path, monkeypatch, reply
):
    exchanges, before, checked = [], {}, []
    original_submit = integration.Exchange.submit_order

    def submit(exchange, *args, **kwargs):
        if exchange not in exchanges:
            exchanges.append(exchange)
        return original_submit(exchange, *args, **kwargs)

    monkeypatch.setattr(integration.Exchange, "submit_order", submit)

    def counts():
        return sum(len(e.orders) for e in exchanges), sum(len(e.trades) for e in exchanges)

    async def capture_client(folder, handler):
        def record(request):
            if request.method == "POST":
                user = json.loads(json.loads(request.content)["messages"][1]["content"])
                if user["observation"].get("phase") == "post_close_style_review":
                    state = user["agent_state"]
                    before[user["observation"]["session_date"], state["persona"]["agent_id"]] = {
                        "portfolio": deepcopy(state["portfolio"]),
                        "style": state["current_style"],
                        "counts": counts(),
                    }
            return handler(request)

        return await integration.memory_client(folder, record)

    original_mean_field = integration.event_runner.compute_mean_field

    def after_application(agents, *args, **kwargs):
        session = args[1] if len(args) >= 2 else kwargs.get("as_of_session")
        if all((session, a.persona.agent_id) in before for a in agents):
            for agent in agents:
                snapshot = before[session, agent.persona.agent_id]
                assert agent.context()["portfolio"] == snapshot["portfolio"]
                assert counts() == snapshot["counts"]
                expected = (
                    snapshot["style"] if reply == "stay" else integration.other(snapshot["style"])
                )
                assert agent.current_style == expected
            checked.append(session)
        return original_mean_field(agents, *args, **kwargs)

    monkeypatch.setattr(integration.event_runner, "compute_mean_field", after_application)
    asyncio.run(
        integration.run_path(
            tmp_path,
            monkeypatch,
            mode="endogenous",
            reply=reply,
            client_factory=capture_client,
            treatment="vanilla_llm",
        )
    )
    assert len(checked) == 2 and len(before) == 4
    assert counts()[0] >= 4 and counts()[1] >= 1  # Nonempty real production exchange fixture.


def test_writer_is_local_deterministic_and_keeps_original_study_separate(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = builder.write_materials(first)
    b = builder.write_materials(second)
    assert a == b
    assert len(a["cases"]) == len({c["planned_payload_sha256"] for c in a["cases"]}) == 4
    assert all(c["status"] == "unsubmitted" for c in a["cases"])
    assert all(path.read_bytes() == (second / path.name).read_bytes() for path in first.iterdir())
    assert "completed18 majority experiment" in a["historical_study"]
    with pytest.raises(FileExistsError):
        builder.write_materials(first)


@pytest.mark.parametrize("style,condition", [("other", "positive"), ("fundamental", "other")])
def test_only_the_predeclared_controls_can_be_generated(style, condition):
    with pytest.raises(ValueError):
        builder.make_case(style, condition)
