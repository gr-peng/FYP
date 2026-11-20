"""MockTransport only: bounded strict style collection, dynamic replay and quota."""

import asyncio
import copy
import json
from contextlib import contextmanager

import httpx
import pandas as pd
import pytest
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.simulation import event_runner
from scripts import collect_hybrid_style_pilot as tool


def plan():
    return {
        "run_id": tool.RUN_ID,
        "prior_POSTs": 229,
        "identities": tool.identities(),
        "evidence_kind": "synthetic_tool_test",
    }


def request_input(identity):
    return "Fictional transport test", json.dumps(
        {
            "agent_state": {
                "current_style": "fundamental",
                "persona": {"agent_id": identity["agent_id"]},
            }
        }
    )


def handler_factory(failure=None, fail_post=1):
    calls = []

    async def handler(request):
        calls.append(request.method)
        if request.method == "GET":
            model = "unavailable" if failure == "registry" else tool.OPTIONS["model"]
            return httpx.Response(200, json={"data": [{"id": model}]})
        index = calls.count("POST")
        bad = failure if index == fail_post else None
        if bad == "transport":
            raise httpx.ConnectError("fictional disconnected request")
        if bad == "cancel":
            raise asyncio.CancelledError
        if bad == "http":
            return httpx.Response(503, text="fictional service unavailable")
        if bad == "json":
            return httpx.Response(200, text="not JSON")
        wire = json.loads(request.content)
        current = json.loads(wire["messages"][1]["content"])["agent_state"]["current_style"]
        body = {
            "id": f"synthetic-{index}",
            "created": 1700000000 + index,
            "model": "wrong-model" if bad == "model" else tool.OPTIONS["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "decision": "switch" if bad == "coherence" else "stay",
                                "target_style": current,
                                "rationale": "Synthetic transport answer; not model evidence.",
                            }
                        ),
                    },
                }
            ],
        }
        if bad == "reasoning":
            body["choices"][0]["message"]["reasoning_content"] = "synthetic reasoning field"
        return httpx.Response(
            200, json=body, headers={"x-litellm-cache-key": "synthetic"} if bad == "cache" else {}
        )

    return handler, calls


def test_120_strict_responses_then_stop_and_rebuild_raw_logs(tmp_path):
    handler, calls = handler_factory()

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http:
            adapter = tool.StrictAdapter(http, tmp_path / "logs", plan())
            await adapter.verify()
            for identity in tool.identities():
                system, user = request_input(identity)
                decision, meta = await adapter.decide(system, user, StyleDecision, identity)
                assert decision.decision == "stay" and meta["repair_count"] == 0
            with pytest.raises(ValueError, match="unavailable"):
                await adapter.decide(system, user, StyleDecision, identity)

    asyncio.run(execute())
    assert calls == ["GET"] + ["POST"] * 120
    result = tool.check_logs(tmp_path / "logs", plan())
    assert result["complete"] and result["attempts"] == 120
    assert result["compatible_responses"] == 120 and result["independent_generation_count"] is None


@pytest.mark.parametrize(
    "failure",
    ["registry", "transport", "cancel", "http", "json", "model", "reasoning", "cache", "coherence"],
)
def test_first_failure_stops_without_repair_and_keeps_raw(tmp_path, failure):
    handler, calls = handler_factory(failure)

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http:
            adapter = tool.StrictAdapter(http, tmp_path / "logs", plan())
            if failure == "registry":
                with pytest.raises(ValueError, match="model missing"):
                    await adapter.verify()
            else:
                await adapter.verify()
                identity = tool.identities()[0]
                system, user = request_input(identity)
                error = asyncio.CancelledError if failure == "cancel" else ValueError
                with pytest.raises(error):
                    await adapter.decide(system, user, StyleDecision, identity)
            with pytest.raises(ValueError, match="unavailable"):
                await adapter.decide("test", "{}", StyleDecision, tool.identities()[0])

    asyncio.run(execute())
    count = 0 if failure == "registry" else 1
    assert calls.count("POST") == count
    logs = [tool.records(tmp_path / "logs" / name) for name in tool.LOGS]
    assert [len(rows) for rows in logs] == [count] * 3
    result = tool.check_logs(tmp_path / "logs", plan())
    assert not result["complete"] and result["compatible_responses"] == 0
    if failure == "json":
        assert logs[2][0]["response"] == {"_invalid_json_text": "not JSON"}


def test_wrong_identity_has_no_dispatch(tmp_path):
    handler, calls = handler_factory()

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http:
            adapter = tool.StrictAdapter(http, tmp_path / "logs", plan())
            await adapter.verify()
            identity = tool.identities()[1]
            system, user = request_input(identity)
            with pytest.raises(ValueError, match="identity/order"):
                await adapter.decide(system, user, StyleDecision, identity)

    asyncio.run(execute())
    assert calls == ["GET"]
    assert all(not tool.records(tmp_path / "logs" / name) for name in tool.LOGS)


def configured_executor(tmp_path, monkeypatch, failure=None):
    packet = plan()
    root = tmp_path / "synthetic_auto"
    folder = root / tool.RUN_ID
    folder.mkdir(parents=True)
    monkeypatch.setattr(tool, "AUTO_ROOT", root)
    monkeypatch.setattr(tool, "checked_plan", lambda _folder: packet)
    tool.save(
        folder / "independent_preflight.json",
        {
            "accepted_for_execution": True,
            "mustfix": [],
            "plan_sha256": tool.digest(packet),
            "evidence_kind": "synthetic_test_fixture_no_actual_approval",
        },
    )
    fixed, bars = tool.setup()
    hybrid = copy.deepcopy(fixed)
    hybrid["behavior"].pop("rule_style_adaptation")
    hybrid["llm"] = {"providers": {"aigc_relay": tool.OPTIONS}}
    tool.save(folder / "fixed_config.json", fixed)
    tool.save(folder / "hybrid_config.json", hybrid)
    bars.to_csv(folder / "warmup.csv", index=False)
    budget = tmp_path / "synthetic_budget.json"
    tool.save(
        budget,
        {
            "global_authorized_completion_POST_cap": 500,
            "used_completion_POSTs": 229,
            "reserved_next_completion_POSTs": 0,
            "available_unreserved_POSTs": 271,
            "active_run_id": None,
            "version": 8,
        },
    )
    monkeypatch.setattr(tool.shared, "BUDGET_PATH", budget)
    monkeypatch.setattr(event_runner, "source_hash", lambda: "a" * 64)
    import dotenv

    monkeypatch.setattr(
        dotenv, "dotenv_values", lambda _path: {"AIGC_API_KEY": "synthetic-never-real"}
    )
    handler, calls = handler_factory(failure, fail_post=3)
    monkeypatch.setattr(
        tool.shared,
        "school_client",
        lambda key, route: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
    )
    return folder, budget, calls


def test_complete_dynamic_run_replays_exactly_and_charges_120(tmp_path, monkeypatch):
    folder, budget, calls = configured_executor(tmp_path, monkeypatch)
    state = asyncio.run(tool.execute(folder))
    assert state["new_attempts"] == 120 and state["status"] == "complete_pending_independent_audit"
    ledger = tool.read(budget)
    assert ledger["used_completion_POSTs"] == 349 and ledger["active_run_id"] is None
    assert ledger["reserved_next_completion_POSTs"] == 0
    replay = asyncio.run(tool.replay_complete_run(folder, tmp_path / "recorded_replay"))
    assert replay["exact_dynamic_prompts_checked"] == 120 and replay["new_model_POSTs"] == 0
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    assert calls == ["GET"] + ["POST"] * 120
    assert tool.read(folder / "market_comparison.json")["all_trading_equal_through"] == "2026-01-30"
    # With no switch, even a difference after day5 must be rejected automatically.
    path = folder / "hybrid/orders.parquet"
    orders = pd.read_parquet(path)
    orders.loc[orders.session_date == "2026-01-10", "rationale"] = "unexpected later difference"
    orders.to_parquet(path, index=False)
    with pytest.raises(AssertionError):
        tool.summarize_market(folder, tool.read(folder / "response_audit.json"))


@pytest.mark.parametrize("failure", ["registry", "model", "cancel"])
def test_executor_releases_unused_quota_and_preserves_pending(tmp_path, monkeypatch, failure):
    folder, budget, calls = configured_executor(tmp_path, monkeypatch, failure)
    error = asyncio.CancelledError if failure == "cancel" else ValueError
    with pytest.raises(error):
        asyncio.run(tool.execute(folder))
    count = 0 if failure == "registry" else 3
    ledger = tool.read(budget)
    assert ledger["used_completion_POSTs"] == 229 + count
    assert ledger["available_unreserved_POSTs"] == 271 - count
    assert ledger["reserved_next_completion_POSTs"] == 0 and ledger["active_run_id"] is None
    assert calls.count("POST") == count
    assert tool.read(folder / "execution_state.json")["status"] == "stopped_incomplete"
    if count:
        meta = tool.read(folder / "hybrid/run_metadata.json")
        assert meta["completed_market_sessions"] == 5 and meta["applied_review_rows"] == 0
        rows = tool.records(folder / "hybrid/hybrid_review_batches.jsonl")
        assert rows[-1]["valid_unapplied"] == 2 and len(rows[-1]["unsent_identities"]) == 17


def test_zero_post_start_race_is_rejected_inside_lock_before_data_or_credentials(
    tmp_path, monkeypatch
):
    folder, budget, calls = configured_executor(tmp_path, monkeypatch)
    original_lock = tool.shared.exclusive_budget
    previous_budget = budget.read_bytes()
    previous_start = {"run_id": tool.RUN_ID, "status": "earlier_zero_post_attempt"}

    @contextmanager
    def race(path):
        with original_lock(path):
            tool.save(folder / "execution_started.json", previous_start)
            yield

    monkeypatch.setattr(tool.shared, "exclusive_budget", race)
    import dotenv

    def forbidden(*_args, **_kwargs):
        raise AssertionError("raced start must reject before credentials")

    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    assert tool.read(folder / "execution_started.json") == previous_start
    assert budget.read_bytes() == previous_budget and not (folder / "fixed").exists()
    assert calls == []


@pytest.mark.parametrize("change", ["source", "input", "synthetic", "identity", "allowance"])
def test_preparation_and_freeze_reject_changed_inputs_without_credentials(
    tmp_path, monkeypatch, change
):
    import dotenv

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline preparation must not read credentials")

    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    folder = tmp_path / "candidate"
    packet = tool.prepare(folder)
    assert tool.checked_plan(folder) == packet
    if change == "source":
        path = folder / "sources/scripts/collect_hybrid_style_pilot.py"
        path.write_text(path.read_text() + "\n# altered frozen source\n")
    elif change == "input":
        config = tool.read(folder / "hybrid_config.json")
        config["population"]["initial_shares"] = 99
        tool.save(folder / "hybrid_config.json", config)
    else:
        if change == "synthetic":
            packet["evidence_kind"] = "synthetic_tool_test"
        elif change == "identity":
            packet["identities"][0]["kind"] = "order"
        else:
            packet["maximum_cumulative_POSTs"] = 501
        tool.save(folder / "plan.json", packet)
        (folder / "plan.sha256").write_text(tool.digest(packet) + "\n")
    with pytest.raises(ValueError, match="source changed|input changed|fixed protocol"):
        tool.checked_plan(folder)
