"""Exercise the production review path with fictional data and an in-memory API."""

import asyncio
import copy
import json
import random
from dataclasses import asdict
from decimal import Decimal

import httpx
import pandas as pd
import pytest
import requests

from behavioral_market.agents.policy import rule_order
from behavioral_market.llm.client import FatalAPIError, MockLLMClient, OpenAICompatibleClient
from behavioral_market.llm.prompts import PROMPT_VERSION, prompts
from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.market.exchange import Exchange
from behavioral_market.simulation import event_runner

MODEL = "fictional-style-model"


def provider_config():
    return {
        "model": MODEL,
        "api_key_env": "FICTIONAL_UNUSED_KEY",
        "allowed_host": "fixture.invalid",
        "chat_completions_url": "https://fixture.invalid/completions",
        "models_url": "https://fixture.invalid/models",
        "cache_dir": "unused-fictional-cache",
        "max_concurrency": 2,
        "timeout_seconds": 1,
        "temperature": 0,
        "max_tokens": 200,
        "enable_thinking": False,
        "transport_retries": 0,
        "trust_env": False,
    }


def envelope(content, finish_reason="stop"):
    return {
        "model": MODEL,
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
    }


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No credentials, external HTTP, datasets or source-tree scan")

    async def forbidden_async(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(event_runner, "load_dotenv", forbidden)
    monkeypatch.setattr(event_runner, "load_dataset", forbidden)
    monkeypatch.setattr(event_runner, "source_hash", lambda: "fictional-source")
    monkeypatch.setattr(event_runner, "git_state", lambda: ("fictional-commit", False))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


async def memory_client(tmp_path, handler):
    client = OpenAICompatibleClient(
        provider_config(), "fictional-key", tmp_path / "logs", tmp_path / "cache"
    )
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    return client


def fictional_market(interval=1, days=2):
    dates = pd.date_range("2025-12-01", periods=31 + days).strftime("%Y-%m-%d")
    closes = [100.0] * 31 + [102.0 + i for i in range(days)]
    bars = pd.DataFrame(
        {
            "symbol": "FIXTURE",
            "session_date": dates,
            "open": closes,
            "high": [p + 5 for p in closes],
            "low": [p - 5 for p in closes],
            "close": closes,
            "volume": 100,
        }
    )
    config = {
        "experiment": {"seed": 17},
        "data": {
            "primary_symbol": "FIXTURE",
            "context_symbols": [],
            "experiment_start": "2026-01-01",
            "experiment_end": dates[-1],
            "calendar": "24/7",
        },
        "population": {
            "n_agents": 2,
            "initial_fundamental_share": 0.5,
            "initial_cash": "1000",
            "initial_shares": 2,
        },
        "behavior": {"memory_window": 20, "strategy_switch_interval": interval},
        "fundamental_shock": {"enabled": False, "magnitude": 0},
        "events": {"condition": "no_event"},
        "llm": {"providers": {"aigc_relay": provider_config()}},
    }
    return config, bars


def other(style):
    return "technical" if style == "fundamental" else "fundamental"


async def run_path(
    tmp_path,
    monkeypatch,
    *,
    mode="historical",
    interval=1,
    reply="switch",
    change_future=False,
    unfilled_buy=False,
    client_factory=memory_client,
    treatment="low_herding",
    opening_limit_price=101,
    second_session_order_policy=None,
    mean_field_visibility="default",
    order_batch_size=None,
):
    config, bars = fictional_market(interval, days=max(2, interval + 1))
    if mean_field_visibility != "default":
        config["behavior"]["mean_field_visibility"] = mean_field_visibility
    if order_batch_size is not None:
        config["behavior"]["order_decision_batch_size"] = order_batch_size
    if change_future:
        bars.loc[bars.session_date >= "2026-01-02", ["open", "high", "low", "close"]] *= 10
    if unfilled_buy:
        bars.loc[bars.session_date == "2026-01-01", "low"] = 102
    captured, agents = [], []
    create_population = event_runner.population

    def population(*args):
        agents.extend(create_population(*args))
        for agent in agents:
            agent.memory.append({"session": "2025-12-31", "owner": agent.persona.agent_id})
        return agents

    monkeypatch.setattr(event_runner, "population", population)

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        state = user["agent_state"]
        style = "post_close_style_review" == user["observation"].get("phase")
        captured.append((style, payload["messages"][0]["content"], user))
        if style:
            action = "stay" if reply == "stay" else "switch"
            target = state["current_style"] if action == "stay" else other(state["current_style"])
            if reply == "inconsistent":
                target = state["current_style"]
            if reply == "invalid":
                return httpx.Response(200, json=envelope("not JSON"))
            if reply == "mixed" and state["persona"]["agent_id"] != "A000":
                action, target = "stay", state["current_style"]
            if reply == "switch_once" and (
                state["persona"]["agent_id"] != "A000"
                or user["observation"]["session_date"] != "2026-01-01"
            ):
                action, target = "stay", state["current_style"]
            content = {"decision": action, "target_style": target, "rationale": "fictional"}
        else:
            if (
                second_session_order_policy is not None
                and user["observation"]["session_date"] == "2026-01-02"
            ):
                agent = next(
                    a for a in agents if a.persona.agent_id == state["persona"]["agent_id"]
                )
                decision = second_session_order_policy(agent, user["observation"])
                return httpx.Response(200, json=envelope(decision.model_dump_json()))
            action = "buy" if state["persona"]["agent_id"] == "A000" else "sell"
            content = {
                "action": action,
                "quantity": 1,
                "limit_price": opening_limit_price,
                "rationale": "fictional",
                "sentiment": "neutral",
            }
        return httpx.Response(200, json=envelope(json.dumps(content)))

    client = await client_factory(tmp_path, respond)
    try:
        metrics = await event_runner.run_experiment(
            config,
            mode,
            treatment,
            "aigc_relay",
            tmp_path / "run",
            bars=bars,
            events=[],
            client=client,
            manifest_hash="fully-fictional-fixture",
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
    return captured, agents, metrics


@pytest.mark.parametrize("mode", ["historical", "endogenous"])
@pytest.mark.parametrize("interval", [1, 2])
def test_close_review_uses_close_facts_and_actual_block(tmp_path, monkeypatch, mode, interval):
    captured, _, _ = asyncio.run(run_path(tmp_path, monkeypatch, mode=mode, interval=interval))
    for style, system, user in captured:
        if not style:
            assert "permitted_price_band" in user["observation"]
            continue
        obs = user["observation"]
        assert obs["order_book"]["last_trade"] == obs["last_close"]
        assert "permitted_price_band" not in obs
        assert obs["review_block"]["completed_sessions"] == interval
        assert obs["review_block"]["end_session"] == obs["session_date"]
        expected_start = pd.Timestamp(obs["session_date"]) - pd.Timedelta(days=interval - 1)
        assert obs["review_block"]["start_session"] == expected_start.strftime("%Y-%m-%d")
        assert f"{interval}-session block" in user["decision_instruction"]
        assert "You submit one limit order" not in system
        assert "form the order" not in system
        assert obs["mean_field"]["as_of_session"] < obs["session_date"]
        if mode == "endogenous":
            assert obs["order_book"]["scope"] == "post_close_after_day_order_expiry"


@pytest.mark.parametrize("reply", ["stay", "switch", "inconsistent", "invalid", "mixed"])
def test_real_parse_apply_preserves_private_history_and_accounts(tmp_path, monkeypatch, reply):
    captured, agents, metrics = asyncio.run(run_path(tmp_path, monkeypatch, reply=reply))
    reviews = {u["agent_state"]["persona"]["agent_id"]: u for s, _, u in captured if s}
    orders = [u for s, _, u in captured if not s]
    first_reviews = {}
    for style, _, user in captured:
        if style:
            first_reviews.setdefault(user["agent_state"]["persona"]["agent_id"], user)
    fields = [u["observation"]["mean_field"] for u in first_reviews.values()]
    assert fields[0] == fields[1]
    assert {u["agent_state"]["current_style"] for u in first_reviews.values()} == {
        "fundamental",
        "technical",
    }
    for user in orders:
        state, obs = user["agent_state"], user["observation"]
        agent_id = state["persona"]["agent_id"]
        assert state["memory"][0]["owner"] == agent_id
        assert all(m["session"] < obs["session_date"] for m in state["memory"])
        if obs["session_date"] == "2026-01-02":
            previous = first_reviews[agent_id]["agent_state"]
            should_switch = reply == "switch" or reply == "mixed" and agent_id == "A000"
            expected = (
                other(previous["current_style"]) if should_switch else previous["current_style"]
            )
            assert state["current_style"] == expected
            assert state["portfolio"] == previous["portfolio"]
            assert state["memory"] == previous["memory"]
            assert state["counterfactual"]["actual_block_return"] == 0
    for agent in agents:
        assert agent.portfolio.cash >= 0 and agent.portfolio.position >= 0
        assert (
            reviews[agent.persona.agent_id]["agent_state"]["memory"][0]["owner"]
            == agent.persona.agent_id
        )
    switches = pd.read_csv(tmp_path / "run/style_switches.csv")
    if reply in {"inconsistent", "invalid"}:
        expected_status = (
            "inconsistent_style" if reply == "inconsistent" else "invalid_json_or_schema"
        )
        assert set(switches.validation_status) == {expected_status}
        assert set(switches.decision) == {"stay"}
        assert not switches.choice_valid.any()
        assert metrics["valid_style_decisions"] == 0
        assert metrics["invalid_style_decisions"] == len(switches)
        assert metrics["valid_style_switch_rate"] is None
        assert metrics["switch_rate"] == 0
        assert metrics["fallback_rate"] == 0  # All order answers were valid.
        if reply == "invalid":
            assert switches.requested_decision.isna().all()
            assert switches.requested_target_style.isna().all()
            assert switches.fallback.all()
        else:
            assert set(switches.requested_decision) == {"switch"}
            assert not switches.fallback.any()
        assert json.loads((tmp_path / "run/run_metadata.json").read_text())["quality"] == "degraded"
    else:
        assert switches.choice_valid.all()
        assert metrics["valid_style_decisions"] == len(switches)
        assert metrics["invalid_style_decisions"] == 0


@pytest.mark.parametrize("mode", ["historical", "endogenous"])
def test_future_bars_cannot_change_earlier_orders_or_reviews(tmp_path, monkeypatch, mode):
    prefixes = []
    for altered in (False, True):
        with monkeypatch.context() as scoped:
            captured, _, _ = asyncio.run(
                run_path(tmp_path / str(altered), scoped, mode=mode, change_future=altered)
            )
        prefixes.append(
            [
                (style, system, user)
                for style, system, user in captured
                if user["observation"]["session_date"] == "2026-01-01"
            ]
        )
    assert prefixes[0] == prefixes[1]
    assert len(prefixes[0]) == 4  # Two orders and two closing reviews.


def test_unfilled_order_is_not_added_to_account_as_a_trade(tmp_path, monkeypatch):
    captured, _, _ = asyncio.run(run_path(tmp_path, monkeypatch, unfilled_buy=True))
    state = next(
        user["agent_state"]
        for style, _, user in captured
        if style
        and user["observation"]["session_date"] == "2026-01-01"
        and user["agent_state"]["persona"]["agent_id"] == "A000"
    )
    assert state["portfolio"]["cash"] == 1000
    assert state["portfolio"]["position"] == 2
    assert state["memory"][-1]["action"] == "buy"
    assert state["memory"][-1]["filled_quantity"] == 0
    assert state["memory"][-1]["pnl_change"] == 4  # Mark-to-market only.


def bad_response(variant):
    content = json.dumps({"decision": "stay", "target_style": "fundamental", "rationale": "x"})
    body = envelope(content)
    if variant == "duplicate_content":
        body["choices"][0]["message"]["content"] = (
            '{"decision":"switch","decision":"stay","target_style":"fundamental","rationale":"x"}'
        )
    elif variant == "duplicate_envelope":
        text = json.dumps(body)
        return text.replace('"model":', '"model":"conflicting-model","model":', 1)
    elif variant == "missing_finish":
        del body["choices"][0]["finish_reason"]
    else:
        body["choices"][0]["finish_reason"] = variant
    return json.dumps(body)


@pytest.mark.parametrize("from_cache", [False, True])
@pytest.mark.parametrize(
    "variant",
    [
        "length",
        "content_filter",
        "tool_calls",
        None,
        "missing_finish",
        "duplicate_content",
        "duplicate_envelope",
    ],
)
def test_incomplete_or_ambiguous_reply_never_becomes_valid(tmp_path, variant, from_cache):
    async def execute():
        calls = []

        def respond(request):
            calls.append(request)
            content = (
                json.dumps(
                    envelope('{"decision":"stay","target_style":"fundamental","rationale":"x"}')
                )
                if from_cache
                else bad_response(variant)
            )
            return httpx.Response(200, text=content)

        client = await memory_client(tmp_path, respond)
        user = '{"agent_state":{"current_style":"fundamental"}}'
        try:
            if from_cache:
                await client.decide("system", user, StyleDecision, {"fixture": 1})
                cached = next((tmp_path / "cache").glob("*.json"))
                cached.write_text(bad_response(variant))
                before = cached.read_bytes()
                client.cache_only = True
            with pytest.raises(FatalAPIError):
                await client.decide("system", user, StyleDecision, {"fixture": 1})
            assert len(calls) == 1  # No repair or retry for these contract failures.
            if from_cache:
                assert cached.read_bytes() == before
            else:
                assert not list((tmp_path / "cache").glob("*.json"))
                if variant == "duplicate_envelope":
                    response = json.loads(
                        (tmp_path / "logs/llm_responses.jsonl").read_text().splitlines()[-1]
                    )
                    assert response["response_text"] == bad_response(variant)
                    assert response["response"] is None
                    assert response["retry_count"] == 0
                    assert response["latency_seconds"] >= 0
        finally:
            await client.close()

    asyncio.run(execute())


@pytest.mark.parametrize("style", ["fundamental", "technical"])
def test_prompt_is_a_snapshot_and_does_not_mutate_agent(style):
    config, _ = fictional_market()
    agent = event_runner.population(config, "low_herding", "100")[0]
    agent.current_style = style
    agent.memory.append({"session": "2025-12-31", "private_note": agent.persona.agent_id})
    before = copy.deepcopy(asdict(agent))
    _, text = prompts(agent, {"session_date": "2026-01-01"}, "low_herding", "style", 17)
    saved = json.loads(text)
    agent.memory.append({"session": "2026-01-01"})
    assert saved["agent_state"]["memory"] == before["memory"]
    agent.memory.pop()
    assert asdict(agent) == before


def fixture_log(tmp_path, name):
    return [json.loads(line) for line in (tmp_path / "logs" / name).read_text().splitlines()]


def assert_review_bindings(tmp_path):
    """Join only this test's fresh logs, rather than trust the new row columns."""
    requests = fixture_log(tmp_path, "llm_requests.jsonl")
    responses = fixture_log(tmp_path, "llm_responses.jsonl")
    rows = pd.read_parquet(tmp_path / "run/style_switches.parquet")
    for row in rows.to_dict("records"):
        matches = [record for record in requests if record["request_hash"] == row["request_hash"]]
        assert len(matches) == 1
        request = matches[0]
        payload = json.loads(request["payload"]["messages"][1]["content"])
        state, obs = payload["agent_state"], payload["observation"]
        assert row["agent_id"] == request["agent_id"] == state["persona"]["agent_id"]
        assert row["session_date"] == request["session_date"] == obs["session_date"]
        assert row["from_style"] == state["current_style"]
        assert row["phase"] == obs["phase"] == "post_close_style_review"
        assert row["decision_at_utc"] == obs["decision_at_utc"]
        assert row["prompt_version"] == PROMPT_VERSION
        assert row["decision_source"] == "OpenAICompatibleClient"
        assert row["repair_count"] == request["repair"]
        response = next(r for r in responses if r["request_hash"] == row["request_hash"])
        assert row["cache_hit"] == response["cached"]
    return rows


def test_reverse_completion_keeps_each_review_bound_to_its_own_request(tmp_path, monkeypatch):
    completed = []

    async def factory(path, handler):
        client = await memory_client(path, handler)
        decide = client.decide
        gates = {}

        async def reversed_decide(system, user, schema, identity):
            if schema is not StyleDecision:
                return await decide(system, user, schema, identity)
            gate = gates.setdefault(identity["session_date"], asyncio.Event())
            if identity["agent_id"] == "A000":
                await asyncio.wait_for(gate.wait(), timeout=2)
            result = await decide(system, user, schema, identity)
            completed.append((identity["session_date"], identity["agent_id"]))
            if identity["agent_id"] == "A001":
                gate.set()
            return result

        client.decide = reversed_decide
        return client

    asyncio.run(run_path(tmp_path, monkeypatch, client_factory=factory))
    rows = assert_review_bindings(tmp_path)
    for session in rows.session_date.unique():
        assert [agent for date, agent in completed if date == session] == ["A001", "A000"]
    assert not rows.cache_hit.any()
    assert set(rows.repair_count) == {0}


def test_repaired_review_points_to_final_answer_not_first_failed_attempt(tmp_path, monkeypatch):
    async def factory(path, handler):
        def first_bad(request):
            if request.method == "POST":
                payload = json.loads(request.content)
                user = json.loads(payload["messages"][1]["content"])
                if (
                    user["observation"].get("phase") == "post_close_style_review"
                    and len(payload["messages"]) == 2
                ):
                    return httpx.Response(200, json=envelope("invalid JSON"))
            return handler(request)

        return await memory_client(path, first_bad)

    asyncio.run(run_path(tmp_path, monkeypatch, client_factory=factory))
    rows = assert_review_bindings(tmp_path)
    requests = fixture_log(tmp_path, "llm_requests.jsonl")
    assert set(rows.repair_count) == {1}
    for row in rows.to_dict("records"):
        pair = [
            r
            for r in requests
            if r["kind"] == "style"
            and r["agent_id"] == row["agent_id"]
            and r["session_date"] == row["session_date"]
        ]
        assert [r["repair"] for r in pair] == [0, 1]
        assert row["request_hash"] == pair[1]["request_hash"] != pair[0]["request_hash"]


def test_cached_review_records_cache_hit_without_another_fixture_post(tmp_path, monkeypatch):
    shared_cache = tmp_path / "shared_cache"
    shared_cache.mkdir()
    fresh_posts, replay_posts = [], []

    def factory(posts, cache_only):
        async def create(path, handler):
            def counted(request):
                if request.method == "POST":
                    posts.append(request)
                return handler(request)

            client = await memory_client(path, counted)
            client.cache = shared_cache
            client.cache_only = cache_only
            return client

        return create

    captured = []
    for name, posts, cache_only in [("fresh", fresh_posts, False), ("replay", replay_posts, True)]:
        with monkeypatch.context() as scoped:
            captured.append(
                asyncio.run(
                    run_path(tmp_path / name, scoped, client_factory=factory(posts, cache_only))
                )
            )
    assert fresh_posts and replay_posts == []
    fresh = assert_review_bindings(tmp_path / "fresh")
    replay = assert_review_bindings(tmp_path / "replay")
    assert not fresh.cache_hit.any() and replay.cache_hit.all()
    assert list(fresh.request_hash) == list(replay.request_hash)
    assert captured[0][0] and captured[1][0] == []  # No transport for cache hits.
    for name in ["agents", "orders", "market", "mean_field"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "fresh/run" / f"{name}.parquet"),
            pd.read_parquet(tmp_path / "replay/run" / f"{name}.parquet"),
        )
    fresh_requests = fixture_log(tmp_path / "fresh", "llm_requests.jsonl")
    replay_requests = fixture_log(tmp_path / "replay", "llm_requests.jsonl")
    assert [r["payload"] for r in fresh_requests] == [r["payload"] for r in replay_requests]


@pytest.mark.parametrize("source", ["rule", "MockLLMClient", "injected_client"])
def test_non_api_sources_do_not_invent_request_metadata(tmp_path, monkeypatch, source):
    class InjectedClient:
        async def decide(self, *args):
            return await MockLLMClient().decide(*args)

    async def factory(path, handler):
        return InjectedClient() if source == "injected_client" else MockLLMClient()

    asyncio.run(
        run_path(
            tmp_path,
            monkeypatch,
            client_factory=factory,
            treatment="rule_abm" if source == "rule" else "low_herding",
        )
    )
    rows = pd.read_parquet(tmp_path / "run/style_switches.parquet")
    assert set(rows.decision_source) == {source}
    assert rows[["request_hash", "cache_hit", "repair_count"]].isna().all().all()
    assert not (tmp_path / "logs").exists()


def test_failed_parse_leaves_missing_request_metadata_unknown(tmp_path, monkeypatch):
    asyncio.run(run_path(tmp_path, monkeypatch, reply="invalid"))
    rows = pd.read_parquet(tmp_path / "run/style_switches.parquet")
    assert set(rows.decision_source) == {"OpenAICompatibleClient"}
    assert rows[["request_hash", "cache_hit", "repair_count"]].isna().all().all()
    assert rows.requested_decision.isna().all() and not rows.choice_valid.any()


@pytest.mark.parametrize("mode", ["historical", "endogenous"])
def test_population_feedback_tracks_applied_styles_without_anticipating_reviews(
    tmp_path, monkeypatch, mode
):
    first_day, next_day = "2026-01-01", "2026-01-02"
    prior_inputs, prior_tables, prior_reviews = None, None, None
    summaries = []
    for condition in ["stay", "mixed", "switch", "invalid"]:
        directory = tmp_path / condition
        with monkeypatch.context() as scoped:
            captured, _, metrics = asyncio.run(
                run_path(directory, scoped, mode=mode, reply=condition)
            )
        opening = {
            user["agent_state"]["persona"]["agent_id"]: user
            for style, _, user in captured
            if not style and user["observation"]["session_date"] == first_day
        }
        assert set(opening) == {"A000", "A001"}
        initial = {agent: user["agent_state"]["current_style"] for agent, user in opening.items()}
        assert set(initial.values()) == {"fundamental", "technical"}
        # Gold answers come from the initial input and the scripted intervention,
        # not from the runner's output or its population aggregation function.
        changed = (
            {"A000", "A001"}
            if condition == "switch"
            else {"A000"}
            if condition == "mixed"
            else set()
        )
        expected = {
            agent: (other(style) if agent in changed else style) for agent, style in initial.items()
        }
        expected_fundamental = list(expected.values()).count("fundamental") / 2
        expected_switch_rate = len(changed) / 2

        switches = pd.read_parquet(directory / "run/style_switches.parquet")
        closing_reviews = switches.loc[switches.session_date == first_day]
        assert len(closing_reviews) == 2
        for row in closing_reviews.to_dict("records"):
            agent = row["agent_id"]
            assert row["from_style"] == initial[agent]
            assert row["target_style"] == expected[agent]
            assert row["decision"] == ("switch" if agent in changed else "stay")
            assert row["choice_valid"] == (condition != "invalid")
            if condition == "invalid":
                assert row["requested_decision"] is None
                assert row["requested_target_style"] is None
                assert row["fallback"]
                assert row["validation_status"] == "invalid_json_or_schema"
            else:
                assert row["requested_decision"] == row["decision"]
                assert row["requested_target_style"] == expected[agent]
        assert (closing_reviews.decision == "switch").mean() == expected_switch_rate
        if condition == "invalid":
            assert metrics["valid_style_decisions"] == 0
            assert metrics["valid_style_switch_rate"] is None

        population = pd.read_parquet(directory / "run/mean_field.parquet")
        closed = population.loc[population.session_date == first_day].iloc[0].to_dict()
        assert closed["as_of_session"] == first_day
        assert closed["fundamental_share"] == expected_fundamental
        assert closed["technical_share"] == 1 - expected_fundamental
        snapshot = {
            key: None if pd.isna(value) else value
            for key, value in closed.items()
            if key != "session_date"
        }
        next_inputs = [
            (style, user)
            for style, _, user in captured
            if user["observation"]["session_date"] == next_day
        ]
        assert {u["agent_state"]["persona"]["agent_id"] for s, u in next_inputs if not s} == set(
            initial
        )
        assert {u["agent_state"]["persona"]["agent_id"] for s, u in next_inputs if s} == set(
            initial
        )
        for _, user in next_inputs:
            state, obs = user["agent_state"], user["observation"]
            assert state["current_style"] == expected[state["persona"]["agent_id"]]
            assert obs["mean_field"] == snapshot
            assert obs["mean_field"]["as_of_session"] < obs["session_date"]

        before_application = {}
        for style, _, user in captured:
            if style and user["observation"]["session_date"] == first_day:
                agent = user["agent_state"]["persona"]["agent_id"]
                before_application.setdefault(agent, user)
                assert user["agent_state"]["current_style"] == initial[agent]
                assert user["observation"]["mean_field"]["as_of_session"] == "2025-12-31"
                assert user["observation"]["mean_field"]["fundamental_share"] == 0.5
        assert set(before_application) == set(initial)
        first_tables = {}
        for name in ["orders", "trades", "agents", "market"]:
            table = pd.read_parquet(directory / "run" / f"{name}.parquet")
            first_tables[name] = table.loc[table.session_date == first_day].reset_index(drop=True)
        if prior_inputs is None:
            prior_inputs, prior_tables, prior_reviews = opening, first_tables, before_application
        else:
            assert opening == prior_inputs
            assert before_application == prior_reviews
            for name, table in first_tables.items():
                pd.testing.assert_frame_equal(table, prior_tables[name])
        if condition == "switch":
            assert expected_switch_rate == 1
            assert expected_fundamental == 0.5  # Everyone switched, yet the mix is unchanged.
        if condition == "mixed":
            assert expected_fundamental in {0, 1}
        summaries.append(
            {
                "condition": condition,
                "applied_switch_rate": float((closing_reviews.decision == "switch").mean()),
                "closing_fundamental_share": closed["fundamental_share"],
                "closing_technical_share": closed["technical_share"],
                "next_session_feedback_matches": True,
            }
        )
    print(json.dumps({"mode": mode, "feedback_cases": summaries}, sort_keys=True))


def test_one_style_change_reaches_rule_orders_matching_and_accounts(tmp_path, monkeypatch):
    first_day, next_day = "2026-01-01", "2026-01-02"
    seeds = {"A000": 17000, "A001": 17001}
    original_submit = Exchange.submit_order
    runs = {}

    for condition in ["stay", "switch_once"]:
        directory = tmp_path / condition
        rngs = {agent: random.Random(seed) for agent, seed in seeds.items()}
        rule_calls, submissions = [], []

        def second_session_policy(agent, observation, *, rngs=rngs, rule_calls=rule_calls):
            agent_id = agent.persona.agent_id
            rng = rngs[agent_id]
            before = rng.getstate()
            decision = rule_order(agent, observation, rng)
            rule_calls.append(
                (
                    agent_id,
                    copy.deepcopy(agent.context()),
                    copy.deepcopy(observation),
                    before,
                    rng.getstate(),
                )
            )
            return decision

        def submit(exchange, agent_id, side, price, quantity, *, submissions=submissions):
            # Observe actual arrival and settlement without changing either.
            assert set(exchange.portfolios) == set(seeds)  # No external liquidity leg.
            before = {a: (p.cash, p.position) for a, p in exchange.portfolios.items()}
            day = exchange.day
            order, fills = original_submit(exchange, agent_id, side, price, quantity)
            expected = dict(before)
            for fill in fills:
                cash, shares = expected[fill.buyer_id]
                expected[fill.buyer_id] = (
                    cash - fill.price * fill.quantity,
                    shares + fill.quantity,
                )
                cash, shares = expected[fill.seller_id]
                expected[fill.seller_id] = (
                    cash + fill.price * fill.quantity,
                    shares - fill.quantity,
                )
            assert {a: (p.cash, p.position) for a, p in exchange.portfolios.items()} == expected
            assert sum(cash for cash, _ in expected.values()) == Decimal("2000")
            assert sum(shares for _, shares in expected.values()) == 4
            submissions.append(
                {
                    "day": day,
                    "agent_id": agent_id,
                    "order_id": order.order_id,
                    "action": order.side.value,
                    "limit_price": str(order.price),
                    "quantity": order.quantity,
                }
            )
            return order, fills

        with monkeypatch.context() as scoped:
            scoped.setattr(Exchange, "submit_order", submit)
            captured, agents, _ = asyncio.run(
                run_path(
                    directory,
                    scoped,
                    mode="endogenous",
                    reply=condition,
                    opening_limit_price=104,
                    second_session_order_policy=second_session_policy,
                )
            )
        tables = {
            name: pd.read_parquet(directory / "run" / f"{name}.parquet")
            for name in ["orders", "trades", "agents", "market", "style_switches"]
        }
        reviews = tables["style_switches"]
        changed = reviews.loc[reviews.decision == "switch"]
        assert len(changed) == (1 if condition == "switch_once" else 0)
        if condition == "switch_once":
            row = changed.iloc[0]
            assert (row.session_date, row.agent_id, row.from_style) == (
                first_day,
                "A000",
                "fundamental",
            )
            assert row.requested_decision == "switch" and row.choice_valid and not row.fallback
            assert row.requested_target_style == row.target_style == "technical"

        calls = {
            agent_id: (state, obs, before, after)
            for agent_id, state, obs, before, after in rule_calls
        }
        assert len(rule_calls) == len(calls) == 2
        next_orders = (
            tables["orders"].loc[tables["orders"].session_date == next_day].set_index("agent_id")
        )
        for agent_id, (state, obs, before, after) in calls.items():
            order_input = next(
                user
                for style, _, user in captured
                if not style
                and user["agent_state"]["persona"]["agent_id"] == agent_id
                and user["observation"]["session_date"] == next_day
            )
            assert state == order_input["agent_state"]
            assert obs == order_input["observation"]
            assert obs["last_close"] == 104 and obs["fundamental_value"] == 100
            assert obs["momentum_5"] == pytest.approx(0.04)
            gold_rng = random.Random(seeds[agent_id])
            assert before == gold_rng.getstate()
            noise = gold_rng.gauss(0, 0.008)
            assert after == gold_rng.getstate()
            # Independent direction from pre-order facts, without calling signal().
            fundamental = obs["fundamental_value"] / obs["last_close"] - 1
            momentum = obs["momentum_5"]
            if agent_id == "A000":
                assert fundamental + noise < -0.003 and momentum + noise > 0.003
            expected_style = (
                "fundamental" if agent_id == "A000" and condition == "stay" else "technical"
            )
            assert state["current_style"] == expected_style
            score = (fundamental if expected_style == "fundamental" else momentum) + noise
            expected_side = "buy" if score > 0.003 else "sell"
            row = next_orders.loc[agent_id]
            assert row["style"] == expected_style
            assert row.action == row.requested_action == expected_side
            assert row.quantity == row.requested_quantity > 0
            assert row.validation_status == "ok" and not row.fallback
            if expected_side == "buy":
                assert Decimal(str(row.limit_price)) * int(row.quantity) <= Decimal(
                    str(state["portfolio"]["cash"])
                )
            else:
                assert row.quantity <= state["portfolio"]["position"]

        # Price is independently tied to the earlier, resting order at each match.
        for day, session in enumerate([first_day, next_day]):
            submitted = [r for r in submissions if r["day"] == day]
            assert len(submitted) == 2
            recorded = tables["orders"].loc[tables["orders"].session_date == session]
            assert set(recorded.order_id) == {r["order_id"] for r in submitted}
            lookup = {r["order_id"]: r for r in submitted}
            position = {r["order_id"]: i for i, r in enumerate(submitted)}
            fills = tables["trades"].loc[tables["trades"].session_date == session]
            for fill in fills.itertuples():
                buy, sell = lookup[fill.buy_order_id], lookup[fill.sell_order_id]
                assert buy["action"] == "buy" and sell["action"] == "sell"
                assert fill.buyer_id == buy["agent_id"] and fill.seller_id == sell["agent_id"]
                assert Decimal(buy["limit_price"]) >= Decimal(sell["limit_price"])
                maker = min([buy, sell], key=lambda row: position[row["order_id"]])
                assert Decimal(str(fill.price)) == Decimal(maker["limit_price"])
                assert fill.quantity == min(buy["quantity"], sell["quantity"])
            assert recorded.filled_quantity.sum() == 2 * fills.quantity.sum()

        # Reconstruct closing holdings from the initial ledger and actual fills.
        ledger = {a: (Decimal("1000"), 2) for a in seeds}
        for session in [first_day, next_day]:
            fills = tables["trades"].loc[tables["trades"].session_date == session]
            for fill in fills.itertuples():
                price = Decimal(str(fill.price))
                cash, shares = ledger[fill.buyer_id]
                ledger[fill.buyer_id] = cash - price * fill.quantity, shares + fill.quantity
                cash, shares = ledger[fill.seller_id]
                ledger[fill.seller_id] = cash + price * fill.quantity, shares - fill.quantity
            close = Decimal(str(tables["market"].set_index("session_date").loc[session, "close"]))
            accounts = tables["agents"].loc[tables["agents"].session_date == session]
            for account in accounts.itertuples():
                cash, shares = ledger[account.agent_id]
                assert cash >= 0 and shares >= 0
                assert Decimal(str(account.cash)) == cash and account.position == shares
                assert Decimal(str(account.total_equity)) == cash + close * shares
            assert sum(cash for cash, _ in ledger.values()) == Decimal("2000")
            assert sum(shares for _, shares in ledger.values()) == 4
        assert {
            a.persona.agent_id: (a.portfolio.cash, a.portfolio.position) for a in agents
        } == ledger
        runs[condition] = (captured, tables, submissions, calls)

    control, switched = runs["stay"], runs["switch_once"]
    # Everything available before the first review is the same in both runs.
    assert [
        entry for entry in control[0] if entry[2]["observation"]["session_date"] == first_day
    ] == [entry for entry in switched[0] if entry[2]["observation"]["session_date"] == first_day]
    for name in ["orders", "trades", "agents", "market"]:
        before = control[1][name]
        after = switched[1][name]
        pd.testing.assert_frame_equal(
            before.loc[before.session_date == first_day].reset_index(drop=True),
            after.loc[after.session_date == first_day].reset_index(drop=True),
        )
    assert [(r["day"], r["agent_id"]) for r in control[2]] == [
        (r["day"], r["agent_id"]) for r in switched[2]
    ]
    for agent_id in seeds:
        assert control[3][agent_id][2:] == switched[3][agent_id][2:]  # Identical random streams.
    assert control[3]["A001"][0] == switched[3]["A001"][0]
    assert {k: v for k, v in control[3]["A001"][1].items() if k != "mean_field"} == {
        k: v for k, v in switched[3]["A001"][1].items() if k != "mean_field"
    }  # Its lagged group snapshot changes; the rule does not use that field.
    summaries = []
    for condition, (_, tables, submissions, _) in runs.items():
        orders = tables["orders"].loc[tables["orders"].session_date == next_day]
        fills = tables["trades"].loc[tables["trades"].session_date == next_day]
        accounts = tables["agents"].loc[tables["agents"].session_date == next_day]
        if condition == "switch_once":
            assert set(orders.action) == {"buy"} and fills.empty
            assert not orders.filled_quantity.any() and set(orders.status) == {"expired"}
            assert tables["market"].set_index("session_date").loc[next_day, "close"] == 104
            first = (
                tables["agents"]
                .loc[tables["agents"].session_date == first_day]
                .set_index("agent_id")
            )
            pd.testing.assert_frame_equal(
                accounts.set_index("agent_id")[["cash", "position"]], first[["cash", "position"]]
            )
        else:
            assert set(orders.action) == {"buy", "sell"} and len(fills) == 1
            assert fills.quantity.sum() > 0
        summaries.append(
            {
                "condition": condition,
                "next_orders": orders[
                    [
                        "agent_id",
                        "style",
                        "action",
                        "quantity",
                        "limit_price",
                        "filled_quantity",
                        "status",
                    ]
                ].to_dict("records"),
                "next_trades": fills[["buyer_id", "seller_id", "quantity", "price"]].to_dict(
                    "records"
                ),
                "next_accounts": accounts[["agent_id", "cash", "position", "total_equity"]].to_dict(
                    "records"
                ),
                "actual_arrivals": submissions,
                "cash_and_shares_conserved": True,
            }
        )
    print(json.dumps({"execution_chain_cases": summaries}, sort_keys=True))


@pytest.mark.parametrize(
    "mode,batch", [("historical", None), ("endogenous", None), ("endogenous", 1)]
)
def test_population_visibility_changes_only_client_summary(tmp_path, monkeypatch, mode, batch):
    results = {}
    for visibility in ["default", "visible", "hidden"]:
        directory = tmp_path / visibility
        wire = []

        async def factory(path, handler, *, wire=wire):
            def capture_request(http_request):
                if http_request.method == "POST":
                    wire.append(http_request.content)
                return handler(http_request)

            return await memory_client(path, capture_request)

        with monkeypatch.context() as scoped:
            captured, agents, metrics = asyncio.run(
                run_path(
                    directory,
                    scoped,
                    mode=mode,
                    reply="stay",
                    client_factory=factory,
                    mean_field_visibility=visibility,
                    order_batch_size=batch,
                )
            )
        resolved = "hidden" if visibility == "hidden" else "visible"
        metadata = json.loads((directory / "run/run_metadata.json").read_text())
        snapshot = json.loads((directory / "run/config_snapshot.json").read_text())
        for record in [metadata, snapshot["run"]]:
            assert record["mean_field_visibility"] == resolved
            assert record["prompt_mean_field_supplied"] == (resolved == "visible")
        tables = {
            name: pd.read_parquet(directory / "run" / f"{name}.parquet")
            for name in [
                "orders",
                "style_switches",
                "agents",
                "market",
                "trades",
                "mean_field",
                "order_book_snapshots",
            ]
        }
        for name in ["orders", "style_switches"]:
            assert set(tables[name].mean_field_visibility) == {resolved}
            assert set(tables[name].prompt_mean_field_supplied) == {resolved == "visible"}
        requests = fixture_log(directory, "llm_requests.jsonl")
        assert len(captured) == len(wire) == len(requests) == 8
        assert {r["kind"] for r in requests} == {"order", "style"}
        for request in requests:
            assert (request.get("mean_field_visibility") == "hidden") == (resolved == "hidden")
        observations = [
            json.loads(line)
            for line in (directory / "run/observations.jsonl").read_text().splitlines()
        ]
        assert all(
            "mean_field" in obs and "mean_field_availability" not in obs for obs in observations
        )
        for _, _, user in captured:
            state = user["agent_state"]
            assert state["memory"][0]["owner"] == state["persona"]["agent_id"]
        results[visibility] = (captured, wire, requests, tables, observations, metrics, agents)

    default, visible, hidden = (results[key] for key in ["default", "visible", "hidden"])
    assert default[0] == visible[0] and default[1] == visible[1]
    assert default[2] == visible[2]  # Identity, body, namespace and request hash remain the same.
    for public, withheld in zip(default[0], hidden[0], strict=True):
        assert public[:2] == withheld[:2]  # Same kind and system instruction.
        expected = copy.deepcopy(public[2])
        expected["observation"].pop("mean_field")
        expected["observation"]["mean_field_availability"] = "withheld_by_experiment"
        assert withheld[2] == expected
    for public, withheld in zip(default[2], hidden[2], strict=True):
        assert public["request_hash"] != withheld["request_hash"]
        assert {
            k: v
            for k, v in withheld.items()
            if k not in {"request_hash", "payload", "mean_field_visibility"}
        } == {k: v for k, v in public.items() if k not in {"request_hash", "payload"}}
    for name, table in default[3].items():
        for candidate in [visible[3][name], hidden[3][name]]:
            drop = [
                key
                for key in ["mean_field_visibility", "prompt_mean_field_supplied", "request_hash"]
                if key in table
            ]
            pd.testing.assert_frame_equal(table.drop(columns=drop), candidate.drop(columns=drop))
    assert default[4] == visible[4] == hidden[4]
    assert default[5] == visible[5] == hidden[5]
    assert (
        [asdict(a) for a in default[6]]
        == [asdict(a) for a in visible[6]]
        == [asdict(a) for a in hidden[6]]
    )
    if batch == 1:
        assert {r[2]["observation"].get("decision_batch") for r in hidden[0] if not r[0]} == {0, 1}
    print(
        json.dumps(
            {
                "mode": mode,
                "batch": batch,
                "matched_prompts": len(default[0]),
                "hidden_changes_only_summary_availability": True,
                "fixed_reply_outputs_unchanged": True,
            }
        )
    )


@pytest.mark.parametrize("treatment", ["vanilla_llm", "low_herding", "high_herding"])
@pytest.mark.parametrize("kind", ["order", "style"])
def test_hidden_summary_prompt_does_not_mutate_private_or_market_state(treatment, kind):
    config, _ = fictional_market()
    agent = event_runner.population(config, treatment, "100")[0]
    agent.memory.append({"session": "2025-12-31", "owner": agent.persona.agent_id})
    observation = {
        "session_date": "2026-01-01",
        "last_close": 100,
        "mean_field": {"majority_action": "buy", "majority_strength": 0.8},
        "order_book": {"best_bid": 99, "best_ask": 101},
    }
    original = copy.deepcopy(observation)
    state = copy.deepcopy(asdict(agent))
    default = prompts(agent, observation, treatment, kind, 17)
    visible = prompts(agent, observation, treatment, kind, 17, mean_field_visibility="visible")
    hidden = prompts(agent, observation, treatment, kind, 17, mean_field_visibility="hidden")
    assert default == visible and hidden[0] == visible[0]
    expected = json.loads(visible[1])
    expected["observation"].pop("mean_field")
    expected["observation"]["mean_field_availability"] = "withheld_by_experiment"
    assert json.loads(hidden[1]) == expected
    assert observation == original and asdict(agent) == state


@pytest.mark.parametrize("value", [None, True, False, 0, "invalid", [], {}])
def test_population_visibility_rejects_bad_setting_before_side_effects(
    tmp_path, monkeypatch, value
):
    config, _ = fictional_market()
    config["behavior"]["mean_field_visibility"] = value
    original = copy.deepcopy(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid visibility must be rejected before other work")

    monkeypatch.setattr(event_runner, "git_state", forbidden)
    monkeypatch.setattr(event_runner, "source_hash", forbidden)
    monkeypatch.setattr(event_runner, "population", forbidden)
    with pytest.raises(ValueError, match="mean_field_visibility"):
        asyncio.run(
            event_runner.run_experiment(
                config, "endogenous", "low_herding", "aigc_relay", tmp_path / "rejected"
            )
        )
    with pytest.raises(ValueError, match="mean_field_visibility"):
        prompts(None, {}, "low_herding", "order", 17, mean_field_visibility=value)
    assert config == original and not (tmp_path / "rejected").exists()


def test_rule_policy_is_a_negative_control_for_population_summary():
    config, _ = fictional_market()
    agents = event_runner.population(config, "low_herding", "100")
    for agent in agents:
        original = copy.deepcopy(asdict(agent))
        base = {"last_close": 104, "fundamental_value": 100, "momentum_5": 0.04}
        observations = [
            {**base, "mean_field": {"majority_action": "buy", "majority_strength": 1}},
            {**base, "mean_field": {"majority_action": "sell", "majority_strength": 1}},
            {**base, "mean_field_availability": "withheld_by_experiment"},
        ]
        orders, states = [], []
        for observation in observations:
            rng = random.Random(17000 + int(agent.persona.agent_id[1:]))
            orders.append(rule_order(agent, observation, rng))
            states.append(rng.getstate())
        assert orders[0] == orders[1] == orders[2]
        assert states[0] == states[1] == states[2]
        assert asdict(agent) == original


@pytest.mark.parametrize("source", ["rule", "MockLLMClient", "injected_client"])
def test_hidden_summary_provenance_does_not_claim_real_model_exposure(
    tmp_path, monkeypatch, source
):
    class InjectedClient:
        async def decide(self, *args):
            return await MockLLMClient().decide(*args)

    async def factory(path, handler):
        return InjectedClient() if source == "injected_client" else MockLLMClient()

    asyncio.run(
        run_path(
            tmp_path,
            monkeypatch,
            mean_field_visibility="hidden",
            client_factory=factory,
            treatment="rule_abm" if source == "rule" else "low_herding",
        )
    )
    for name in ["orders", "style_switches"]:
        table = pd.read_parquet(tmp_path / "run" / f"{name}.parquet")
        assert set(table.mean_field_visibility) == {"hidden"}
        if source == "rule":
            assert table.prompt_mean_field_supplied.isna().all()
        else:
            assert not table.prompt_mean_field_supplied.any()
    styles = pd.read_parquet(tmp_path / "run/style_switches.parquet")
    assert set(styles.decision_source) == {source}
    assert styles.request_hash.isna().all()
    assert not (tmp_path / "logs").exists()


def test_population_visibility_output_paths_keep_default_and_separate_hidden():
    config, _ = fictional_market()
    for mode in ["historical", "endogenous"]:
        default = event_runner.output_path(config, mode, "low_herding", "aigc_relay")
        visible = copy.deepcopy(config)
        visible["behavior"]["mean_field_visibility"] = "visible"
        hidden = copy.deepcopy(config)
        hidden["behavior"]["mean_field_visibility"] = "hidden"
        assert event_runner.output_path(visible, mode, "low_herding", "aigc_relay") == default
        assert event_runner.output_path(
            hidden, mode, "low_herding", "aigc_relay"
        ) == default.with_name(default.name + "_mean_field_hidden")
