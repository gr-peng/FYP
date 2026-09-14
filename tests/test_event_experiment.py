import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pandas as pd
import pytest
from pydantic import ValidationError

from behavioral_market.agents.policy import enforce_portfolio
from behavioral_market.agents.state import population
from behavioral_market.data.alpha_vantage import parse_daily_csv
from behavioral_market.data.calendar import sessions, visible_news
from behavioral_market.data.features import historical_features
from behavioral_market.data.news_dedup import deduplicate, normalize_url
from behavioral_market.data.schemas import DailyBar, MarketEvent, NewsItem
from behavioral_market.environments.observations import available_fundamentals
from behavioral_market.evaluation.event_report import simulated_event_cars
from behavioral_market.evaluation.metrics import event_study, run_metrics
from behavioral_market.llm.client import FatalAPIError, RequestPacer, SiliconFlowClient
from behavioral_market.llm.schemas import OrderDecision
from behavioral_market.simulation.event_runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_alpha_parser_rejects_quota_and_dedup_keeps_earliest():
    raw = "timestamp,open,high,low,close,volume\n2026-07-01,100,102,98,101,1000\n"
    parsed = parse_daily_csv(raw, "NVDA", "2026-09-01T00:00:00Z")
    assert float(parsed[0].close) == 101
    with pytest.raises(ValueError, match="quota"):
        parse_daily_csv('{"Information":"quota exhausted"}', "NVDA", "2026-09-01T00:00:00Z")
    original = NewsItem(
        news_id="1",
        published_at_utc="2026-07-01T14:39:00Z",
        published_at_et="2026-07-01T10:39:00-04:00",
        source="fixture",
        title="Meta considers selling excess compute",
        url="https://example.com/story?utm_source=x",
        fetched_at="2026-09-01T00:00:00Z",
    )
    duplicate = original.model_copy(update={"news_id": "2", "url": "https://example.com/story"})
    assert normalize_url(original.url) == duplicate.url
    assert deduplicate([original, duplicate]) == [original]


@pytest.fixture
def config():
    value = json.loads((ROOT / "config/meta_compute_2026.json").read_text())
    value["population"]["n_agents"] = 4
    return value


@pytest.fixture
def events():
    return [
        MarketEvent.model_validate(x)
        for x in json.loads((ROOT / "config/meta_compute_2026_events.json").read_text())
    ]


@pytest.fixture
def bars():
    rows = []
    for symbol in ["NVDA", "META", "SOXX", "QQQ", "SPY"]:
        for i, day in enumerate(sessions("2026-04-01", "2026-07-17").index):
            close = 100 + i / 10
            rows.append(
                dict(
                    symbol=symbol,
                    session_date=day,
                    open=close,
                    high=close * 1.02,
                    low=close * 0.98,
                    close=close,
                    volume=10000,
                )
            )
    return pd.DataFrame(rows)


def test_calendar_and_event_visibility(events):
    schedule = sessions("2026-06-15", "2026-07-17")
    assert len(schedule) == 23
    assert "2026-06-19" not in schedule.index and "2026-07-03" not in schedule.index
    assert schedule.loc["2026-07-01"].market_open.hour == 13
    assert not visible_news(
        events, schedule.loc["2026-06-30"].market_open, schedule.loc["2026-07-01"].market_open
    )
    assert [
        e.event_id
        for e in visible_news(
            events, schedule.loc["2026-07-01"].market_open, schedule.loc["2026-07-02"].market_open
        )
    ] == [events[0].event_id]
    assert [
        e.event_id
        for e in visible_news(
            events, schedule.loc["2026-07-09"].market_open, schedule.loc["2026-07-10"].market_open
        )
    ] == [events[1].event_id]
    assert sessions("2026-01-05", "2026-01-05").market_open.iloc[0].hour == 14


def test_features_exclude_same_day_and_future(bars):
    primary = bars[bars.symbol == "NVDA"].copy()
    original = historical_features(primary, "2026-07-01")
    primary.loc[primary.session_date >= "2026-07-01", "close"] = 999999
    assert historical_features(primary, "2026-07-01") == original


def test_schema_and_portfolio_constraints(config):
    agent = population(config, "low_herding", "100")[0]
    for quantity in [-1, 1.5, True]:
        with pytest.raises(ValidationError):
            OrderDecision(
                action="buy", quantity=quantity, limit_price=100, rationale="", sentiment="neutral"
            )
    for action, quantity in [("buy", 1000), ("sell", 101)]:
        order = OrderDecision(
            action=action, quantity=quantity, limit_price=100, rationale="", sentiment="neutral"
        )
        converted, status = enforce_portfolio(order, agent.portfolio)
        assert converted.action == "hold" and status == "constraint_violation"
    with pytest.raises(ValidationError):
        DailyBar(
            symbol="NVDA",
            session_date="2026-07-01",
            open=100,
            high=90,
            low=80,
            close=100,
            volume=1,
            source="fixture",
            fetched_at="2026-09-01T00:00:00Z",
        )


def test_treatments_change_only_herding(config):
    low = population(config, "low_herding", "100")
    high = population(config, "high_herding", "100")
    for a, b in zip(low, high, strict=True):
        x, y = asdict(a.persona), asdict(b.persona)
        assert y.pop("herding") > x.pop("herding")
        assert x == y


def test_sec_future_restatements_hidden():
    facts = {
        "facts": {
            "us-gaap": {
                "AssetsCurrent": {
                    "units": {
                        "USD": [
                            {"filed": "2026-06-01", "end": "2026-04-30", "val": 20},
                            {"filed": "2026-08-01", "end": "2026-04-30", "val": 999},
                        ]
                    }
                },
                "LiabilitiesCurrent": {
                    "units": {"USD": [{"filed": "2026-06-01", "end": "2026-04-30", "val": 10}]}
                },
            }
        }
    }
    assert available_fundamentals(facts, "2026-07-01")["current_ratio"] == 2
    assert available_fundamentals(facts, "2026-06-01")["current_ratio"] is None


def test_api_repair_fallback_cache_and_no_key_leak(config, tmp_path):
    async def execute():
        count = 0

        def respond(request):
            nonlocal count
            count += 1
            return httpx.Response(
                200,
                json={
                    "model": config["llm"]["model"],
                    "choices": [{"message": {"content": "bad"}}],
                },
            )

        client = SiliconFlowClient(
            config["llm"], "unit-test-secret", tmp_path / "logs", tmp_path / "cache"
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.siliconflow.cn/v1/"
        )
        decision, meta = await client.decide("system", "{}", OrderDecision, {"seed": 42})
        assert decision.action == "hold" and meta["fallback"] and count == 2
        await client.decide("system", "{}", OrderDecision, {"seed": 42})
        assert count == 2
        await client.decide("system", "{}", OrderDecision, {"seed": 43})
        assert count == 4
        client.log("extra.jsonl", {"value": "unit-test-secret"})
        await client.close()
        assert all(
            "unit-test-secret" not in p.read_text() for p in (tmp_path / "logs").glob("*.jsonl")
        )

    asyncio.run(execute())


def test_shared_pacer_backoff_and_cache_only(config, tmp_path):
    async def execute():
        calls = []

        def respond(request):
            calls.append(asyncio.get_running_loop().time())
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0.05"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"action":"hold","quantity":0,"limit_price":null,'
                                '"rationale":"ok","sentiment":"neutral"}'
                            }
                        }
                    ],
                    "model": config["llm"]["model"],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        pacer = RequestPacer(100)
        client = SiliconFlowClient(
            config["llm"], "unit-test-secret", tmp_path / "live", tmp_path / "cache", pacer=pacer
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://api.siliconflow.cn/v1/"
        )
        result, meta = await client.decide("system", "{}", OrderDecision, {"seed": 99})
        await client.close()
        assert result.action == "hold" and meta["status"] == "ok"
        assert len(calls) == 2 and calls[1] - calls[0] >= 0.045
        attempts = [
            json.loads(x) for x in (tmp_path / "live/llm_attempts.jsonl").read_text().splitlines()
        ]
        assert [a["retry_count"] for a in attempts] == [0, 1]
        responses = [
            json.loads(x) for x in (tmp_path / "live/llm_responses.jsonl").read_text().splitlines()
        ]
        assert [r["http_status"] for r in responses] == [429, 200]
        assert all(r["requested_at"] <= r["response_received_at"] for r in responses)
        offline = SiliconFlowClient(
            config["llm"], None, tmp_path / "offline", tmp_path / "cache", cache_only=True
        )
        replay, replay_meta = await offline.decide("system", "{}", OrderDecision, {"seed": 99})
        assert replay.action == "hold" and replay_meta["cache_hit"]
        with pytest.raises(FatalAPIError, match="missing response"):
            await offline.decide("system", "{}", OrderDecision, {"seed": 100})
        await offline.close()

    asyncio.run(execute())


def test_three_session_drawdown_cannot_be_negative():
    market = pd.DataFrame(
        {
            "session_date": ["2026-07-01", "2026-07-02", "2026-07-06"],
            "close": [101.0, 102.0, 103.0],
            "simulated_volume": [0, 0, 0],
            "fundamental_value": [100.0, 100.0, 100.0],
            "order_imbalance": [0.0, 0.0, 0.0],
        }
    )
    orders = pd.DataFrame(
        {
            "action": ["hold"],
            "quantity": [0],
            "filled_quantity": [0],
            "validation_status": ["ok"],
            "fallback": [False],
            "aggressiveness": [0.0],
        }
    )
    result = run_metrics(market, orders, pd.DataFrame(), 100.0, 100)
    assert result["three_session_drawdown_2026-07-01"] == 0


def test_historical_path_has_zero_car_difference(bars):
    primary = bars[bars.symbol == "NVDA"].sort_values("session_date")
    market = primary[primary.session_date >= "2026-06-15"][["session_date", "close"]]
    initial = float(primary.loc[primary.session_date == "2026-06-12", "close"].iloc[0])
    studies = pd.DataFrame(
        [event_study(bars, "NVDA", "QQQ", event) for event in ["2026-07-01", "2026-07-09"]]
    )
    diagnostics = simulated_event_cars(market, bars, studies, initial)
    assert all(
        abs(value) < 1e-12 for key, value in diagnostics.items() if key.startswith("CAR_difference")
    )


def test_mock_replay_immutable_prices_and_lag(config, bars, events, tmp_path):
    original = bars.copy(deep=True)
    metrics = asyncio.run(
        run_experiment(
            config,
            "historical",
            "low_herding",
            "mock",
            tmp_path / "first",
            bars=bars,
            events=events,
        )
    )
    repeated = asyncio.run(
        run_experiment(
            config,
            "historical",
            "low_herding",
            "mock",
            tmp_path / "second",
            bars=bars,
            events=events,
        )
    )
    assert metrics == repeated
    pd.testing.assert_frame_equal(bars, original)
    market = pd.read_parquet(tmp_path / "first/market.parquet")
    real = bars[(bars.symbol == "NVDA") & (bars.session_date >= "2026-06-15")]
    assert list(market.close) == list(real.close)
    assert list(market.open) == list(real.open)
    assert list(market.high) == list(real.high)
    assert list(market.low) == list(real.low)
    observations = [
        json.loads(x) for x in (tmp_path / "first/observations.jsonl").read_text().splitlines()
    ]
    for obs in observations:
        assert obs["mean_field"]["as_of_session"] < obs["session_date"]
    assert len(pd.read_parquet(tmp_path / "first/style_switches.parquet")) == 8


def test_endogenous_entire_run_cannot_read_future_bars(config, bars, events, tmp_path):
    altered = bars.copy(deep=True)
    altered.loc[
        altered.session_date >= "2026-06-15", ["open", "high", "low", "close", "volume"]
    ] = 999999
    first = asyncio.run(
        run_experiment(
            config,
            "endogenous",
            "high_herding",
            "mock",
            tmp_path / "first",
            bars=bars,
            events=events,
        )
    )
    second = asyncio.run(
        run_experiment(
            copy.deepcopy(config),
            "endogenous",
            "high_herding",
            "mock",
            tmp_path / "second",
            bars=altered,
            events=events,
        )
    )
    assert first == second
    assert (tmp_path / "first/observations.jsonl").read_bytes() == (
        tmp_path / "second/observations.jsonl"
    ).read_bytes()
    accounts = pd.read_parquet(tmp_path / "first/agents.parquet")
    assert (accounts.cash >= 0).all() and (accounts.position >= 0).all()
    for _, day in accounts.groupby("session_date"):
        assert day.position.sum() == 400
        assert day.cash.sum() == pytest.approx(40000)
