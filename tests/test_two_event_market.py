import asyncio
import copy
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.schemas import OrderDecision, hold
from experiments.two_event_market import (
    context,
    exchange_decision,
    lot_history,
    round_schedule,
    run_case,
    run_suite,
    setup,
    system_prompt,
    validate_order,
    visible_events,
)

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    config = json.loads((ROOT / "config/two_event_market_pilot.json").read_text())
    spec = config["scenarios"][1]
    dates = pd.date_range("2024-10-25", "2024-11-07").strftime("%Y-%m-%d")
    bars = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "session_date": dates,
            "open": 60000.0,
            "high": 61000.0,
            "low": 59000.0,
            "close": 60000.0,
            "volume": 12.3456,
        }
    )
    return config, spec, bars


def test_future_prices_cannot_change_any_initial_information_or_units():
    config, spec, bars = fixtures()
    changed = bars.copy()
    changed.loc[changed.session_date >= spec["start"], ["close", "volume"]] *= 1000
    pd.testing.assert_frame_equal(lot_history(bars, spec), lot_history(changed, spec))
    history = lot_history(bars, spec)
    assert history.close.iloc[-1] == 60
    assert history.volume.iloc[-1] == pytest.approx(12345.6)
    payloads = []
    for data in (bars, changed):
        agents, targets, env = setup(config, spec, data, 42, "finite")
        env.start_session(spec["start"])
        env.requote_background()
        payloads.append(
            context(
                config,
                spec,
                agents[0],
                targets[0],
                env,
                "2024-11-05T00:00Z",
                "news",
                "private_targets",
                {},
            )
        )
    assert payloads[0] == payloads[1]
    assert payloads[0]["observation"]["completed_history_through"] == "2024-11-04"
    assert payloads[0]["observation"]["news"] == []


def test_news_cutoffs_and_decision_opportunities_include_intraday_and_weekends():
    _, spec, _ = fixtures()
    assert len(round_schedule(spec)) == 5
    assert len(visible_events(spec, "2024-11-06T07:23:59Z", "news")) == 0
    assert len(visible_events(spec, "2024-11-06T07:24:00Z", "news")) == 1
    assert len(visible_events(spec, "2024-11-06T10:34:00Z", "news")) == 2
    assert visible_events(spec, "2024-11-07T00:00:00Z", "no_news") == []
    weekend = copy.deepcopy(spec)
    weekend.update(start="2024-11-08", end="2024-11-10")
    assert len(round_schedule(weekend)) == 3


def test_paired_rule_negative_control_and_global_conservation(tmp_path):
    config, spec, bars = fixtures()
    results = []
    for condition in ("news", "no_news"):
        out = tmp_path / condition
        result = asyncio.run(
            run_case(config, spec, bars, (42, "private_targets", condition, "finite"), out, "test")
        )
        assert result["cash_conservation"] == "0.000000"
        assert result["lot_conservation"] == 0
        assert result["rejected_orders"] == 0
        assert result["proposed_actions"]["buy"] > 0
        assert result["proposed_actions"]["sell"] > 0
        assert result["trades"] > 0
        assert 0 < result["agent_fill_ratio"] <= 1
        result.pop("event_condition")
        results.append(result)
    assert results[0] == results[1]
    # Changes after the experiment start cannot change the simulated paths either.
    bars.loc[bars.session_date >= spec["start"], "close"] = 1.0
    rerun = asyncio.run(
        run_case(
            config,
            spec,
            bars,
            (42, "private_targets", "news", "finite"),
            tmp_path / "changed_future",
            "test",
        )
    )
    rerun.pop("event_condition")
    assert rerun == results[0]


def test_invalid_model_output_fails_run_instead_of_silent_hold(tmp_path):
    class InvalidClient:
        async def decide(self, *args):
            return hold("transport error"), {"fallback": True}

    config, spec, bars = fixtures()
    with pytest.raises(RuntimeError, match="instead of using fallback"):
        asyncio.run(
            run_case(
                config,
                spec,
                bars,
                (42, "vanilla", "news", "none"),
                tmp_path,
                "test",
                InvalidClient(),
                "mock",
            )
        )
    assert json.loads((tmp_path / "run_metadata.json").read_text())["status"] == "failed"
    assert not (tmp_path / "metrics.json").exists()


def test_cent_rounded_mu_ask_fills_on_declared_grid_and_weights_are_exact():
    config, _, bars = fixtures()
    spec = config["scenarios"][0].copy()
    spec.update(start="2024-11-05", end="2024-11-07", events=[])
    bars["symbol"] = "MU"
    bars[["open", "high", "low", "close"]] = 1145.280029
    outcomes = []
    for fixed in (False, True):
        trial, market = copy.deepcopy(config), copy.deepcopy(spec)
        if fixed:
            trial.update(price_basis="underlying", computed_portfolio_facts=True)
            market["asset_price_tick"] = 0.01
        agents, _, env = setup(trial, market, bars, 42, "finite")
        env.start_session(market["start"])
        env.requote_background()
        payload = context(
            trial,
            market,
            agents[0],
            0.2,
            env,
            "2024-11-05T14:30Z",
            "no_news",
            "private_targets",
            {},
        )
        decision = OrderDecision(
            action="buy",
            quantity=1,
            limit_price=round(payload["observation"]["book"]["best_ask"], 2),
            rationale="copy ask to cents",
            sentiment="neutral",
        )
        assert validate_order(decision, payload) is None
        env.submit(agents[0], exchange_decision(decision, payload))
        outcomes.append(sum(t["quantity"] for t in env._trades))
        if fixed:
            facts = payload["agent_state"]["computed_account_facts"]
            assert facts["actual_asset_weight_fraction"] == pytest.approx(0.49620664)
            assert facts["target_minus_actual_asset_weight_percentage_points"] == pytest.approx(
                -29.620664
            )
        env.finish_session(agents)
    assert outcomes == [0, 1]


def test_btc_underlying_quote_converts_exactly_once_into_lot_cash_cost():
    config, spec, bars = fixtures()
    config.update(price_basis="underlying", computed_portfolio_facts=True)
    spec["asset_price_tick"] = 0.01
    agents, _, env = setup(config, spec, bars, 42, "finite")
    env.start_session(spec["start"])
    env.requote_background()
    payload = context(
        config, spec, agents[0], 0.8, env, "2024-11-05T00:00Z", "no_news", "private_targets", {}
    )
    assert payload["observation"]["last_trade"] == 60000
    assert payload["constraints"]["price_tick"] == 0.01
    assert "quantity * lot_size * limit_price" in system_prompt(config)
    before = agents[0].portfolio.cash
    decision = OrderDecision(
        action="buy",
        quantity=1,
        limit_price=payload["observation"]["book"]["best_ask"],
        rationale="take one lot",
        sentiment="bullish",
    )
    assert validate_order(decision, payload) is None
    internal = exchange_decision(decision, payload)
    env.submit(agents[0], internal)
    assert before - agents[0].portfolio.cash == Decimal(str(decision.limit_price)) * Decimal(".001")
    assert len(env._trades) == 1
    bad = decision.model_copy(update={"limit_price": decision.limit_price + 0.001})
    assert validate_order(bad, payload) == "price_tick"
    env.finish_session(agents)


def test_parallel_disjoint_pairs_preserve_paths_and_pre_news_pair_order(tmp_path):
    config, spec, bars = fixtures()
    config.update(price_basis="underlying", computed_portfolio_facts=True)
    spec["asset_price_tick"] = 0.01
    config["scenarios"] = [spec]
    data_config = {"study_id": "synthetic_pair_test"}
    write_json(tmp_path / config["data_config"], data_config)
    directory = tmp_path / "data/processed" / data_config["study_id"]
    directory.mkdir(parents=True)
    file = directory / f"{spec['scenario_id']}_daily.parquet"
    bars.to_parquet(file)
    write_json(
        directory / "manifest.json",
        {
            "config_hash": digest(json.dumps(data_config, sort_keys=True).encode()),
            "raw_files": [],
            "processed_files": [
                {"path": str(file.relative_to(tmp_path)), "sha256": digest(file.read_bytes())}
            ],
        },
    )
    results = []
    for parallel in (1, 4):
        config.update(pair_concurrency=parallel, experiment_id=f"concurrency_{parallel}")
        results.append(asyncio.run(run_suite(config, "mock", root=tmp_path)))
        output = tmp_path / "outputs" / data_config["study_id"] / config["experiment_id"] / "mock"
        for path in output.rglob("no_news/run_metadata.json"):
            no = json.loads(path.read_text())
            yes = json.loads((path.parent.parent / "news/run_metadata.json").read_text())
            assert no["started_at"] >= yes["completed_at"]
    assert results[0] == results[1]
