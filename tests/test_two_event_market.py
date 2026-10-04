import asyncio
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from behavioral_market.llm.schemas import hold
from experiments.two_event_market import (
    context,
    lot_history,
    round_schedule,
    run_case,
    setup,
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
