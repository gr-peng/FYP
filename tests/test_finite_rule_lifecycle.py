"""Finite comparisons must survive two reviews without borrowing actual trades."""

import asyncio
import copy
import json
from dataclasses import asdict
from decimal import Decimal

import pandas as pd
import pytest
from test_event_report_visibility import report_fixture as report_fixture
from test_style_path_integration import fictional_market
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.agents.policy import rule_style
from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.evaluation import event_report
from behavioral_market.llm.schemas import OrderDecision, StyleDecision, hold
from behavioral_market.market import Portfolio
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.finite_rule_pair import (
    LEGACY_METHOD,
    METHOD,
    FiniteRulePair,
    settings,
)

SPEC = {"half_spread": "0.05", "counterparty_cash": "10000", "counterparty_shares": 100}


def agent():
    p = Portfolio(1000, 2, 100)
    return AgentState(
        AgentPersona("A000", 0.5, 0.5, 0.5, 0.5, 0.5, "fundamental"),
        p,
        "fundamental",
        1200,
        1200,
        peak_equity=1200,
    )


def observation(day, reference=100, momentum=0, anchor=None):
    return {
        "session_date": f"2026-01-{day:02}",
        "history_last_session": "2025-12-31" if day == 1 else f"2026-01-{day - 1:02}",
        "last_close": reference,
        "fundamental_value": anchor or reference,
        "momentum_5": momentum,
        "permitted_price_band": {"minimum": reference * 0.95, "maximum": reference * 1.05},
    }


def test_finite_pair_matches_hand_calculation_and_preserves_actual_account():
    actual = agent()
    original = copy.deepcopy(asdict(actual))
    pair = FiniteRulePair(actual, 100, SPEC, 0)
    all_rows = []
    for obs, close in [
        (observation(1, momentum=0.1), 110),
        (observation(2, reference=110, momentum=-0.1), 100),
    ]:
        pair.open_session(obs)
        with pytest.raises(ValueError, match="unclosed"):
            pair.summary(actual.portfolio)
        all_rows.extend(pair.close_session(close))
    summary = pair.summary(actual.portfolio)
    assert summary["current_rule_block_return"] == 0
    assert summary["alternative_rule_block_return"] == pytest.approx(49.5 / 1200)
    assert summary["gap"] == summary["alternative_rule_block_return"]
    alternative = [r for r in all_rows if r["branch"] == "alternative_rule"]
    assert [r["trades"][0]["price"] for r in alternative] == ["100.05", "109.95"]
    assert [r["trades"][0]["quantity"] for r in alternative] == [5, 5]
    assert Decimal(alternative[-1]["portfolio_after_close"]["cash"]) == Decimal("1049.50")
    for row in all_rows:
        p, cp = row["portfolio_after_close"], row["counterparty_after_close"]
        assert Decimal(p["cash"]) + Decimal(cp["cash"]) == 11000
        assert p["position"] + cp["position"] == 102
    assert asdict(actual) == original
    actual.finite_review = summary
    assert rule_style(actual).decision == "switch"
    # Deliberately change only the actual account: paired rule choice must not change.
    actual.portfolio.cash = Decimal("1")
    actual.portfolio.mark_to_market(100)
    assert pair.summary(actual.portfolio)["gap"] == summary["gap"]
    assert pair.summary(actual.portfolio)["observed_account_return"] != 0


@pytest.mark.parametrize(
    "spread,shares,cash,expected_fills",
    [
        ("1", 100, "10000", [0, 0]),
        ("0.05", 1, "10000", [1, 0]),
        ("0.05", 0, "10000", [0, 0]),
    ],
)
def test_partial_empty_and_non_crossing_quotes_are_not_fabricated(
    spread, shares, cash, expected_fills
):
    pair = FiniteRulePair(
        agent(),
        100,
        {"half_spread": spread, "counterparty_cash": cash, "counterparty_shares": shares},
        0,
    )
    fills = []
    for day in [1, 2]:
        pair.open_session(observation(day, momentum=0.1))
        row = pair.close_session(100)[1]
        fills.append(sum(t["quantity"] for t in row["trades"]))
        assert row["cash_and_shares_conserved"]
    assert fills == expected_fills


def test_counterparty_buy_cash_is_finite_and_price_band_applies():
    pair = FiniteRulePair(agent(), 100, {**SPEC, "counterparty_cash": "50"}, 0)
    pair.open_session(observation(1, momentum=-0.1))
    row = pair.close_session(100)[1]
    assert row["quote_status"] == "counterparty_exhausted" and not row["trades"]
    obs = observation(2, momentum=0.1)
    obs["permitted_price_band"]["maximum"] = 100
    pair.open_session(obs)
    row = pair.close_session(100)[1]
    assert row["constraint_violation"] == "price_band_violation" and not row["trades"]


def test_closing_marks_cannot_change_orders_and_history_cannot_be_future():
    orders = []
    for close in [1, 10000]:
        pair = FiniteRulePair(agent(), 100, SPEC, 0)
        pair.open_session(observation(1, momentum=0.1))
        row = pair.close_session(close)[1]
        orders.append((row["requested_order"], row["trades"]))
    assert orders[0] == orders[1]
    with pytest.raises(ValueError, match="increasing"):
        pair.open_session(observation(1))
    with pytest.raises(ValueError, match="prior completed"):
        pair.open_session({**observation(2), "history_last_session": "2026-01-02"})
    with pytest.raises(ValueError, match="positive"):
        pair = FiniteRulePair(agent(), 0, SPEC, 0)


class ScriptedClient:
    def __init__(self, forced_switch=True):
        self.calls = []
        self.forced_switch = forced_switch

    async def decide(self, system, user, schema, identity):
        data = json.loads(user)
        self.calls.append((system, data, copy.deepcopy(identity)))
        day = data["observation"]["session_date"]
        state = data["agent_state"]
        if schema is StyleDecision:
            change = self.forced_switch and day == "2026-01-05"
            return StyleDecision(
                decision="switch" if change else "stay",
                target_style="technical" if change else state["current_style"],
                rationale="scripted lifecycle test",
            ), {"status": "mock"}
        if day == "2026-01-01":
            return OrderDecision(
                action="buy",
                quantity=8,
                limit_price=105,
                rationale="scripted loss",
                sentiment="neutral",
            ), {"status": "mock"}
        if day == "2026-01-06":
            return OrderDecision(
                action="sell",
                quantity=2,
                limit_price=100,
                rationale="scripted sale",
                sentiment="neutral",
            ), {"status": "mock"}
        return hold("scripted waiting"), {"status": "mock"}


def run_two_blocks(tmp_path, monkeypatch, *, method=METHOD, future=False, mode="historical"):
    config, bars = fictional_market(interval=5, days=10)
    config["population"].update(n_agents=1, initial_fundamental_share=1)
    bars[["open", "close"]] = 100.0
    bars["low"], bars["high"] = 95.0, 105.0
    if future:
        bars.loc[bars.session_date >= "2026-01-06", ["open", "close", "high", "low"]] *= 2
    if method is not None:
        config["behavior"]["strategy_return_method"] = method
    if method == METHOD:
        config["behavior"]["finite_rule_pair"] = SPEC
    created = []
    create_population = event_runner.population

    def capture_population(*args):
        result = create_population(*args)
        created.extend(result)
        return result

    with monkeypatch.context() as scope:
        scope.setattr(event_runner, "population", capture_population)
        client = ScriptedClient()
        asyncio.run(
            event_runner.run_experiment(
                config, mode, "vanilla_llm", "mock", tmp_path, bars=bars, events=[], client=client
            )
        )
    return client, created[0], config


def test_two_reviews_use_paired_returns_and_reset_from_actual_portfolio(tmp_path, monkeypatch):
    client, final, config = run_two_blocks(tmp_path, monkeypatch)
    reviews = [data for _, data, ident in client.calls if ident["kind"] == "style"]
    assert len(reviews) == 2
    first, second = [r["agent_state"]["counterfactual"] for r in reviews]
    assert first["observed_account_return"] == pytest.approx(-40 / 1200)
    assert first["gap"] == second["gap"] == 0
    assert first["current_rule_block_return"] == first["alternative_rule_block_return"] == 0
    assert (second["current_rule_style"], second["alternative_rule_style"]) == (
        "technical",
        "fundamental",
    )
    start = second["block_start_portfolio"]
    assert Decimal(start["cash"]) == 160 and start["position"] == 10
    assert Decimal(start["avg_cost"]) == 104 and Decimal(start["total_equity"]) == 1160
    assert second["observed_account_return"] == 0 and second["completed_sessions"] == 5
    assert final.portfolio.cash == 360 and final.portfolio.position == 8
    assert (
        final.finite_review["block_index"] == 2 and final.finite_review["completed_sessions"] == 0
    )
    assert final.finite_review["block_start_portfolio"]["position"] == 8
    assert final.shadow_growth == 1
    # A baseline review must stay despite the separate actual account's loss.
    hypothetical = agent()
    hypothetical.portfolio = Portfolio(160, 10, 104)
    hypothetical.portfolio.mark_to_market(100)
    hypothetical.finite_review = first
    assert rule_style(hypothetical).decision == "stay"
    switches = pd.read_parquet(tmp_path / "style_switches.parquet")
    accounts = pd.read_parquet(tmp_path / "agents.parquet")
    assert len(switches) == 2 and len(accounts) == 10
    for frame in [switches, accounts]:
        assert set(frame.strategy_return_method) == {METHOD}
        assert "actual_block_return" not in frame and "counterfactual_block_return" not in frame
    for row, review in zip(switches.to_dict("records"), [first, second], strict=True):
        assert row["paired_return_gap"] == review["gap"]
        for key in [
            "observed_account_return",
            "current_rule_block_return",
            "alternative_rule_block_return",
        ]:
            assert row[key] == review[key]
    for data in reviews:
        assert "frictionless long/cash" not in data["decision_instruction"]
        assert "observed_account_return" in data["decision_instruction"]
        assert "permitted_price_band" not in data["observation"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "finite_rule_replay.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 20  # Two branches per day, ten days, one agent.
    reset_rows = [r for r in rows if r["session_date"] == "2026-01-06"]
    assert len(reset_rows) == 2
    for row in reset_rows:
        assert row["portfolio_before_order"] == start
        assert row["portfolio_after_close"]["position"] == 10
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    snapshot = json.loads((tmp_path / "config_snapshot.json").read_text())
    assert metadata["strategy_return_method"] == snapshot["run"]["strategy_return_method"] == METHOD
    assert "finite_rule_replay.jsonl" in metadata["artifact_hashes"]


def test_no_future_price_leaks_into_first_block_or_next_open(tmp_path, monkeypatch):
    first, _, _ = run_two_blocks(tmp_path / "one", monkeypatch)
    changed, _, _ = run_two_blocks(tmp_path / "two", monkeypatch, future=True)
    # Jan 6 order still precedes the first changed close.
    one = [r for r in first.calls if r[1]["observation"]["session_date"] <= "2026-01-06"]
    two = [r for r in changed.calls if r[1]["observation"]["session_date"] <= "2026-01-06"]
    assert one == two


def test_legacy_default_keeps_prompt_identity_and_outputs(tmp_path, monkeypatch):
    default, _, config = run_two_blocks(tmp_path / "default", monkeypatch, method=None)
    explicit, _, _ = run_two_blocks(tmp_path / "explicit", monkeypatch, method=LEGACY_METHOD)
    assert default.calls == explicit.calls
    assert not (tmp_path / "default/finite_rule_replay.jsonl").exists()
    for name in ["orders", "trades", "market", "agents", "style_switches", "mean_field"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "default" / (name + ".parquet")),
            pd.read_parquet(tmp_path / "explicit" / (name + ".parquet")),
        )
    base = event_runner.output_path(config, "historical", "vanilla_llm", "mock")
    config["behavior"].update(strategy_return_method=METHOD, finite_rule_pair=SPEC)
    assert event_runner.output_path(config, "historical", "vanilla_llm", "mock") != base


def test_finite_path_runs_in_endogenous_mode_without_adding_shadow_orders(tmp_path, monkeypatch):
    _, _, _ = run_two_blocks(tmp_path, monkeypatch, mode="endogenous")
    assert len(pd.read_parquet(tmp_path / "orders.parquet")) == 10
    assert len(pd.read_parquet(tmp_path / "trades.parquet")) == 0
    assert len(pd.read_parquet(tmp_path / "style_switches.parquet")) == 2


@pytest.mark.parametrize(
    "update",
    [
        {"strategy_return_method": None},
        {"strategy_return_method": "unknown"},
        {"finite_rule_pair": None},
        {"finite_rule_pair": {**SPEC, "half_spread": "nan"}},
        {"finite_rule_pair": {**SPEC, "counterparty_shares": True}},
        {"finite_rule_pair": {**SPEC, "counterparty_cash": -1}},
        {"strategy_switch_interval": 0},
    ],
)
def test_invalid_settings_fail_before_outputs_or_clients(tmp_path, monkeypatch, update):
    config, _ = fictional_market()
    config["behavior"].update(strategy_return_method=METHOD, finite_rule_pair=SPEC)
    config["behavior"].update(update)
    with pytest.raises(ValueError):
        asyncio.run(
            event_runner.run_experiment(
                config, "historical", "vanilla_llm", "aigc_relay", tmp_path / "invalid"
            )
        )
    assert not (tmp_path / "invalid").exists()


def test_legacy_report_rejects_finite_method_instead_of_pooling(request):
    config, directory, destination, add_run, _ = request.getfixturevalue("report_fixture")
    add_run("finite", "visible", "low_herding", 100)
    path = directory / "finite/run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["strategy_return_method"] = METHOD
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not yet support finite"):
        event_report.evaluate(config, directory, destination)


def test_finite_settings_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="explicit"):
        settings({"finite_rule_pair": SPEC})


@pytest.mark.parametrize(
    "alternative_cash,decision",
    [
        ("1047.999999", "stay"),
        ("1048", "stay"),
        ("1048.000001", "switch"),
    ],
)
def test_paired_gap_strict_one_percent_boundary(alternative_cash, decision):
    actual = agent()
    pair = FiniteRulePair(actual, 100, SPEC, 0)
    for name, cash in [("current_rule", "1036"), ("alternative_rule", alternative_cash)]:
        branch = pair.branches[name]
        branch["agent"].portfolio.cash = Decimal(cash)
        branch["counterparty"].cash = Decimal(11000) - Decimal(cash)
        branch["agent"].portfolio.mark_to_market(100)
    actual.finite_review = pair.summary(actual.portfolio)
    if alternative_cash == "1048":
        assert actual.finite_review["gap"] == 0.01
    assert rule_style(actual).decision == decision


@pytest.mark.parametrize("metadata_method", [None, LEGACY_METHOD])
def test_report_checks_snapshot_when_metadata_method_missing_or_wrong(request, metadata_method):
    config, directory, destination, add_run, _ = request.getfixturevalue("report_fixture")
    add_run("finite_snapshot", "visible", "low_herding", 100)
    path = directory / "finite_snapshot"
    metadata = json.loads((path / "run_metadata.json").read_text())
    if metadata_method is not None:
        metadata["strategy_return_method"] = metadata_method
    (path / "run_metadata.json").write_text(json.dumps(metadata))
    snapshot = {
        "behavior": {"strategy_return_method": METHOD},
        "run": {"strategy_return_method": METHOD},
    }
    (path / "config_snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="finite|contradictory"):
        event_report.evaluate(config, directory, destination)
