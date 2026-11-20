"""Cross-agent fills and fractional average costs must survive the read-only audit."""

import asyncio
import json

import pandas as pd
from test_finite_rule_lifecycle import SPEC, ScriptedClient
from test_style_path_integration import fictional_market
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.llm.schemas import OrderDecision
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.finite_rule_pair import METHOD
from scripts.audit_finite_lifecycle import audit


def test_two_accounts_with_actual_crossing_orders_and_two_resets(tmp_path, monkeypatch):
    config, bars = fictional_market(interval=5, days=10)
    config["population"].update(n_agents=2, initial_fundamental_share=1)
    config["behavior"].update(strategy_return_method=METHOD, finite_rule_pair=SPEC)
    bars[["open", "close"]] = 100.0
    bars["low"], bars["high"] = 95.0, 105.0
    monkeypatch.setattr(event_runner, "source_hash", lambda: "f" * 64)

    class TwoAccountClient(ScriptedClient):
        async def decide(self, system, user, schema, identity):
            data = json.loads(user)
            day = data["observation"]["session_date"]
            if schema is OrderDecision and identity["agent_id"] == "A001":
                if day in {"2026-01-01", "2026-01-06"}:
                    return OrderDecision(
                        action="sell" if day.endswith("01") else "buy",
                        quantity=1,
                        limit_price=105 if day.endswith("01") else 100,
                        sentiment="neutral",
                        rationale="scripted second counterparty",
                    ), {"status": "mock"}
            return await super().decide(system, user, schema, identity)

    asyncio.run(
        event_runner.run_experiment(
            config,
            "endogenous",
            "vanilla_llm",
            "mock",
            tmp_path,
            bars=bars,
            events=[],
            client=TwoAccountClient(),
        )
    )
    result = audit(tmp_path)
    assert result["checked_agent_days"] == 20
    assert result["checked_branch_days"] == 40
    assert result["checked_reviews"] == 4
    assert len(pd.read_parquet(tmp_path / "trades.parquet")) == 2
    accounts = pd.read_parquet(tmp_path / "agents.parquet")
    first_buyer = accounts[(accounts.agent_id == "A000") & (accounts.session_date == "2026-01-01")]
    assert first_buyer.avg_cost.iloc[0] == 101.666667
    assert accounts.groupby("session_date").cash.sum().eq(2000).all()
    assert accounts.groupby("session_date").position.sum().eq(4).all()
