"""Fixed rule controls must still review and reset on the same clock."""

import asyncio
import json

import pandas as pd
import pytest
from test_finite_lifecycle_audit import rehash
from test_finite_rule_lifecycle import SPEC
from test_style_path_integration import fictional_market
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.finite_rule_pair import METHOD
from scripts.audit_finite_lifecycle import audit


class NoModelClient:
    async def decide(self, *args, **kwargs):
        raise AssertionError("rule comparison must never request a model decision")


def test_fixed_and_adaptive_review_same_blocks_without_model_calls(tmp_path, monkeypatch):
    config, bars = fictional_market(interval=5, days=10)
    config["population"].update(n_agents=1, initial_fundamental_share=1)
    config["behavior"].update(strategy_return_method=METHOD, finite_rule_pair=SPEC)
    monkeypatch.setattr(event_runner, "source_hash", lambda: "a" * 64)
    seen = []

    def switch(agent):
        seen.append((agent.finite_review["block_index"], agent.finite_review["completed_sessions"]))
        target = "technical" if agent.current_style == "fundamental" else "fundamental"
        return StyleDecision(decision="switch", target_style=target, rationale="injected test rule")

    monkeypatch.setattr(event_runner, "rule_style", switch)
    for mode in ["fixed", "adaptive"]:
        config["behavior"]["rule_style_adaptation"] = mode
        output = tmp_path / mode
        asyncio.run(
            event_runner.run_experiment(
                config,
                "historical",
                "rule_abm",
                "mock",
                output,
                bars=bars,
                events=[],
                client=NoModelClient(),
            )
        )
        reviews = pd.read_parquet(output / "style_switches.parquet")
        assert len(reviews) == 2 and set(reviews.rule_style_adaptation) == {mode}
        assert set(reviews.decision_source) == {"fixed_rule" if mode == "fixed" else "rule"}
        assert set(reviews.decision) == {"stay" if mode == "fixed" else "switch"}
        assert reviews.request_hash.isna().all()
        metadata = json.loads((output / "run_metadata.json").read_text())
        snapshot = json.loads((output / "config_snapshot.json").read_text())
        assert metadata["rule_style_adaptation"] == snapshot["run"]["rule_style_adaptation"] == mode
        result = audit(output)
        assert result["checked_reviews"] == 2
        if mode == "fixed":
            assert seen == []
    assert seen == [(0, 5), (1, 5)]
    for name in ["market", "orders", "trades", "agents"]:
        frames = [
            pd.read_parquet(tmp_path / arm / f"{name}.parquet") for arm in ["fixed", "adaptive"]
        ]
        frames = [f[f.session_date <= "2026-01-05"].reset_index(drop=True) for f in frames]
        pd.testing.assert_frame_equal(*frames)
    config["behavior"]["rule_style_adaptation"] = "fixed"
    fixed = event_runner.output_path(config, "historical", "rule_abm", "mock")
    config["behavior"]["rule_style_adaptation"] = "adaptive"
    assert event_runner.output_path(config, "historical", "rule_abm", "mock") != fixed


@pytest.mark.parametrize("value", [None, True, 0, "unknown", []])
def test_bad_adaptation_rejected_before_io(tmp_path, value):
    config, _ = fictional_market()
    config["behavior"].update(
        strategy_return_method=METHOD, finite_rule_pair=SPEC, rule_style_adaptation=value
    )
    with pytest.raises(ValueError, match="rule_style_adaptation"):
        asyncio.run(event_runner.run_experiment(config, "historical", "rule_abm", "mock", tmp_path))


def test_rule_only_setting_cannot_silently_change_llm_treatment(tmp_path):
    config, _ = fictional_market()
    config["behavior"].update(
        strategy_return_method=METHOD, finite_rule_pair=SPEC, rule_style_adaptation="fixed"
    )
    with pytest.raises(ValueError, match="requires finite rule_abm"):
        asyncio.run(
            event_runner.run_experiment(config, "historical", "vanilla_llm", "mock", tmp_path)
        )


@pytest.mark.parametrize("change", ["declaration", "review_arm", "source", "fixed_switch"])
def test_audit_rejects_contradictory_rule_controls(tmp_path, monkeypatch, change):
    config, bars = fictional_market(interval=5, days=5)
    config["population"].update(n_agents=1, initial_fundamental_share=1)
    config["behavior"].update(
        strategy_return_method=METHOD, finite_rule_pair=SPEC, rule_style_adaptation="fixed"
    )
    monkeypatch.setattr(event_runner, "source_hash", lambda: "a" * 64)
    asyncio.run(
        event_runner.run_experiment(
            config,
            "historical",
            "rule_abm",
            "mock",
            tmp_path,
            bars=bars,
            events=[],
            client=NoModelClient(),
        )
    )
    assert audit(tmp_path)["checked_reviews"] == 1
    if change == "declaration":
        path = tmp_path / "run_metadata.json"
        data = json.loads(path.read_text())
        data["rule_style_adaptation"] = "adaptive"
        path.write_text(json.dumps(data))
        expected = "declarations mismatch"
    else:
        path = tmp_path / "style_switches.parquet"
        data = pd.read_parquet(path)
        if change == "review_arm":
            data.loc[0, "rule_style_adaptation"] = "adaptive"
        elif change == "source":
            data.loc[0, "decision_source"] = "OpenAICompatibleClient"
        else:
            data.loc[0, "decision"] = "switch"
        data.to_parquet(path, index=False)
        rehash(tmp_path, path.name)
        expected = "fixed rule policy switched" if change == "fixed_switch" else "source mismatch"
    with pytest.raises(ValueError, match=expected):
        audit(tmp_path)
