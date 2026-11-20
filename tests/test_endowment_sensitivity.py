"""Offline preparation and access boundaries; no prospective study runs."""

import asyncio
import copy
import json

import pytest

from scripts import run_endowment_sensitivity as tool
from scripts.diagnose_rule_market import capacity_detail


def design():
    return {
        "protocol": tool.PROTOCOL,
        "seeds": [17, 29, 43],
        "accounts": 20,
        "sessions": 30,
        "arms": tool.ARMS,
        "new_model_POSTs": 0,
        "analysis": ["Synthetic test of freezing; not an actual study authorization."],
    }


def prepared(tmp_path):
    path = tmp_path / "test-design.json"
    path.write_text(json.dumps(design()))
    packet = tmp_path / "packet"
    tool.prepare(packet, path)
    return packet


def test_only_endowment_changes_and_starting_wealth_is_preserved():
    for seed in tool.SEEDS:
        original = tool.configuration(seed, "original")
        alternative = tool.configuration(seed, "equal_value")
        assert original["population"] != alternative["population"]
        for config in (original, alternative):
            spec = config["population"]
            assert tool.number(spec["initial_cash"]) + 100 * spec["initial_shares"] == 1200
            assert config["behavior"]["rule_style_adaptation"] == "fixed"
        alternative["population"].update(tool.ARMS["original"])
        assert original == alternative


def test_equal_initial_values_do_not_claim_equal_executable_capacity():
    held = {
        "requested_action": "hold",
        "requested_quantity": 0,
        "action": "hold",
        "rationale": "test",
    }
    counts = []
    for arm in tool.ARMS:
        state = tool.ARMS[arm]
        row = capacity_detail(
            held, {"cash": state["initial_cash"], "position": state["initial_shares"]}, 100, 0.5
        )
        counts.append(
            (row["conditional_buy_capacity_at_derived_limit"], row["conditional_sell_capacity"])
        )
        assert row["desired_side_before_clipping"] is None
    assert counts == [(9, 2), (5, 6)]


def test_frozen_packet_contains_all_six_runs(tmp_path):
    packet = prepared(tmp_path)
    plan = tool.checked_plan(packet)
    assert len(plan["runs"]) == 6 and len(plan["inputs"]) == 8
    assert plan["new_model_POSTs"] == 0


@pytest.mark.parametrize(
    "changed", ["warmup.csv", "configs/seed17_equal_value.json", "design.json"]
)
def test_changed_input_is_rejected(tmp_path, changed):
    packet = prepared(tmp_path)
    path = packet / changed
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen input changed"):
        tool.checked_plan(packet)


def test_changed_source_snapshot_is_rejected(tmp_path):
    packet = prepared(tmp_path)
    path = packet / "sources/scripts/diagnose_rule_market.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        tool.checked_plan(packet)


@pytest.mark.parametrize(
    "preflight",
    [
        {"accepted_for_execution": False, "mustfix": [], "plan_sha256": "use_actual"},
        {"accepted_for_execution": True, "mustfix": ["unresolved"], "plan_sha256": "use_actual"},
        {"accepted_for_execution": True, "mustfix": [], "plan_sha256": "wrong"},
    ],
)
def test_unaccepted_plan_does_not_run_or_create_output(tmp_path, monkeypatch, preflight):
    packet = prepared(tmp_path)
    review = copy.deepcopy(preflight)
    if review["plan_sha256"] == "use_actual":
        review["plan_sha256"] = tool.digest((packet / "plan.json").read_bytes())
    (packet / "independent_preflight.json").write_text(json.dumps(review))
    monkeypatch.setattr(
        tool.event_runner, "run_experiment", lambda *_a, **_k: pytest.fail("simulation started")
    )
    with pytest.raises(ValueError, match="independent preflight"):
        asyncio.run(tool.execute(packet, tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_any_accidental_model_decision_fails():
    with pytest.raises(RuntimeError, match="forbids model"):
        asyncio.run(tool.NoModelClient().decide("test", "test"))


def test_report_preserves_zero_activity_and_negative_difference():
    def row(volume, close):
        return {
            "decisions": {"buy": 0, "sell": 0, "hold": 600},
            "single_side_traded_quantity": volume,
            "zero_trade_days": 30,
            "daily_closes": [close] * 30,
            "final_zero_share_accounts": 0,
            "final_valuation": {
                "total_cash": "20000",
                "total_shares": 40,
                "wealth_change_from_price_mark": "0",
            },
        }

    report = tool.describe(
        {
            "pairs": [
                {
                    "seed": 7,
                    "arms": {"original": row(3, 101), "equal_value": row(0, 100)},
                    "equal_minus_original_volume": -3,
                    "equal_minus_original_final_close": "-1",
                }
            ]
        }
    )
    assert "成交0股" in report and "成交量差-3股" in report and "末价差-1" in report
    assert "不能单独证明" in report and "不能据此说AI更像真人" in report
