"""The auditor must recompute balances, rather than merely trust saved totals."""

import json
import shutil
from decimal import Decimal

import pandas as pd
import pytest
from test_finite_rule_lifecycle import run_two_blocks
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.simulation import event_runner
from scripts import audit_finite_lifecycle as checker


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    directory = tmp_path / "source"
    # The shared offline fixture uses a human-readable dummy source hash. This
    # auditor requires the real hash format; still no source scan or external I/O.
    monkeypatch.setattr(event_runner, "source_hash", lambda: "f" * 64)
    run_two_blocks(directory, monkeypatch)
    return directory


def rehash(directory, name):
    path = directory / "run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["artifact_hashes"][name] = checker.sha(directory / name)
    path.write_text(json.dumps(metadata))


def test_two_blocks_recomputed_from_daily_evidence(completed_run):
    result = checker.audit(completed_run)
    assert result["status"] == "accounting_checks_passed"
    assert (
        result["checked_agent_days"],
        result["checked_branch_days"],
        result["checked_reviews"],
    ) == (10, 20, 2)
    first, second = result["reviews"]
    assert Decimal(first["observed_account_return"]) == pytest.approx(-Decimal(40) / 1200)
    assert Decimal(first["paired_return_gap"]) == 0
    assert Decimal(second["block_start_cash"]) == 160 and second["block_start_shares"] == 10
    assert Decimal(second["block_start_equity"]) == 1160
    assert result["new_model_POSTs"] == 0
    assert "真人" in checker.describe(result)


@pytest.mark.parametrize(
    "change",
    [
        "missing_row",
        "duplicate_row",
        "fake_cash",
        "bad_reset",
        "future_history",
        "changed_style",
        "different_opening",
    ],
)
def test_hash_consistent_but_false_accounting_is_rejected(completed_run, change):
    name = "finite_rule_replay.jsonl"
    path = completed_run / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing_row":
        rows.pop()
    elif change == "duplicate_row":
        rows.append(rows[0])
    elif change == "fake_cash":
        rows[0]["portfolio_after_close"]["cash"] = "999"
    elif change == "bad_reset":
        rows[10]["portfolio_before_order"]["cash"] = "1000"
    elif change == "future_history":
        rows[0]["history_last_session"] = rows[0]["session_date"]
    elif change == "changed_style":
        rows[2]["style"] = "technical"
    else:
        rows[1]["opening_observation"]["momentum_5"] = 0.5
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rehash(completed_run, name)
    with pytest.raises(ValueError):
        checker.audit(completed_run)


@pytest.mark.parametrize("change", ["missing_review", "false_gap", "next_style"])
def test_saved_review_cannot_override_recomputed_evidence(completed_run, change):
    name = "style_switches.parquet"
    frame = pd.read_parquet(completed_run / name)
    if change == "missing_review":
        frame = frame.iloc[:1]
    elif change == "false_gap":
        frame.loc[0, "paired_return_gap"] = 0.1
    else:
        frame.loc[0, "target_style"] = "fundamental"
    frame.to_parquet(completed_run / name, index=False)
    rehash(completed_run, name)
    with pytest.raises(ValueError):
        checker.audit(completed_run)


def test_artifact_and_configuration_hashes_are_required(completed_run, tmp_path):
    copied = tmp_path / "copied"
    shutil.copytree(completed_run, copied)
    (copied / "finite_rule_replay.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        checker.audit(copied)
    config = json.loads((completed_run / "config_snapshot.json").read_text())
    config["experiment"]["seed"] += 1
    (completed_run / "config_snapshot.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config hash"):
        checker.audit(completed_run)


def test_reduced_metadata_count_does_not_hide_missing_calendar_day(completed_run):
    name = "market.parquet"
    pd.read_parquet(completed_run / name).iloc[:-1].to_parquet(completed_run / name, index=False)
    rehash(completed_run, name)
    path = completed_run / "run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["sessions"] -= 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="configured schedule"):
        checker.audit(completed_run)


def test_fill_recomputation_checks_both_accounts_and_limits():
    # Independent literal ledger: rule buys two at100.05, then closes at110.
    before = {"cash": "1000", "position": 2, "avg_cost": "100", "realized_pnl": "0"}
    row = {
        "portfolio_before_order": before,
        "portfolio_after_close": {
            "cash": "799.90",
            "position": 4,
            "avg_cost": "100.025",
            "realized_pnl": "0",
            "total_equity": "1239.90",
            "unrealized_pnl": "39.90",
        },
        "counterparty_after_close": {
            "cash": "10200.10",
            "position": 98,
            "total_equity": "20980.10",
        },
        "assigned_quote": "100.05",
        "assigned_quantity": 2,
        "quote_status": "assigned_quote_submitted",
        "expired_orders": 0,
        "opening_observation": {"last_close": 100},
        "permitted_price_band": {"minimum": 95, "maximum": 105},
        "requested_order": {"action": "buy", "quantity": 2, "limit_price": 100.3},
        "validated_order": {"action": "buy", "quantity": 2, "limit_price": 100.3},
        "constraint_violation": None,
        "market_close_used_after_order": "110",
        "orders_before_expiry": [
            {
                "order_id": "B",
                "agent_id": "RULE",
                "side": "buy",
                "price": "100.30",
                "quantity": 2,
                "remaining_quantity": 0,
                "status": "filled",
            },
            {
                "order_id": "S",
                "agent_id": "ASSIGNED_COUNTERPARTY",
                "side": "sell",
                "price": "100.05",
                "quantity": 2,
                "remaining_quantity": 0,
                "status": "filled",
            },
        ],
        "trades": [
            {
                "trade_id": "T",
                "buy_order_id": "B",
                "sell_order_id": "S",
                "buyer_id": "RULE",
                "seller_id": "ASSIGNED_COUNTERPARTY",
                "price": "100.05",
                "quantity": 2,
            }
        ],
    }
    spec = {"counterparty_cash": "10000", "counterparty_shares": 100, "half_spread": "0.05"}
    checker.verify_day(row, None, spec)
    with pytest.raises(ValueError, match="mismatch"):
        checker.verify_day(row, None, {**spec, "half_spread": "0.10"})
    row["orders_before_expiry"][0]["price"] = "100"
    with pytest.raises(ValueError, match="mismatch"):
        checker.verify_day(row, None, spec)


def test_missing_actual_trade_cannot_be_hidden_by_rehashing(completed_run):
    name = "trades.parquet"
    pd.read_parquet(completed_run / name).iloc[1:].to_parquet(completed_run / name, index=False)
    rehash(completed_run, name)
    with pytest.raises(ValueError, match="fill count|accounting mismatch"):
        checker.audit(completed_run)


@pytest.mark.parametrize(
    "change",
    [
        "effective_review",
        "phantom_expiry",
        "phantom_order",
        "invalid_choice_changed_style",
        "fallback_changed_style",
    ],
)
def test_contract_contradictions_do_not_pass_even_with_recomputed_hashes(completed_run, change):
    if change in {"effective_review", "invalid_choice_changed_style", "fallback_changed_style"}:
        name = "style_switches.parquet"
        frame = pd.read_parquet(completed_run / name)
        if change == "effective_review":
            frame.loc[0, "decision"] = "stay"
        elif change == "invalid_choice_changed_style":
            frame.loc[0, "choice_valid"] = False
        else:
            frame.loc[0, "fallback"] = True
        frame.to_parquet(completed_run / name, index=False)
    else:
        name = "finite_rule_replay.jsonl"
        rows = [json.loads(line) for line in (completed_run / name).read_text().splitlines()]
        if change == "phantom_expiry":
            rows[0]["expired_orders"] = 99
        else:
            rows[0]["validated_order"].update(action="buy", quantity=5, limit_price=100.3)
        (completed_run / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rehash(completed_run, name)
    with pytest.raises(ValueError):
        checker.audit(completed_run)


@pytest.mark.parametrize(
    "change", ["actual_position", "shadow_position", "counterparty_position", "boolean_expiry"]
)
def test_share_counts_do_not_inherit_money_rounding_tolerance(completed_run, change):
    if change == "actual_position":
        name = "agents.parquet"
        frame = pd.read_parquet(completed_run / name)
        frame["position"] = frame.position.astype(float)
        frame.loc[0, "position"] += 1e-7
        frame.to_parquet(completed_run / name, index=False)
    else:
        name = "finite_rule_replay.jsonl"
        rows = [json.loads(line) for line in (completed_run / name).read_text().splitlines()]
        if change == "boolean_expiry":
            rows[0]["expired_orders"] = False
        else:
            key = (
                "portfolio_after_close"
                if change == "shadow_position"
                else "counterparty_after_close"
            )
            rows[0][key]["position"] += 1e-7
        (completed_run / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rehash(completed_run, name)
    with pytest.raises(ValueError, match="integer count"):
        checker.audit(completed_run)
