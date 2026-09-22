#!/usr/bin/env python3
"""Training-only schema, timing and conservation audit after partition freeze."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finance_data import load_development_rows, sha256  # noqa: E402
from experiments.scopic_states import (  # noqa: E402
    ASSETS,
    pre_ranking_decisions,
    sell_opportunities,
)


def audit():
    plan_path = ROOT / "config/frozen/financial_data_partitions_20260922.json"
    plan = json.loads(plan_path.read_text())
    cda = pd.DataFrame(load_development_rows(ROOT, plan, "cda"))
    scopic_rows = load_development_rows(ROOT, plan, "scopic")
    scopic = pd.DataFrame(scopic_rows)
    numeric = [c for c in scopic if c.startswith(("player.quantity", "player.price"))]
    numeric += ["subsession.round_number", "player.result_final"]
    scopic[numeric] = scopic[numeric].apply(pd.to_numeric)
    scopic = scopic.sort_values(["participant.code", "subsession.round_number"])
    def qcols(prefix):
        return [f"player.{prefix}_{a}" for a in ASSETS]
    conservation = (
        scopic[qcols("quantity_pre")].to_numpy()
        + scopic[qcols("quantity_buy")].to_numpy()
        - scopic[qcols("quantity_sell")].to_numpy()
        - scopic[qcols("quantity")].to_numpy()
    )
    wealth_errors, current_wealth_errors, terminal_errors = [], [], []
    min_cash = 10000
    reset_people = 0
    for _, frame in scopic.groupby("participant.code"):
        cash = 10000
        rows = frame.to_dict("records")
        for i, row in enumerate(rows):
            t = int(row["subsession.round_number"])
            if t == 4:
                cash = 10000
                reset_people += int(all(row[f"player.quantity_pre_{a}"] == 0 for a in ASSETS))
            cash += sum(
                (row[f"player.quantity_sell_{a}"] - row[f"player.quantity_buy_{a}"])
                * row[f"player.price_{a}"]
                for a in ASSETS
            )
            min_cash = min(min_cash, cash)
            wealth_now = cash + sum(
                row[f"player.quantity_{a}"] * row[f"player.price_{a}"] for a in ASSETS
            )
            if 4 <= t < 17:
                wealth_next = cash + sum(
                    row[f"player.quantity_{a}"] * rows[i + 1][f"player.price_{a}"] for a in ASSETS
                )
                wealth_errors.append(wealth_next - row["player.result_final"])
                current_wealth_errors.append(wealth_now - row["player.result_final"])
            elif t == 17:
                terminal_errors.append(wealth_now - row["player.result_final"])
    decisions = pre_ranking_decisions(scopic_rows)
    opportunities = sell_opportunities(decisions)
    trades = cda.loc[cda["Transaction (YES / NO)"] == "YES"]
    trade_groups = list(trades.groupby(["SessionID", "Round"]))
    report = {
        "scope": "train only; development and test behavior unopened",
        "partition_sha256": sha256(plan_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scopic": {
            "rows": len(scopic),
            "participants": scopic["participant.code"].nunique(),
            "sessions": scopic["session.code"].nunique(),
            "share_conservation_bad_rows": int(np.any(conservation != 0, axis=1).sum()),
            "raw_round_4_zero_holdings_participants": reset_people,
            "minimum_reconstructed_cash": int(min_cash),
            "next_price_wealth_checks": len(wealth_errors),
            "next_price_wealth_exact": sum(x == 0 for x in wealth_errors),
            "current_price_wealth_exact": sum(x == 0 for x in current_wealth_errors),
            "terminal_wealth_checks": len(terminal_errors),
            "terminal_current_price_wealth_exact": sum(x == 0 for x in terminal_errors),
            "terminal_future_price": "unavailable; do not infer using participant outcomes",
            "pre_ranking_main_decisions": len(decisions),
            "pre_ranking_owned_asset_opportunities": len(opportunities),
            "pre_ranking_sell_any_count": sum(x["sell_any"] for x in opportunities),
            "sample_cash_and_next_wealth_are_diagnostics_not_behavioral_inputs": True,
        },
        "cda": {
            "quote_rows": len(cda),
            "sessions": cda.SessionID.nunique(),
            "completed_trades": len(trades),
            "quote_ids_unique_within_session_round": not cda.duplicated(
                ["SessionID", "Round", "contractID"]
            )
            .any()
            .item(),
            "trade_ids_unique_within_session_round": not trades.duplicated(
                ["SessionID", "Round", "Trade ID"]
            )
            .any()
            .item(),
            "rounds_with_trades": len(trade_groups),
            "rounds_with_nonmonotone_trade_order_in_quote_rows": sum(
                not frame["Trade ID"].is_monotonic_increasing for _, frame in trade_groups
            ),
            "filled_market_price_equals_eventual_execution_price": bool(
                (trades.Price == trades["Market Price"]).all()
            ),
            "reconstruction": "Quote rows carry eventual acceptance annotations; quote and trade "
            "orders are distinct. Joint interleaving/timing and realized dividends are unresolved. "
            "Market Price, Accepter, Transaction, Trade ID are not established pre-quote inputs.",
        },
    }
    destination = ROOT / "outputs/financial_training_audit_20260922"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "scopic_pre_ranking_decisions.json").write_text(
        json.dumps(decisions, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
