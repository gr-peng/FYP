#!/usr/bin/env python3
"""Summarize order-book visibility, order validity, liquidity, and herding checks."""

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def diagnose_run(metadata_path):
    path = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        return None
    orders = pd.read_parquet(path / "orders.parquet")
    trades_path = path / "trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    active = orders.action.isin(["buy", "sell"])
    daily_sides = orders.groupby("session_date").action.agg(
        lambda values: {"buy", "sell"}.issubset(set(values))
    )
    eligible = (
        orders.prior_majority_action.isin(["buy", "sell"])
        if "prior_majority_action" in orders
        else pd.Series(False, index=orders.index)
    )
    followed = (
        orders.action.eq(orders.prior_majority_action)
        if "prior_majority_action" in orders
        else pd.Series(False, index=orders.index)
    )
    seen_bid = orders.get("seen_best_bid", pd.Series(index=orders.index, dtype=float))
    seen_ask = orders.get("seen_best_ask", pd.Series(index=orders.index, dtype=float))
    status = orders.validation_status.astype(str)
    row = {
        "symbol": metadata.get("symbol", "unknown"),
        "event_condition": metadata.get(
            "event_condition", "e1_e2" if metadata.get("include_secondary") else "e1_only"
        ),
        "mode": metadata.get("mode"),
        "treatment": metadata.get("treatment"),
        "seed": metadata.get("seed"),
        "orders": len(orders),
        "active_orders": int(active.sum()),
        "hold_rate": float((orders.action == "hold").mean()),
        "invalid_or_rejected_rate": float((~status.isin(["ok", "mock", "rule"])).mean()),
        "constraint_violation_rate": float((status == "constraint_violation").mean()),
        "price_band_violation_rate": float((status == "price_band_violation").mean()),
        "fallback_rate": float(orders.fallback.astype(bool).mean()),
        "book_context_visible_rate": float((seen_bid.notna() | seen_ask.notna()).mean()),
        "mean_seen_spread_bps": safe_mean(orders.get("seen_spread_bps", [])),
        "mean_bid_distance_to_ask_bps": safe_mean(
            orders.get("bid_distance_to_ask_bps", [])[orders.action == "buy"]
            if "bid_distance_to_ask_bps" in orders
            else []
        ),
        "mean_ask_distance_to_bid_bps": safe_mean(
            orders.get("ask_distance_to_bid_bps", [])[orders.action == "sell"]
            if "ask_distance_to_bid_bps" in orders
            else []
        ),
        "both_sides_sessions": int(daily_sides.sum()),
        "sessions": int(orders.session_date.nunique()),
        "trades": len(trades),
        "majority_signal_opportunities": int(eligible.sum()),
        "follow_majority_rate": float(followed[eligible].mean()) if eligible.any() else None,
        "active_follow_majority_rate": float(followed[eligible & active].mean())
        if (eligible & active).any()
        else None,
        "run_directory": str(path),
    }
    return row


def diagnose(root):
    rows = [diagnose_run(path) for path in sorted(root.rglob("run_metadata.json"))]
    frame = pd.DataFrame(row for row in rows if row is not None)
    if frame.empty:
        raise RuntimeError(f"no completed runs under {root}")
    frame.to_csv(root / "liquidity_diagnostics.csv", index=False)
    columns = [
        "symbol",
        "event_condition",
        "treatment",
        "seed",
        "hold_rate",
        "invalid_or_rejected_rate",
        "fallback_rate",
        "book_context_visible_rate",
        "mean_seen_spread_bps",
        "mean_bid_distance_to_ask_bps",
        "mean_ask_distance_to_bid_bps",
        "both_sides_sessions",
        "sessions",
        "trades",
        "follow_majority_rate",
        "active_follow_majority_rate",
    ]
    print(frame[columns].to_string(index=False))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    diagnose(parser.parse_args().directory)
