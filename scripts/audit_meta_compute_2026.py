#!/usr/bin/env python3
"""Audit completed real or mock runs against external data and accounting invariants."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.data.archive import digest, write_json  # noqa: E402
from behavioral_market.data.calendar import sessions  # noqa: E402
from behavioral_market.simulation.event_runner import load_dataset  # noqa: E402


def audit(provider="siliconflow", config_path=None, directory=None):
    config_path = config_path or ROOT / "config/meta_compute_2026.json"
    config = json.loads(Path(config_path).read_text())
    bars, _, _, manifest_hash = load_dataset(config)
    expected_sessions = list(
        sessions(config["data"]["experiment_start"], config["data"]["experiment_end"]).index
    )
    root = Path(directory) if directory else ROOT / "outputs/meta_compute_2026" / provider
    completed = []
    for metadata_path in sorted(root.rglob("run_metadata.json")):
        if "_superseded" in metadata_path.parts:
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata["status"] != "complete" or metadata["n_agents"] != 30:
            continue
        path = metadata_path.parent
        symbol = metadata.get("symbol", config["data"]["primary_symbol"])
        primary = bars[bars.symbol == symbol].set_index("session_date")
        real_closes = primary.loc[expected_sessions, "close"].astype(float).to_numpy()
        initial_price = float(
            primary.loc[primary.index < expected_sessions[0], "close"].iloc[-1]
        )
        assert metadata["data_manifest_hash"] == manifest_hash, path
        for name, expected in metadata["artifact_hashes"].items():
            assert digest((path / name).read_bytes()) == expected, (path, name)
        market = pd.read_parquet(path / "market.parquet")
        orders = pd.read_parquet(path / "orders.parquet")
        accounts = pd.read_parquet(path / "agents.parquet")
        trades = pd.read_parquet(path / "trades.parquet")
        switches = pd.read_parquet(path / "style_switches.parquet")
        seen = pd.read_parquet(path / "events_seen.parquet")
        observations = [
            json.loads(line) for line in (path / "observations.jsonl").read_text().splitlines()
        ]
        assert market.session_date.tolist() == expected_sessions, path
        assert len(observations) == 23 and len(orders) == 690 and len(accounts) == 690, path
        assert len(switches) == 60, path
        assert (accounts.cash >= 0).all() and (accounts.position >= 0).all(), path
        assert (orders.quantity >= 0).all(), path
        assert (orders.filled_quantity <= orders.quantity).all(), path
        assert len(trades) == market.trade_count.sum(), path
        condition = metadata.get(
            "event_condition", "e1_e2" if metadata["include_secondary"] else "e1_only"
        )
        expected_seen = []
        if condition != "no_event":
            expected_seen.append(("2026-07-02", "META_COMPUTE_20260701"))
        if condition == "e1_e2":
            expected_seen.append(("2026-07-10", "META_COMPUTE_20260709"))
        actual_seen = (
            list(seen[["session_date", "event_id"]].itertuples(index=False, name=None))
            if len(seen)
            else []
        )
        assert actual_seen == expected_seen, path
        for observation in observations:
            day = observation["session_date"]
            assert observation["history_last_session"] < day, (path, day)
            assert observation["mean_field"]["as_of_session"] < day, (path, day)
            assert all(
                filing < day for filing in observation.get("fundamentals_filing_dates", [])
            ), (path, day)
        shock = metadata["shock"]
        anchor = np.where(
            market.session_date < "2026-07-02", initial_price, initial_price * (1 + shock)
        )
        assert np.allclose(market.fundamental_value, anchor, rtol=0, atol=1e-5), path
        if metadata["mode"] == "historical":
            assert np.allclose(market.close, real_closes, rtol=0, atol=1e-6), path
            assert orders.filled_quantity.sum() == market.simulated_volume.sum(), path
        else:
            assert all(not observation["context_returns"] for observation in observations), path
            assert all(
                value == 3000 for value in accounts.groupby("session_date").position.sum()
            ), path
            assert np.allclose(
                accounts.groupby("session_date").cash.sum(), 300000, rtol=0, atol=0.02
            ), path
            assert orders.filled_quantity.sum() == 2 * market.simulated_volume.sum(), path
        completed.append(
            {
                "mode": metadata["mode"],
                "symbol": symbol,
                "event_condition": condition,
                "treatment": metadata["treatment"],
                "seed": metadata["seed"],
                "shock": shock,
                "trades": int(len(trades)),
                "source_hash": metadata["source_hash"],
            }
        )
    result = {
        "provider": provider,
        "complete_runs": len(completed),
        "historical_runs": sum(row["mode"] == "historical" for row in completed),
        "endogenous_runs": sum(row["mode"] == "endogenous" for row in completed),
        "verified": completed,
    }
    write_json(root / "evaluation/audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "verified"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["siliconflow", "aigc_relay", "mock"], default="siliconflow"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    audit(args.provider, args.config, args.directory)
