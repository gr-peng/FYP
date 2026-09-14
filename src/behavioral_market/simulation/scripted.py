"""Reproducible scripted-agent smoke run for the MVP-0 exchange."""

import argparse
import csv
import json
import random
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from behavioral_market.market import Exchange, Portfolio, Side


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict, output: Path) -> dict:
    """Run one-asset scripted trading; orders, not an exogenous path, set the price."""
    seed = int(config["experiment"]["seed"])
    rng = random.Random(seed)
    market = config["market"]
    population = config["population"]
    n_agents = int(population["n_agents"])
    periods = int(market["periods"])
    if n_agents < 2 or periods < 1:
        raise ValueError("at least two agents and one period are required")
    exchange = Exchange(market["asset"], market["initial_price"])
    for index in range(n_agents):
        exchange.register(
            f"A{index:03d}",
            Portfolio(
                cash=Decimal(str(population["initial_cash"])),
                position=int(population["initial_shares"]),
                avg_cost=exchange.market_price,
            ),
        )
    initial_cash = sum(portfolio.cash for portfolio in exchange.portfolios.values())
    initial_shares = sum(portfolio.position for portfolio in exchange.portfolios.values())
    market_rows: list[dict] = []
    for day in range(periods):
        opening_price = exchange.market_price
        # Limit prices are scripted relative to the last endogenous transaction.
        proposals = []
        for agent_id in exchange.portfolios:
            side = rng.choice([Side.BUY, Side.SELL])
            offset = Decimal(rng.randint(-2, 2)) / 100
            price = (opening_price * (1 + offset)).quantize(Decimal("0.01"))
            proposals.append((agent_id, side, price, rng.randint(1, 8)))
        rng.shuffle(proposals)
        for agent_id, side, price, quantity in proposals:
            try:
                exchange.submit_order(agent_id, side, price, quantity)
            except ValueError:
                # Scripted proposals can exceed a portfolio's remaining capacity.
                continue
        market_rows.append(
            {
                "day": day,
                "open": str(opening_price),
                "close": str(exchange.market_price),
                "volume": exchange.day_volume,
                "trade_count": len([t for t in exchange.trades if t.day == day]),
            }
        )
        exchange.end_day()

    assert sum(p.cash for p in exchange.portfolios.values()) == initial_cash
    assert sum(p.position for p in exchange.portfolios.values()) == initial_shares
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "orders.csv",
        [
            {**asdict(order), "side": order.side.value, "status": order.status.value}
            for order in exchange.orders
        ],
        [
            "order_id",
            "agent_id",
            "side",
            "price",
            "quantity",
            "timestamp",
            "day",
            "remaining_quantity",
            "status",
        ],
    )
    _write_csv(
        output / "trades.csv",
        [asdict(trade) for trade in exchange.trades],
        list(asdict(exchange.trades[0]))
        if exchange.trades
        else [
            "trade_id",
            "buyer_id",
            "seller_id",
            "price",
            "quantity",
            "buy_order_id",
            "sell_order_id",
            "timestamp",
            "day",
        ],
    )
    _write_csv(
        output / "market.csv", market_rows, ["day", "open", "close", "volume", "trade_count"]
    )
    _write_csv(
        output / "agents.csv",
        [
            {"agent_id": agent_id, **asdict(portfolio)}
            for agent_id, portfolio in exchange.portfolios.items()
        ],
        [
            "agent_id",
            "cash",
            "position",
            "avg_cost",
            "realized_pnl",
            "unrealized_pnl",
            "total_equity",
        ],
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    metadata = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": None,
        "prompt_version": None,
        "asset": market["asset"],
        "periods": periods,
        "n_agents": n_agents,
        "order_ttl": "day",
        "transaction_price_rule": "resting_order",
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "trades": len(exchange.trades),
        "final_price": str(exchange.market_price),
    }
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/mvp0.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
