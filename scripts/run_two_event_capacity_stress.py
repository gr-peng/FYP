# ruff: noqa: E402
"""Synthetic all-buy/all-sell capacity checks, never counted as LLM or human results."""

import asyncio
import copy
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.schemas import OrderDecision, hold
from experiments.two_event_data import verify_manifest
from experiments.two_event_market import run_case


class ConstantDirection:
    def __init__(self, direction):
        self.direction = direction

    async def decide(self, system, user, schema, identity):
        payload = json.loads(user)
        constraint = payload["constraints"]
        portfolio = payload["agent_state"]["portfolio"]
        tick = constraint["price_tick"]
        # Cross every reachable level, subject to the same band and account caps.
        if self.direction == "buy":
            price = math.floor(constraint["max_price"] / tick) * tick
            capacity = math.floor(
                portfolio["cash"] / (price * payload["order_notional_multiplier"])
            )
        else:
            price = math.ceil(constraint["min_price"] / tick) * tick
            capacity = portfolio["lots"]
        quantity = min(capacity, constraint["max_quantity"])
        if quantity <= 0:
            return hold("Synthetic stress participant has exhausted its resources"), {
                "status": "stress"
            }
        return OrderDecision(
            action=self.direction,
            quantity=quantity,
            limit_price=round(price, 2),
            rationale="Synthetic constant-direction capacity stress, not an investor model",
            sentiment="neutral",
        ), {"status": "stress"}


async def main():
    config = json.loads((ROOT / "config/two_event_month_extension_v6.json").read_text())
    sensitivity = json.loads((ROOT / "config/two_event_liquidity_sensitivity_v6.json").read_text())
    config = copy.deepcopy(config)
    config["experiment_id"] = "two_event_capacity_stress_v6"
    config["liquidity_profiles"] = sensitivity["liquidity_profiles"]
    config["interpretation"] = (
        "Mechanics-only all-buy/all-sell stress. No human/model inference; "
        "both directions tested for both assets. Bounds are scenario-specific, not a mathematical "
        "bound over all possible strategies. No post-start prices enter the market."
    )
    study = json.loads((ROOT / config["data_config"]).read_text())
    data = ROOT / "data/processed" / study["study_id"]
    manifest = json.loads((data / "manifest.json").read_text())
    verify_manifest(manifest, ROOT)
    out = ROOT / "outputs" / study["study_id"] / config["experiment_id"]
    rows = []
    for spec in config["scenarios"]:
        bars = pd.read_parquet(data / f"{spec['scenario_id']}_daily.parquet")
        for liquidity in ("thin", "finite", "deep"):
            for direction in ("buy", "sell"):
                result = await run_case(
                    config,
                    spec,
                    bars,
                    (42, "vanilla", "no_news", liquidity),
                    out / spec["symbol"] / liquidity / direction,
                    digest((data / "manifest.json").read_bytes()),
                    ConstantDirection(direction),
                    f"synthetic_stress_{direction}",
                )
                assert result["rejected_orders"] == 0
                rows.append(result)
    write_json(out / "summary.json", rows)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "scenario",
                        "liquidity",
                        "provider",
                        "return",
                        "trades",
                        "zero_volume_sessions",
                        "background_final_lots",
                    )
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
