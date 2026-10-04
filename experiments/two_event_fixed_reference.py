"""Fixed real daily bars with the explicitly simplified OHLC-touch execution proxy.

This is an environment/unit check, not historical fill reconstruction. The rule
policy ignores news and synthetic private allocation targets are not calibrated.
"""

import random
from decimal import Decimal
from types import SimpleNamespace

from behavioral_market.environments.historical_replay import HistoricalReplayEnvironment
from experiments.two_event_market import (
    context,
    exchange_decision,
    round_schedule,
    rule_decision,
    setup,
    validate_order,
)


def run_reference(config, spec, bars, seed, condition):
    agents, targets, _ = setup(config, spec, bars, seed, "none")
    data = bars[bars.symbol.eq(spec["symbol"])].copy()
    tick, lot = Decimal(str(spec["asset_price_tick"])), Decimal(str(spec["lot_size"]))
    for column in ("open", "high", "low", "close"):
        data[column] = data[column].map(
            lambda value: float((Decimal(str(value)) / tick).to_integral_value() * tick * lot)
        )
    data["volume"] = data.volume.astype(float) / float(lot)
    replay = HistoricalReplayEnvironment(data, spec["start"])
    initial = float(replay.history.iloc[-1].close)
    # One order opportunity at daily open; intraday news is first visible next open.
    openings = {}
    for day, cutoff in round_schedule(spec):
        openings.setdefault(day, cutoff)
    decisions, days = [], []
    previous = {key: 0 for key in ("buy", "sell", "hold", "rejected")}
    for index, (day, cutoff) in enumerate(openings.items()):
        price = float(replay.history.iloc[-1].close)
        for agent in agents:
            agent.portfolio.mark_to_market(price)
        view = SimpleNamespace(
            history=replay.history,
            exchange=SimpleNamespace(market_price=Decimal(str(price))),
            _opening=price,
            book_context=lambda price=price: {
                "best_bid": None,
                "best_ask": None,
                "spread": None,
                "spread_bps": None,
                "last_trade": price,
                "bid_depth": None,
                "ask_depth": None,
                "scope": "no_historical_book; last_trade_proxy_is_previous_daily_close",
            },
        )
        payloads = [
            context(config, spec, a, t, view, cutoff, condition, "private_targets", previous)
            for a, t in zip(agents, targets, strict=True)
        ]
        proposed = [rule_decision(p) for p in payloads]
        if any(validate_order(d, p) for d, p in zip(proposed, payloads, strict=True)):
            raise ValueError("reference rule generated infeasible proposal")
        internal = [exchange_decision(d, p) for d, p in zip(proposed, payloads, strict=True)]
        arrival = list(range(len(agents)))
        random.Random(seed * 100003 + index).shuffle(arrival)
        close, volume, orders, fills = replay.settle(day, agents, internal, arrival)
        by_id = {r["agent_id"]: r for r in orders}
        previous = {key: 0 for key in ("buy", "sell", "hold", "rejected")}
        for agent, payload, decision in zip(agents, payloads, proposed, strict=True):
            previous[decision.action] += 1
            row = {
                "decision_at": cutoff,
                "agent_id": agent.persona.agent_id,
                "payload": payload,
                "decision": decision.model_dump(),
                "execution_at_daily_settlement": by_id[agent.persona.agent_id],
                "end_equity": float(agent.portfolio.total_equity),
            }
            decisions.append(row)
            agent.memory.append(
                {
                    "decision_at": cutoff,
                    "decision": decision.model_dump(),
                    "equity": row["end_equity"],
                    "filled_lots": by_id[agent.persona.agent_id]["filled_quantity"],
                }
            )
        days.append(
            {
                "session_date": day,
                "close_per_lot": close,
                "external_volume_lots": volume,
                "proxy_fills": len(fills),
            }
        )
    return {
        "scenario": spec["scenario_id"],
        "symbol": spec["symbol"],
        "calendar": spec["calendar"],
        "condition": condition,
        "seed": seed,
        "scope": "Rule-only fixed history and OHLC-touch-at-limit proxy; not actual fills",
        "reference_price_return": days[-1]["close_per_lot"] / initial - 1,
        "agent_mean_return": sum(
            float(a.portfolio.total_equity) / a.initial_equity - 1 for a in agents
        )
        / len(agents),
        "decisions": decisions,
        "daily": days,
    }
