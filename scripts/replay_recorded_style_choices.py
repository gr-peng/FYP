"""Explore recorded styles through paired one-step synthetic books, with no model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
from itertools import product
from pathlib import Path

from behavioral_market.agents.policy import enforce_portfolio, rule_order
from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.evaluation.rp_metrics import market_metrics
from behavioral_market.market import Exchange, Portfolio
from scripts import collect_style_review_economic_continuation as study

LEVELS = (-0.1, 0.0, 0.1)


class ZeroNoise:
    def gauss(self, mean, sigma):
        return 0.0


def simulate(case, target_style, fundamental_signal, technical_signal):
    """A constructed one-step observation, never a chronological future price series."""
    if target_style not in ("fundamental", "technical"):
        raise ValueError("unknown style")
    if fundamental_signal not in LEVELS or technical_signal not in LEVELS:
        raise ValueError("use all predefined signal levels")
    state = deepcopy(case["construction_not_sent"]["final_state"])
    persona = AgentPersona(**state["persona"])
    values = state["portfolio"]
    portfolio = Portfolio(values["cash"], values["position"], values["avg_cost"])
    portfolio.mark_to_market(100)
    if (portfolio.cash, portfolio.position, portfolio.avg_cost) != (Decimal(1000), 2, Decimal(100)):
        raise ValueError("this diagnostic requires the accepted fixed account snapshot")
    agent = AgentState(
        persona,
        portfolio,
        case["original_style"],
        1200,
        1200,
        memory=deepcopy(state["memory"]),
        peak_equity=1200,
    )
    exchange = Exchange("SYNTH", 100)
    exchange.register(persona.agent_id, portfolio)
    exchange.register("FICTIONAL_BID_COUNTERPARTY", Portfolio(10000))
    exchange.register("FICTIONAL_ASK_COUNTERPARTY", Portfolio(0, 100, 100))
    initial_cash = sum(p.cash for p in exchange.portfolios.values())
    initial_shares = sum(p.position for p in exchange.portfolios.values())
    exchange.submit_order("FICTIONAL_BID_COUNTERPARTY", "buy", "99.95", 10)
    exchange.submit_order("FICTIONAL_ASK_COUNTERPARTY", "sell", "100.05", 10)
    before = asdict(portfolio)
    agent.current_style = target_style
    assert asdict(portfolio) == before and len(exchange.orders) == 2 and not exchange.trades
    observation = {
        "last_close": 100,
        "fundamental_value": 100 * (1 + fundamental_signal),
        "momentum_5": technical_signal,
    }
    proposal = rule_order(agent, observation, ZeroNoise())
    decision, violation = enforce_portfolio(proposal, portfolio)
    assert violation is None
    if decision.action != "hold":
        exchange.submit_order(
            persona.agent_id, decision.action, decision.limit_price, decision.quantity
        )
    assert sum(p.cash for p in exchange.portfolios.values()) == initial_cash
    assert sum(p.position for p in exchange.portfolios.values()) == initial_shares
    accepted = [
        {
            "order_id": order.order_id,
            "side": order.side.value,
            "quantity": order.quantity,
            "accepted": True,
        }
        for order in exchange.orders
    ]
    trades = [asdict(trade) for trade in exchange.trades]
    metrics = market_metrics(
        [100, float(exchange.market_price)],
        [observation["fundamental_value"]] * 2,
        accepted_orders=accepted,
        trades=trades,
        evidence_kind="synthetic_one_step_after_recorded_style",
    )
    return {
        "target_style": target_style,
        "observation": observation,
        "decision": decision.model_dump(),
        "agent_order_submitted": decision.action != "hold",
        "agent_filled_quantity": sum(trade.quantity for trade in exchange.trades),
        "agent_cash_after": format(portfolio.cash, ".2f"),
        "agent_shares_after": portfolio.position,
        "agent_equity_after": format(portfolio.total_equity, ".2f"),
        "cash_and_share_conservation": True,
        "accepted_orders": accepted,
        "trades": [{**trade, "price": str(trade["price"])} for trade in trades],
        "metrics": metrics,
    }


def paired_branch(case, choice, fundamental_signal, technical_signal):
    if (choice["agent_id"], choice["from_style"], choice["case_id"]) != (
        case["case_id"].split(".")[0],
        case["original_style"],
        case["case_id"],
    ) or choice["compatible"] is not True:
        raise ValueError("recorded choice and accepted case differ")
    target = choice["choice"]["target_style"]
    control = simulate(case, case["original_style"], fundamental_signal, technical_signal)
    applied = simulate(case, target, fundamental_signal, technical_signal)
    return {
        "slot_id": choice["slot_id"],
        "agent_id": choice["agent_id"],
        "condition": choice["condition"],
        "case_id": choice["case_id"],
        "fundamental_signal": fundamental_signal,
        "technical_signal": technical_signal,
        "original_style": case["original_style"],
        "recorded_target_style": target,
        "style_changed": target != case["original_style"],
        "action_changed": control["decision"]["action"] != applied["decision"]["action"],
        "holdings_changed": control["agent_shares_after"] != applied["agent_shares_after"],
        "cash_changed": control["agent_cash_after"] != applied["agent_cash_after"],
        "fixed_original_style_control": control,
        "applied_recorded_style": applied,
    }


def run(source, destination):
    plan, packet = study.load_packet(source)
    raw = source / "live_execution"
    result = study.evaluator(plan)(
        packet, *(study.original.read_jsonl(raw / name) for name in study.original.LOGS)
    )
    if (
        result["evidence_kind"] != "recorded_model_logs"
        or not result["protocol_complete"]
        or result["logged_dispatch_attempts"] != 12
    ):
        raise ValueError("only the completed recorded12 study can provide choices")
    materials, _, _ = study.original.checked_materials()
    cases = {case["case_id"]: case for case in materials["cases"]}
    pairs = [
        paired_branch(cases[row["case_id"]], row, f, t)
        for row in result["rows"]
        for f, t in product(LEVELS, repeat=2)
    ]
    assert len(pairs) == 108
    grid = []
    for f, t in product(LEVELS, repeat=2):
        rows = [
            row for row in pairs if (row["fundamental_signal"], row["technical_signal"]) == (f, t)
        ]
        cells = []
        for agent, condition in product(("A000", "A001"), ("positive", "negative")):
            cell = [
                row for row in rows if (row["agent_id"], row["condition"]) == (agent, condition)
            ]
            cells.append(
                {
                    "agent_id": agent,
                    "condition": condition,
                    "source_choices": len(cell),
                    "control_actions": dict(
                        Counter(
                            row["fixed_original_style_control"]["decision"]["action"]
                            for row in cell
                        )
                    ),
                    "applied_actions": dict(
                        Counter(row["applied_recorded_style"]["decision"]["action"] for row in cell)
                    ),
                    "action_changed_pairs": sum(row["action_changed"] for row in cell),
                }
            )
        grid.append(
            {
                "fundamental_signal": f,
                "technical_signal": t,
                "cells": cells,
                "action_changed_pairs": sum(row["action_changed"] for row in rows),
            }
        )
    destination.mkdir(parents=True, exist_ok=False)
    summary = {
        "evidence_kind": "exploratory_synthetic_translation_of_recorded_choices",
        "source_run_id": packet["run_id"],
        "source_packet_sha256": packet["packet_sha256"],
        "source_requested_model": packet["model"],
        "source_choices": 12,
        "grid_states": 9,
        "paired_branches": 108,
        "simulations": 216,
        "style_changed_pairs": sum(row["style_changed"] for row in pairs),
        "action_changed_pairs": sum(row["action_changed"] for row in pairs),
        "holdings_changed_pairs": sum(row["holdings_changed"] for row in pairs),
        "all_cash_and_shares_conserved": all(
            row[key]["cash_and_share_conservation"]
            for row in pairs
            for key in ("fixed_original_style_control", "applied_recorded_style")
        ),
        "grid": grid,
        "new_remote_calls": 0,
        "limits": "Chosen after source answers;one-step constructed observations,not time series. "
        "Recorded choices are held fixed,not predicted behavior in new states. Assigned finite "
        "counterparties determine fills/prices. Twelve returns are not12people. No human "
        "alignment,market realism,collective feedback or independent generation proof.",
    }
    study.save(destination / "summary.json", summary)
    study.save(destination / "paired_branches.json", pairs)
    study.save(destination / "source_evaluation.json", result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.source, args.output)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "paired_branches",
                    "simulations",
                    "action_changed_pairs",
                    "all_cash_and_shares_conserved",
                    "new_remote_calls",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
