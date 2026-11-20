"""Compare a long/cash signal proxy with constrained rule-order replay, offline only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from behavioral_market.agents.policy import enforce_portfolio, rule_order
from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.market import Exchange, Portfolio
from scripts import collect_style_review_economic_controls as source

SPREADS = ("0.00", "0.05", "1.00")


def replay(case, style, half_spread):
    """Given historical marks, not a generated market or a forecast of future performance."""
    source.require(style in ("fundamental", "technical"), "unknown style")
    source.require(half_spread in SPREADS, "use the predefined execution assumptions")
    spread = Decimal(half_spread)
    portfolio, counterparty = Portfolio(1000, 2, 100), Portfolio(10000, 100, 100)
    agent = AgentState(
        AgentPersona("FOCAL", 0.5, 0.5, 0.5, 0.5, 0.5, style),
        portfolio,
        style,
        1200,
        1200,
        peak_equity=1200,
    )
    daily = []
    for row in case["construction_not_sent"]["daily_trace"]:
        source.require(row["history_last_session"] < row["session_date"], "future observation")
        reference = Decimal(str(row["start_reference"]))
        observation = {
            "last_close": float(reference),
            "fundamental_value": row["fundamental_anchor"],
            "momentum_5": row["momentum_5"],
        }
        portfolio.mark_to_market(reference)
        counterparty.mark_to_market(reference)
        decision = rule_order(agent, observation, source.materials.ZeroNoise())
        decision, violation = enforce_portfolio(decision, portfolio)
        source.require(violation is None, "rule order violated portfolio")
        exchange = Exchange("SYNTH", reference)
        exchange.register("FOCAL", portfolio)
        exchange.register("ASSIGNED_COUNTERPARTY", counterparty)
        if decision.action != "hold":
            side = "sell" if decision.action == "buy" else "buy"
            quote = reference + spread if side == "sell" else reference - spread
            # One finite, direction-conditioned counterparty avoids counterparties
            # trading with each other at zero spread. Its quote/quantity is assigned,
            # not inferred from an observed book. It need not cross the focal limit.
            exchange.submit_order("ASSIGNED_COUNTERPARTY", side, quote, decision.quantity)
            exchange.submit_order("FOCAL", decision.action, decision.limit_price, decision.quantity)
        source.require(
            sum(p.cash for p in exchange.portfolios.values()) == 11000, "cash not conserved"
        )
        source.require(
            sum(p.position for p in exchange.portfolios.values()) == 102, "shares not conserved"
        )
        close = Decimal(str(row["close"]))
        for p in exchange.portfolios.values():
            p.mark_to_market(close)
        filled = sum(t.quantity for t in exchange.trades)
        orders = [
            {**asdict(o), "side": o.side.value, "status": o.status.value, "price": str(o.price)}
            for o in exchange.orders
        ]
        expired = len(exchange.end_day())
        agent.memory.append(
            {
                "session": row["session_date"],
                "action": decision.action,
                "filled_quantity": filled,
                "close": float(close),
            }
        )
        daily.append(
            {
                "session": row["session_date"],
                "history_last_session": row["history_last_session"],
                "observation_used_for_order": observation,
                "decision": decision.model_dump(),
                "filled_quantity": filled,
                "assigned_half_spread": half_spread,
                "orders_before_expiry": orders,
                "trades": [{**asdict(t), "price": str(t.price)} for t in exchange.trades],
                "expired_orders": expired,
                "exogenous_close_used_only_after_order": str(close),
                "cash_after": str(portfolio.cash),
                "shares_after": portfolio.position,
                "equity_after": str(portfolio.total_equity),
                "counterparty_cash_after": str(counterparty.cash),
                "counterparty_shares_after": counterparty.position,
                "cash_and_shares_conserved": True,
            }
        )
    return {
        "style": style,
        "initial_cash": "1000",
        "initial_shares": 2,
        "initial_equity": "1200",
        "half_spread": half_spread,
        "fees": "0",
        "final_cash": str(portfolio.cash),
        "final_shares": portfolio.position,
        "final_equity": str(portfolio.total_equity),
        "return": str(portfolio.total_equity / Decimal(1200) - 1),
        "daily": daily,
        "all_cash_and_shares_conserved": True,
    }


def audit(package):
    rows = []
    for case in package["cases"]:
        original = case["original_style"]
        alternative = "technical" if original == "fundamental" else "fundamental"
        proxy = case["construction_not_sent"]["final_state"]["counterfactual"]
        for spread in SPREADS:
            current, other = replay(case, original, spread), replay(case, alternative, spread)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "original_proxy_condition": case["economic_condition"],
                    "proxy_method": proxy["method"],
                    "proxy_alternative_return": proxy["alternative_block_return"],
                    "half_spread": spread,
                    "original_rule_replay": current,
                    "alternative_rule_replay": other,
                    "finite_alternative_minus_original": str(
                        Decimal(other["return"]) - Decimal(current["return"])
                    ),
                }
            )
    return {
        "evidence_kind": "exploratory_fictional_finite_account_replay",
        "source": "accepted four hypothetical economic cases; no human/market raw files",
        "rows": rows,
        "new_model_POSTs": 0,
        "limits": "Proxy arithmetic is preserved, not corrected or relabeled as a trading profit. "
        "Finite replay uses the production rule with zero noise and one assigned finite "
        "direction-conditioned counterparty each session,zero fees,and external closing marks. "
        "Quotes and full liquidity where limits cross are construction assumptions,not real "
        "execution estimates. The original labels positive/negative refer only to the old proxy; "
        "they are not required signs of finite-account returns. Not endogenous market generation, "
        "future profitability,human alignment or validation of a replacement model prompt.",
    }


def describe(result):
    lines = ["替代策略的简化收益，与真正按账户约束下单，不是同一个计算。", ""]
    for row in result["rows"]:
        lines.append(
            f"{row['case_id']} / 每边价差{row['half_spread']}：旧简化收益"
            f"{row['proxy_alternative_return'] * 100:.4f}%，"
            f"原策略账户收益{Decimal(row['original_rule_replay']['return']) * 100:.4f}%，"
            f"替代策略账户收益{Decimal(row['alternative_rule_replay']['return']) * 100:.4f}%。"
        )
    lines += [
        "",
        "简化算法无信号就全换现金；当前下单规则无信号只等待，仍持有之前买入的股票。",
        "以上成交价、对手和收盘价均为构造条件；旧模型回答保留，不据此改写旧结果。",
        "新提示如要展示可执行策略收益，应先用同样账户约束和成交假设另建材料。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = source.checked_materials()[0]
    result = audit(package)
    args.output.mkdir(parents=True, exist_ok=False)
    source.save(args.output / "audit.json", result)
    (args.output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
    print(
        source.materials.canonical(
            {"case_execution_pairs": len(result["rows"]), "new_model_POSTs": 0}
        )
    )


if __name__ == "__main__":
    main()
