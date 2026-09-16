from decimal import Decimal

from behavioral_market.llm.schemas import OrderDecision, StyleDecision, hold


def enforce_portfolio(decision, portfolio, price_floor=None, price_ceiling=None):
    if decision.action != "hold" and (
        (price_floor is not None and decision.limit_price < price_floor)
        or (price_ceiling is not None and decision.limit_price > price_ceiling)
    ):
        return hold("limit price outside permitted band"), "price_band_violation"
    if (
        decision.action == "buy"
        and Decimal(str(decision.limit_price)) * decision.quantity > portfolio.cash
    ):
        return hold("insufficient cash"), "constraint_violation"
    if decision.action == "sell" and decision.quantity > portfolio.position:
        return hold("insufficient shares"), "constraint_violation"
    return decision, None


def signal(style, observation):
    if style == "fundamental":
        return observation["fundamental_value"] / observation["last_close"] - 1
    return observation.get("momentum_5") or 0.0


def rule_order(agent, observation, rng):
    value = signal(agent.current_style, observation) + rng.gauss(0, 0.008)
    if abs(value) < 0.003:
        return hold("rule signal below threshold")
    side = "buy" if value > 0 else "sell"
    price = round(observation["last_close"] * (1 + (0.003 if side == "buy" else -0.003)), 2)
    quantity = max(1, round(1 + 8 * agent.persona.risk_tolerance))
    if side == "buy":
        quantity = min(quantity, int(agent.portfolio.cash // Decimal(str(price))))
    else:
        quantity = min(quantity, agent.portfolio.position)
    if not quantity:
        return hold("rule portfolio bound")
    return OrderDecision(
        action=side,
        quantity=quantity,
        limit_price=price,
        rationale="seeded fundamental or momentum rule",
        sentiment="bullish" if value > 0 else "bearish",
    )


def rule_style(agent):
    gap = agent.shadow_growth - float(agent.portfolio.total_equity) / agent.block_start_equity
    target = "technical" if agent.current_style == "fundamental" else "fundamental"
    switch = gap > 0.01
    return StyleDecision(
        decision="switch" if switch else "stay",
        target_style=target if switch else agent.current_style,
        rationale="counterfactual gap threshold",
    )
