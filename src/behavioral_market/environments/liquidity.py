"""Finite, explicitly synthetic background liquidity for mechanism experiments.

This participant never reads future prices, events, treatment labels or model output.
Cash and inventory persist. Requoting does not replenish either resource.
"""

from decimal import Decimal

from behavioral_market.market import Portfolio


class FiniteLiquidityProvider:
    agent_id = "BACKGROUND_LP"

    def __init__(self, exchange, spec):
        self.exchange = exchange
        self.half_spread_bps = Decimal(str(spec["half_spread_bps"]))
        self.level_spacing_bps = Decimal(str(spec["level_spacing_bps"]))
        self.level_quantities = tuple(spec["level_quantities"])
        self.inventory_skew_bps = Decimal(str(spec["inventory_skew_bps"]))
        if not (
            0 < self.half_spread_bps < 1000
            and 0 < self.level_spacing_bps < 1000
            and 0 <= self.inventory_skew_bps < 1000
            and self.level_quantities
            and all(type(q) is int and q > 0 for q in self.level_quantities)
            and type(spec["initial_shares"]) is int
            and spec["initial_shares"] > 0
        ):
            raise ValueError("invalid finite liquidity parameters")
        self.initial_shares = spec["initial_shares"]
        self.portfolio = Portfolio(spec["initial_cash"], self.initial_shares, exchange.market_price)
        exchange.register(self.agent_id, self.portfolio)

    def requote(self, reference):
        """Cancel own outstanding orders and quote within remaining finite resources."""
        reference = Decimal(str(reference))
        if not reference.is_finite() or reference <= 0:
            raise ValueError("invalid background reference")
        for order in list(self.exchange.book.resting_orders()):
            if order.agent_id == self.agent_id:
                self.exchange.cancel_order(order.order_id, self.agent_id)
        inventory_deviation = (
            Decimal(self.initial_shares - self.portfolio.position) / self.initial_shares
        )
        # Bound skew even if repeated sell flow greatly increases LP inventory.
        inventory_deviation = max(Decimal(-1), min(Decimal(1), inventory_deviation))
        center = reference * (1 + self.inventory_skew_bps * inventory_deviation / 10000)
        records, trades = [], []
        for level, target_quantity in enumerate(self.level_quantities):
            offset = (self.half_spread_bps + level * self.level_spacing_bps) / 10000
            if offset >= 1:
                raise ValueError("background quote ladder reaches nonpositive prices")
            for side, price in (("buy", center * (1 - offset)), ("sell", center * (1 + offset))):
                price = price.quantize(Decimal("0.000001"))
                if price <= 0:
                    raise ValueError("background price below simulation precision")
                capacity = (
                    int(self.exchange._available_cash(self.agent_id) // price)
                    if side == "buy"
                    else self.exchange._available_shares(self.agent_id)
                )
                quantity = min(target_quantity, capacity)
                if quantity <= 0:
                    continue
                order, fills = self.exchange.submit_order(self.agent_id, side, price, quantity)
                records.append(
                    {
                        "order_id": order.order_id,
                        "action": side,
                        "price": str(price),
                        "quantity": quantity,
                        "level": level,
                        "reference": str(reference),
                    }
                )
                trades.extend(fills)
        return records, trades
