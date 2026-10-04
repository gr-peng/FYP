from decimal import Decimal

from behavioral_market.environments.liquidity import FiniteLiquidityProvider
from behavioral_market.market import Exchange, Portfolio


def make_market():
    exchange = Exchange("test", "100")
    exchange.register("trader", Portfolio("10000", 20, "100"))
    background = FiniteLiquidityProvider(
        exchange,
        {
            "initial_cash": "500",
            "initial_shares": 5,
            "half_spread_bps": 20,
            "level_spacing_bps": 10,
            "level_quantities": [3, 3],
            "inventory_skew_bps": 50,
        },
    )
    return exchange, background


def test_one_sided_buys_exhaust_finite_inventory_without_creating_resources():
    exchange, background = make_market()
    cash = sum(p.cash for p in exchange.portfolios.values())
    shares = sum(p.position for p in exchange.portfolios.values())
    records, _ = background.requote("100")
    assert sum(row["quantity"] for row in records if row["action"] == "sell") == 5
    _, trades = exchange.submit_order("trader", "buy", "110", 8)
    assert sum(t.quantity for t in trades) == 5
    assert background.portfolio.position == 0
    exchange.end_day()
    for _ in range(3):
        records, _ = background.requote("100")
        assert not any(row["action"] == "sell" for row in records)
    assert sum(p.cash for p in exchange.portfolios.values()) == cash
    assert sum(p.position for p in exchange.portfolios.values()) == shares


def test_one_sided_sells_respect_reserved_cash_and_requotes_release_it():
    exchange, background = make_market()
    for _ in range(2):
        background.requote("100")
        assert exchange._available_cash(background.agent_id) >= 0
        bids = [order for order in exchange.book.bids if order.agent_id == background.agent_id]
        assert sum(order.quantity for order in bids) == 5
    _, trades = exchange.submit_order("trader", "sell", "90", 8)
    assert sum(t.quantity for t in trades) == 5
    assert background.portfolio.cash >= 0
    exchange.end_day()
    records, _ = background.requote("100")
    assert not any(row["action"] == "buy" for row in records)


def test_inventory_skew_changes_quotes_without_marking_up_the_market():
    exchange, background = make_market()
    records, _ = background.requote("100")
    first_ask = min(Decimal(r["price"]) for r in records if r["action"] == "sell")
    exchange.submit_order("trader", "buy", "101", 1)
    last_trade = exchange.market_price
    records, _ = background.requote("100")
    next_ask = min(Decimal(r["price"]) for r in records if r["action"] == "sell")
    assert next_ask > first_ask
    assert exchange.market_price == last_trade
