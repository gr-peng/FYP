from decimal import Decimal

import pytest

from behavioral_market.market import Exchange, Portfolio


def test_buy_sell_and_marks() -> None:
    portfolio = Portfolio("1000")
    portfolio.apply_buy("10", 10)
    assert (portfolio.cash, portfolio.position, portfolio.avg_cost) == (
        Decimal("900"),
        10,
        Decimal("10"),
    )
    portfolio.mark_to_market("12")
    assert portfolio.unrealized_pnl == Decimal("20")
    assert portfolio.total_equity == Decimal("1020")
    portfolio.apply_sell("12", 4)
    assert (portfolio.cash, portfolio.position, portfolio.realized_pnl) == (
        Decimal("948"),
        6,
        Decimal("8"),
    )
    assert portfolio.unrealized_pnl == Decimal("12")
    portfolio.apply_sell("8", 6)
    assert portfolio.position == 0
    assert portfolio.avg_cost == 0
    assert portfolio.realized_pnl == Decimal("-4")


def test_invariants_and_atomic_rejection() -> None:
    portfolio = Portfolio("100", 2, "10")
    with pytest.raises(ValueError, match="cash"):
        portfolio.apply_buy("101", 1)
    with pytest.raises(ValueError, match="shares"):
        portfolio.apply_sell("10", 3)
    assert (portfolio.cash, portfolio.position) == (Decimal("100"), 2)
    with pytest.raises(ValueError):
        Portfolio("-1")


def test_exchange_conserves_cash_and_shares_with_price_improvement() -> None:
    exchange = Exchange("TEST", "100")
    exchange.register("buyer", Portfolio("1000"))
    exchange.register("seller", Portfolio("100", 5, "80"))
    exchange.submit_order("seller", "sell", "90", 3)
    _, trades = exchange.submit_order("buyer", "buy", "100", 3)
    assert trades[0].price == Decimal("90")
    assert exchange.portfolios["buyer"].cash == Decimal("730")
    assert exchange.portfolios["seller"].cash == Decimal("370")
    assert exchange.portfolios["seller"].realized_pnl == Decimal("30")
    assert sum(p.cash for p in exchange.portfolios.values()) == Decimal("1100")
    assert sum(p.position for p in exchange.portfolios.values()) == 5
