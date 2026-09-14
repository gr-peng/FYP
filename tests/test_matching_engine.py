from decimal import Decimal

import pytest

from behavioral_market.market import Exchange, OrderStatus, Portfolio


@pytest.fixture
def exchange() -> Exchange:
    market = Exchange("TEST", "100")
    for agent in ("B", "S1", "S2", "S3"):
        market.register(agent, Portfolio("1000", 10, "100"))
    return market


@pytest.mark.parametrize(
    ("bid", "ask", "expected"), [("99", "100", 0), ("100", "100", 1), ("101", "100", 1)]
)
def test_crossing_rule(exchange: Exchange, bid: str, ask: str, expected: int) -> None:
    exchange.submit_order("S1", "sell", ask, 1)
    _, trades = exchange.submit_order("B", "buy", bid, 1)
    assert len(trades) == expected
    if trades:
        assert trades[0].price == Decimal(ask)
    assert exchange.market_price == (Decimal(ask) if expected else Decimal("100"))


def test_price_priority_beats_arrival_time(exchange: Exchange) -> None:
    exchange.submit_order("S1", "sell", "102", 1)
    exchange.submit_order("S2", "sell", "101", 1)
    _, trades = exchange.submit_order("B", "buy", "103", 2)
    assert [trade.seller_id for trade in trades] == ["S2", "S1"]
    assert [trade.price for trade in trades] == [Decimal("101"), Decimal("102")]


def test_time_priority_at_same_price(exchange: Exchange) -> None:
    exchange.submit_order("S1", "sell", "100", 1)
    exchange.submit_order("S2", "sell", "100", 1)
    _, trades = exchange.submit_order("B", "buy", "100", 2)
    assert [trade.seller_id for trade in trades] == ["S1", "S2"]


def test_partial_and_multi_order_fill(exchange: Exchange) -> None:
    first, _ = exchange.submit_order("S1", "sell", "100", 2)
    second, _ = exchange.submit_order("S2", "sell", "101", 3)
    bid, trades = exchange.submit_order("B", "buy", "101", 4)
    assert [trade.quantity for trade in trades] == [2, 2]
    assert first.status == OrderStatus.FILLED
    assert second.status == OrderStatus.PARTIALLY_FILLED
    assert second.remaining_quantity == 1
    assert bid.status == OrderStatus.FILLED
    assert exchange.book.asks == [second]
    assert exchange.day_volume == 4


def test_bid_priority_and_resting_bid_price(exchange: Exchange) -> None:
    exchange.submit_order("B", "buy", "101", 1)
    exchange.submit_order("S1", "buy", "102", 1)
    _, trades = exchange.submit_order("S2", "sell", "100", 2)
    assert [trade.buyer_id for trade in trades] == ["S1", "B"]
    assert [trade.price for trade in trades] == [Decimal("102"), Decimal("101")]


def test_day_expiration_and_cancel_release_reservations(exchange: Exchange) -> None:
    bid, _ = exchange.submit_order("B", "buy", "100", 9)
    with pytest.raises(ValueError, match="cash"):
        exchange.submit_order("B", "buy", "100", 2)
    exchange.cancel_order(bid.order_id, "B")
    assert bid.status == OrderStatus.CANCELLED
    exchange.submit_order("B", "buy", "100", 10)
    expired = exchange.end_day()
    assert len(expired) == 1 and expired[0].status == OrderStatus.EXPIRED
    assert exchange.book.resting_orders() == ()
    assert exchange.market_price == Decimal("100")
    assert exchange.day_volume == 0
    exchange.submit_order("B", "buy", "100", 10)


def test_no_self_trade_and_no_overcommit(exchange: Exchange) -> None:
    exchange.submit_order("S1", "sell", "100", 8)
    with pytest.raises(ValueError, match="shares"):
        exchange.submit_order("S1", "sell", "101", 3)
    with pytest.raises(ValueError, match="self-crossing"):
        exchange.submit_order("S1", "buy", "100", 1)
    assert len(exchange.orders) == 1
    assert len(exchange.trades) == 0


def test_market_price_only_changes_on_execution(exchange: Exchange) -> None:
    exchange.submit_order("B", "buy", "90", 1)
    exchange.submit_order("S1", "sell", "110", 1)
    assert exchange.market_price == Decimal("100")
    exchange.end_day()
    assert exchange.market_price == Decimal("100")
