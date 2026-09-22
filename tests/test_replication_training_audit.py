import numpy as np
import pandas as pd
import pytest

from experiments.replication_training_audit import account_ledger, attach_trade_actors, macro_panels


def test_trade_and_dividend_ledgers_keep_period_cash_timing_separate():
    common = {"session": "s", "repeat": 1, "group": 1}
    rounds = pd.DataFrame(
        [
            {
                **common,
                "period": period,
                "player": player,
                "end_cash": cash,
                "end_assets": assets,
                "dividend": dividend,
                "dividend_payment": assets * dividend,
                "next_cash": cash + assets * dividend,
            }
            for period, player, cash, assets, dividend in [
                (1, 1, 2850, 23, 10),
                (1, 2, 1150, 57, 10),
                (2, 1, 3160, 21, 0),
                (2, 2, 1640, 59, 0),
            ]
        ]
    )
    trades = pd.DataFrame(
        [
            {**common, "period": 1, "ask_player": 2, "bid_player": 1, "quantity": 3, "price": 50},
            {**common, "period": 2, "ask_player": 1, "bid_player": 2, "quantity": 2, "price": 40},
        ]
    )
    ledger = account_ledger(rounds, trades)
    assert ledger.cash_residual.eq(0).all() and ledger.asset_residual.eq(0).all()
    assert ledger.dividend_residual.eq(0).all() and ledger.next_cash_residual.eq(0).all()
    assert ledger.query("player == 1 and period == 2").iloc[0].start_cash == 3080
    assert ledger.query("player == 1 and period == 1").iloc[0].start_cash == 3000
    altered = rounds.copy()
    altered.loc[(altered.player == 1) & (altered.period == 2), "end_cash"] += 1
    assert account_ledger(altered, trades).cash_residual.abs().sum() == 1


def test_market_macro_prices_are_volume_weighted_and_empty_period_stays_missing():
    common = {"session": "s", "repeat": 1, "group": 1}
    rounds = pd.DataFrame(
        [
            {
                **common,
                "period": period,
                "end_assets": 80,
                "dividend": 0,
                "average_price": 50 if period == 1 else np.nan,
                "closing_price": 60 if period == 1 else np.nan,
            }
            for period in (1, 2)
        ]
    )
    trades = pd.DataFrame(
        [
            {**common, "period": 1, "uuid": "t1", "quantity": 1, "price": 40},
            {**common, "period": 1, "uuid": "t2", "quantity": 9, "price": 60},
        ]
    )
    markets = pd.DataFrame([{"slab": "s", "mid": 1, "mlab": "m", "study": 1, "cond": 1}])
    periods, outcomes = macro_panels(rounds, trades, markets)
    first, second = periods.iloc[0], periods.iloc[1]
    assert first.vwap == 58 and first.unweighted_mean == 50
    assert first.fundamental_value == 50
    assert second.volume == 0 and pd.isna(second.vwap) and pd.isna(second.rad)
    assert outcomes.iloc[0].priced_periods == 1
    assert outcomes.iloc[0].rad == pytest.approx(8 / 27.5)
    assert outcomes.iloc[0].turnover == 0.125


def test_trade_reference_cannot_cross_markets():
    common = {"session": "s", "repeat": 1, "period": 1, "quantity": 1, "price": 50, "created": 10}
    orders = pd.DataFrame(
        [
            {**common, "group": 2, "uuid": "ask", "player": 1, "kind": "limit", "side": "ask"},
            {**common, "group": 1, "uuid": "bid", "player": 2, "kind": "market", "side": "bid"},
        ]
    )
    trades = pd.DataFrame(
        [{**common, "group": 1, "uuid": "t", "ask_uuid": "ask", "bid_uuid": "bid"}]
    )
    with pytest.raises(ValueError, match="another market"):
        attach_trade_actors(orders, trades)
