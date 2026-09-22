import copy

import pandas as pd

from experiments.replication_quote_states import quote_states


def fixture():
    key = {"session": "s", "repeat": 1, "group": 1, "period": 2}
    orders = pd.DataFrame(
        [
            {
                **key,
                "uuid": "past",
                "created": 2,
                "player": 1,
                "side": "bid",
                "kind": "market",
                "quantity": 3,
                "price": 40,
                "is_replacement": 0,
            },
            {
                **key,
                "uuid": "target",
                "created": 5,
                "player": 1,
                "side": "ask",
                "kind": "market",
                "quantity": 2,
                "price": 42,
                "is_replacement": 0,
            },
            {
                **key,
                "uuid": "future",
                "created": 9,
                "player": 1,
                "side": "bid",
                "kind": "limit",
                "quantity": 1,
                "price": 99,
                "is_replacement": 0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                **key,
                "created": 2,
                "price": 40,
                "quantity": 3,
                "ask_player": 2,
                "bid_player": 1,
                "ask_uuid": "old_ask",
                "bid_uuid": "past",
            },
            {
                **key,
                "created": 5,
                "price": 42,
                "quantity": 2,
                "ask_player": 1,
                "bid_player": 2,
                "ask_uuid": "target",
                "bid_uuid": "old_bid",
            },
            {
                **key,
                "created": 9,
                "price": 99,
                "quantity": 1,
                "ask_player": 2,
                "bid_player": 1,
                "ask_uuid": "later_ask",
                "bid_uuid": "future",
            },
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                **key,
                "player": 1,
                "participant": "p",
                "start_cash": 1000,
                "start_assets": 60,
                "end_cash": 1,
                "end_assets": 1,
            }
        ]
    )
    periods = pd.DataFrame(
        [
            {
                **key,
                "period": p,
                "reported_average": 50 + p,
                "reported_close": 60 + p,
                "dividend": 10,
            }
            for p in (1, 2, 3)
        ]
    )
    markets = pd.DataFrame([{"slab": "s", "mid": 1, "cond": 1}])
    return [orders, trades, ledger, periods, markets]


def test_observation_excludes_target_fill_and_future_prices_actions_balances():
    frames = fixture()
    result, exclusions = quote_states(*frames, selected_ids=["target"])
    assert not exclusions and len(result) == 1
    state = result[0]["state"]
    assert state["cash"] == 880 and state["shares"] == 63
    assert state["earlier_current_period_trades"]["mean_price"] == 40
    assert len(state["earlier_periods"]) == 1
    assert state["own_earlier_current_period_submissions"] == {"buy": 1, "sell": 0}
    changed = copy.deepcopy(frames)
    changed[0].loc[changed[0].uuid == "target", ["side", "price", "quantity"]] = ["bid", 900, 99]
    changed[0].loc[changed[0].created > 5, ["side", "price", "quantity"]] = ["ask", 700, 88]
    changed[1].loc[changed[1].created >= 5, ["price", "quantity"]] = [900, 99]
    changed[2][["end_cash", "end_assets"]] = -100000
    changed[3].loc[changed[3].period >= 2, ["reported_average", "reported_close", "dividend"]] = 999
    altered, _ = quote_states(*changed, selected_ids=["target"])
    assert altered[0]["state"] == state
    assert altered[0]["target"] != result[0]["target"]
    assert "participant" not in state and "available_cash" not in state


def test_same_second_ambiguity_is_excluded_without_uuid_tiebreak():
    frames = fixture()
    extra = frames[0].iloc[[1]].copy()
    extra["uuid"] = "other"
    extra["player"] = 2
    frames[0] = pd.concat([frames[0], extra], ignore_index=True)
    result, exclusions = quote_states(*frames, selected_ids=["target"])
    assert result == [] and exclusions == {"multiple_submissions_in_target_second": 1}
    frames = fixture()
    frames[1].loc[1, "ask_uuid"] = "unrelated"
    result, exclusions = quote_states(*frames, selected_ids=["target"])
    assert result == [] and exclusions == {"other_trade_in_target_second": 1}
