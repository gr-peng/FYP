import copy

import pandas as pd
import pytest

from experiments.replication_quote_memory import extend_memory


def example():
    rows = []
    for uid, repeat, period, second, player, side in [
        ("rep1", 1, 10, 10, 1, "bid"),
        ("earlier", 2, 1, 10, 1, "ask"),
        ("current_past", 2, 2, 1, 1, "bid"),
        ("target", 2, 2, 5, 1, "ask"),
        ("future_current", 2, 2, 8, 1, "bid"),
        ("future_period", 2, 3, 1, 1, "bid"),
        ("peer", 1, 1, 1, 2, "ask"),
    ]:
        rows.append(
            {
                "session": "s",
                "repeat": repeat,
                "period": period,
                "created": second,
                "player": player,
                "side": side,
                "group": 1,
                "uuid": uid,
                "is_replacement": 0,
            }
        )
    orders = pd.DataFrame(rows)
    ledger = orders[["session", "repeat", "period", "group", "player"]].drop_duplicates()
    ledger["participant"] = ledger.player.map({1: "same_person", 2: "another_person"})
    records = [
        {
            "provenance": {"session": "s", "repeat": 2, "period": 2, "participant": "same_person"},
            "state": {"cash": 3000, "shares": 20},
            "target": {"buy": 0},
        }
    ]
    return records, orders, ledger


def test_memory_crosses_account_reset_but_not_target_or_future_boundary():
    records, orders, ledger = example()
    original = copy.deepcopy(records)
    result = extend_memory(records, orders, ledger)
    memory = result[0]["state"]["own_completed_period_memory"]
    assert memory["previous_repetitions"] == {"buy": 1, "sell": 0}
    assert memory["earlier_periods_this_repetition"] == {"buy": 0, "sell": 1}
    assert result[0]["state"]["cash"] == 3000
    changed = orders.copy()
    changed.loc[
        changed.uuid.isin(["target", "future_current", "future_period", "peer"]), "side"
    ] = "bid"
    assert extend_memory(records, changed, ledger) == result
    assert records == original


def test_earlier_real_history_changes_memory_without_importing_peer_behavior():
    records, orders, ledger = example()
    before = extend_memory(records, orders, ledger)[0]["state"]["own_completed_period_memory"]
    orders.loc[orders.uuid.eq("rep1"), "side"] = "ask"
    after = extend_memory(records, orders, ledger)[0]["state"]["own_completed_period_memory"]
    assert after["previous_repetitions"] == {"buy": 0, "sell": 1}
    assert before["earlier_periods_this_repetition"] == after["earlier_periods_this_repetition"]


def test_first_period_and_ambiguous_subject_mapping():
    records, orders, ledger = example()
    records[0]["provenance"].update(repeat=1, period=1)
    memory = extend_memory(records, orders, ledger)[0]["state"]["own_completed_period_memory"]
    assert (
        memory["previous_repetitions"]
        == memory["earlier_periods_this_repetition"]
        == {"buy": 0, "sell": 0}
    )
    assert memory["last_completed_period_with_own_orders"] is None
    conflict = ledger.iloc[[0]].copy()
    conflict["participant"] = "wrong"
    with pytest.raises(ValueError, match="ambiguous"):
        extend_memory(records, orders, pd.concat([ledger, conflict]))
