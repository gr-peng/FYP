import copy

import numpy as np
import pytest

from experiments.financial_probability import Demonstrations, question, select_targets
from experiments.market_quote_memory_baselines import FEATURES
from experiments.replication_development_data import selected_development_rows


def sample(i):
    return {
        "provenance": {"session": "s", "participant": f"p{i // 2}", "order_id": str(i)},
        "state": {
            "cash": 1000 + i,
            "shares": 10,
            "displayed_expected_remaining_dividends_per_share": 50,
            "repetition": 1,
            "period": 1,
            "elapsed_seconds": i,
            "earlier_periods": [],
            "earlier_current_period_trades": {
                "count": 0,
                "mean_price": None,
                "mean_price_in_last_completed_trade_second": None,
            },
            "own_earlier_current_period_submissions": {"buy": 0, "sell": 0},
            "pre_market_task": {"task": "video", "condition": "calm"},
            "own_completed_period_memory": {
                "earlier_periods_this_repetition": {"buy": 0, "sell": 0},
                "previous_repetitions": {"buy": 0, "sell": 0},
                "last_completed_period_with_own_orders": None,
            },
        },
        "target": {"buy": i % 2, "price": 10, "quantity": 1, "kind": "limit"},
    }


def test_target_sampling_and_retrieval_never_use_current_target_labels():
    train = [sample(i) for i in range(12)]
    model = {"mean": [0] * len(FEATURES), "scale": [1] * len(FEATURES)}
    original = Demonstrations(train, model)
    changed = copy.deepcopy(train)
    for record in changed:
        record["target"] = {
            "buy": 1 - record["target"]["buy"],
            "price": 999,
            "quantity": 999,
            "kind": "market",
        }
    shuffled = Demonstrations(changed, model)
    assert [r["provenance"] for r in select_targets(train, {"s"}, 3)] == [
        r["provenance"] for r in select_targets(changed, {"s"}, 3)
    ]
    indices = original.retrieve(sample(99)["state"])
    assert indices == shuffled.retrieve(sample(99)["state"])
    assert len({train[i]["provenance"]["participant"] for i in indices}) == 4
    left, right = original.bundle(sample(99)["state"]), shuffled.bundle(sample(99)["state"])
    for method in ("no_examples", "neutral_labels"):
        assert question(left, method, "buy_first") == question(right, method, "buy_first")
    assert sorted(original.shuffled) == sorted(r["target"]["buy"] for r in train)


def test_both_menu_orders_reverse_labels_without_reversing_behavior():
    model = {"mean": np.zeros(len(FEATURES)), "scale": np.ones(len(FEATURES))}
    bundle = Demonstrations([sample(i) for i in range(12)], model).bundle(sample(99)["state"])
    left = question(bundle, "real_labels", "buy_first")
    right = question(bundle, "real_labels", "sell_first")
    assert left[3] == "A" and right[3] == "B"
    assert left[0]["target"] == right[0]["target"]
    for a, b in zip(left[0]["examples"], right[0]["examples"], strict=True):
        assert a["state"] == b["state"]
        assert a["observed_choice_encoding"]["A"] == b["observed_choice_encoding"]["B"]
    with pytest.raises(ValueError):
        question(bundle, "unregistered", "buy_first")


def test_development_loader_does_not_read_train_or_test_behavior():
    class ClosedRow(dict):
        def __getitem__(self, key):
            if key != "session":
                raise AssertionError("closed behavior read")
            return super().__getitem__(key)

    assignments = {"train": ["a"], "development": ["b"], "test": ["c"]}
    rows = [ClosedRow(session="a"), {"session": "b", "side": "buy"}, ClosedRow(session="c")]
    assert list(selected_development_rows(rows, assignments, ["session", "side"], "session")) == [
        {"session": "b", "side": "buy"}
    ]
