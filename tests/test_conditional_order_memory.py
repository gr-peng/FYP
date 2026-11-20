"""Synthetic-only memory masking, temporal boundaries and training isolation."""

import copy
import socket

import dotenv
import numpy as np
import pandas as pd
import pytest
from test_replication_quote_memory import example
from test_replication_quote_states import fixture

from experiments import conditional_order_memory as tool
from experiments import market_quote_baselines as base
from experiments import market_quote_memory_baselines as extended
from experiments import replication_development_data, replication_training_data
from experiments.replication_quote_memory import extend_memory
from experiments.replication_quote_states import quote_states


@pytest.fixture(autouse=True)
def no_external_access(monkeypatch):
    def reject(*_args, **_kwargs):
        pytest.fail("synthetic memory tests cannot access network, credentials or human loaders")

    monkeypatch.setattr(socket, "socket", reject)
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(dotenv, "dotenv_values", reject)
    monkeypatch.setattr(pd, "read_csv", reject)
    monkeypatch.setattr(pd, "read_excel", reject)
    monkeypatch.setattr(pd, "read_parquet", reject)
    monkeypatch.setattr(replication_training_data, "load_training_frames", reject)
    monkeypatch.setattr(replication_training_data, "load_training_macro_reference", reject)
    monkeypatch.setattr(replication_development_data, "load_development_frames", reject)


def state():
    result, _ = quote_states(*fixture(), selected_ids=["target"])
    s = result[0]["state"]
    s[tool.COMPLETED] = {
        "earlier_periods_this_repetition": {"buy": 2, "sell": 1},
        "previous_repetitions": {"buy": 0, "sell": 0},
        "last_completed_period_with_own_orders": {"buy": 2, "sell": 1},
    }
    return s


def records(prefix="train", n=24):
    result = []
    for i in range(n):
        s = state()
        s["cash"] += 31 * i
        s[tool.CURRENT] = {"buy": i % 5, "sell": (i + 2) % 7}
        s[tool.COMPLETED]["earlier_periods_this_repetition"] = {"buy": i % 3, "sell": i % 4}
        earlier = s[tool.COMPLETED]["earlier_periods_this_repetition"]
        s[tool.COMPLETED]["last_completed_period_with_own_orders"] = (
            dict(earlier) if sum(earlier.values()) else None
        )
        result.append(
            {
                "provenance": {
                    "session": f"{prefix}-s{i % 3}",
                    "participant": f"{prefix}-p{i % 6}",
                    "order_id": f"q{i}",
                },
                "state": s,
                "target": {"buy": int(i % 4 in (0, 1))},
            }
        )
    return result


def test_exact_nested_features_and_no_unmasked_derived_aliases():
    s = state()
    names = list(tool.ARMS)
    vectors = [tool.features(s, arm) for arm in names]
    assert [len(v) for v in vectors] == [15, 17, 23]
    keep = [i for i, name in enumerate(base.FEATURES) if name not in tool.CURRENT_FEATURES]
    np.testing.assert_array_equal(vectors[0], vectors[1][keep])
    np.testing.assert_array_equal(vectors[1], vectors[2][:17])
    s["derived_memory_alias"] = {"buy": 9999}
    s["target"] = {"buy": 1}
    s["order_id"] = "future-target"
    for arm, before in zip(names, vectors, strict=True):
        np.testing.assert_array_equal(tool.features(s, arm), before)
        assert not {"derived_memory_alias", "target", "order_id"} & set(tool.project(s, arm))
    assert tool.CURRENT not in tool.project(s, names[0])
    assert tool.COMPLETED not in tool.project(s, names[1])


def test_own_history_changes_only_the_declared_feature_bundles():
    before = state()
    before["repetition"] = 2
    current = copy.deepcopy(before)
    current[tool.CURRENT] = {"buy": 0, "sell": 30}
    completed = copy.deepcopy(before)
    completed[tool.COMPLETED]["previous_repetitions"] = {"buy": 30, "sell": 0}
    a15, a17, a23 = tool.ARMS
    np.testing.assert_array_equal(tool.features(before, a15), tool.features(current, a15))
    assert not np.array_equal(tool.features(before, a17), tool.features(current, a17))
    np.testing.assert_array_equal(tool.features(before, a17), tool.features(completed, a17))
    assert not np.array_equal(tool.features(before, a23), tool.features(completed, a23))


def test_absent_history_is_not_silently_zero_filled():
    s = state()
    del s[tool.CURRENT]
    assert len(tool.features(s, "account_public_task_15")) == 15
    with pytest.raises(ValueError, match="missing current history"):
        tool.features(s, "plus_current_period_17")
    s = state()
    del s[tool.COMPLETED]["last_completed_period_with_own_orders"]
    with pytest.raises(ValueError, match="missing last-period"):
        tool.features(s, "plus_completed_history_23")


@pytest.mark.parametrize("invalid", [None, -1, True, 2.0, "2"])
def test_unknown_or_invalid_history_counts_fail(invalid):
    s = state()
    s[tool.CURRENT]["buy"] = invalid
    with pytest.raises(ValueError, match="history counts"):
        tool.features(s, "plus_current_period_17")


def test_known_empty_completed_history_has_explicit_laplace_defaults():
    s = state()
    s[tool.COMPLETED] = {
        "earlier_periods_this_repetition": {"buy": 0, "sell": 0},
        "previous_repetitions": {"buy": 0, "sell": 0},
        "last_completed_period_with_own_orders": None,
    }
    np.testing.assert_array_equal(
        tool.features(s, "plus_completed_history_23")[-6:], [0.5, 0, 0.5, 0, 0.5, 0]
    )


@pytest.mark.parametrize(
    "earlier,previous,last",
    [
        ({"buy": 2, "sell": 1}, {"buy": 0, "sell": 0}, None),
        ({"buy": 0, "sell": 0}, {"buy": 0, "sell": 0}, {"buy": 2, "sell": 1}),
        ({"buy": 1, "sell": 1}, {"buy": 0, "sell": 0}, {"buy": 2, "sell": 0}),
        ({"buy": 0, "sell": 1}, {"buy": 3, "sell": 2}, {"buy": 2, "sell": 0}),
        ({"buy": 2, "sell": 1}, {"buy": 0, "sell": 0}, {"buy": 0, "sell": 0}),
    ],
)
def test_contradictory_completed_history_fails_before_fitting(earlier, previous, last):
    s = state()
    s["repetition"] = 2
    s[tool.COMPLETED] = {
        "earlier_periods_this_repetition": earlier,
        "previous_repetitions": previous,
        "last_completed_period_with_own_orders": last,
    }
    with pytest.raises(ValueError, match="inconsistent completed history"):
        tool.features(s, "plus_completed_history_23")


def test_last_active_period_can_belong_to_a_previous_repetition():
    s = state()
    s["repetition"] = 2
    s[tool.COMPLETED] = {
        "earlier_periods_this_repetition": {"buy": 0, "sell": 0},
        "previous_repetitions": {"buy": 3, "sell": 2},
        "last_completed_period_with_own_orders": {"buy": 1, "sell": 2},
    }
    assert tool.project(s, "plus_completed_history_23")[tool.COMPLETED] == s[tool.COMPLETED]


def test_current_target_and_future_do_not_change_predecision_features():
    frames = fixture()
    before, _ = quote_states(*frames, selected_ids=["target"])
    changed = copy.deepcopy(frames)
    changed[0].loc[changed[0].created >= 5, ["side", "quantity", "price"]] = ["bid", 99, 999]
    changed[1].loc[changed[1].created >= 5, ["quantity", "price"]] = [99, 999]
    changed[2][["end_cash", "end_assets"]] = -999
    changed[3].loc[changed[3].period >= 2, ["reported_average", "reported_close", "dividend"]] = 999
    after, _ = quote_states(*changed, selected_ids=["target"])
    assert before[0]["target"] != after[0]["target"]
    for rows in (before, after):
        rows[0]["state"][tool.COMPLETED] = state()[tool.COMPLETED]
    fitted = {arm: tool.fit(records(), arm) for arm in tool.ARMS}
    for arm in tool.ARMS:
        np.testing.assert_array_equal(
            tool.features(before[0]["state"], arm), tool.features(after[0]["state"], arm)
        )
        np.testing.assert_array_equal(
            tool.predict(fitted[arm], before), tool.predict(fitted[arm], after)
        )


def test_same_second_actions_are_excluded_instead_of_ordered_arbitrarily():
    frames = fixture()
    extra = frames[0].iloc[[1]].copy()
    extra["uuid"] = "same-second-peer"
    extra["player"] = 2
    frames[0] = pd.concat([frames[0], extra], ignore_index=True)
    result, exclusions = quote_states(*frames, selected_ids=["target"])
    assert result == [] and exclusions == {"multiple_submissions_in_target_second": 1}


def test_reset_retains_only_declared_past_and_never_peer_or_other_session_history():
    target, orders, ledger = example()
    target[0]["state"] = {**state(), "cash": 3000, "shares": 20, "repetition": 2}
    original = extend_memory(target, orders, ledger)
    other_orders, other_ledger = orders.copy(), ledger.copy()
    other_orders["session"] = other_ledger["session"] = "unrelated-session"
    other_orders["side"] = "bid"
    changed = orders.copy()
    changed.loc[
        changed.uuid.isin(["target", "future_current", "future_period", "peer"]), "side"
    ] = "bid"
    after = extend_memory(
        target, pd.concat([changed, other_orders]), pd.concat([ledger, other_ledger])
    )
    assert original == after
    s = after[0]["state"]
    assert s["cash"] == 3000 and s["shares"] == 20
    assert s[tool.COMPLETED]["previous_repetitions"] == {"buy": 1, "sell": 0}
    changed.loc[changed.uuid.eq("rep1"), "side"] = "ask"
    altered = extend_memory(target, changed, ledger)[0]["state"]
    for arm in list(tool.ARMS)[:2]:
        np.testing.assert_array_equal(tool.features(s, arm), tool.features(altered, arm))
    assert not np.array_equal(
        tool.features(s, "plus_completed_history_23"),
        tool.features(altered, "plus_completed_history_23"),
    )


def test_each_arm_refits_its_own_dimensions_and_matches_existing_reference_fits():
    train = records()
    for arm, reference in [
        ("plus_current_period_17", base),
        ("plus_completed_history_23", extended),
    ]:
        model = tool.fit(train, arm)
        expected = reference.fit(train)
        for key in ("mean", "scale", "beta"):
            np.testing.assert_allclose(model[key], expected[key], rtol=1e-7, atol=1e-9)
    reduced = tool.fit(train, "account_public_task_15")
    assert len(reduced["mean"]) == 15 and len(reduced["beta"]) == 16


def test_evaluation_features_and_labels_do_not_fit_scalers_constants_or_coefficients():
    train, evaluation = records(), records("evaluation", 8)
    original_train = copy.deepcopy(train)
    models, _ = tool.split_predictions(train, evaluation)
    for row in evaluation:
        row["target"]["buy"] = 1 - row["target"]["buy"]
        row["state"]["cash"] *= 1000
        row["state"][tool.CURRENT]["buy"] += 30
    after, predictions = tool.split_predictions(train, evaluation)
    assert models == after and train == original_train
    assert len(predictions) == 7 and all(len(p) == 8 for p in predictions.values())
    for values in predictions.values():
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


@pytest.mark.parametrize("field", ["session", "participant"])
def test_people_and_repetitions_cannot_cross_training_and_evaluation(field):
    train, evaluation = records(), records("evaluation", 8)
    evaluation[0]["provenance"][field] = train[0]["provenance"][field]
    with pytest.raises(ValueError, match=field + " crossing"):
        tool.split_predictions(train, evaluation)


def test_bad_coverage_or_fitting_failure_never_drops_hard_targets(monkeypatch):
    train = records()
    with pytest.raises(ValueError, match="duplicate targets"):
        tool.fit(train + [copy.deepcopy(train[0])], "account_public_task_15")
    train[0]["state"]["cash"] = float("nan")
    with pytest.raises(ValueError, match="invalid feature vector"):
        tool.fit(train, "account_public_task_15")

    class Failure:
        success = False

    monkeypatch.setattr(tool, "minimize", lambda *_a, **_k: Failure())
    with pytest.raises(ValueError, match="fit failed"):
        tool.fit(records(), "account_public_task_15")


def test_weights_balance_sessions_then_people_not_folds_or_number_of_orders():
    sample = [
        {"provenance": {"session": s, "participant": p}, "target": {"buy": 0}}
        for s, p in [("s1", "a"), ("s1", "a"), ("s1", "b"), ("s2", "c")]
    ]
    np.testing.assert_array_equal(base.weights(sample), [0.125, 0.125, 0.25, 0.5])
