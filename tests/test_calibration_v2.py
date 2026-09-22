import math
from collections import Counter

import pandas as pd
import pytest
from test_choices13k_data_driven import make_item

from experiments.calibration_v2 import (
    METHODS,
    build_prompt_v2,
    check_matrix,
    fit_thresholds,
    jobs_for,
    label_permutation,
    observable_features,
    retrieve,
    visible_fingerprint,
)


def pool_items():
    return [make_item(i, (i % 10) / 10, ambiguity=bool(i % 2)) for i in range(1, 41)]


def test_hidden_probabilities_and_latent_statistics_cannot_change_retrieval_or_prompt():
    pool = pool_items()
    target = make_item(100, 0.123, ambiguity=True)
    altered = target.model_copy(
        update={
            "option_b": [(0.99, 120.0), (0.01, 1.0)],
            "ev_a": -1000.0,
            "ev_b": 1e9,
            "var_b": 1e12,
            "n_humans": 999,
            "human_b_rate": 0.987,
            "correlation": -1,
        }
    )
    assert visible_fingerprint(target) == visible_fingerprint(altered)
    assert observable_features(target) == observable_features(altered)
    thresholds = fit_thresholds(pool)
    first = retrieve(pool, target, thresholds, 9)
    second = retrieve(pool, altered, thresholds, 9)
    assert [x.item_id for x in first] == [x.item_id for x in second]
    lookup = {x.item_id: x for x in pool}
    permutation = label_permutation(pool, 9)
    for method in METHODS:
        assert build_prompt_v2(target, first, method, "A_first", lookup, permutation, 9) == (
            build_prompt_v2(altered, second, method, "A_first", lookup, permutation, 9)
        )


def test_hidden_probability_zero_does_not_remove_displayed_payoff():
    original = make_item(10, ambiguity=True)
    altered = original.model_copy(update={"option_b": [(1.0, 30.0), (0.0, 1.0)]})
    assert visible_fingerprint(original) == visible_fingerprint(altered)
    assert observable_features(original) == observable_features(altered)


def test_all_pool_labels_and_nonobservable_metadata_do_not_affect_retrieval():
    pool = pool_items()
    modified = [
        x.model_copy(update={"human_b_rate": 1 - x.human_b_rate, "n_humans": 500, "problem_id": -1})
        for x in pool
    ]
    target = make_item(100)
    assert fit_thresholds(pool) == fit_thresholds(modified)
    for strategy in ("matched", "random"):
        first = retrieve(pool, target, fit_thresholds(pool), 12, strategy)
        second = retrieve(modified, target, fit_thresholds(modified), 12, strategy)
        assert [x.item_id for x in first] == [x.item_id for x in second]


def test_fingerprint_handles_swap_split_outcomes_and_opaque_ids():
    first = make_item(10)
    changed = first.model_copy(
        update={
            "item_id": 999,
            "problem_id": 999,
            "option_a": first.option_b,
            "option_b": [(0.25, 20.0), (0.25, 20.0), (0.5, 2.0), (0.0, 99.0)],
        }
    )
    assert visible_fingerprint(first) == visible_fingerprint(changed)
    target = make_item(100)
    target_clone = target.model_copy(update={"item_id": 999})
    pool = [target_clone, *pool_items()]
    assert 999 not in [x.item_id for x in retrieve(pool, target, fit_thresholds(pool), 9)]


def test_permutation_preserves_label_multiset_without_fixed_ids():
    pool = pool_items()
    lookup = {x.item_id: x for x in pool}
    permutation = label_permutation(pool, 17)
    assert all(int(key) != value for key, value in permutation.items())
    assert Counter(x.human_b_rate for x in pool) == Counter(
        lookup[i].human_b_rate for i in permutation.values()
    )


def test_neutral_prompt_is_invariant_to_all_training_labels():
    pool = pool_items()
    target = make_item(100)
    changed = [x.model_copy(update={"human_b_rate": 1 - x.human_b_rate}) for x in pool]
    first = build_prompt_v2(
        target,
        pool[:4],
        "C_neutral",
        "A_first",
        {x.item_id: x for x in pool},
        label_permutation(pool, 9),
        9,
    )
    second = build_prompt_v2(
        target,
        changed[:4],
        "C_neutral",
        "A_first",
        {x.item_id: x for x in changed},
        label_permutation(changed, 9),
        9,
    )
    assert first == second


def test_response_matrix_checks_identity_and_counterbalance_on_resume():
    plan = {"target_ids": [1, 2, 3]}
    jobs = jobs_for(plan, 10, 8)
    assert len(jobs) == 150
    assert all(
        Counter(o for m, i, r, o in jobs if i == item and m == method)
        == {"A_first": 5, "B_first": 5}
        for item in plan["target_ids"]
        for method in METHODS
    )
    rows = [
        {"prompt_id": m, "item_id": i, "repeat_id": r, "display_order": o} for m, i, r, o in jobs
    ]
    assert len(check_matrix(pd.DataFrame(rows), jobs, complete=True)) == 150
    with pytest.raises(RuntimeError, match="incomplete"):
        check_matrix(pd.DataFrame(rows[:-1]), jobs, complete=True)
    with pytest.raises(RuntimeError, match="identities"):
        check_matrix(pd.DataFrame([*rows, rows[0]]), jobs)
    rows[0]["display_order"] = "B_first" if rows[0]["display_order"] == "A_first" else "A_first"
    with pytest.raises(RuntimeError, match="counterbalance"):
        check_matrix(pd.DataFrame(rows), jobs)


def test_probability_and_sampled_baselines_are_distinct():
    # A finite action sampler has error even when the true probability is exactly 0.5.
    expected_error = sum(math.comb(10, k) / 1024 * abs(k / 10 - 0.5) for k in range(11))
    assert expected_error == pytest.approx(0.123046875)
