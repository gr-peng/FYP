import asyncio
import json

import pandas as pd
import pytest

from behavioral_market.benchmarks.data_driven import (
    run_data_driven_stage,
    select_development_method,
)
from behavioral_market.benchmarks.evaluation import aggregate_responses, bootstrap
from behavioral_market.benchmarks.retrieval import (
    build_few_shot_prompt,
    pseudo_human_choices,
    select_demonstrations,
)
from behavioral_market.benchmarks.sampling import data_driven_splits, orientations
from behavioral_market.benchmarks.schemas import Choices13kItem


def make_item(item_id, human_b_rate=0.5, *, loss=False, ambiguity=False):
    low = -2.0 if loss else 2.0
    return Choices13kItem(
        item_id=item_id,
        problem_id=item_id,
        n_humans=20,
        feedback=False,
        human_b_rate=human_b_rate,
        option_a=[(0.5, 10.0 + item_id), (0.5, low)],
        option_b=[(0.25, 20.0 + item_id), (0.75, 1.0)],
        ambiguity=ambiguity,
        correlation=0,
        block=1,
        has_loss=loss,
        ev_a=6.0,
        ev_b=5.75,
        var_a=16.0,
        var_b=67.0,
    )


def split_frame(count=180):
    rows = []
    for item_id in range(count):
        rows.append(
            {
                "item_id": item_id,
                "feedback": False,
                "ambiguity": bool(item_id & 1),
                "has_loss": bool(item_id & 2),
                "ev_gap": float(item_id + 1),
                "human_b_rate": (item_id % 10) / 10,
            }
        )
    return pd.DataFrame(rows)


def test_data_driven_splits_exclude_all_previous_items():
    excluded = set(range(20))
    splits = data_driven_splits(split_frame(), excluded, 60, 30, 40, 20260918)
    assert all(set(frame.item_id).isdisjoint(excluded) for frame in splits)
    assert all(not frame.feedback.any() for frame in splits)


def test_calibration_dev_test_are_disjoint():
    splits = data_driven_splits(split_frame(), range(20), 60, 30, 40, 20260918)
    ids = [set(frame.item_id) for frame in splits]
    assert [len(frame) for frame in splits] == [60, 30, 40]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))


def test_human_target_not_used_for_retrieval():
    calibration = [make_item(i, i / 20, loss=bool(i & 1)) for i in range(1, 20)]
    target = make_item(100, 0.1, loss=True)
    changed = target.model_copy(update={"human_b_rate": 0.9})
    bins = {item.item_id: ["small", "medium", "large"][item.item_id % 3] for item in calibration}
    bins[target.item_id] = "medium"
    first = select_demonstrations(calibration, target, bins, 8, "matched", 7)
    second = select_demonstrations(calibration, changed, bins, 8, "matched", 7)
    assert [item.item_id for item in first] == [item.item_id for item in second]


def test_target_brate_never_enters_prompt():
    demos = [make_item(i, i / 10) for i in range(1, 5)]
    target = make_item(100, 0.13, ambiguity=True)
    changed = target.model_copy(update={"human_b_rate": 0.87})
    first = build_few_shot_prompt(target, demos, "A_first", 10, 7)
    second = build_few_shot_prompt(changed, demos, "A_first", 10, 7)
    assert first == second
    assert "human_b_rate" not in first[1] and "bRate" not in first[1]


def test_demonstrations_only_come_from_calibration_pool():
    calibration = [make_item(i) for i in range(20)]
    target = make_item(100)
    bins = {item.item_id: "small" for item in [*calibration, target]}
    selected = select_demonstrations(calibration, target, bins, 16, "random", 9)
    assert len({item.item_id for item in selected}) == 16
    assert {item.item_id for item in selected} <= {item.item_id for item in calibration}


def test_final_test_not_read_during_method_selection():
    metrics = {
        "B2_random_k4": {"mae": 0.3},
        "B2_matched_k4": {"mae": 0.2},
    }
    order = ["B2_random_k4", "B2_matched_k4"]
    assert select_development_method(metrics, order) == "B2_matched_k4"


def test_ab_counterbalance_remains_exact():
    for item_id in range(20):
        values = orientations(item_id, 10, 20260918)
        assert values.count("A_first") == values.count("B_first") == 5


def test_method_selection_is_deterministic():
    metrics = {
        "B2_random_k4": {"mae": 0.2},
        "B2_random_k8": {"mae": 0.2},
    }
    order = ["B2_random_k4", "B2_random_k8"]
    assert select_development_method(metrics, order) == "B2_random_k4"
    assert select_development_method(metrics, order) == "B2_random_k4"


def test_data_driven_bootstrap_is_item_level():
    items = pd.DataFrame(
        [
            {
                **make_item(item_id, 0.5).model_dump(exclude={"option_a", "option_b"}),
                "option_a_json": json.dumps(make_item(item_id).option_a),
                "option_b_json": json.dumps(make_item(item_id).option_b),
            }
            for item_id in range(3)
        ]
    )
    responses = []
    for method_id in ["B0", "B2_random_k4"]:
        for item_id in range(3):
            for repeat_id in range(10):
                responses.append(
                    {
                        "split": "final_test",
                        "prompt_id": method_id,
                        "item_id": item_id,
                        "repeat_id": repeat_id,
                        "status": "ok",
                        "display_order": "A_first" if repeat_id < 5 else "B_first",
                        "original_b_choice": repeat_id < 5,
                        "first_display_selected": repeat_id % 2 == 0,
                    }
                )
    rates = aggregate_responses(pd.DataFrame(responses), items, 10)
    result = bootstrap(rates, 10, 5, "B0", "B2_random_k4")
    assert len(result) == 30
    assert result.replicate.nunique() == 10


def test_selected_method_is_frozen_before_final_test(tmp_path):
    root = tmp_path
    (root / "config").mkdir()
    source_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    prompt_source = source_root / "config/choices13k_prompts.json"
    (root / "config/choices13k_prompts.json").write_text(prompt_source.read_text())
    config = json.loads((source_root / "config/choices13k_data_driven.json").read_text())
    with pytest.raises(RuntimeError, match="selected_method.json does not exist"):
        asyncio.run(run_data_driven_stage(config, root, "final_test", provider="mock"))


def test_changing_final_test_human_labels_cannot_change_selected_method():
    metrics = {
        "B2_random_k4": {"mae": 0.19},
        "B2_matched_k4": {"mae": 0.21},
    }
    order = ["B2_random_k4", "B2_matched_k4"]
    original_final_labels = [0.1, 0.8]
    changed_final_labels = [0.9, 0.2]
    assert original_final_labels != changed_final_labels
    assert select_development_method(metrics, order) == "B2_random_k4"


def test_pseudo_human_decisions_are_deterministic():
    item = make_item(5, 0.7)
    first = pseudo_human_choices(item, 10, 20260918)
    assert first == pseudo_human_choices(item, 10, 20260918)
    assert first.count("B") == 7 and first.count("A") == 3
