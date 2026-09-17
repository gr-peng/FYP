import json
from copy import deepcopy

import pandas as pd
import pytest

from behavioral_market.benchmarks.choices13k import items_frame, load_official
from behavioral_market.benchmarks.confirmatory import validate_confirmatory_config
from behavioral_market.benchmarks.evaluation import aggregate_responses, bootstrap, prompt_metrics
from behavioral_market.benchmarks.prompts import build_prompt
from behavioral_market.benchmarks.sampling import (
    confirmatory_split,
    orientations,
    stratified_split,
)


def fixture_data(tmp_path):
    csv_path = tmp_path / "c13k_selections.csv"
    csv_path.write_text(
        "Problem,Feedback,n,Block,Ha,pHa,La,Hb,pHb,Lb,LotShapeB,LotNumB,Amb,Corr,bRate,bRate_std\n"
        "7,True,15,2,26,.95,-1,23,.05,21,0,1,False,0,.6,.1\n"
        "7,False,12,1,14,.6,-18,8,.25,-5,1,2,True,0,.3,.2\n"
    )
    problems_path = tmp_path / "c13k_problems.json"
    problems_path.write_text(
        json.dumps(
            {
                "0": {"A": [[0.95, 26], [0.05, -1]], "B": [[0.95, 21], [0.05, 23]]},
                "1": {"A": [[0.6, 14], [0.4, -18]], "B": [[0.75, -5], [0.25, 8]]},
            }
        )
    )
    return csv_path, problems_path


def test_parser_uses_row_identity_and_derives_analysis_features(tmp_path):
    items = load_official(*fixture_data(tmp_path))
    assert [item.item_id for item in items] == [0, 1]
    assert [item.problem_id for item in items] == [7, 7]
    assert items[0].has_loss and items[0].ev_a == 24.65
    assert items[1].ambiguity and items[1].feedback is False


def test_prompt_counterbalances_and_hides_labels_and_ambiguous_probabilities(tmp_path):
    item = load_official(*fixture_data(tmp_path))[1]
    _, user, display = build_prompt(item, "Choose naturally.", "B_first")
    assert display == {"A": "B", "B": "A"}
    assert "undisclosed probabilities" in user
    assert "75%" not in user and "25%" not in user
    for forbidden in ["human_b_rate", "bRate", "expected value", "riskier"]:
        assert forbidden not in user
    values = orientations(item.item_id, 10, 20260916)
    assert values.count("A_first") == values.count("B_first") == 5
    assert values == orientations(item.item_id, 10, 20260916)


def test_stratified_split_is_deterministic_and_disjoint():
    rows = []
    for item_id in range(96):
        rows.append(
            {
                "item_id": item_id,
                "feedback": bool(item_id & 1),
                "ambiguity": bool(item_id & 2),
                "has_loss": bool(item_id & 4),
                "ev_gap": float(item_id + 1),
            }
        )
    frame = pd.DataFrame(rows)
    first = stratified_split(frame, 40, 20, 20260916)
    second = stratified_split(frame, 40, 20, 20260916)
    assert [len(value) for value in first] == [40, 20, 60]
    assert set(first[0].item_id).isdisjoint(first[1].item_id)
    for left, right in zip(first, second, strict=True):
        assert left.item_id.tolist() == right.item_id.tolist()


def test_item_level_metrics_do_not_treat_repeats_as_independent(tmp_path):
    items = items_frame(load_official(*fixture_data(tmp_path)))
    responses = []
    for item_id in [0, 1]:
        for repeat_id in range(10):
            responses.append(
                {
                    "split": "heldout",
                    "prompt_id": "P0",
                    "item_id": item_id,
                    "repeat_id": repeat_id,
                    "status": "ok",
                    "original_b_choice": repeat_id < (6 if item_id == 0 else 3),
                    "first_display_selected": repeat_id % 2 == 0,
                    "display_order": "A_first" if repeat_id < 5 else "B_first",
                }
            )
    rates = aggregate_responses(pd.DataFrame(responses), items, 10)
    metrics = prompt_metrics(rates)
    assert len(rates) == 2
    assert metrics["mae"] < 1e-12
    assert metrics["invalid_rate"] == 0
    assert metrics["first_option_rate"] == 0.5


def confirmatory_frame():
    rows = []
    for item_id in range(240):
        rows.append(
            {
                "item_id": item_id,
                "feedback": item_id % 5 == 0,
                "ambiguity": bool(item_id & 1),
                "has_loss": bool(item_id & 2),
                "ev_gap": float(item_id + 1),
                "human_b_rate": (item_id % 11) / 10,
            }
        )
    return pd.DataFrame(rows)


def test_confirmatory_split_excludes_pilot_items_and_uses_no_feedback():
    frame = confirmatory_frame()
    excluded = set(range(30))
    result = confirmatory_split(frame, excluded, 100, 20260917)
    assert len(result) == 100
    assert set(result.item_id).isdisjoint(excluded)
    assert not result.feedback.any()


def test_confirmatory_split_is_frozen_and_does_not_use_human_target():
    frame = confirmatory_frame()
    changed_targets = frame.copy()
    changed_targets["human_b_rate"] = 1 - changed_targets.human_b_rate
    first = confirmatory_split(frame, range(30), 100, 20260917)
    second = confirmatory_split(changed_targets, range(30), 100, 20260917)
    assert first.item_id.tolist() == second.item_id.tolist()
    repeated = confirmatory_split(frame, range(30), 100, 20260917)
    assert first.item_id.tolist() == repeated.item_id.tolist()


def frozen_config(prompt_path):
    return {
        "experiment": {"seed": 20260917},
        "dataset": {
            "confirmatory_items": 1000,
            "primary_feedback": False,
            "ev_gap_quantiles": 3,
            "confirmatory_split_sha256": "a" * 64,
        },
        "sampling": {"repeats_per_item": 10, "counterbalance_option_order": True},
        "prompts": {
            "file": str(prompt_path),
            "candidate_ids": ["P0", "P1"],
            "vanilla_id": "P0",
            "treatment_id": "P1",
        },
        "llm": {"temperature": 0.7, "enable_thinking": False},
        "evaluation": {"primary_metric": "mae", "bootstrap_samples": 5000},
    }


def test_only_p0_p1_and_other_confirmatory_gates_are_allowed(tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "version": "test",
                "P0": "Choose the option you prefer.",
                "P1": (
                    "Respond as an ordinary adult participant making the choice for themselves."
                ),
            }
        )
    )
    config = frozen_config(prompt_path)
    validate_confirmatory_config(config, tmp_path)
    for mutate in [
        lambda value: value["prompts"].update(candidate_ids=["P0", "P1", "P2"]),
        lambda value: value["dataset"].update(primary_feedback=True),
        lambda value: value["sampling"].update(repeats_per_item=8),
        lambda value: value["llm"].update(temperature=0.8),
    ]:
        changed = deepcopy(config)
        mutate(changed)
        with pytest.raises(ValueError, match="protocol drift"):
            validate_confirmatory_config(changed, tmp_path)


def test_confirmatory_counterbalance_is_exact():
    for item_id in range(25):
        order = orientations(item_id, 10, 20260917)
        assert order.count("A_first") == order.count("B_first") == 5


def test_position_bias_metric(tmp_path):
    items = items_frame(load_official(*fixture_data(tmp_path)))
    responses = []
    for repeat_id in range(10):
        b_first = repeat_id >= 5
        responses.append(
            {
                "split": "confirmatory",
                "prompt_id": "P0",
                "item_id": 0,
                "repeat_id": repeat_id,
                "status": "ok",
                "display_order": "B_first" if b_first else "A_first",
                "original_b_choice": b_first,
                "first_display_selected": True,
            }
        )
    rates = aggregate_responses(pd.DataFrame(responses), items[items.item_id == 0], 10)
    metrics = prompt_metrics(rates)
    assert metrics["position_bias"] == 0.5
    assert metrics["b_position_effect"] == 1.0


def test_bootstrap_is_item_level(tmp_path):
    items = items_frame(load_official(*fixture_data(tmp_path)))
    responses = []
    for prompt_id in ["P0", "P1"]:
        for item_id in items.item_id:
            for repeat_id in range(10):
                responses.append(
                    {
                        "split": "confirmatory",
                        "prompt_id": prompt_id,
                        "item_id": item_id,
                        "repeat_id": repeat_id,
                        "status": "ok",
                        "display_order": "A_first" if repeat_id < 5 else "B_first",
                        "original_b_choice": repeat_id < (5 + item_id),
                        "first_display_selected": repeat_id % 2 == 0,
                    }
                )
    rates = aggregate_responses(pd.DataFrame(responses), items, 10)
    result = bootstrap(rates, 20, 7, "P0", "P1")
    assert len(result) == 60
    assert set(result.kind) == {"prompt", "paired_delta"}
