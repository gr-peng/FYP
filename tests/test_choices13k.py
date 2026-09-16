import json

import pandas as pd

from behavioral_market.benchmarks.choices13k import items_frame, load_official
from behavioral_market.benchmarks.evaluation import aggregate_responses, prompt_metrics
from behavioral_market.benchmarks.prompts import build_prompt
from behavioral_market.benchmarks.sampling import orientations, stratified_split


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
                }
            )
    rates = aggregate_responses(pd.DataFrame(responses), items, 10)
    metrics = prompt_metrics(rates)
    assert len(rates) == 2
    assert metrics["mae"] < 1e-12
    assert metrics["invalid_rate"] == 0
    assert metrics["first_option_rate"] == 0.5
