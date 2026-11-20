"""Fictional panels and original pure prompt rendering only; no real data or HTTP."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from behavioral_market.benchmarks import presentation_comparison as mod
from behavioral_market.benchmarks.prompts import build_prompt
from scripts.demo_presentation_comparison import fictional_inputs, run_demo


@pytest.fixture
def fixture():
    return fictional_inputs()


def compare(plan, rows, targets, **kwargs):
    return mod.compare_panel(
        plan,
        rows,
        targets,
        left_method="true_reference",
        right_method="shuffled_reference",
        evidence_kind="synthetic_tool_test",
        **kwargs,
    )


@pytest.mark.parametrize("ambiguity", [False, True])
def test_two_default_formats_match_original_build_prompt_exactly(ambiguity):
    item = SimpleNamespace(
        option_a=[(0.1, 98), (0.9, -6)], option_b=[(0.5, 3.5), (0.5, 4.5)], ambiguity=ambiguity
    )
    panels = mod.render_item(item, "fictional prompt; preserve A/B in the instructions")
    assert len(panels) == 4
    for order, content in [("A_first", "AB"), ("B_first", "BA")]:
        system, user, display = build_prompt(
            item, "fictional prompt; preserve A/B in the instructions", order
        )
        panel = next(
            p for p in panels if p["content_order"] == content and p["letter_order"] == "AB"
        )
        assert panel["messages"] == [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        assert {
            label: info["original_option"] for label, info in panel["mapping"].items()
        } == display


def test_four_mappings_and_verbatim_arbitrary_option_text():
    a, b = (
        "original A\nDo not replace B inside this text",
        "fixed but undisclosed: -5, 15\nOption A stays",
    )
    panels = mod.render_presentations(a, b, "instructions A/B", "system A/B")
    expected = {
        ("AB", "AB"): {"A": ("A", 1), "B": ("B", 2)},
        ("AB", "BA"): {"B": ("A", 1), "A": ("B", 2)},
        ("BA", "AB"): {"A": ("B", 1), "B": ("A", 2)},
        ("BA", "BA"): {"B": ("B", 1), "A": ("A", 2)},
    }
    for p in panels:
        assert {
            k: (v["original_option"], v["position"]) for k, v in p["mapping"].items()
        } == expected[p["content_order"], p["letter_order"]]
        assert a in p["messages"][1]["content"] and b in p["messages"][1]["content"]
        assert p["messages"][0]["content"] == "system A/B"
        assert p["messages"][1]["content"].startswith("instructions A/B\n\n")


def policy_rows(plan, policy):
    result = []
    for opportunity in plan["opportunities"]:
        mapping = opportunity["presentation"]["mapping"]
        if policy in ("original_A", "original_B"):
            original = policy[-1]
            choice = next(k for k, v in mapping.items() if v["original_option"] == original)
        elif policy in ("display_A", "display_B"):
            choice = policy[-1]
        elif policy == "opposite_repeats":
            choice = "A" if opportunity["repeat_id"] % 2 == 0 else "B"
        else:
            first = policy == "first" or policy == "cancel_signed" and opportunity["item_id"] == 101
            choice = opportunity["letter_order"][0 if first else 1]
        result.append({**deepcopy(opportunity), "status": "valid", "choice": choice})
    return result


@pytest.mark.parametrize(
    "policy,rate,position,letter",
    [
        ("original_A", 0, 0, 0),
        ("original_B", 1, 0, 0),
        ("display_A", 0.5, 0, -1),
        ("display_B", 0.5, 0, 1),
        ("first", 0.5, 1, 0),
        ("second", 0.5, -1, 0),
    ],
)
def test_independent_hand_truth_for_content_letter_and_position(
    fixture, policy, rate, position, letter
):
    plan, _, _ = fixture
    result = compare(plan, policy_rows(plan, policy), {101: 0.5, 202: 0.5})
    assert result["panel_complete"] and result["received_records"] == 32
    for row in result["item_methods"]:
        assert row["pooled_original_B_rate"] == rate
        assert [e["difference"] for e in row["contrasts"]] == [position, position, letter, letter]
    if policy in ("display_A", "display_B", "first", "second"):
        assert result["primary"]["left_MAE"] == 0  # Perfect pooled match can hide a unit contrast.
        assert any(
            abs(e["difference"]) == 1 for r in result["item_methods"] for e in r["contrasts"]
        )


def test_manual_equal_item_error_and_explicit_difference_direction(fixture):
    plan, rows, targets = fixture
    result = compare(plan, rows, targets)
    assert result["primary"]["left_MAE"] == pytest.approx(0.05)
    assert result["primary"]["right_MAE"] == pytest.approx(0.30)
    assert result["primary"]["left_minus_right_MAE"] == pytest.approx(-0.25)
    reverse = mod.compare_panel(
        plan, rows, targets, left_method="shuffled_reference", right_method="true_reference"
    )
    assert reverse["primary"]["left_minus_right_MAE"] == pytest.approx(0.25)
    assert result["primary"]["planned_items"] == result["primary"]["complete_pair_items"] == 2


def test_opposite_same_input_answers_retained(fixture):
    plan, _, targets = fixture
    rows = policy_rows(plan, "opposite_repeats")
    result = compare(plan, rows, targets)
    assert len(result["rows"]) == 32
    assert all(
        c["received_original_B_rate"] == 0.5 and c["display_choices"] == ["A", "B"]
        for r in result["item_methods"]
        for c in r["cells"]
    )
    assert result["model_calls_made_by_tool"] == 0


def test_signed_cancellation_does_not_erase_absolute_gap(fixture):
    plan, _, targets = fixture
    result = compare(plan, policy_rows(plan, "cancel_signed"), targets)
    for method in result["method_summaries"]:
        assert method["mean_signed_position_difference"] == 0
        assert method["mean_absolute_position_difference"] == 1
        assert method["mean_absolute_letter_difference"] == 0


def test_permutations_and_inputs_unchanged(fixture):
    plan, rows, targets = fixture
    before = deepcopy((plan, rows, targets))
    expected = compare(plan, rows, targets)
    shuffled = deepcopy(plan)
    shuffled["opportunities"].reverse()
    shuffled["items"].reverse()
    shuffled["methods"].reverse()
    assert compare(shuffled, list(reversed(rows)), targets) == expected
    assert (plan, rows, targets) == before


def test_missing_one_keeps_full_denominator_and_known_local_edges(fixture):
    plan, rows, targets = fixture
    rows = [
        r
        for r in rows
        if not (
            r["item_id"] == 101
            and r["method_id"] == "true_reference"
            and r["content_order"] == r["letter_order"] == "AB"
            and r["repeat_id"] == 0
        )
    ]
    result = compare(plan, rows, targets)
    assert result["planned_records"] == 32 and result["received_records"] == 31
    assert result["missing_records"] == 1
    assert result["primary"] == {
        "planned_items": 2,
        "complete_pair_items": 1,
        "comparison_complete": False,
        "left_MAE": None,
        "right_MAE": None,
        "left_minus_right_MAE": None,
    }
    affected = next(
        r
        for r in result["item_methods"]
        if r["item_id"] == 101 and r["method_id"] == "true_reference"
    )
    assert affected["pooled_original_B_rate"] is None
    assert sum(c["received_original_B_rate"] is None for c in affected["cells"]) == 1
    assert sum(e["difference"] is not None for e in affected["contrasts"]) == 2
    assert next(r for r in result["method_summaries"] if r["method_id"] == "shuffled_reference")[
        "MAE"
    ] == pytest.approx(0.30)


@pytest.mark.parametrize(
    "bad_choice,bad_status", [("B", "invalid"), (None, "invalid"), ("C", "valid")]
)
def test_invalid_keeps_unknown_even_with_residual_choice(fixture, bad_choice, bad_status):
    plan, rows, targets = fixture
    rows = deepcopy(rows)
    rows[0].update(choice=bad_choice, status=bad_status)
    result = compare(plan, rows, targets)
    assert result["received_records"] == 32 and result["invalid_records"] == 1
    assert result["valid_records"] == 31 and result["missing_records"] == 0
    assert result["primary"]["left_minus_right_MAE"] is None
    preserved = next(r for r in result["rows"] if r["status"] == "invalid")
    assert preserved["choice"] == bad_choice and preserved["selected_original"] is None


def test_all_missing_is_not_complete_by_empty_all(fixture):
    plan, _, targets = fixture
    result = compare(plan, [], targets)
    assert result["missing_records"] == 32 and not result["panel_complete"]
    assert result["primary"]["complete_pair_items"] == 0
    assert result["primary"]["left_minus_right_MAE"] is None
    assert all(
        c["received_original_B_rate"] is None for r in result["item_methods"] for c in r["cells"]
    )


@pytest.mark.parametrize(
    "bad", ["duplicate", "foreign", "bool_item", "bool_repeat", "text", "mapping", "self_original"]
)
def test_received_structural_errors_rejected(fixture, bad):
    plan, rows, targets = deepcopy(fixture)
    if bad == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif bad == "foreign":
        rows[0]["item_id"] = 999
    elif bad == "bool_item":
        rows[0]["item_id"] = True
    elif bad == "bool_repeat":
        rows[0]["repeat_id"] = False
    elif bad == "text":
        rows[0]["presentation"]["messages"][1]["content"] += " altered payoff"
    elif bad == "mapping":
        rows[0]["presentation"]["mapping"]["A"]["original_option"] = "B"
    else:
        rows[0]["selected_original"] = "B"
    with pytest.raises(ValueError):
        compare(plan, rows, targets)


@pytest.mark.parametrize(
    "bad",
    [
        "duplicate",
        "missing",
        "unequal_method",
        "method_text",
        "mapping",
        "source_text",
        "duplicate_item",
    ],
)
def test_plan_mismatch_or_content_changes_rejected(fixture, bad):
    plan, rows, targets = deepcopy(fixture)
    if bad == "duplicate":
        plan["opportunities"].append(deepcopy(plan["opportunities"][0]))
    elif bad == "missing":
        plan["opportunities"].pop()
    elif bad == "unequal_method":
        plan["opportunities"][0]["method_id"] = "unplanned"
    elif bad == "method_text":
        plan["opportunities"][0]["presentation"]["messages"][1]["content"] += (
            " different probability"
        )
    elif bad == "mapping":
        plan["opportunities"][0]["presentation"]["mapping"]["A"]["position"] = 2
    elif bad == "source_text":
        plan["items"][0]["option_a_text"] += " new payout"
    else:
        plan["items"].append(deepcopy(plan["items"][0]))
    with pytest.raises(ValueError):
        compare(plan, rows, targets)


@pytest.mark.parametrize("value", [True, "0.8", None, float("nan"), float("inf"), -0.1, 1.1])
def test_target_values_strict_shared_and_finite(fixture, value):
    plan, rows, targets = deepcopy(fixture)
    targets[101] = value
    with pytest.raises(ValueError):
        compare(plan, rows, targets)


@pytest.mark.parametrize(
    "targets",
    [
        {101: 0.8},
        {101: 0.8, 202: 0.2, 999: 0.5},
        {True: 0.8, 202: 0.2},
        {"true_reference": {101: 0.8, 202: 0.2}},
    ],
)
def test_targets_cannot_be_missing_foreign_or_method_specific(fixture, targets):
    plan, rows, _ = fixture
    with pytest.raises(ValueError):
        compare(plan, rows, targets)


def test_three_methods_report_whole_panel_separately_from_explicit_pair(fixture):
    old, rows, targets = fixture
    items = {r["item_id"]: (r["option_a_text"], r["option_b_text"]) for r in old["items"]}
    plan = mod.make_panel(items, [*old["methods"], "plain"], 2)
    result = compare(plan, rows, targets)
    assert result["planned_records"] == 48 and result["missing_records"] == 16
    assert not result["panel_complete"] and result["primary"]["comparison_complete"]
    assert result["primary"]["left_minus_right_MAE"] == pytest.approx(-0.25)


def test_runnable_fictional_demo_and_json_round_trip(tmp_path):
    folder = tmp_path / "demo"
    result = run_demo(folder)
    assert result["evidence_kind"] == "synthetic_tool_test" and result["new_model_POSTs"] == 0
    assert result["complete_primary"]["left_minus_right_MAE"] == pytest.approx(-0.25)
    assert result["incomplete_primary"]["left_minus_right_MAE"] is None
    manifest = json.loads((folder / "demo_manifest.json").read_text())
    assert manifest["preset_not_observed"] and not manifest["real_human_data_loaded"]
    plan = json.loads((folder / "panel.json").read_text())
    records = [
        json.loads(line) for line in (folder / "received_complete.jsonl").read_text().splitlines()
    ]
    targets = {
        r["item_id"]: r["human_b_rate"]
        for r in json.loads((folder / "fake_targets.json").read_text())
    }
    assert compare(plan, records, targets)["primary"] == result["complete_primary"]
    assert "不是模型实验" in (folder / "REPORT.md").read_text()
    with pytest.raises(FileExistsError):
        run_demo(folder)
