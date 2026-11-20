import copy
import hashlib
import json

import pytest

from behavioral_market.evaluation.style_diagnostics import (
    bind_archived_answer,
    condition_counts,
    normalize_review,
    summarize_reviews,
)
from scripts.audit_style_switching import load_contexts, verify_answer


@pytest.fixture
def row():
    return {
        "session_date": "2026-06-29",
        "agent_id": "A000",
        "from_style": "technical",
        "target_style": "fundamental",
        "decision": "switch",
        "validation_status": "ok",
        "actual_block_return": -0.02,
        "counterfactual_block_return": 0.01,
        "rationale": "archived answer",
    }


@pytest.fixture
def context():
    return {
        "agent_state": {
            "persona": {"agent_id": "A000"},
            "current_style": "technical",
            "memory": [{"session": "2026-06-29"}],
            "counterfactual": {
                "actual_block_return": -0.02,
                "alternative_block_return": 0.01,
                "gap": 0.03,
                "method": "long_cash_signal_proxy",
            },
        },
        "observation": {
            "session_date": "2026-06-29",
            "phase": "post_close_style_review",
            "decision_at_utc": "2026-06-29T20:00:00+00:00",
            "history_last_session": "2026-06-29",
            "last_close": 100,
            "fundamental_value": 105,
            "fundamental_method": "pre_start_close_anchor_scenario",
            "mean_field": {
                "as_of_session": "2026-06-26",
                "fundamental_share": 0.7,
                "technical_share": 0.3,
            },
        },
    }


def test_features_use_the_matching_predecision_account_and_lagged_population(row, context):
    result = normalize_review(row, "r", 42, "vanilla_llm", context)
    assert result["loss"] is True and result["proxy_better"] is True
    assert result["other_style_majority"] is True
    assert result["context_verified"] is True
    assert result["valuation_gap"] == pytest.approx(0.05)


@pytest.mark.parametrize("as_of", ["2026-06-29", "2026-06-30"])
def test_same_day_or_future_population_cannot_be_counted_as_a_prior_signal(row, context, as_of):
    context["observation"]["mean_field"]["as_of_session"] = as_of
    with pytest.raises(ValueError, match="Population"):
        normalize_review(row, "r", 42, "vanilla_llm", context)


def test_future_personal_history_is_rejected(row, context):
    context["agent_state"]["memory"].append({"session": "2026-06-30"})
    with pytest.raises(ValueError, match="Future memory"):
        normalize_review(row, "r", 42, "vanilla_llm", context)


@pytest.mark.parametrize(
    "field,value", [("history_last_session", "2026-06-30"), ("phase", "pre_open_order")]
)
def test_next_session_or_wrong_phase_cannot_enter_review_inputs(row, context, field, value):
    context["observation"][field] = value
    with pytest.raises(ValueError):
        normalize_review(row, "r", 42, "vanilla_llm", context)


def test_mismatched_agent_is_not_joined_by_date_alone(row, context):
    context["agent_state"]["persona"]["agent_id"] = "A001"
    with pytest.raises(ValueError, match="another agent"):
        normalize_review(row, "r", 42, "vanilla_llm", context)


def test_after_the_fact_replacement_return_is_detected(row, context):
    context["agent_state"]["counterfactual"]["actual_block_return"] = 0.2
    with pytest.raises(ValueError, match="Published return"):
        normalize_review(row, "r", 42, "vanilla_llm", context)


def test_missing_prompt_is_not_reconstructed_from_later_outcomes(row):
    with pytest.raises(ValueError, match="archived context"):
        normalize_review(row, "r", 42, "vanilla_llm")


def test_tied_styles_are_unavailable_for_direction_not_false_majority(row, context):
    context["observation"]["mean_field"].update(fundamental_share=0.5, technical_share=0.5)
    result = normalize_review(row, "r", 42, "vanilla_llm", context)
    summary = condition_counts([result], "other_style_majority")
    assert summary["condition_unavailable_or_tied"] == 1
    assert summary["condition_false"]["n"] == summary["condition_true"]["n"] == 0
    assert summary["condition_false"]["switch_rate"] is None
    assert summary["rate_difference"] is None


def test_corrected_or_default_stay_does_not_count_as_a_valid_llm_choice(row, context):
    row.update(decision="stay", target_style="technical", validation_status="inconsistent_style")
    record = normalize_review(row, "r", 42, "vanilla_llm", context)
    summary = condition_counts([record], "loss")
    assert summary["total_opportunities"] == summary["invalid_choices"] == 1
    assert summary["condition_true"]["n"] == 0


def test_complete_loss_only_group_cannot_be_reported_as_zero_loss_effect(row, context):
    record = normalize_review(row, "r", 42, "vanilla_llm", context)
    summary = summarize_reviews([record], {"r": 1})
    condition = summary["per_treatment"][0]["conditions"]["loss"]
    assert condition["runs_with_both_conditions"] == 0
    assert condition["equal_run_mean_difference"] is None
    assert summary["human_alignment_established"] is False


def test_one_seed_with_no_comparison_is_not_silently_dropped(row, context):
    a = normalize_review(row, "r1", 42, "vanilla_llm", context)
    b = copy.deepcopy(a)
    b.update(
        agent_id="A001",
        loss=False,
        switched=False,
        decision="stay",
        target_style="technical",
        actual_block_return=0.02,
    )
    c = copy.deepcopy(a)
    c.update(run_id="r2", seed=43)
    summary = summarize_reviews([a, b, c], {"r1": 2, "r2": 1})
    condition = summary["per_treatment"][0]["conditions"]["loss"]
    assert condition["runs_with_both_conditions"] == 1
    assert condition["expected_runs"] == 2
    assert condition["equal_run_mean_difference"] is None


@pytest.mark.parametrize("separation", ["review_date", "from_style"])
def test_pooled_loss_effect_does_not_hide_missing_same_date_and_style_controls(
    row,
    context,
    separation,
):
    a = normalize_review(row, "r", 42, "vanilla_llm", context)
    b = copy.deepcopy(a)
    b.update(
        agent_id="A001",
        loss=False,
        switched=False,
        decision="stay",
        target_style="technical",
        actual_block_return=0.02,
    )
    b[separation] = "2026-07-14" if separation == "review_date" else "fundamental"
    b["target_style"] = b["from_style"]
    result = summarize_reviews([a, b], {"r": 2})
    condition = result["per_treatment"][0]["conditions"]["loss"]
    assert condition["rate_difference"] == 1
    assert condition["runs_with_both_conditions"] == 1
    assert condition["same_run_date_style_blocks_with_both_conditions"] == 0
    assert condition["same_run_date_style_blocks"] == 2


def test_duplicate_and_missing_reviews_cannot_be_hidden(row, context):
    a = normalize_review(row, "r", 42, "vanilla_llm", context)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_reviews([a, a], {"r": 2})
    with pytest.raises(ValueError, match="Missing review"):
        summarize_reviews([a], {"r": 2})


def test_raw_model_answer_must_match_the_stored_decision(row):
    requests = {"h": {"key": ("A000", "2026-06-29")}}
    answer = {k: row[k] for k in ["decision", "target_style", "rationale"]}
    assert verify_answer(row, requests, {"h": answer}) == "h"
    answer["decision"] = "stay"
    with pytest.raises(ValueError, match="raw answer"):
        verify_answer(row, requests, {"h": answer})


def test_program_rule_is_labeled_as_programmed_not_human_or_llm(row):
    row["validation_status"] = "rule"
    record = normalize_review(row, "r", 42, "rule_abm")
    assert record["authority"] == "programmed_rule"
    assert record["context_verified"] is False


@pytest.mark.parametrize("treatment", ["vanilla_llm", "low_herding", "high_herding"])
def test_program_rule_status_cannot_be_counted_as_an_ai_answer(row, context, treatment):
    row["validation_status"] = "rule"
    with pytest.raises(ValueError, match="status"):
        normalize_review(row, "r", 42, treatment, context)
    with pytest.raises(ValueError, match="status"):
        verify_answer(row, {}, {})


@pytest.mark.parametrize("treatment", ["rule_abm", "vanilla_llm", "invented_group"])
def test_unknown_or_cross_group_status_cannot_silently_enter_the_summary(row, context, treatment):
    row["validation_status"] = "ok" if treatment == "rule_abm" else "invented_status"
    with pytest.raises(ValueError):
        normalize_review(row, "r", 42, treatment, context)


@pytest.mark.parametrize(
    "status", ["invalid_json", "invalid_json_or_schema", "transport_exhausted"]
)
def test_known_fallbacks_remain_visible_without_becoming_valid_choices(row, context, status):
    row.update(validation_status=status, decision="stay", target_style="technical")
    record = normalize_review(row, "r", 42, "vanilla_llm", context)
    assert not bind_archived_answer(record, None)["valid_choice"]


@pytest.mark.parametrize("request_hash", [None, "", "h", "g" * 64])
def test_valid_ai_choice_requires_a_bound_full_request_identity(row, context, request_hash):
    record = normalize_review(row, "r", 42, "vanilla_llm", context)
    with pytest.raises(ValueError, match="answer"):
        bind_archived_answer(record, request_hash)
    assert bind_archived_answer(record, "a" * 64)["raw_answer_request_hash"] == "a" * 64


def test_programmed_choice_cannot_be_given_an_ai_answer_identity(row):
    row["validation_status"] = "rule"
    record = normalize_review(row, "r", 42, "rule_abm")
    with pytest.raises(ValueError, match="programmed"):
        bind_archived_answer(record, "a" * 64)


@pytest.mark.parametrize("rationale", [123, None, "x" * 2001])
def test_raw_answer_must_pass_the_original_choice_schema(row, rationale):
    row["rationale"] = rationale
    raw = {key: row[key] for key in ("decision", "target_style", "rationale")}
    with pytest.raises(ValueError, match="raw answer"):
        verify_answer(row, {"h": {"key": ("A000", "2026-06-29")}}, {"h": raw})


def test_extra_fields_in_raw_answer_cannot_be_silently_ignored(row):
    raw = {key: row[key] for key in ("decision", "target_style", "rationale")}
    raw["extra"] = "not permitted"
    with pytest.raises(ValueError, match="raw answer"):
        verify_answer(row, {"h": {"key": ("A000", "2026-06-29")}}, {"h": raw})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_missing_or_nonfinite_returns_cannot_become_zero_profit(row, context, bad):
    row["actual_block_return"] = bad
    with pytest.raises(ValueError):
        normalize_review(row, "r", 42, "vanilla_llm", context)


def test_repair_cannot_change_the_input_account_then_reuse_the_initial_features(tmp_path):
    meta = {
        "seed": 42,
        "treatment": "vanilla_llm",
        "mode": "historical",
        "shock": 0,
        "include_secondary": True,
        "prompt_version": "test",
    }
    identity = {k: v for k, v in meta.items() if k != "prompt_version"}
    identity.update(agent_id="A000", session_date="2026-06-29", kind="style")
    lines = []
    for repair in (0, 1):
        context = {"simulation_seed": 42, "changed_account": repair}
        payload = {"messages": [{"role": "user", "content": json.dumps(context)}]}
        call_id = {**identity, "repair": repair}
        key = hashlib.sha256(json.dumps([payload, call_id, "test"], sort_keys=True).encode())
        lines.append({**call_id, "request_hash": key.hexdigest(), "payload": payload})
    (tmp_path / "llm_requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (tmp_path / "llm_responses.jsonl").write_text("")
    with pytest.raises(ValueError, match="Repair changed decision context"):
        load_contexts(tmp_path, meta, {("A000", "2026-06-29")})
