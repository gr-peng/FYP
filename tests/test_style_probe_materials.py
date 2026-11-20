import copy
import json

import pytest

from behavioral_market.evaluation.style_probe_materials import (
    build_share_probes,
    payload_digest,
    validate_share_probes,
)


@pytest.fixture
def sources():
    result = []
    for i, style in enumerate(("fundamental", "technical")):
        context = {
            "agent_state": {
                "persona": {"agent_id": f"A{i:03}"},
                "current_style": style,
                "portfolio": {"cash": 5000, "total_equity": 9800},
                "memory": [{"session": "2026-06-29", "pnl_change": -200}],
                "counterfactual": {"actual_block_return": -0.02, "alternative_block_return": 0.01},
            },
            "observation": {
                "session_date": "2026-06-29",
                "phase": "post_close_style_review",
                "last_close": 100,
                "fundamental_value": 105,
                "mean_field": {
                    "as_of_session": "2026-06-26",
                    "fundamental_share": 0.5,
                    "technical_share": 0.5,
                    "buy_sell_imbalance": 0.6,
                    "positive_sentiment_share": 0.7,
                },
            },
        }
        payload = {
            "model": "unchanged-original-model",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "Original system prompt.\n"},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
        }
        h = payload_digest(payload)
        result.append(
            {
                "identity": {
                    "run_id": "fixed_run",
                    "agent_id": f"A{i:03}",
                    "from_style": style,
                    "review_date": "2026-06-29",
                    "treatment": "vanilla_llm",
                    "original_request_hash": h,
                    "matched_answer_request_hash": h,
                },
                "payload": payload,
            }
        )
    return result


def change_user(case, mutate):
    context = json.loads(case["payload"]["messages"][1]["content"])
    mutate(context)
    case["payload"]["messages"][1]["content"] = json.dumps(context, indent=2)
    case["payload_sha256"] = payload_digest(case["payload"])


def test_two_styles_have_complete_opposite_majorities_with_other_prompt_bytes_preserved(sources):
    original = copy.deepcopy(sources)
    cases = build_share_probes(sources)
    checks = validate_share_probes(cases, sources)
    assert checks["pairs"] == 2 and checks["input_cases"] == 4
    assert checks["execution_status"] == "materials_ready_not_run"
    assert checks["model_answers_present"] == 0
    assert sources == original
    for case in cases:
        source = next(s for s in sources if s["identity"] == case["source"])
        text = source["payload"]["messages"][1]["content"]
        own = 0.8 if case["condition"] == "own_style_majority" else 0.2
        fundamental = own if case["source"]["from_style"] == "fundamental" else 1 - own
        # Independent, literal expectation also preserves whitespace and all other fields.
        expected = text.replace(
            '"fundamental_share": 0.5', f'"fundamental_share": {round(fundamental, 1)}'
        )
        expected = expected.replace(
            '"technical_share": 0.5', f'"technical_share": {round(1 - fundamental, 1)}'
        )
        assert case["payload"]["messages"][1]["content"] == expected


@pytest.mark.parametrize("count", [0, 2, 3])
def test_incomplete_material_sets_are_rejected(sources, count):
    with pytest.raises(ValueError, match="four"):
        validate_share_probes(build_share_probes(sources)[:count], sources)


def test_duplicate_case_cannot_replace_a_missing_comparison(sources):
    cases = build_share_probes(sources)
    cases[1] = copy.deepcopy(cases[0])
    with pytest.raises(ValueError):
        validate_share_probes(cases, sources)


def test_same_majority_on_both_sides_is_not_a_valid_contrast(sources):
    cases = build_share_probes(sources)
    cases[1]["payload"] = copy.deepcopy(cases[0]["payload"])
    cases[1]["payload_sha256"] = payload_digest(cases[1]["payload"])
    with pytest.raises(ValueError, match="reverse"):
        validate_share_probes(cases, sources)


@pytest.mark.parametrize("field", ["last_close", "fundamental_value", "session_date"])
def test_price_valuation_or_date_cannot_change_with_the_population_factor(sources, field):
    cases = build_share_probes(sources)
    change_user(cases[0], lambda c: c["observation"].update({field: "changed"}))
    with pytest.raises(ValueError, match="beyond"):
        validate_share_probes(cases, sources)


@pytest.mark.parametrize("field", ["buy_sell_imbalance", "positive_sentiment_share"])
def test_order_direction_and_sentiment_are_separate_unchanged_signals(sources, field):
    cases = build_share_probes(sources)
    change_user(cases[0], lambda c: c["observation"]["mean_field"].update({field: 0.1}))
    with pytest.raises(ValueError, match="beyond"):
        validate_share_probes(cases, sources)


def test_account_memory_and_return_cannot_silently_change(sources):
    cases = build_share_probes(sources)
    change_user(
        cases[0], lambda c: c["agent_state"]["counterfactual"].update(actual_block_return=0.1)
    )
    with pytest.raises(ValueError, match="beyond"):
        validate_share_probes(cases, sources)


@pytest.mark.parametrize("field", ["model", "system"])
def test_model_settings_and_system_prompt_cannot_change(sources, field):
    cases = build_share_probes(sources)
    if field == "model":
        cases[0]["payload"]["model"] = "another-model"
    else:
        cases[0]["payload"]["messages"][0]["content"] = "changed"
    cases[0]["payload_sha256"] = payload_digest(cases[0]["payload"])
    with pytest.raises(ValueError, match="envelope"):
        validate_share_probes(cases, sources)


@pytest.mark.parametrize("as_of", ["2026-06-29", "2026-06-30"])
def test_same_day_and_future_population_summaries_are_rejected(sources, as_of):
    for source in sources:
        context = json.loads(source["payload"]["messages"][1]["content"])
        context["observation"]["mean_field"]["as_of_session"] = as_of
        source["payload"]["messages"][1]["content"] = json.dumps(context, indent=2)
    with pytest.raises(ValueError, match="strictly precede"):
        build_share_probes(sources)


def test_new_answers_or_completion_claims_cannot_be_added_to_input_records(sources):
    cases = build_share_probes(sources)
    cases[0]["model_answer"] = {"decision": "switch"}
    with pytest.raises(ValueError, match="no model results"):
        validate_share_probes(cases, sources)


def test_changed_source_identity_or_payload_fingerprint_is_detected(sources):
    cases = build_share_probes(sources)
    cases[0]["payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        validate_share_probes(cases, sources)
    cases = build_share_probes(sources)
    cases[0]["source"]["original_request_hash"] = "f" * 64
    with pytest.raises(ValueError, match="source identity"):
        validate_share_probes(cases, sources)


@pytest.mark.parametrize("shares", [(0.8, 0.8), (1.2, -0.2), (True, 0), (0.5, 0.5)])
def test_invalid_or_tied_shares_cannot_be_published_as_the_planned_contrast(sources, shares):
    cases = build_share_probes(sources)
    change_user(
        cases[0],
        lambda c: c["observation"]["mean_field"].update(
            fundamental_share=shares[0], technical_share=shares[1]
        ),
    )
    with pytest.raises(ValueError):
        validate_share_probes(cases, sources)
