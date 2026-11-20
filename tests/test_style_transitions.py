import copy

import pytest

from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.evaluation.style_transitions import decompose_transitions
from scripts.decompose_style_transitions import cell_text, readable_report

DATES = ["2026-06-29", "2026-07-14"]
AGENTS = ["A000", "A001"]
RUNS = {
    "run-one": {"treatment": "vanilla_llm", "seed": 42},
    "run-two": {"treatment": "high_herding", "seed": 43},
}


@pytest.fixture
def sources():
    reviews, bindings = [], []
    for run, spec in RUNS.items():
        line = 0
        for day in DATES:
            for agent in AGENTS:
                line += 1
                if run == "run-one":
                    origin = "technical" if day == DATES[0] else "fundamental"
                    target = "fundamental"
                else:
                    origin = target = "fundamental" if agent == "A000" else "technical"
                row = {
                    "run_id": run,
                    "agent_id": agent,
                    "review_date": day,
                    **spec,
                    "raw_answer_request_hash": payload_digest([run, agent, day]),
                    "authority": "archived_llm",
                    "context_verified": True,
                    "from_style": origin,
                    "target_style": target,
                    "decision": "stay" if origin == target else "switch",
                    "switched": origin != target,
                    "valid_choice": True,
                    "validation_status": "ok",
                }
                reviews.append(row)
                bindings.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "run_id",
                                "agent_id",
                                "review_date",
                                "raw_answer_request_hash",
                                "authority",
                                "treatment",
                            )
                        },
                        "status": "matched",
                        "response_line": line,
                        "cache_state": "nonlocal_cache",
                    }
                )
    for day in DATES:
        for agent in AGENTS:
            row = {
                "run_id": "rule-run",
                "agent_id": agent,
                "review_date": day,
                "authority": "programmed_rule",
                "treatment": "rule_abm",
                "raw_answer_request_hash": None,
            }
            reviews.append(row)
            bindings.append({**row, "status": "excluded_programmed_rule"})
    return reviews, bindings


def analyze(sources):
    return decompose_transitions(*sources, RUNS, DATES, AGENTS, ["rule-run"])


def context(result, run="run-one", day=DATES[0]):
    return next(
        row for row in result["per_run_date"] if row["run_id"] == run and row["review_date"] == day
    )


def cell(ctx, origin="technical", target="fundamental"):
    return next(
        row for row in ctx["cells"] if row["from_style"] == origin and row["target_style"] == target
    )


def test_complete_trace_counts_and_longitudinal_links_conserve_all_opportunities(sources):
    before = copy.deepcopy(sources)
    result = analyze(sources)
    s = result["summary"]
    assert s["scheduled_AI_choices"] == s["verified_choices"] == 8
    assert s["excluded_rule_review_rows"] == s["excluded_rule_binding_rows"] == 4
    assert s["complete_switch_count"] == 2
    assert s["complete_direction_counts"]["technical_to_fundamental"] == 2
    assert s["continuity_opportunities"] == s["continuity_consistent"] == 4
    assert s["all_continuity_pairs_consistent"] is True
    assert all(sum(c["count"] for c in ctx["cells"]) == 2 for ctx in result["per_run_date"])
    assert sources == before


def test_confirmed_zero_origin_has_zero_counts_but_null_rate(sources):
    result = analyze(sources)
    empty = cell(context(result), "fundamental", "technical")
    assert empty["confirmed_zero_origin"] and empty["origin_opportunities"] == 0
    assert empty["count"] == 0 and empty["rate"] is None
    assert cell_text(empty) == "0机会"
    assert cell(context(result))["rate"] == 1


def test_missing_review_retains_slot_unknown_origin_and_does_not_create_a_zero(sources):
    sources[0].pop(0)
    result = analyze(sources)
    assert result["summary"]["scheduled_AI_choices"] == 8
    assert result["summary"]["verified_choices"] == 7
    assert result["summary"]["unknown_origins"] == 1
    assert result["summary"]["complete_switch_count"] is None
    assert all(
        c["count"] is None and not c["confirmed_zero_origin"] for c in context(result)["cells"]
    )
    assert result["summary"]["continuity_unknown"] == 1


def test_known_origin_with_missing_binding_keeps_denominator_and_unknown_choice(sources):
    sources[1].pop(0)
    result = analyze(sources)
    c = cell(context(result))
    assert c["origin_opportunities"] == c["known_origin_opportunities"] == 2
    assert c["unknown_choices_with_known_origin"] == 1
    assert c["observed_verified_count"] == 1 and c["count"] is None and c["rate"] is None
    assert "未知" in cell_text(c)
    assert result["summary"]["complete_direction_counts"] is None


def test_unknown_second_origin_not_filled_from_first_target(sources):
    sources[0][2]["from_style"] = None
    result = analyze(sources)
    assert result["summary"]["continuity_unknown"] == 1
    pair = next(
        p for p in result["continuity"] if p["run_id"] == "run-one" and p["agent_id"] == "A000"
    )
    assert pair["first_target"] == "fundamental" and pair["second_origin"] is None
    assert pair["consistent"] is None
    assert all(not c["confirmed_zero_origin"] for c in context(result, day=DATES[1])["cells"])


@pytest.mark.parametrize("stream", [0, 1])
def test_different_hashes_for_same_opportunity_are_rejected(sources, stream):
    duplicate = copy.deepcopy(sources[stream][0])
    duplicate["raw_answer_request_hash"] = "a" * 64
    sources[stream].append(duplicate)
    with pytest.raises(ValueError, match="one opportunity"):
        analyze(sources)


@pytest.mark.parametrize("stream", [0, 1])
def test_one_answer_identity_cannot_support_multiple_agent_opportunities(sources, stream):
    sources[stream][1]["raw_answer_request_hash"] = sources[stream][0]["raw_answer_request_hash"]
    with pytest.raises(ValueError, match="multiple opportunities"):
        analyze(sources)


def test_one_raw_response_line_cannot_be_bound_to_two_slots(sources):
    sources[1][1]["response_line"] = sources[1][0]["response_line"]
    with pytest.raises(ValueError, match="raw response"):
        analyze(sources)


@pytest.mark.parametrize(
    "alter", ["different_hash", "unknown_status", "false_valid", "bad_direction", "wrong_authority"]
)
def test_unknown_or_inconsistent_choice_never_defaults_to_stay(sources, alter):
    if alter == "different_hash":
        sources[1][0]["raw_answer_request_hash"] = "b" * 64
    elif alter == "unknown_status":
        sources[1][0]["status"] = "unmatched"
    elif alter == "false_valid":
        sources[0][0]["valid_choice"] = False
    elif alter == "bad_direction":
        sources[0][0]["decision"] = "stay"
    else:
        sources[0][0]["authority"] = "programmed_rule"
    result = analyze(sources)
    row = result["choices"][0]
    assert not row["verified_choice"] and row["target_style"] is None and row["decision"] is None
    assert result["summary"]["unverified_choices"] == 1
    assert result["summary"]["excluded_rule_review_rows"] == 4


def test_longitudinal_mismatch_is_reported_without_fixing_source(sources):
    sources[0][2].update(
        from_style="technical", target_style="technical", decision="stay", switched=False
    )
    before = copy.deepcopy(sources)
    result = analyze(sources)
    assert result["summary"]["continuity_mismatched"] == 1
    assert result["summary"]["all_continuity_pairs_consistent"] is False
    assert result["choices"][2]["from_style"] == "technical"
    report = readable_report(result)
    assert "两期衔接有不一致或未知" in report
    assert "两期逐代理衔接全部对应" not in report
    assert sources == before


def test_second_answer_unknown_does_not_hide_known_second_origin_link(sources):
    sources[1].pop(2)
    result = analyze(sources)
    assert result["summary"]["verified_choices"] == 7
    assert result["summary"]["continuity_consistent"] == 4
    assert result["summary"]["continuity_unknown"] == 0


def test_cache_label_is_retained_without_proving_independent_generation(sources):
    sources[1][0]["cache_state"] = "local_cache"
    result = analyze(sources)
    assert result["choices"][0]["cache_state"] == "local_cache"
    assert result["summary"]["verified_choices"] == 8
    assert result["summary"]["independent_generation_count"] is None
    assert not result["summary"]["human_alignment_established"]


def test_empty_sources_keep_all_scheduled_AI_and_rule_missing_counts():
    result = analyze(([], []))
    s = result["summary"]
    assert s["scheduled_AI_choices"] == s["unverified_choices"] == s["unknown_origins"] == 8
    assert s["missing_rule_review_rows"] == s["missing_rule_binding_rows"] == 4
    assert s["all_continuity_pairs_consistent"] is None
    assert all(c["count"] is None for ctx in result["per_run_date"] for c in ctx["cells"])
    assert "记录未齐" in readable_report(result)


def test_foreign_run_is_not_silently_added_or_dropped(sources):
    foreign = copy.deepcopy(sources[0][0])
    foreign["run_id"] = "not-in-scope"
    sources[0].append(foreign)
    with pytest.raises(ValueError, match="Foreign run"):
        analyze(sources)
