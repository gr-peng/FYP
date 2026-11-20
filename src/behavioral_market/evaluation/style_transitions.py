"""Trace two-period strategy transitions; counts are descriptive, never causal evidence."""

import re
from collections import Counter

STYLES = ("fundamental", "technical")
JOIN_FIELDS = ("run_id", "agent_id", "review_date", "raw_answer_request_hash")


def slot_key(row):
    return row.get("run_id"), row.get("agent_id"), row.get("review_date")


def answer_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def index_records(records, runs, rule_runs, dates, agents, *, binding):
    indexed, rules, hashes, response_refs = {}, set(), set(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Transition sources must contain JSON objects")
        run, agent, day = slot_key(row)
        if day not in dates or agent not in agents:
            raise ValueError("Foreign agent or date in transition source")
        if run in rule_runs:
            if row.get("authority") != "programmed_rule" or row.get("treatment") != "rule_abm":
                raise ValueError("Programmed-rule metadata differs")
            if slot_key(row) in rules:
                raise ValueError("Duplicate rule opportunity")
            rules.add(slot_key(row))
            continue
        if run not in runs:
            raise ValueError("Foreign run in transition source")
        key = slot_key(row)
        if key in indexed:
            raise ValueError(
                "One run,agent,date is one opportunity even with different answer hashes"
            )
        h = row.get("raw_answer_request_hash")
        if answer_hash(h):
            if (run, h) in hashes:
                raise ValueError("One answer identity cannot bind multiple opportunities")
            hashes.add((run, h))
        if binding and row.get("status") == "matched":
            line = row.get("response_line")
            if type(line) is int and line > 0:
                if (run, line) in response_refs:
                    raise ValueError("One raw response cannot bind multiple opportunities")
                response_refs.add((run, line))
        indexed[key] = row
    return indexed, rules


def normalize_slot(run, agent, day, spec, review, binding):
    result = {
        "run_id": run,
        "agent_id": agent,
        "review_date": day,
        "treatment": spec["treatment"],
        "seed": spec["seed"],
        "review_present": review is not None,
        "binding_present": binding is not None,
        "raw_answer_request_hash": review.get("raw_answer_request_hash") if review else None,
        "origin_known": False,
        "from_style": None,
        "verified_choice": False,
        "decision": None,
        "target_style": None,
        "response_line": None,
        "cache_state": binding.get("cache_state", "unknown") if binding else "unknown",
        "status": "missing_review",
    }
    if review is None:
        return result
    if not (
        review.get("authority") == "archived_llm"
        and review.get("treatment") == spec["treatment"]
        and type(review.get("seed")) is int
        and review["seed"] == spec["seed"]
        and review.get("context_verified") is True
        and review.get("from_style") in STYLES
    ):
        result["status"] = "invalid_or_unknown_origin"
        return result
    result.update(origin_known=True, from_style=review["from_style"])
    if binding is None:
        result["status"] = "missing_binding"
        return result
    if not answer_hash(review.get("raw_answer_request_hash")) or any(
        review.get(k) != binding.get(k) for k in JOIN_FIELDS
    ):
        result["status"] = "missing_or_different_answer_identity"
        return result
    if not (
        binding.get("authority") == "archived_llm"
        and binding.get("treatment") == spec["treatment"]
        and binding.get("status") == "matched"
        and type(binding.get("response_line")) is int
        and binding["response_line"] > 0
    ):
        result["status"] = "unverified_binding"
        return result
    decision, target = review.get("decision"), review.get("target_style")
    if not (
        review.get("valid_choice") is True
        and review.get("validation_status") == "ok"
        and decision in {"stay", "switch"}
        and target in STYLES
        and (decision == "stay") == (target == review["from_style"])
        and type(review.get("switched")) is bool
        and review["switched"] == (decision == "switch")
    ):
        result["status"] = "invalid_or_inconsistent_choice"
        return result
    result.update(
        status="verified",
        verified_choice=True,
        decision=decision,
        target_style=target,
        response_line=binding["response_line"],
    )
    return result


def summarize_context(rows, identity):
    origins_complete = all(row["origin_known"] for row in rows)
    cells = []
    for origin in STYLES:
        group = [row for row in rows if row["from_style"] == origin]
        valid = [row for row in group if row["verified_choice"]]
        known_n = len(group)
        unknown_choices = known_n - len(valid)
        complete = origins_complete and unknown_choices == 0
        for target in STYLES:
            observed = sum(row["target_style"] == target for row in valid)
            cells.append(
                {
                    "from_style": origin,
                    "target_style": target,
                    "known_origin_opportunities": known_n,
                    "origin_opportunities": known_n if origins_complete else None,
                    "unknown_choices_with_known_origin": unknown_choices,
                    "observed_verified_count": observed,
                    "count": observed if complete else None,
                    "rate": observed / known_n if complete and known_n else None,
                    "confirmed_zero_origin": origins_complete and known_n == 0,
                    "transition_complete": complete,
                }
            )
    return {
        **identity,
        "scheduled_opportunities": len(rows),
        "review_records_present": sum(row["review_present"] for row in rows),
        "origin_known_count": sum(row["origin_known"] for row in rows),
        "unknown_origin_count": sum(not row["origin_known"] for row in rows),
        "verified_choices": sum(row["verified_choice"] for row in rows),
        "unverified_choices": sum(not row["verified_choice"] for row in rows),
        "all_origins_known": origins_complete,
        "all_choices_verified": all(row["verified_choice"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "rate_aggregation": "conditional_counts_not_equal_run_mean",
        "cells": cells,
    }


def decompose_transitions(reviews, bindings, runs, dates, agents, rule_runs=()):
    if len(dates) != 2 or dates[0] >= dates[1] or not agents or len(set(agents)) != len(agents):
        raise ValueError("Require two ordered dates and unique nonempty agent identities")
    if not runs or set(runs) & set(rule_runs):
        raise ValueError("AI and rule run scopes must be explicit and disjoint")
    review_map, excluded_reviews = index_records(
        reviews, runs, rule_runs, dates, agents, binding=False
    )
    binding_map, excluded_bindings = index_records(
        bindings, runs, rule_runs, dates, agents, binding=True
    )
    choices = [
        normalize_slot(
            run,
            agent,
            day,
            spec,
            review_map.get((run, agent, day)),
            binding_map.get((run, agent, day)),
        )
        for run, spec in sorted(runs.items())
        for day in dates
        for agent in agents
    ]
    per_run = [
        summarize_context(
            [row for row in choices if row["run_id"] == run and row["review_date"] == day],
            {"run_id": run, **spec, "review_date": day},
        )
        for run, spec in sorted(runs.items())
        for day in dates
    ]
    per_group = [
        summarize_context(
            [row for row in choices if row["treatment"] == group and row["review_date"] == day],
            {
                "treatment": group,
                "review_date": day,
                "runs": sum(spec["treatment"] == group for spec in runs.values()),
            },
        )
        for group in sorted({spec["treatment"] for spec in runs.values()})
        for day in dates
    ]
    by_slot = {slot_key(row): row for row in choices}
    continuity = []
    for run in sorted(runs):
        for agent in agents:
            first, second = [by_slot[run, agent, day] for day in dates]
            comparable = first["verified_choice"] and second["origin_known"]
            continuity.append(
                {
                    "run_id": run,
                    "agent_id": agent,
                    "first_date": dates[0],
                    "second_date": dates[1],
                    "first_target": first["target_style"],
                    "second_origin": second["from_style"],
                    "comparable": comparable,
                    "consistent": first["target_style"] == second["from_style"]
                    if comparable
                    else None,
                }
            )
    complete = all(row["verified_choice"] for row in choices)
    direction_counts = {
        f"{a}_to_{b}": sum(
            row["verified_choice"] and row["from_style"] == a and row["target_style"] == b
            for row in choices
        )
        for a in STYLES
        for b in STYLES
    }
    pairs_complete = all(row["comparable"] for row in continuity)
    summary = {
        "analysis_status": "retrospective_descriptive_transition_decomposition",
        "AI_run_count": len(runs),
        "review_dates": list(dates),
        "scheduled_AI_choices": len(choices),
        "AI_review_rows_present": len(review_map),
        "AI_binding_rows_present": len(binding_map),
        "excluded_rule_review_rows": len(excluded_reviews),
        "excluded_rule_binding_rows": len(excluded_bindings),
        "rule_scope_counts_match": excluded_reviews == excluded_bindings,
        "scheduled_rule_opportunities": len(rule_runs) * len(dates) * len(agents),
        "missing_rule_review_rows": len(rule_runs) * len(dates) * len(agents)
        - len(excluded_reviews),
        "missing_rule_binding_rows": len(rule_runs) * len(dates) * len(agents)
        - len(excluded_bindings),
        "verified_choices": sum(row["verified_choice"] for row in choices),
        "unverified_choices": sum(not row["verified_choice"] for row in choices),
        "unknown_origins": sum(not row["origin_known"] for row in choices),
        "all_choices_verified": complete,
        "observed_verified_direction_counts": direction_counts,
        "complete_direction_counts": direction_counts if complete else None,
        "complete_switch_count": sum(row["decision"] == "switch" for row in choices)
        if complete
        else None,
        "continuity_opportunities": len(continuity),
        "continuity_comparable": sum(row["comparable"] for row in continuity),
        "continuity_consistent": sum(row["consistent"] is True for row in continuity),
        "continuity_mismatched": sum(row["consistent"] is False for row in continuity),
        "continuity_unknown": sum(not row["comparable"] for row in continuity),
        "all_continuity_pairs_consistent": all(row["consistent"] for row in continuity)
        if pairs_complete
        else None,
        "independent_generation_count": None,
        "causal_effect_established": False,
        "human_alignment_established": False,
        "formal_behavior_study_complete": False,
        "new_model_calls": 0,
    }
    return {
        "summary": summary,
        "choices": choices,
        "per_run_date": per_run,
        "per_group_date": per_group,
        "continuity": continuity,
    }
