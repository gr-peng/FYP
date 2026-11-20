"""Recompute two complete raw-log phases; four records remain a tiny descriptive check."""

from behavioral_market.evaluation.style_probe_mode_v2 import (
    CONDITIONS,
    PAIR_ROLES,
    evaluate_phase,
)


def combine_phases(cases, base, sources, batches):
    if len(batches) != 2:
        raise ValueError("Require both predeclared phases, including empty planned phases")
    evaluated = []
    scheduled_sets = []
    for batch in batches:
        if set(batch) != {"manifest", "requests", "attempts", "responses"}:
            raise ValueError("Recompute from raw phase logs, never from a claimed summary")
        result = evaluate_phase(
            cases,
            base,
            sources,
            batch["manifest"],
            batch["requests"],
            batch["attempts"],
            batch["responses"],
        )
        evaluated.append(result)
        scheduled_sets.append(set(batch["manifest"]["scheduled_case_ids"]))
    if scheduled_sets[0] & scheduled_sets[1] or set.union(*scheduled_sets) != {
        c["case_id"] for c in cases
    }:
        raise ValueError("Phases duplicate, overlap or fail to cover the fixed four cases")
    if len({r["evidence_kind"] for r in evaluated}) != 1:
        raise ValueError("Never mix synthetic tests and recorded response logs")
    if len({r["run_id"] for r in evaluated}) != 2:
        raise ValueError("Use two explicit, distinct phase run identities")
    rows = [row for phase in evaluated for row in phase["cases"] if row["scheduled"]]
    by_id = {r["case_id"]: r for r in rows}
    pairs = []
    for pair_id, (_agent, style) in sorted(PAIR_ROLES.items()):
        a, b = [by_id[pair_id + "_" + condition] for condition in CONDITIONS]
        complete = a["compatible"] and b["compatible"]
        pairs.append(
            {
                "pair_id": pair_id,
                "from_style": style,
                "comparable": complete,
                "own_majority": {
                    "case_id": a["case_id"],
                    "answer_status": a["answer_contract_status"],
                    "mode_status": a["mode_contract_status"],
                    "choice": a["choice"],
                },
                "other_majority": {
                    "case_id": b["case_id"],
                    "answer_status": b["answer_contract_status"],
                    "mode_status": b["mode_contract_status"],
                    "choice": b["choice"],
                },
                "choice_changed": a["choice"]["target_style"] != b["choice"]["target_style"]
                if complete
                else None,
            }
        )
    complete = all(r["compatible"] for r in rows)
    kinds = evaluated[0]["evidence_kind"]
    targets = {row["choice"]["target_style"] for row in rows if row["compatible"]}
    return {
        "evidence_kind": kinds,
        "evidence_scope": "synthetic_tool_test"
        if kinds == "synthetic_tool_test"
        else "local_logs_not_externally_authenticated",
        "candidate_count": 4,
        "pair_opportunities": 2,
        "valid_compatible_choices": sum(r["compatible"] for r in rows),
        "logged_post_attempts": sum(r["logged_post_attempts"] for r in evaluated),
        "missing_scheduled_cases": sum(r["scheduled_missing"] for r in evaluated),
        "comparable_pairs": sum(p["comparable"] for p in pairs),
        "changed_pairs": sum(p["choice_changed"] for p in pairs) if complete else None,
        "all_valid_choices_target_same_style": next(iter(targets))
        if complete and len(targets) == 1
        else None,
        "tiny_four_case_probe_complete": complete,
        "formal_behavior_study_complete": False,
        "human_alignment_established": False,
        "causal_effect_established": False,
        "real_model_execution_verified": False,
        "research_eligible": False,
        "phases": [{"run_id": r["run_id"], "phase_status": r["phase_status"]} for r in evaluated],
        "pairs": pairs,
        "cases": sorted(rows, key=lambda r: r["case_id"]),
        "limits": [
            "Two preselected contexts with one response per condition, not a human sample.",
            "A fixed target across both conditions is not a majority-response effect.",
            "No herding-rate metric, survivor averaging, significance test or causal claim.",
        ],
    }
