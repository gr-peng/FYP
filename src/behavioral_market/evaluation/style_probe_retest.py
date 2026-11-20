"""Fixed repeated-input check: sixteen slots, two contexts, no population inference."""

import copy
import hashlib
import json
import re
from collections import Counter

from behavioral_market.evaluation.style_probe_intake import aware_time, parse_choice
from behavioral_market.evaluation.style_probe_materials import payload_digest
from behavioral_market.evaluation.style_probe_mode_v2 import (
    CONDITIONS,
    PAIR_ROLES,
    phase_manifest,
    returned_mode,
    validate_variant,
)

NAMESPACE = "style-share-retest-v1"
STAGE = "same_input_retest"
SEQUENCES = ("ABDC", "BCAD", "CDBA", "DACB")
CODES = {
    "A": "A001_2026-06-29_own_style_majority",
    "B": "A001_2026-06-29_other_style_majority",
    "C": "A000_2026-06-29_own_style_majority",
    "D": "A000_2026-06-29_other_style_majority",
}
MISSING = {"unsubmitted", "request_record_only", "awaiting_response"}


def planned_schedule(cases):
    for pair_id in PAIR_ROLES:
        phase_manifest(cases, "role-check", "synthetic_tool_test", pair_id=pair_id)
    by_id = {case["case_id"]: case for case in cases}
    return [
        {
            "record_type": "unsent_retest_payload_proposal",
            "planned_id": f"r{repeat:02d}-p{position}-{code}",
            "local_repeat_id": f"style-share-retest-r{repeat:02d}",
            "repeat": repeat,
            "position": position,
            "code": code,
            **{
                key: copy.deepcopy(by_id[CODES[code]][key])
                for key in ("case_id", "pair_id", "condition", "payload", "payload_sha256")
            },
        }
        for repeat, sequence in enumerate(SEQUENCES, start=1)
        for position, code in enumerate(sequence, start=1)
    ]


def validate_schedule(cases, planned):
    if payload_digest(planned) != payload_digest(planned_schedule(cases)):
        raise ValueError("Retain the exact frozen sixteen-slot schedule and original payloads")


def run_manifest(cases, planned, run_id, evidence_kind):
    validate_schedule(cases, planned)
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid bounded retest run identifier")
    if evidence_kind not in {"synthetic_tool_test", "recorded_model_logs"}:
        raise ValueError("Unknown evidence kind")
    return {
        "protocol_version": NAMESPACE,
        "stage": STAGE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "candidates_sha256": payload_digest(cases),
        "schedule_sha256": payload_digest(planned),
        "scheduled_ids": [row["planned_id"] for row in planned],
        "max_completion_posts": 16,
        "transport_retries": 0,
        "format_repairs": 0,
        "discovery_answers_in_primary_counts": 0,
    }


def request_identity(slot, manifest):
    return {
        "kind": "style_share_retest",
        "probe_namespace": NAMESPACE,
        "stage": STAGE,
        "probe_run_id": manifest["run_id"],
        "evidence_kind": manifest["evidence_kind"],
        **{
            key: slot[key]
            for key in (
                "planned_id",
                "local_repeat_id",
                "repeat",
                "position",
                "case_id",
                "pair_id",
                "condition",
            )
        },
        "repair": 0,
    }


def expected_request(slot, manifest):
    origin = request_identity(slot, manifest)
    h = hashlib.sha256(
        json.dumps([slot["payload"], origin, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    return {**origin, "request_hash": h, "payload": copy.deepcopy(slot["payload"])}


def evaluate_retest(cases, base, sources, planned, manifest, requests, attempts, responses):
    validate_variant(cases, base, sources)
    expected_manifest = run_manifest(cases, planned, manifest["run_id"], manifest["evidence_kind"])
    if payload_digest(manifest) != payload_digest(expected_manifest):
        raise ValueError("Retest manifest differs from the fixed execution scope")
    by_id = {case["case_id"]: case for case in cases}
    by_slot = {slot["planned_id"]: slot for slot in planned}
    scheduled = manifest["scheduled_ids"]
    maps = []
    for kind, records in (("request", requests), ("attempt", attempts), ("response", responses)):
        mapped = {}
        for row in records:
            sid = row.get("planned_id")
            if not isinstance(sid, str) or sid not in by_slot or sid in mapped:
                raise ValueError("Foreign or duplicate retest slot")
            wanted = expected_request(by_slot[sid], manifest)
            if any(row.get(k) != v for k, v in wanted.items() if k != "payload") or any(
                type(row.get(k)) is not int for k in ("repair", "repeat", "position")
            ):
                raise ValueError("Retest identity or request hash differs")
            if kind == "request" and payload_digest(row) != payload_digest(wanted):
                raise ValueError("Retest payload or local identity changed")
            if kind == "attempt" or kind == "response" and row.get("cached") is not True:
                if type(row.get("retry_count")) is not int or row["retry_count"] != 0:
                    raise ValueError("Retest retries are forbidden")
            mapped[sid] = row
        if list(mapped) != scheduled[: len(mapped)]:
            raise ValueError("Raw retest logs must be a prefix of the fixed global order")
        maps.append(mapped)
    submitted, dispatched, received = maps
    if not set(dispatched) <= set(submitted) or not set(received) <= set(submitted):
        raise ValueError("Retest attempt or response lacks its request")
    rows = []
    for i, slot in enumerate(planned):
        sid = slot["planned_id"]
        answer, mode, choice = "unsubmitted", "unobserved", None
        cached = False
        if sid in submitted:
            if i and not rows[-1]["compatible"]:
                raise ValueError("Request after failure or missing response violates stop policy")
            answer = "request_record_only"
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            if i and start < aware_time(
                received[planned[i - 1]["planned_id"]]["response_received_at"]
            ):
                raise ValueError("Retest calls must be sequential")
            answer = "awaiting_response"
        if sid in received:
            row = received[sid]
            if type(row.get("cached")) is not bool:
                raise ValueError("Explicit cache status required")
            cached = row["cached"]
            if cached:
                if sid in dispatched:
                    raise ValueError("A replay cannot claim a new dispatch")
                answer, mode = "cache_replay", "not_new_execution"
            else:
                if sid not in dispatched:
                    raise ValueError("A new retest response needs its dispatch")
                start, finish = (
                    aware_time(row["requested_at"]),
                    aware_time(row["response_received_at"]),
                )
                if start != aware_time(dispatched[sid]["dispatched_at"]) or finish < start:
                    raise ValueError("Retest response clocks differ from its dispatch")
                status = row["http_status"]
                if not (
                    type(status) is int
                    and 100 <= status <= 599
                    or status in ("transport_error", "cancelled_in_flight")
                ):
                    raise ValueError("Invalid HTTP/transport status")
                if status == 200:
                    answer, choice = parse_choice(row["response"], by_id[slot["case_id"]])
                    mode = returned_mode(row["response"])
                else:
                    if row["response"] is not None:
                        raise ValueError("Failed transport cannot contain an answer")
                    answer, mode = "transport_failed", "unobserved"
        rows.append(
            {
                **{
                    k: slot[k]
                    for k in ("planned_id", "repeat", "position", "case_id", "pair_id", "condition")
                },
                "answer_status": answer,
                "mode_status": mode,
                "choice": choice,
                "cache_replay": cached,
                "compatible": answer == "valid_linked_answer"
                and mode == "no_returned_reasoning_observed"
                and not cached,
            }
        )
    conditions = []
    for case in cases:
        group = [row for row in rows if row["case_id"] == case["case_id"]]
        valid = [row for row in group if row["compatible"]]
        decisions = Counter(row["choice"]["decision"] for row in valid)
        targets = Counter(row["choice"]["target_style"] for row in valid)
        conditions.append(
            {
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "from_style": case["source"]["from_style"],
                "condition": case["condition"],
                "scheduled_count": 4,
                "valid_count": len(valid),
                "missing_count": sum(row["answer_status"] in MISSING for row in group),
                "error_count": sum(
                    not row["compatible"] and row["answer_status"] not in MISSING for row in group
                ),
                "decision_counts": {k: decisions[k] for k in ("stay", "switch")},
                "target_counts": {k: targets[k] for k in ("fundamental", "technical")},
                "distinct_observed_targets": len(targets),
                "variation_observed": True
                if len(targets) > 1
                else False
                if len(valid) == 4
                else None,
                "all_four_valid": len(valid) == 4,
            }
        )
    rounds = []
    pairs = []
    for pair_id, (_agent, style) in sorted(PAIR_ROLES.items()):
        group = []
        for repeat in range(1, 5):
            a, b = [
                next(
                    row
                    for row in rows
                    if row["repeat"] == repeat
                    and row["pair_id"] == pair_id
                    and row["condition"] == condition
                )
                for condition in CONDITIONS
            ]
            complete = a["compatible"] and b["compatible"]
            group.append(
                {
                    "pair_id": pair_id,
                    "from_style": style,
                    "repeat": repeat,
                    "own_majority_target": a["choice"]["target_style"] if a["compatible"] else None,
                    "other_majority_target": b["choice"]["target_style"]
                    if b["compatible"]
                    else None,
                    "comparable": complete,
                    "choice_changed": a["choice"]["target_style"] != b["choice"]["target_style"]
                    if complete
                    else None,
                }
            )
        rounds.extend(group)
        complete = all(row["comparable"] for row in group)
        pairs.append(
            {
                "pair_id": pair_id,
                "from_style": style,
                "scheduled_rounds": 4,
                "comparable_rounds": sum(row["comparable"] for row in group),
                "all_rounds_comparable": complete,
                "changed_rounds": sum(row["choice_changed"] for row in group) if complete else None,
            }
        )
    complete = all(row["compatible"] for row in rows)
    synthetic = manifest["evidence_kind"] == "synthetic_tool_test"
    return {
        "protocol_version": NAMESPACE,
        "run_id": manifest["run_id"],
        "evidence_kind": manifest["evidence_kind"],
        "evidence_scope": "synthetic_tool_test"
        if synthetic
        else "local_logs_not_externally_authenticated",
        "scheduled_count": 16,
        "original_context_count": 2,
        "original_condition_count": 4,
        "submitted_count": len(submitted),
        "logged_post_attempts": len(dispatched),
        "valid_count": sum(row["compatible"] for row in rows),
        "missing_count": sum(row["answer_status"] in MISSING for row in rows),
        "error_count": sum(
            not row["compatible"] and row["answer_status"] not in MISSING for row in rows
        ),
        "cache_replays": sum(row["cache_replay"] for row in rows),
        "retest_complete": complete,
        "discovery_answers_in_primary_counts": 0,
        "real_model_execution_verified": False,
        "formal_behavior_study_complete": False,
        "human_alignment_established": False,
        "causal_effect_established": False,
        "independent_sampling_established": False,
        "research_eligible": False,
        "model_calls_performed_by_evaluator": 0,
        "conditions": conditions,
        "pairs": pairs,
        "rounds": rounds,
        "slots": rows,
        "limits": [
            "Sixteen repeated calls concern two original contexts, not sixteen investors.",
            "Four identical choices do not prove zero variability; "
            "different reasons are not different choices.",
            "Local non-cache records do not prove the service has no cache "
            "or independent sampling.",
            "Discovery answers excluded; incomplete context comparisons remain null.",
        ],
    }
