"""Explicit v2 materials and a two-input compatibility phase; never a four-case study."""

import copy
import hashlib
import json
import re

from behavioral_market.evaluation.style_probe_intake import aware_time, parse_choice
from behavioral_market.evaluation.style_probe_materials import payload_digest, validate_share_probes

NAMESPACE = "style-share-probe-v2"
STAGE = "compatibility_pair"
MODE_CONTROL = {"thinking": False, "enable_thinking": False}
PAIR_ID = "A001_2026-06-29"
PAIR_ROLES = {
    PAIR_ID: ("A001", "fundamental"),
    "A000_2026-06-29": ("A000", "technical"),
}
CONDITIONS = ("own_style_majority", "other_style_majority")


def build_variant(base_cases, base_sources):
    validate_share_probes(base_cases, base_sources)
    cases = copy.deepcopy(base_cases)
    for case in cases:
        if "chat_template_kwargs" in case["payload"]:
            raise ValueError("Base material already has a mode override")
        case["payload"]["chat_template_kwargs"] = copy.deepcopy(MODE_CONTROL)
        case["payload_sha256"] = payload_digest(case["payload"])
    validate_variant(cases, base_cases, base_sources)
    return cases


def validate_variant(cases, base_cases, base_sources):
    validate_share_probes(base_cases, base_sources)
    base = {c["case_id"]: c for c in base_cases}
    if len(cases) != 4 or {c["case_id"] for c in cases} != set(base):
        raise ValueError("Retain all four unique candidate materials")
    for case in cases:
        original = base[case["case_id"]]
        stripped = copy.deepcopy(case)
        mode = stripped["payload"].pop("chat_template_kwargs", None)
        if payload_digest(mode) != payload_digest(MODE_CONTROL):
            raise ValueError("Only the fixed boolean-false mode controls are allowed")
        if case["payload_sha256"] != payload_digest(case["payload"]):
            raise ValueError("Variant payload fingerprint differs")
        stripped["payload_sha256"] = original["payload_sha256"]
        if payload_digest(stripped) != payload_digest(original):
            raise ValueError("Variant changed data, messages, settings or source identity")
    return {
        "candidate_count": 4,
        "allowed_payload_delta": {"chat_template_kwargs": MODE_CONTROL},
        "all_original_messages_verbatim": True,
        "four_case_behavior_study_complete": False,
    }


def phase_manifest(cases, run_id, evidence_kind, *, pair_id=PAIR_ID):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid bounded phase identifier")
    if evidence_kind not in {"recorded_model_logs", "synthetic_tool_test"}:
        raise ValueError("Unknown evidence kind")
    if pair_id not in PAIR_ROLES:
        raise ValueError("Only the two predeclared strategy pairs are allowed")
    agent_id, from_style = PAIR_ROLES[pair_id]
    selected = [pair_id + "_" + condition for condition in CONDITIONS]
    if len(cases) != 4 or len({c["case_id"] for c in cases}) != 4:
        raise ValueError("Require the four candidate inputs")
    by_id = {c["case_id"]: c for c in cases}
    if any(
        cid not in by_id
        or by_id[cid]["pair_id"] != pair_id
        or by_id[cid]["source"]["agent_id"] != agent_id
        or by_id[cid]["source"]["from_style"] != from_style
        for cid in selected
    ):
        raise ValueError("Require the fixed complete pair and matching starting strategy")
    result = {
        "protocol_version": NAMESPACE,
        "stage": STAGE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "candidates_sha256": payload_digest(cases),
        "candidate_case_ids": sorted(by_id),
        "scheduled_case_ids": selected,
        "max_completion_posts": 2,
        "transport_retries": 0,
        "format_repairs": 0,
    }
    if pair_id != PAIR_ID:
        result["extension_pair_id"] = pair_id
    return result


def request_identity(case, manifest):
    return {
        "kind": "style_share_mode_compatibility",
        "probe_namespace": NAMESPACE,
        "stage": STAGE,
        "probe_run_id": manifest["run_id"],
        "evidence_kind": manifest["evidence_kind"],
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "condition": case["condition"],
        "repair": 0,
    }


def expected_request(case, manifest):
    origin = request_identity(case, manifest)
    h = hashlib.sha256(
        json.dumps([case["payload"], origin, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    return {**origin, "request_hash": h, "payload": copy.deepcopy(case["payload"])}


def returned_mode(body):
    """No returned reasoning is an observation, never proof about internal computation."""
    if not isinstance(body, dict):
        return "unavailable"
    choices = body.get("choices", [])
    content = (
        any(
            isinstance(c, dict)
            and isinstance(c.get("message"), dict)
            and bool(c["message"].get("reasoning_content"))
            for c in choices
        )
        if isinstance(choices, list)
        else False
    )
    usage = body.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        return "invalid_mode_evidence"
    if content or tokens is not None and tokens > 0:
        return "reasoning_observed"
    return "no_returned_reasoning_observed"


def evaluate_phase(cases, base_cases, base_sources, manifest, requests, attempts, responses):
    validate_variant(cases, base_cases, base_sources)
    if payload_digest(manifest) != payload_digest(
        phase_manifest(
            cases,
            manifest["run_id"],
            manifest["evidence_kind"],
            pair_id=manifest.get("extension_pair_id", PAIR_ID),
        )
    ):
        raise ValueError("Phase manifest differs from fixed candidate and execution scope")
    by_id = {c["case_id"]: c for c in cases}
    scheduled = manifest["scheduled_case_ids"]
    maps = []
    for kind, records in (("request", requests), ("attempt", attempts), ("response", responses)):
        result = {}
        for row in records:
            cid = row.get("case_id")
            if not isinstance(cid, str) or cid not in scheduled or cid in result:
                raise ValueError("Foreign, unscheduled or duplicate phase record")
            wanted = expected_request(by_id[cid], manifest)
            identity = {k: v for k, v in wanted.items() if k != "payload"}
            if (
                any(row.get(k) != v for k, v in identity.items())
                or type(row.get("repair")) is not int
            ):
                raise ValueError("Phase record identity or request hash differs")
            if kind == "request" and payload_digest(row) != payload_digest(wanted):
                raise ValueError("Request payload differs from the frozen v2 variant")
            if kind == "attempt" or kind == "response" and row.get("cached") is not True:
                if type(row.get("retry_count")) is not int or row["retry_count"] != 0:
                    raise ValueError("No retries are allowed in this two-input phase")
            result[cid] = row
        maps.append(result)
    submitted, dispatched, received = maps
    if not set(dispatched) <= set(submitted) or not set(received) <= set(submitted):
        raise ValueError("Attempt or response lacks its unique request")
    rows = []
    for case in cases:
        cid = case["case_id"]
        answer, mode, choice = "not_scheduled", "not_scheduled", None
        cached = False
        if cid in scheduled:
            answer, mode = "unsubmitted", "unobserved"
            if cid in submitted:
                answer = "request_record_only"
            if cid in dispatched:
                aware_time(dispatched[cid]["dispatched_at"])
                answer = "awaiting_response"
            if cid in received:
                row = received[cid]
                if type(row.get("cached")) is not bool:
                    raise ValueError("Response must explicitly identify cache replay")
                cached = row["cached"]
                if cached:
                    if cid in dispatched:
                        raise ValueError("A cached answer cannot also claim a new dispatch")
                    answer, mode = "cache_replay", "not_new_execution"
                else:
                    if cid not in dispatched:
                        raise ValueError("Response lacks a dispatch")
                    start, finish = (
                        aware_time(row["requested_at"]),
                        aware_time(row["response_received_at"]),
                    )
                    if start != aware_time(dispatched[cid]["dispatched_at"]) or finish < start:
                        raise ValueError("Response clocks differ from its dispatch")
                    status = row["http_status"]
                    if not (
                        type(status) is int
                        and 100 <= status <= 599
                        or status in ("transport_error", "cancelled_in_flight")
                    ):
                        raise ValueError("Invalid HTTP/transport status")
                    if status == 200:
                        answer, choice = parse_choice(row["response"], case)
                        mode = returned_mode(row["response"])
                    else:
                        if row["response"] is not None:
                            raise ValueError("Failed transport cannot contain an answer")
                        answer, mode = "transport_failed", "unobserved"
        rows.append(
            {
                "case_id": cid,
                "scheduled": cid in scheduled,
                "answer_contract_status": answer,
                "mode_contract_status": mode,
                "choice": choice,
                "cache_replay": cached,
                "compatible": answer == "valid_linked_answer"
                and mode == "no_returned_reasoning_observed"
                and not cached,
            }
        )
    selected = {r["case_id"]: r for r in rows if r["scheduled"]}
    a, b = [selected[cid] for cid in scheduled]
    if scheduled[1] in submitted and not a["compatible"]:
        raise ValueError("Second request violates stop-on-first-failure phase policy")
    if scheduled[1] in dispatched and aware_time(
        dispatched[scheduled[1]]["dispatched_at"]
    ) < aware_time(received[scheduled[0]]["response_received_at"]):
        raise ValueError("Phase dispatches must be sequential")
    complete = a["compatible"] and b["compatible"]
    synthetic = manifest["evidence_kind"] == "synthetic_tool_test"
    return {
        "protocol_version": NAMESPACE,
        "stage": STAGE,
        "run_id": manifest["run_id"],
        "evidence_kind": manifest["evidence_kind"],
        "evidence_scope": "synthetic_tool_test"
        if synthetic
        else "local_logs_not_externally_authenticated",
        "candidate_count": 4,
        "scheduled_count": 2,
        "not_scheduled_count": 2,
        "submitted_count": len(submitted),
        "logged_post_attempts": len(dispatched),
        "scheduled_missing": sum(
            r["answer_contract_status"]
            in {"unsubmitted", "request_record_only", "awaiting_response"}
            for r in selected.values()
        ),
        "compatible_count": sum(r["compatible"] for r in selected.values()),
        "phase_status": (
            "synthetic_compatibility_pair_passed" if synthetic else "compatibility_pair_passed"
        )
        if complete
        else "incomplete",
        "choice_changed": a["choice"]["target_style"] != b["choice"]["target_style"]
        if complete
        else None,
        "cases": rows,
        "internal_thinking_disabled_proven": False,
        "real_model_execution_verified": False,
        "research_eligible": False,
        "four_case_behavior_study_complete": False,
        "human_alignment_established": False,
        "model_calls_performed_by_evaluator": 0,
    }
