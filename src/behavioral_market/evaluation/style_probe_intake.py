"""Link local probe execution logs without authenticating a real external model call."""

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import ValidationError

from behavioral_market.evaluation.style_probe_materials import payload_digest, validate_share_probes
from behavioral_market.llm.schemas import StyleDecision

NAMESPACE = "style-share-probe-v1"
EVIDENCE_KINDS = {"recorded_model_logs", "synthetic_tool_test"}
MAX_ATTEMPTS = 3


def strict_json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result

    return json.loads(
        text, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite_float
    )


def make_run_manifest(cases, run_id, evidence_kind):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Use a nonempty, bounded probe run identifier")
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError("Unknown evidence kind; a real=true label is not evidence")
    return {
        "protocol_version": NAMESPACE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "materials_sha256": payload_digest(cases),
        "expected_case_ids": sorted(c["case_id"] for c in cases),
        "repair": 0,
        "max_transport_attempts": MAX_ATTEMPTS,
    }


def request_identity(case, manifest):
    return {
        "kind": "style_share_probe",
        "probe_run_id": manifest["run_id"],
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "condition": case["condition"],
        "evidence_kind": manifest["evidence_kind"],
        "repair": 0,
    }


def expected_request(case, manifest):
    """Use the original client's request-hash algorithm, not the payload-only digest."""
    identity = request_identity(case, manifest)
    key = hashlib.sha256(
        json.dumps([case["payload"], identity, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    return {**identity, "request_hash": key, "payload": copy.deepcopy(case["payload"])}


def aware_time(value):
    if not isinstance(value, str):
        raise ValueError("Log clocks must be aware ISO timestamps")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Log clocks must be aware ISO timestamps")
    return result


def retry_index(row):
    value = row.get("retry_count")
    if type(value) is not int or not 0 <= value < MAX_ATTEMPTS:
        raise ValueError("Retry index exceeds the fixed transport-attempt contract")
    return value


def check_identity(row, expected):
    for key, value in expected.items():
        if key != "payload" and (key not in row or row[key] != value):
            raise ValueError("Log identity or request hash does not match the frozen case")
    if type(row.get("repair")) is not int or row["repair"] != 0:
        raise ValueError("Format repairs are not allowed in this paired probe")


def parse_choice(body, case):
    if not isinstance(body, dict):
        return "invalid_response_envelope", None
    if body.get("model") != case["payload"]["model"]:
        return "model_mismatch", None
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return "invalid_response_envelope", None
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        return "invalid_response_envelope", None
    try:
        decision = StyleDecision.model_validate(strict_json(message["content"])).model_dump()
    except (ValidationError, ValueError):
        return "invalid_answer_schema", None
    coherent = (decision["decision"] == "stay") == (
        decision["target_style"] == case["source"]["from_style"]
    )
    return ("valid_linked_answer" if coherent else "inconsistent_style"), decision


def _evaluate(cases, sources, manifest, requests, attempts, responses, evidence_kind):
    validate_share_probes(cases, sources)
    if payload_digest(manifest) != payload_digest(
        make_run_manifest(cases, manifest["run_id"], evidence_kind)
    ):
        raise ValueError("Run manifest differs from the fixed inputs, namespace or evidence mode")
    expected = {c["case_id"]: expected_request(c, manifest) for c in cases}
    request_map, attempt_map, response_map, cached_map = (
        {},
        defaultdict(dict),
        defaultdict(dict),
        {},
    )
    for row in requests:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected:
            raise ValueError("Unknown request case")
        wanted = expected[case_id]
        if payload_digest(row) != payload_digest(wanted) or type(row.get("repair")) is not int:
            raise ValueError(
                "Request payload, settings, hash or identity differs from frozen input"
            )
        if case_id in request_map:
            raise ValueError("Duplicate request; do not select a preferred answer")
        request_map[case_id] = row
    for row in attempts:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in request_map:
            raise ValueError("Transport attempt has no matching request")
        check_identity(row, expected[case_id])
        allowed = set(request_identity(cases[0], manifest)) | {
            "request_hash",
            "retry_count",
            "dispatched_at",
        }
        if set(row) != allowed:
            raise ValueError("Unexpected transport-attempt fields")
        index = retry_index(row)
        aware_time(row["dispatched_at"])
        if index in attempt_map[case_id]:
            raise ValueError("Duplicate transport attempt")
        attempt_map[case_id][index] = row
    for row in responses:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in request_map:
            raise ValueError("Response has no matching request")
        check_identity(row, expected[case_id])
        base = set(request_identity(cases[0], manifest)) | {"request_hash", "cached", "response"}
        if type(row.get("cached")) is not bool:
            raise ValueError("Response must explicitly identify cache replay")
        if row["cached"]:
            if set(row) != base | {"status"} or row["status"] != "ok":
                raise ValueError("Unexpected cache-response format")
            if case_id in cached_map or response_map[case_id]:
                raise ValueError("Multiple responses or mixed cache/live records")
            cached_map[case_id] = row
            continue
        required = base | {
            "http_status",
            "retry_count",
            "requested_at",
            "response_received_at",
            "latency_seconds",
        }
        if not required <= set(row) or set(row) - required - {"trace_id", "retry_after_seconds"}:
            raise ValueError("Unexpected live-response format")
        index = retry_index(row)
        if index in response_map[case_id] or case_id in cached_map:
            raise ValueError("Duplicate response or mixed cache/live records")
        if index not in attempt_map[case_id]:
            raise ValueError("Response has no matching transport attempt")
        start = aware_time(row["requested_at"])
        finish = aware_time(row["response_received_at"])
        if start != aware_time(attempt_map[case_id][index]["dispatched_at"]) or finish < start:
            raise ValueError("Response clock differs from its matching dispatch")
        latency = row["latency_seconds"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid logged latency")
        # Client latency includes queuing/throttling before dispatched_at; no equality assertion.
        status = row["http_status"]
        if not (
            type(status) is int
            and 100 <= status <= 599
            or isinstance(status, str)
            and status in {"transport_error", "cancelled_in_flight"}
        ):
            raise ValueError("Unknown HTTP/transport status")
        if status != 200 and row["response"] is not None:
            raise ValueError("Failed transport must not carry a model answer")
        response_map[case_id][index] = row
    rows = []
    for case in cases:
        case_id = case["case_id"]
        dispatched, received = attempt_map[case_id], response_map[case_id]
        indices = sorted(dispatched)
        if indices != list(range(len(indices))):
            raise ValueError("Transport retry sequence has gaps")
        successful = [i for i, row in received.items() if row["http_status"] == 200]
        if len(successful) > 1:
            raise ValueError("Multiple successful responses; never select the last or best")
        for i in indices[1:]:
            if i - 1 not in received or received[i - 1]["http_status"] == 200:
                raise ValueError("Retry lacks a failed preceding response")
            if aware_time(dispatched[i]["dispatched_at"]) < aware_time(
                received[i - 1]["response_received_at"]
            ):
                raise ValueError("Retry precedes the preceding response")
        choice = None
        if case_id in cached_map:
            if dispatched:
                raise ValueError("Cache replay cannot also claim a new dispatch")
            status, choice = parse_choice(cached_map[case_id]["response"], case)
            status = "cache_replay_not_new_execution" if status == "valid_linked_answer" else status
        elif case_id not in request_map:
            status = "unsubmitted"
        elif not dispatched:
            status = "request_record_only"
        elif successful:
            status, choice = parse_choice(received[successful[0]]["response"], case)
        elif indices[-1] not in received:
            status = "awaiting_response"
        else:
            status = "transport_failed"
        rows.append(
            {
                "case_id": case_id,
                "pair_id": case["pair_id"],
                "condition": case["condition"],
                "from_style": case["source"]["from_style"],
                "status": status,
                "request_hash": expected[case_id]["request_hash"]
                if case_id in request_map
                else None,
                "payload_sha256": case["payload_sha256"],
                "logged_transport_attempts": len(dispatched),
                "failed_transport_responses": sum(
                    r["http_status"] != 200 for r in received.values()
                ),
                "cache_replay": case_id in cached_map,
                "valid_choice": status == "valid_linked_answer" and case_id not in cached_map,
                "choice": choice,
            }
        )
    pairs = []
    for pair_id in sorted({r["pair_id"] for r in rows}):
        pair = {r["condition"]: r for r in rows if r["pair_id"] == pair_id}
        a, b = pair["own_style_majority"], pair["other_style_majority"]
        usable = a["valid_choice"] and b["valid_choice"]
        pairs.append(
            {
                "pair_id": pair_id,
                "from_style": a["from_style"],
                "comparable": usable,
                "choice_changed": a["choice"]["target_style"] != b["choice"]["target_style"]
                if usable
                else None,
                "switched_under_own_majority": a["choice"]["decision"] == "switch"
                if usable
                else None,
                "switched_under_other_majority": b["choice"]["decision"] == "switch"
                if usable
                else None,
                "case_statuses": {r["condition"]: r["status"] for r in pair.values()},
            }
        )
    complete = all(r["valid_choice"] for r in rows)
    synthetic = evidence_kind == "synthetic_tool_test"
    state = "local_logs_complete" if complete else "incomplete" if requests else "not_started"
    if synthetic:
        state = "synthetic_test_complete" if complete else "synthetic_test_incomplete"
    errors = {
        "transport_failed",
        "invalid_response_envelope",
        "model_mismatch",
        "invalid_answer_schema",
        "inconsistent_style",
    }
    return {
        "run_id": manifest["run_id"],
        "evidence_kind": evidence_kind,
        "evidence_scope": "synthetic_tool_test_no_model_execution"
        if synthetic
        else "local_execution_logs_linked_not_externally_authenticated",
        "execution_status": state,
        "all_four_answers_usable": complete,
        "opportunities": 4,
        "pair_opportunities": 2,
        "valid_choice_count": sum(r["valid_choice"] for r in rows),
        "error_cases": sum(r["status"] in errors for r in rows),
        "missing_cases": sum(
            r["status"] in {"unsubmitted", "request_record_only", "awaiting_response"} for r in rows
        ),
        "cache_replays": len(cached_map),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "comparable_pairs": sum(p["comparable"] for p in pairs),
        "changed_pairs": sum(p["choice_changed"] for p in pairs) if complete else None,
        "logged_transport_attempts": len(attempts),
        "failed_transport_responses": sum(r["failed_transport_responses"] for r in rows),
        "cases": rows,
        "pairs": pairs,
        "model_calls_performed_by_intake": 0,
        "real_model_execution_verified": False,
        "research_eligible": False,
        "human_alignment_established": False,
        "causal_effect_established": False,
    }


def evaluate_recorded_probe(cases, sources, manifest, requests, attempts, responses):
    return _evaluate(cases, sources, manifest, requests, attempts, responses, "recorded_model_logs")


def evaluate_synthetic_probe(cases, sources, manifest, requests, attempts, responses):
    return _evaluate(cases, sources, manifest, requests, attempts, responses, "synthetic_tool_test")
