"""Allowlisted aggregate-reference adapter and matched collection authentication.

The target rate is deliberately absent from this module's packet builder. Source
authentication and actual transport belong to the bounded collection script.
"""

import math
from collections import defaultdict
from copy import deepcopy

from behavioral_market.evaluation.choice_presentation import parse_answer, signature
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time

from . import presentation_comparison as panel_tool

PROTOCOL = "aggregate-reference-comparison-v1"
TARGET = 384
REFERENCES = (166, 211, 542)
METHODS = ("true_reference", "shuffled_reference")
PAIRINGS = {METHODS[0]: (166, 211, 542), METHODS[1]: (211, 542, 166)}
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
MODEL = "DeepSeek-V4-Flash"
LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
TEXT_FIELDS = {"item_id", "option_a_text", "option_b_text"}
CONTROLS = {
    "model": MODEL,
    "temperature": 0.7,
    "max_tokens": 32,
    "enable_thinking": False,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    "cache": {"no-cache": True, "no-store": True},
}
INTRO = (
    "Aggregate choices on other questions are provided as reference information. "
    "Each fraction describes participants choosing that reference's original Option B. "
    "Use these references to guide your choice on the new question. "
    "They do not give the new question's participant choice rate."
)

require = panel_tool.require


def _text(row, expected_id, extra=()):
    require(isinstance(row, dict) and set(row) == TEXT_FIELDS | set(extra), "source schema")
    require(type(row["item_id"]) is int and row["item_id"] == expected_id, "source item ID")
    require(
        all(isinstance(row[k], str) and row[k].strip() for k in TEXT_FIELDS - {"item_id"}),
        "source option text",
    )


def checked_materials(materials):
    require(
        isinstance(materials, dict) and set(materials) == {"target", "references"},
        "only target text and reference rows allowed",
    )
    _text(materials["target"], TARGET)
    refs = materials["references"]
    require(isinstance(refs, list) and len(refs) == 3, "three fixed references")
    target_pair = tuple(materials["target"][k] for k in ("option_a_text", "option_b_text"))
    for row, item in zip(refs, REFERENCES, strict=True):
        _text(row, item, ("human_b_rate",))
        rate = row["human_b_rate"]
        require(
            type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1,
            "finite reference proportion",
        )
        pair = tuple(row[k] for k in ("option_a_text", "option_b_text"))
        require(pair != target_pair and pair[::-1] != target_pair, "reference equals target")
    require(len({r["human_b_rate"] for r in refs}) > 1, "reference conditions are identical")
    return deepcopy(materials)


def reference_cards(materials, method):
    """Local source IDs select proportions; neither method nor source IDs enter wire."""
    materials = checked_materials(materials)
    require(method in METHODS, "unknown method")
    rates = {r["item_id"]: r["human_b_rate"] for r in materials["references"]}
    return [
        {
            "option_a_text": row["option_a_text"],
            "option_b_text": row["option_b_text"],
            "assigned_rate": rates[source],
        }
        for row, source in zip(materials["references"], PAIRINGS[method], strict=True)
    ]


def _context(cards):
    return INTRO + "".join(
        f"\n\nReference {i}:\nOriginal Option A:\n{r['option_a_text']}"
        f"\n\nOriginal Option B:\n{r['option_b_text']}"
        f"\n\nFraction choosing original Option B: {canonical(r['assigned_rate'])}"
        for i, r in enumerate(cards, 1)
    )


def make_packet(materials, run_id, evidence_kind="synthetic_tool_test"):
    materials = checked_materials(materials)
    require(isinstance(run_id, str) and bool(run_id.strip()), "run ID")
    require(evidence_kind in {"synthetic_tool_test", "recorded_model_logs"}, "evidence kind")
    target = materials["target"]
    panel = panel_tool.make_panel(
        {TARGET: (target["option_a_text"], target["option_b_text"])}, list(METHODS), 2
    )
    presentations = {
        (p["content_order"], p["letter_order"]): p["presentation"] for p in panel["opportunities"]
    }
    requests = []
    for content, letters in panel_tool.FORMATS:
        for repeat in range(2):
            for method in METHODS if repeat == 0 else METHODS[::-1]:
                presentation = presentations[content, letters]
                payload = deepcopy(CONTROLS)
                payload["messages"] = deepcopy(presentation["messages"])
                payload["messages"][1]["content"] = (
                    _context(reference_cards(materials, method))
                    + "\n\nNew question:\n"
                    + payload["messages"][1]["content"]
                )
                ident = {
                    "run_id": run_id,
                    "evidence_kind": evidence_kind,
                    "slot_id": f"slot_{len(requests) + 1:02d}",
                    "position": len(requests) + 1,
                    "item_id": TARGET,
                    "method_id": method,
                    "content_order": content,
                    "letter_order": letters,
                    "repeat_id": repeat,
                    "wire_sha256": digest(payload),
                    "presentation_sha256": digest(presentation),
                }
                requests.append(
                    {
                        **ident,
                        "request_hash": digest([payload, ident, PROTOCOL]),
                        "payload": payload,
                    }
                )
    core = {
        "protocol": PROTOCOL,
        "evidence_kind": evidence_kind,
        "run_id": run_id,
        "materials": materials,
        "panel": panel,
        "requests": requests,
        "endpoint": ENDPOINT,
    }
    return {**core, "packet_sha256": digest(core)}


def validate_packet(packet):
    rebuilt = make_packet(packet["materials"], packet["run_id"], packet["evidence_kind"])
    require(canonical(packet) == canonical(rebuilt), "packet differs from rebuilt materials")
    return rebuilt


def identity(request):
    return {k: v for k, v in request.items() if k != "payload"}


def evaluate(packet, requests, attempts, responses, targets):
    """Authenticate a sequential log prefix before adapting to the common panel."""
    packet = validate_packet(packet)
    expected = {r["slot_id"]: r for r in packet["requests"]}
    maps = []
    for name, records in zip(LOGS, (requests, attempts, responses), strict=True):
        observed = {}
        for row in records:
            sid = row.get("slot_id")
            require(sid in expected and sid not in observed, "foreign or duplicate log slot")
            base = identity(expected[sid])
            require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
                "log identity differs",
            )
            if name == LOGS[0]:
                require(canonical(row) == canonical(expected[sid]), "actual wire differs")
            else:
                fields = (
                    {"dispatched_at", "retry_count"}
                    if name == LOGS[1]
                    else {
                        "requested_at",
                        "response_received_at",
                        "retry_count",
                        "cached",
                        "http_status",
                        "latency_seconds",
                        "response",
                        "cache_key_header",
                    }
                )
                require(set(row) == set(base) | fields, "unexpected log fields")
                require(type(row["retry_count"]) is int and row["retry_count"] == 0, "retry")
            observed[sid] = row
        require(list(observed) == list(expected)[: len(observed)], "fixed log order differs")
        maps.append(observed)
    submitted, dispatched, received = maps
    require(set(received) <= set(dispatched) <= set(submitted), "orphan log")
    planned = {panel_tool.key(r): r for r in packet["panel"]["opportunities"]}
    rows, adapted, previous_finish = [], [], None
    for req in expected.values():
        sid = req["slot_id"]
        if sid in submitted:
            require(not rows or rows[-1]["compatible"], "request after incomplete/invalid reply")
        body, choice, mode = None, None, None
        status = "request_record_only" if sid in submitted else "unsubmitted"
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            require(previous_finish is None or start >= previous_finish, "overlapping dispatch")
            status = "awaiting_response"
        if sid in received:
            row = received[sid]
            require(row["cached"] is False, "local replay")
            start, finish = aware_time(row["requested_at"]), aware_time(row["response_received_at"])
            require(
                start == aware_time(dispatched[sid]["dispatched_at"]) and finish >= start, "clock"
            )
            previous_finish = finish
            latency, http, header = (
                row["latency_seconds"],
                row["http_status"],
                row["cache_key_header"],
            )
            require(
                type(latency) in (int, float) and math.isfinite(latency) and latency >= 0, "latency"
            )
            require(
                (type(http) is int and 100 <= http <= 599)
                or http in ("transport_error", "cancelled_in_flight"),
                "HTTP status",
            )
            require(
                isinstance(header, dict)
                and set(header) == {"present", "value_sha256"}
                and type(header["present"]) is bool
                and (
                    header["value_sha256"] is None
                    if not header["present"]
                    else isinstance(header["value_sha256"], str)
                    and len(header["value_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in header["value_sha256"])
                ),
                "cache header",
            )
            if http == 200:
                body = row["response"]
                choice, mode = parse_answer(body)
                status = "compatible" if choice is not None else "invalid_answer_or_mode"
            else:
                require(row["response"] is None, "failed transport body")
                status = "transport_failed"
            common = planned[panel_tool.key(req)]
            adapted.append(
                {**deepcopy(common), "status": "valid" if choice else "invalid", "choice": choice}
            )
        rows.append(
            {
                **identity(req),
                "status": status,
                "compatible": choice is not None,
                "choice": choice,
                "returned_mode": mode,
                "strong_signature": signature(body),
                "body_sha256": digest(body) if body is not None else None,
            }
        )
    comparison = panel_tool.compare_panel(
        packet["panel"], adapted, targets, left_method=METHODS[0], right_method=METHODS[1]
    )
    signatures = defaultdict(list)
    for row in rows:
        if row["strong_signature"] is not None:
            signatures[digest(row["strong_signature"])].append(row["slot_id"])
    cells = defaultdict(list)
    for row in rows:
        cells[row["method_id"], row["content_order"], row["letter_order"]].append(row)
    variability = [
        {
            "method_id": k[0],
            "content_order": k[1],
            "letter_order": k[2],
            "slot_ids": [r["slot_id"] for r in group],
            "both_valid": all(r["compatible"] for r in group),
            "different_display_choices": len({r["choice"] for r in group}) > 1
            if all(r["compatible"] for r in group)
            else None,
        }
        for k, group in cells.items()
    ]
    return {
        "evidence_kind": packet["evidence_kind"],
        "local_logs_match_frozen_packet": True,
        "protocol_complete": all(r["compatible"] for r in rows),
        "logged_dispatch_attempts": len(dispatched),
        "used_completion_POSTs": 75 + len(dispatched),
        "rows": rows,
        "comparison": comparison,
        "same_input_variability": variability,
        "repeated_strong_signature_groups": [g for g in signatures.values() if len(g) > 1],
        "independent_generation_count": None,
        "overall_human_alignment_established": False,
        "limit": "One fixed calibration target; authenticates local logs, not independent "
        "generation, general human alignment, personal memory or market realism.",
    }
