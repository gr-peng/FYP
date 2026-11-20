"""Two-item presentation diagnostic; probabilities describe received records only."""

from collections import defaultdict
from copy import deepcopy

from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import MODE_CONTROL, returned_mode

NAMESPACE = "choices13k-presentation-2x2-v1"
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
MODEL = "DeepSeek-V4-Flash"
CACHE = {"no-cache": True, "no-store": True}
LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
SUFFIX = "\n\nReturn JSON only with exactly one field named choice whose value is A or B."
FORMATS = (("AB", "AB"), ("AB", "BA"), ("BA", "BA"), ("BA", "AB"))
SCHEDULE = [
    {
        "slot_id": f"slot_{n + 1:02d}",
        "item_id": item,
        "content_order": c,
        "letter_order": labels,
        "repeat_id": repeat,
    }
    for n, (item, repeat, c, labels) in enumerate(
        (item, repeat, c, labels)
        for repeat, item, formats in (
            (0, 166, FORMATS),
            (0, 211, FORMATS[::-1]),
            (1, 211, FORMATS),
            (1, 166, FORMATS[::-1]),
        )
        for c, labels in formats
    )
]


def require(value, message):
    if not value:
        raise ValueError(message)


def split_user(text):
    require(isinstance(text, str) and text.endswith(SUFFIX), "fixed user suffix required")
    core = text[: -len(SUFFIX)]
    require(
        core.count("\n\nOption A:\n") == core.count("\n\nOption B:\n") == 1,
        "exact original option titles required",
    )
    head, rest = core.split("\n\nOption A:\n")
    first, second = rest.split("\n\nOption B:\n")
    require(bool(first) and bool(second), "empty original option block")
    return head, first, second


def materialize(bases):
    require(set(bases) == {"166", "211"}, "only two fixed items")
    materials = {}
    for item, pair in bases.items():
        require(set(pair) == {"A_first", "B_first"}, "both old orientations required")
        parsed = {}
        for order, payload in pair.items():
            require(
                set(payload)
                == {
                    "model",
                    "temperature",
                    "max_tokens",
                    "enable_thinking",
                    "response_format",
                    "messages",
                },
                "old payload keys differ",
            )
            require(
                payload["model"] == MODEL
                and type(payload["temperature"]) is float
                and payload["temperature"] == 0.7
                and type(payload["max_tokens"]) is int
                and payload["max_tokens"] == 32
                and payload["enable_thinking"] is False
                and payload["response_format"] == {"type": "json_object"},
                "old settings differ",
            )
            messages = payload["messages"]
            require(
                len(messages) == 2
                and [m["role"] for m in messages] == ["system", "user"]
                and all(set(m) == {"role", "content"} for m in messages),
                "old messages differ",
            )
            parsed[order] = split_user(messages[1]["content"])
        a, b = parsed["A_first"], parsed["B_first"]
        require(a == (b[0], b[2], b[1]), "original option blocks do not reverse exactly")
        left, right = deepcopy(pair["A_first"]), deepcopy(pair["B_first"])
        left["messages"][1]["content"] = right["messages"][1]["content"] = "[option blocks]"
        require(canonical(left) == canonical(right), "non-option messages or settings changed")
        materials[item] = {
            "base_payload": deepcopy(pair["A_first"]),
            "head": a[0],
            "originals": {"A": a[1], "B": a[2]},
        }
    return materials


def packet(plan, bases):
    require(
        plan["schedule"] == SCHEDULE and plan["selected_item_ids"] == [166, 211],
        "fixed16 schedule differs",
    )
    require(
        plan["endpoint"] == ENDPOINT
        and plan["model"] == MODEL
        and type(plan["prior_POSTs"]) is int
        and plan["prior_POSTs"] == 59
        and type(plan["max_new_completion_POSTs"]) is int
        and plan["max_new_completion_POSTs"] == 16
        and plan["batch_cumulative_stop"] == 75
        and plan["compatibility_calls_outside_allocation"] == 0,
        "fixed route/budget differs",
    )
    require(
        plan["evidence_kind"] in {"recorded_model_logs", "synthetic_tool_test"}, "evidence kind"
    )
    require(
        canonical(plan["technical_delta"])
        == canonical({"chat_template_kwargs": MODE_CONTROL, "cache": CACHE}),
        "fixed technical delta differs",
    )
    materials, requests = materialize(bases), []
    for i, slot in enumerate(SCHEDULE):
        content, labels = slot["content_order"], slot["letter_order"]
        source = materials[str(slot["item_id"])]
        user = (
            source["head"]
            + "".join(
                f"\n\nOption {label}:\n{source['originals'][original]}"
                for original, label in zip(content, labels, strict=True)
            )
            + SUFFIX
        )
        payload = deepcopy(source["base_payload"])
        payload["messages"][1]["content"] = user
        payload.update(chat_template_kwargs=deepcopy(MODE_CONTROL), cache=deepcopy(CACHE))
        identity = {
            "kind": NAMESPACE,
            "probe_run_id": plan["frozen_run_id"],
            "evidence_kind": plan["evidence_kind"],
            **slot,
            "position": i + 1,
            "repair": 0,
        }
        requests.append(
            {**identity, "request_hash": digest([payload, identity, NAMESPACE]), "payload": payload}
        )
    core = {
        "protocol": NAMESPACE,
        "plan": deepcopy(plan),
        "bases": deepcopy(bases),
        "requests": requests,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "prior_POSTs": 59,
        "batch_cumulative_stop": 75,
        "budget_sources": plan["budget_sources"],
    }
    return {**core, "packet_sha256": digest(core)}


def identity(request):
    return {k: v for k, v in request.items() if k != "payload"}


def parse_answer(body):
    mode = returned_mode(body)
    try:
        require(isinstance(body, dict) and body.get("model") == MODEL, "model")
        choices = body["choices"]
        require(isinstance(choices, list) and len(choices) == 1, "one choice")
        require(choices[0]["finish_reason"] == "stop", "finish")
        msg = choices[0]["message"]
        require(msg["role"] == "assistant" and isinstance(msg["content"], str), "message")
        answer = strict_json(msg["content"])
        require(
            isinstance(answer, dict)
            and set(answer) == {"choice"}
            and answer["choice"] in ("A", "B"),
            "strict A/B contract",
        )
        return answer["choice"] if mode == "no_returned_reasoning_observed" else None, mode
    except (KeyError, IndexError, TypeError, ValueError):
        return None, mode


def signature(body):
    if not isinstance(body, dict):
        return None
    identifier, created = body.get("id"), body.get("created")
    if not (
        isinstance(identifier, str) and identifier.strip() and type(created) is int and created >= 0
    ):
        return None
    try:
        text = body["choices"][0]["message"]["content"]
        return [identifier, created, digest(text)] if isinstance(text, str) else None
    except (KeyError, IndexError, TypeError):
        return None


def evaluate(package, requests, attempts, responses):
    require(
        canonical(package) == canonical(packet(package["plan"], package["bases"])),
        "packet differs from fixed materials",
    )
    expected, maps = package["requests"], []
    wanted = {r["slot_id"]: r for r in expected}
    for kind, records in zip(LOGS, (requests, attempts, responses), strict=True):
        mapped = {}
        for row in records:
            sid = row.get("slot_id")
            require(sid in wanted and sid not in mapped, "foreign or duplicate slot")
            base = identity(wanted[sid])
            require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
                "slot identity differs",
            )
            if kind == LOGS[0]:
                require(canonical(row) == canonical(wanted[sid]), "wire payload differs")
            else:
                fields = (
                    {"dispatched_at", "retry_count"}
                    if kind == LOGS[1]
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
                require(
                    type(row["retry_count"]) is int and row["retry_count"] == 0, "retry forbidden"
                )
            mapped[sid] = row
        require(list(mapped) == list(wanted)[: len(mapped)], "fixed log prefix differs")
        maps.append(mapped)
    submitted, dispatched, received = maps
    require(set(received) <= set(dispatched) <= set(submitted), "orphan logs")
    rows, previous_finish = [], None
    for req in expected:
        sid, body = req["slot_id"], None
        if sid in submitted:
            require(not rows or rows[-1]["compatible"], "request after protocol failure")
        choice, mode, status, header = None, None, "unsubmitted", None
        if sid in submitted:
            status = "request_record_only"
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            require(previous_finish is None or start >= previous_finish, "overlap")
            status = "awaiting_response"
        if sid in received:
            r = received[sid]
            require(r["cached"] is False, "local replay forbidden")
            start, finish = aware_time(r["requested_at"]), aware_time(r["response_received_at"])
            require(
                start == aware_time(dispatched[sid]["dispatched_at"]) and finish >= start,
                "response clock differs",
            )
            previous_finish = finish
            require(
                type(r["latency_seconds"]) in (int, float)
                and 0 <= r["latency_seconds"] < float("inf"),
                "latency",
            )
            http, header = r["http_status"], r["cache_key_header"]
            require(
                type(http) is int
                and 100 <= http <= 599
                or http in ("transport_error", "cancelled_in_flight"),
                "HTTP status",
            )
            require(
                set(header) == {"present", "value_sha256"}
                and type(header["present"]) is bool
                and (
                    header["value_sha256"] is None
                    if not header["present"]
                    else isinstance(header["value_sha256"], str)
                    and len(header["value_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in header["value_sha256"])
                ),
                "header diagnostic",
            )
            if http == 200:
                body = r["response"]
                choice, mode = parse_answer(body)
                status = "compatible" if choice is not None else "invalid_answer_or_mode"
            else:
                require(r["response"] is None, "failed transport carries body")
                status = "transport_failed"
        pos = req["letter_order"].index(choice) + 1 if choice is not None else None
        original = req["content_order"][pos - 1] if pos is not None else None
        bpos = req["content_order"].index("B") + 1
        rows.append(
            {
                "slot_id": sid,
                "item_id": req["item_id"],
                "content_order": req["content_order"],
                "letter_order": req["letter_order"],
                "repeat_id": req["repeat_id"],
                "original_B_position": bpos,
                "original_B_letter": req["letter_order"][bpos - 1],
                "status": status,
                "compatible": choice is not None,
                "choice": choice,
                "selected_original": original,
                "selected_position": pos,
                "strong_signature": signature(body),
                "body_sha256": digest(body) if body else None,
                "cache_key_header": header,
            }
        )
    cells, contrasts = [], []
    for item in (166, 211):
        rates = {}
        for content, labels in FORMATS:
            group = [
                r
                for r in rows
                if (r["item_id"], r["content_order"], r["letter_order"]) == (item, content, labels)
            ]
            complete = all(r["compatible"] for r in group)
            rate = sum(r["selected_original"] == "B" for r in group) / 2 if complete else None
            bpos, bletter = group[0]["original_B_position"], group[0]["original_B_letter"]
            rates[bpos, bletter] = rate
            cells.append(
                {
                    "item_id": item,
                    "content_order": content,
                    "letter_order": labels,
                    "original_B_position": bpos,
                    "original_B_letter": bletter,
                    "slot_ids": [r["slot_id"] for r in group],
                    "display_choices": [r["choice"] for r in group],
                    "original_choices": [r["selected_original"] for r in group],
                    "selected_positions": [r["selected_position"] for r in group],
                    "valid_returns": sum(r["compatible"] for r in group),
                    "received_original_B_rate": rate,
                }
            )
        for letter in ("A", "B"):
            hi, lo = rates[1, letter], rates[2, letter]
            contrasts.append(
                {
                    "item_id": item,
                    "factor": "position_first_minus_second",
                    "fixed_original_B_letter": letter,
                    "difference": hi - lo if hi is not None and lo is not None else None,
                }
            )
        for position in (1, 2):
            hi, lo = rates[position, "B"], rates[position, "A"]
            contrasts.append(
                {
                    "item_id": item,
                    "factor": "letter_B_minus_A",
                    "fixed_original_B_position": position,
                    "difference": hi - lo if hi is not None and lo is not None else None,
                }
            )
    groups = defaultdict(list)
    for row in rows:
        if row["strong_signature"] is not None:
            groups[digest(row["strong_signature"])].append(row["slot_id"])
    return {
        "protocol_complete": all(r["compatible"] for r in rows),
        "rows": rows,
        "cells": cells,
        "contrasts": contrasts,
        "logged_dispatch_attempts": len(dispatched),
        "valid_returns": sum(r["compatible"] for r in rows),
        "repeated_id_created_text_groups": [g for g in groups.values() if len(g) > 1],
        "unknown_strong_signature_slots": [
            r["slot_id"] for r in rows if r["strong_signature"] is None
        ],
        "independent_generation_count": None,
        "human_alignment_established": False,
        "cache_control_effect_established": False,
        "limit": "Two previously inspected items;shared descriptive contrasts,not independent "
        "sampling,causal preference attribution or human alignment confirmation.",
    }
