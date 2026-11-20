"""Authenticate a fixed matched-choice log prefix and collect without retries.

Callers must rebuild/validate packets against their frozen sources and authorize
the entire batch before constructing a real authenticated HTTP client.
"""

import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime

import httpx

from behavioral_market.benchmarks import presentation_comparison as panel_tool
from behavioral_market.evaluation.choice_presentation import parse_answer, signature
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
require = panel_tool.require


def identity(request):
    return {k: v for k, v in request.items() if k != "payload"}


def authenticate_logs(packet, requests, attempts, responses):
    """Match complete wire and evidence identity before common-panel adaptation."""
    expected = {r["slot_id"]: r for r in packet["requests"]}
    maps = []
    for name, records in zip(LOGS, (requests, attempts, responses), strict=True):
        observed = {}
        for row in records:
            require(isinstance(row, dict), "log row schema")
            sid = row.get("slot_id")
            require(sid in expected and sid not in observed, "foreign/duplicate log slot")
            base = identity(expected[sid])
            require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
                "log identity differs",
            )
            if name == LOGS[0]:
                require(canonical(row) == canonical(expected[sid]), "actual wire differs")
            else:
                extra = (
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
                require(set(row) == set(base) | extra, "unexpected log fields")
                require(
                    type(row["retry_count"]) is int and row["retry_count"] == 0, "retry forbidden"
                )
            observed[sid] = row
        require(list(observed) == list(expected)[: len(observed)], "fixed log order differs")
        maps.append(observed)
    submitted, dispatched, received = maps
    require(set(received) <= set(dispatched) <= set(submitted), "orphan log")
    common = {panel_tool.key(r): r for r in packet["panel"]["opportunities"]}
    rows, adapted, finish_previous = [], [], None
    for request in expected.values():
        sid = request["slot_id"]
        if sid in submitted:
            require(not rows or rows[-1]["compatible"], "request after incomplete/invalid reply")
        choice, mode, body, header = None, None, None, None
        status = "request_record_only" if sid in submitted else "unsubmitted"
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            require(finish_previous is None or start >= finish_previous, "overlapping dispatch")
            status = "awaiting_response"
        if sid in received:
            row = received[sid]
            require(row["cached"] is False, "local replay forbidden")
            start, finish = aware_time(row["requested_at"]), aware_time(row["response_received_at"])
            require(
                start == aware_time(dispatched[sid]["dispatched_at"]) and finish >= start,
                "clock differs",
            )
            finish_previous = finish
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
            adapted.append(
                {
                    **deepcopy(common[panel_tool.key(request)]),
                    "status": "valid" if choice else "invalid",
                    "choice": choice,
                }
            )
        rows.append(
            {
                **identity(request),
                "compatible": choice is not None,
                "status": status,
                "choice": choice,
                "returned_mode": mode,
                "cache_key_header": header,
                "strong_signature": signature(body),
                "body_sha256": digest(body) if body is not None else None,
            }
        )
    groups, cells = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["strong_signature"] is not None:
            groups[digest(row["strong_signature"])].append(row["slot_id"])
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
        "local_logs_match_frozen_packet": True,
        "protocol_complete": all(r["compatible"] for r in rows),
        "logged_dispatch_attempts": len(dispatched),
        "rows": rows,
        "adapted": adapted,
        "same_input_variability": variability,
        "repeated_strong_signature_groups": [g for g in groups.values() if len(g) > 1],
        "unknown_strong_signature_slots": [
            r["slot_id"] for r in rows if r["strong_signature"] is None
        ],
        "independent_generation_count": None,
    }


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines()]


async def collect(packet, output, client, targets, evaluate, authorization, secret=""):
    """Require quota for the complete frozen batch; never silently shorten it."""
    prior, cap = authorization["prior_POSTs"], authorization["global_cap"]
    count = len(packet["requests"])
    require(
        type(prior) is int and type(cap) is int and prior >= 0 and cap >= prior + count,
        "complete batch exceeds authorized cap",
    )
    evaluate(packet, [], [], [], targets)
    output.mkdir(parents=True, exist_ok=False)
    for name in LOGS:
        (output / name).touch()
    _save(output / "packet.json", packet)
    state = {"run_id": packet["run_id"], "status": "running", "stop_reason": None}

    def scored():
        return evaluate(packet, *(read_jsonl(output / n) for n in LOGS), targets)

    def log(name, row):
        text = canonical(row)
        if secret:
            text = text.replace(secret, "[REDACTED]")
        with (output / name).open("a") as handle:
            handle.write(text + "\n")

    def ledger(running):
        sent = len(read_jsonl(output / LOGS[1]))
        reserved = count - sent if running else 0
        _save(
            output / "budget_ledger.json",
            {
                "global_authorization_source": authorization["grant_source"],
                "global_authorized_completion_POST_cap": cap,
                "used_completion_POSTs": prior + sent,
                "new_logged_dispatches": sent,
                "reserved_next_completion_POSTs": reserved,
                "available_unreserved_POSTs": cap - prior - sent - reserved,
                "active_run_id": state["run_id"] if running else None,
                "status": "running" if running else "terminal_pending_audit",
                "no_resume": True,
            },
        )

    _save(output / "execution_state.json", state)
    ledger(True)
    try:
        registry = await client.get("https://aigc-api.hkust-gz.edu.cn/api/v2/models")
        require(registry.status_code == 200, "registry HTTP")
        ids = [r["id"] for r in strict_json(registry.text)["data"]]
        require(packet["model"] in ids, "model absent")
        _save(output / "model_registry.json", {"model_ids": ids})
        for request in packet["requests"]:
            before = scored()
            index = request["position"] - 1
            require(before["logged_dispatch_attempts"] == index, "dispatch prefix")
            require(all(r["compatible"] for r in before["rows"][:index]), "prior invalid")
            require(
                all(r["status"] == "unsubmitted" for r in before["rows"][index:]),
                "unexpected pending",
            )
            log(LOGS[0], request)
            base, start = identity(request), datetime.now(UTC).isoformat()
            log(LOGS[1], {**base, "dispatched_at": start, "retry_count": 0})
            ledger(True)
            begun, http, body, cancelled = time.perf_counter(), "transport_error", None, False
            header = {"present": False, "value_sha256": None}
            try:
                response = await client.post(packet["endpoint"], json=request["payload"])
                http = response.status_code
                cache_key = response.headers.get("x-litellm-cache-key")
                header = {
                    "present": cache_key is not None,
                    "value_sha256": hashlib.sha256(cache_key.encode()).hexdigest()
                    if cache_key is not None
                    else None,
                }
                if http == 200:
                    try:
                        body = strict_json(response.text)
                    except ValueError:
                        body = {"_invalid_json_text": response.text}
            except asyncio.CancelledError:
                http, cancelled = "cancelled_in_flight", True
            except (httpx.HTTPError, ValueError):
                pass
            log(
                LOGS[2],
                {
                    **base,
                    "requested_at": start,
                    "response_received_at": datetime.now(UTC).isoformat(),
                    "retry_count": 0,
                    "cached": False,
                    "http_status": http,
                    "latency_seconds": time.perf_counter() - begun,
                    "response": body,
                    "cache_key_header": header,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            result = scored()
            _save(output / "evaluation.json", result)
            if not result["rows"][index]["compatible"]:
                state.update(status="stopped", stop_reason="protocol failure")
                break
            print(
                canonical({"completed_slots": index + 1, "used_POSTs": prior + index + 1}),
                flush=True,
            )
        else:
            state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="cancelled in flight; no resume")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        _save(output / "execution_state.json", state)
        ledger(False)
    result = scored()
    _save(output / "evaluation.json", result)
    return result
