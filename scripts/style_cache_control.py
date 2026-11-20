"""Four fixed ABBA requests test documented joint proxy-cache controls."""

import argparse
import asyncio
import hashlib
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import httpx

from behavioral_market.evaluation import style_bank_square as square
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from scripts import style_bank_square as cli
from scripts.style_fact_check import fingerprint, read_jsonl, save

NAMESPACE = "documented-joint-cache-control-v1"
MODES = ("default", "no_read_no_write", "no_read_no_write", "default")
CONTROL = {"no-cache": True, "no-store": True}
LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")


def packet_for(workspace, plan):
    for path, fp in plan["source_files"].items():
        square.require(fingerprint(workspace / path) == fp, "source bytes differ")
    cli.checked_path(workspace, plan["documentation"])
    source = square.make_package(read_jsonl(workspace / BANK_PATH), plan)
    cli.verify_budget_sources(workspace, source)
    root = Path(source["execution_output"]).resolve()
    square.require(
        root.is_relative_to((workspace / "iterations").resolve()), "output outside iterations"
    )
    square.require(plan["probe_case_id"] == "synthetic_0100", "fixed original0100 required")
    square.require(plan["schedule"] == list(MODES), "fixed ABBA order required")
    case = next(c for c in source["cases"] if c["case_id"] == plan["probe_case_id"])
    slots = []
    for i, mode in enumerate(MODES):
        identity = {
            "kind": NAMESPACE,
            "probe_run_id": source["run_id"],
            "evidence_kind": source["evidence_kind"],
            "slot_id": f"slot_{i + 1}",
            "position": i + 1,
            "case_id": case["case_id"],
            "control_mode": mode,
            "repair": 0,
        }
        payload = deepcopy(case["payload"])
        if mode == "no_read_no_write":
            payload["cache"] = deepcopy(CONTROL)
        slots.append(
            {
                **identity,
                "request_hash": digest([payload, identity, NAMESPACE]),
                "payload": payload,
            }
        )
    core = {
        "protocol": NAMESPACE,
        "plan": deepcopy(plan),
        "source_square": source,
        "probe_case": case,
        "requests": slots,
        "purpose": "Joint cache-control compatibility observations, not behavior replication.",
        "independent_generation_count": None,
        "separate_no_cache_and_no_store_effects": None,
    }
    return {**core, "packet_sha256": digest(core)}


def checked_packet(workspace, packet):
    square.require(
        canonical(packet) == canonical(packet_for(workspace, packet["plan"])),
        "packet differs from source, schedule, control or budget",
    )


def identity(request):
    return {k: v for k, v in request.items() if k != "payload"}


def strong_signature(row):
    identifier, created, text = row["provider_id"], row["created"], row["text_sha256"]
    if (
        isinstance(identifier, str)
        and identifier.strip()
        and type(created) is int
        and created >= 0
        and text is not None
    ):
        return [identifier, created, text]
    return None


def evaluate(packet, requests, attempts, responses):
    """Rebuild prefix outcomes; no execution state flag can establish completion."""
    expected = packet["requests"]
    wanted = {r["slot_id"]: r for r in expected}
    maps = []
    for kind, records in zip(
        ("request", "attempt", "response"), (requests, attempts, responses), strict=True
    ):
        mapped = {}
        for row in records:
            sid = row.get("slot_id")
            square.require(sid in wanted and sid not in mapped, "foreign or duplicate slot")
            base = identity(wanted[sid])
            square.require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
                "slot identity/hash differs",
            )
            if kind == "request":
                square.require(canonical(row) == canonical(wanted[sid]), "wire payload differs")
            else:
                fields = (
                    {"dispatched_at", "retry_count"}
                    if kind == "attempt"
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
                square.require(set(row) == set(base) | fields, "unexpected log fields")
                square.require(
                    type(row["retry_count"]) is int and row["retry_count"] == 0,
                    "retry forbidden",
                )
            mapped[sid] = row
        square.require(list(mapped) == list(wanted)[: len(mapped)], "fixed prefix order differs")
        maps.append(mapped)
    submitted, dispatched, received = maps
    square.require(set(received) <= set(dispatched) <= set(submitted), "orphan logs")
    rows, previous_finish = [], None
    for request in expected:
        sid = request["slot_id"]
        if sid in submitted:
            square.require(not rows or rows[-1]["compatible"], "request after protocol failure")
        status, answer, mode, choice, body = "unsubmitted", None, None, None, None
        compatible = False
        if sid in submitted:
            status = "request_record_only"
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            square.require(previous_finish is None or start >= previous_finish, "overlap")
            status = "awaiting_response"
        header = None
        if sid in received:
            response = received[sid]
            square.require(response["cached"] is False, "local replay forbidden")
            start, finish = (
                aware_time(response["requested_at"]),
                aware_time(response["response_received_at"]),
            )
            square.require(
                start == aware_time(dispatched[sid]["dispatched_at"]) and finish >= start,
                "response clock differs",
            )
            previous_finish = finish
            latency = response["latency_seconds"]
            square.require(type(latency) in (int, float) and 0 <= latency < float("inf"), "latency")
            header = response["cache_key_header"]
            square.require(
                set(header) == {"present", "value_sha256"}
                and type(header["present"]) is bool
                and (
                    header["value_sha256"] is None
                    if not header["present"]
                    else isinstance(header["value_sha256"], str)
                    and len(header["value_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in header["value_sha256"])
                ),
                "cache header diagnostic differs",
            )
            http = response["http_status"]
            square.require(
                type(http) is int
                and 100 <= http <= 599
                or http in ("transport_error", "cancelled_in_flight"),
                "HTTP status differs",
            )
            if http == 200:
                body = response["response"]
                answer, mode, parsed = square.choice_status(body, packet["probe_case"])
                compatible = (
                    answer == "valid_linked_answer" and mode == "no_returned_reasoning_observed"
                )
                status = answer if compatible else "invalid_answer_or_mode"
                choice = parsed if compatible else None
            else:
                square.require(response["response"] is None, "failed transport carries answer")
                status = "transport_failed"
        text = body["choices"][0]["message"]["content"] if compatible else None
        rows.append(
            {
                "slot_id": sid,
                "control_mode": request["control_mode"],
                "status": status,
                "compatible": compatible,
                "answer_status": answer,
                "mode_status": mode,
                "choice": choice,
                "provider_id": body.get("id") if isinstance(body, dict) else None,
                "created": body.get("created") if isinstance(body, dict) else None,
                "text_sha256": digest(text) if text is not None else None,
                "body_sha256": digest(body) if body is not None else None,
                "cache_key_header": header,
            }
        )
    complete = all(r["compatible"] for r in rows)
    groups = {}
    for field in ("provider_id", "created", "text_sha256", "body_sha256", "id_created_text"):
        seen = {}
        for row in rows:
            value = strong_signature(row) if field == "id_created_text" else row.get(field)
            if field == "provider_id" and not (isinstance(value, str) and value.strip()):
                value = None
            if field == "created" and not (type(value) is int and value >= 0):
                value = None
            if value is not None:
                seen.setdefault(digest(value), []).append(row["slot_id"])
        groups[field] = [v for v in seen.values() if len(v) > 1]
    return {
        "protocol_complete": complete,
        "rows": rows,
        "logged_dispatch_attempts": len(dispatched),
        "return_reuse_groups": groups,
        "unobserved_strong_signature_slots": [
            row["slot_id"] for row in rows if strong_signature(row) is None
        ],
        "joint_controls_sent": deepcopy(CONTROL),
        "cache_control_effect_established": False,
        "separate_control_effects_established": False,
        "independent_generation_count": None,
        "human_alignment_established": False,
        "limit": (
            "HTTP200 does not prove cache controls executed. Missing cache-key header does "
            "not prove a miss. Even distinct signatures do not prove independent generations. "
            "ABBA only checks the joint controls; repeated returns remain outcomes."
        ),
    }


def scored(packet, output):
    return evaluate(packet, *(read_jsonl(output / n) for n in LOGS))


def describe(result):
    lines = [
        "# 四条接口缓存控制检查",
        "",
        f"记录到{result['logged_dispatch_attempts']}次发送尝试。",
        "",
        "| 次序 | 设置 | 状态 | 目标 | 缓存键响应头出现 |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        header = row["cache_key_header"]
        lines.append(
            f"| {row['slot_id']} | {row['control_mode']} | {row['status']} | "
            f"{row['choice']['target_style'] if row['choice'] else '未知'} | "
            f"{header['present'] if header else '未知'} |"
        )
    lines += [
        "",
        "重复分组：" + canonical(result["return_reuse_groups"]),
        "",
        "这是两个开关一起使用的接口检查。HTTP成功不证明开关生效；没有缓存键响应头不证明没有命中。",
        "原问题不变。相同或不同编号、正文都不能证明独立生成，也不能证明AI像真人。",
        "",
    ]
    return "\n".join(lines)


async def collect(packet, workspace, client, secret=""):
    checked_packet(workspace, packet)
    output = Path(packet["source_square"]["execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    for name in LOGS:
        (output / name).touch()
    save(output / "packet.json", packet)
    state = {"status": "running", "stop_reason": None}

    def log(name, row):
        text = canonical(row)
        if secret:
            text = text.replace(secret, "[REDACTED]")
        with (output / name).open("a") as handle:
            handle.write(text + "\n")

    try:
        response = await client.get(square.execution_config(packet["source_square"])["models_url"])
        square.require(response.status_code == 200, "model registry HTTP failure")
        registry = response.json()
        square.require(
            packet["probe_case"]["payload"]["model"] in {x["id"] for x in registry["data"]},
            "model absent",
        )
        save(output / "model_registry.json", registry)
        for request in packet["requests"]:
            log(LOGS[0], request)
            base, start = identity(request), datetime.now(UTC).isoformat()
            log(LOGS[1], {**base, "dispatched_at": start, "retry_count": 0})
            begun = time.perf_counter()
            http, body, cancellation = "transport_error", None, False
            header = {"present": False, "value_sha256": None}
            try:
                response = await client.post(square.ENDPOINT, json=request["payload"])
                http = response.status_code
                value = response.headers.get("x-litellm-cache-key")
                header = {
                    "present": value is not None,
                    "value_sha256": hashlib.sha256(value.encode()).hexdigest()
                    if value is not None
                    else None,
                }
                if http == 200:
                    body = response.json()
            except asyncio.CancelledError:
                http, cancellation = "cancelled_in_flight", True
            except (httpx.HTTPError, ValueError):
                pass
            log(
                LOGS[2],
                {
                    **base,
                    "requested_at": start,
                    "response_received_at": datetime.now(UTC).isoformat(),
                    "cached": False,
                    "http_status": http,
                    "retry_count": 0,
                    "latency_seconds": time.perf_counter() - begun,
                    "response": body,
                    "cache_key_header": header,
                },
            )
            if cancellation:
                raise asyncio.CancelledError
            result = scored(packet, output)
            if not result["rows"][request["position"] - 1]["compatible"]:
                state.update(status="stopped", stop_reason="protocol failure")
                break
        else:
            state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="in-flight accounting unknown; never resume")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save(output / "execution_state.json", state)
        attempts = len(read_jsonl(output / LOGS[1]))
        used = packet["source_square"]["prior_POSTs"] + attempts
        save(
            output / "budget_ledger.json",
            {
                "global_authorization_source": str(
                    (workspace / packet["plan"]["budget_sources"]["grant"]["path"]).resolve()
                ),
                "global_authorized_completion_POST_cap": 100,
                "used_completion_POSTs": used,
                "reserved_next_completion_POSTs": 0,
                "available_unreserved_POSTs": 100 - used,
                "active_run_id": None,
                "new_logged_dispatches": attempts,
                "status": "completed_pending_independent_audit"
                if state["status"] == "complete"
                else "stopped_pending_audit",
                "no_resume": True,
            },
        )
    result = scored(packet, output)
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(describe(result))
    return result


def prepare(args):
    packet = packet_for(args.workspace, strict_json(args.plan.read_text()))
    square.require(
        not Path(packet["source_square"]["execution_output"]).exists(), "already executed"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "packet.json", packet)
    (args.output / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in packet["requests"])
    )
    save(args.output / "pending_evaluation.json", evaluate(packet, [], [], []))
    return {"status": "prepared_not_sent", "max_new_POSTs": 4}


def load_packet(workspace, folder):
    packet = strict_json((folder / "packet.json").read_text())
    checked_packet(workspace, packet)
    square.require(
        canonical(read_jsonl(folder / "expected_requests.jsonl")) == canonical(packet["requests"]),
        "frozen wire requests differ",
    )
    return packet


async def execute(args):
    packet = load_packet(args.workspace, args.package)
    square.require(
        packet["source_square"]["evidence_kind"] == "recorded_model_logs", "tool test not live"
    )
    square.require(not Path(packet["source_square"]["execution_output"]).exists(), "never resume")
    from dotenv import dotenv_values

    key = dotenv_values(args.workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    square.require(bool(key), "AIGC_API_KEY missing")
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        timeout=120,
        trust_env=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as client:
        return await collect(packet, args.workspace, client, secret=key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "execute", "evaluate"])
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--package", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "execute":
        result = asyncio.run(execute(args))
        result = {k: result[k] for k in ("protocol_complete", "logged_dispatch_attempts")}
    else:
        packet = load_packet(args.workspace, args.package)
        result = scored(packet, Path(packet["source_square"]["execution_output"]))
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(describe(result))
        result = {k: result[k] for k in ("protocol_complete", "logged_dispatch_attempts")}
    print(canonical(result))


if __name__ == "__main__":
    main()
