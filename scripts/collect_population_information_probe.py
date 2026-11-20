"""Collect the fixed18 strategy reviews; preparing/evaluating never opens credentials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from behavioral_market.evaluation.choice_presentation import signature
from behavioral_market.evaluation.matched_choice_collection import LOGS, collect, read_jsonl
from behavioral_market.evaluation.style_probe_intake import aware_time, parse_choice, strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode
from scripts import build_population_information_probe as materials

WORKSPACE = Path(__file__).resolve().parents[2]
STAGE = WORKSPACE / "iterations/population_information_collection_20261023"
MATERIAL_STAGE = WORKSPACE / "iterations/population_information_probe_20261022"
PROTOCOL = "population-information-collection-v1"
RUN_ID = "population-information-18-20261023-v1"
LIVE_OUTPUT = STAGE / "live_execution"
GRANT_PATH = STAGE / "execution_authorization.json"
PINNED_PLANS = {
    "next_iteration_plan.json": "e6116fa2c7f6276433798dff22e7a85c2e2466284c5cab4d12cff72acd5ac70a",
    "plan.json": "0b30a07b6460798f308c3fc332c43e17d3192077272b672bab73d714d6b58235",
}
EVIDENCE_KINDS = ("recorded_model_logs", "synthetic_tool_test")
APPROVAL_REPLY = "允许新增最多18次，累计最多133次"
HOUR_APPROVAL_REPLY = "同意之后一个小时你的所有操作"
APPROVAL_QUESTION = (
    "是否允许执行已验收的18次群体信息对照，将累计上限从115提高到133？"
    "材料是两个全新假设账户在基本面多数、技术多数、隐藏汇总三种情况下的账户、"
    "本人记忆、平价行情、零策略收益及假设群体比例，每种各问3次；"
    "发送至学校既有接口 https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions "
    "的DeepSeek-V4-Flash。不含真人记录或密钥，逐条发送、不重试、不修复、不补问，"
    "首次传输、身份、格式、模式或策略一致性异常即停。相同选择和重复返回如实保留。"
    "此前115次额度已全部用完，因此需要这次具体的新额度。"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def checked_materials():
    plans = {}
    for name, sha in PINNED_PLANS.items():
        path = MATERIAL_STAGE / name
        require(fingerprint(path)["sha256"] == sha, "accepted plan bytes differ")
        plans[name] = strict_json(path.read_text())
    next_plan, prior = plans["next_iteration_plan.json"], plans["plan.json"]
    sources = {
        entry["path"]: entry["fingerprint"]
        for key in (
            "frozen_readonly_sources",
            "expanded_readonly_sources",
            "inherited_readonly_import_context",
        )
        for entry in prior[key]
    }
    sources.update({e["path"]: e["fingerprint"] for e in next_plan["frozen_readonly_sources"]})
    for path, wanted in sources.items():
        require(fingerprint(WORKSPACE / path) == wanted, "accepted source bytes differ")
    for entry in next_plan["accepted_material_files"]:
        require(fingerprint(WORKSPACE / entry["path"]) == entry["fingerprint"], "materials differ")
    budget = next_plan["budget_source"]
    require(
        fingerprint(WORKSPACE / budget["path"]) == budget["fingerprint"], "budget source differs"
    )
    ledger = strict_json((WORKSPACE / budget["path"]).read_text())
    require(
        ledger["used_completion_POSTs"] == ledger["global_authorized_completion_POST_cap"] == 115
        and ledger["reserved_next_completion_POSTs"] == 0,
        "expected exhausted115-call allowance",
    )
    package = strict_json((MATERIAL_STAGE / "materials_v1/materials.json").read_text())
    materials.validate_package(package)
    return package, sources, budget


def make_packet(evidence_kind="recorded_model_logs"):
    require(evidence_kind in EVIDENCE_KINDS, "unknown evidence kind")
    package, sources, budget = checked_materials()
    cases = {c["case_id"]: c for c in package["cases"]}
    requests = []
    for slot in package["schedule"]:
        case = cases[slot["case_id"]]
        origin = {
            "kind": PROTOCOL,
            "run_id": RUN_ID,
            "evidence_kind": evidence_kind,
            "slot_id": slot["slot_id"],
            "position": slot["position"],
            "case_id": case["case_id"],
            "agent_id": case["agent_id"],
            "from_style": case["from_style"],
            "condition": case["condition"],
            "round": slot["round"],
            "within_account_round_position": slot["within_account_round_position"],
            "repair": 0,
        }
        payload = deepcopy(case["payload"])
        requests.append(
            {
                **origin,
                "request_hash": materials.digest([payload, origin, PROTOCOL]),
                "payload": payload,
            }
        )
    adapter_sources = {
        name: fingerprint(materials.REPO / name)
        for name in (
            "scripts/collect_population_information_probe.py",
            "tests/test_population_information_collection.py",
        )
    }
    core = {
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "evidence_kind": evidence_kind,
        "endpoint": package["endpoint_proposal"],
        "model": requests[0]["payload"]["model"],
        "accepted_materials_sha256": fingerprint(MATERIAL_STAGE / "materials_v1/materials.json")[
            "sha256"
        ],
        "source_receipt": sources,
        "adapter_sources": adapter_sources,
        "budget_basis": budget,
        "requests": requests,
        "transport_retries": 0,
        "format_repairs": 0,
        "expected_records_only": True,
    }
    return {**core, "packet_sha256": materials.digest(core)}


def check_packet(packet):
    require(isinstance(packet, dict), "packet must be an object")
    require(
        materials.canonical(packet)
        == materials.canonical(make_packet(packet.get("evidence_kind"))),
        "packet differs from accepted sources, payloads, identities or schedule",
    )


def identity(request):
    return {k: v for k, v in request.items() if k != "payload"}


def evaluate(packet, requests, attempts, responses, targets=None):
    """Validate raw log prefixes; state/summary files cannot establish an answer."""
    require(targets is None, "A/B targets do not belong in this strategy probe")
    check_packet(packet)
    expected = {r["slot_id"]: r for r in packet["requests"]}
    maps = []
    for name, records in zip(LOGS, (requests, attempts, responses), strict=True):
        require(isinstance(records, list), "log records must be lists")
        mapped = {}
        for row in records:
            require(isinstance(row, dict), "log row must be an object")
            sid = row.get("slot_id")
            require(
                isinstance(sid, str) and sid in expected and sid not in mapped,
                "foreign/duplicate slot",
            )
            base = identity(expected[sid])
            require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
                "log identity/hash differs",
            )
            if name == LOGS[0]:
                require(
                    materials.canonical(row) == materials.canonical(expected[sid]), "wire differs"
                )
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
            mapped[sid] = row
        require(list(mapped) == list(expected)[: len(mapped)], "fixed global prefix differs")
        maps.append(mapped)
    submitted, dispatched, received = maps
    require(set(received) <= set(dispatched) <= set(submitted), "orphan attempt/response")
    rows, previous_finish = [], None
    for request in expected.values():
        sid = request["slot_id"]
        if sid in submitted:
            require(not rows or rows[-1]["compatible"], "request after pending/failed answer")
        status = "request_record_only" if sid in submitted else "unsubmitted"
        answer, mode, parsed, body, header = None, None, None, None, None
        compatible = False
        if sid in dispatched:
            start = aware_time(dispatched[sid]["dispatched_at"])
            require(previous_finish is None or start >= previous_finish, "overlapping dispatches")
            status = "awaiting_response"
        if sid in received:
            response = received[sid]
            require(response["cached"] is False, "local replay forbidden")
            start = aware_time(response["requested_at"])
            finish = aware_time(response["response_received_at"])
            require(
                start == aware_time(dispatched[sid]["dispatched_at"]) and finish >= start,
                "response clock differs",
            )
            previous_finish = finish
            latency, http, header = (
                response["latency_seconds"],
                response["http_status"],
                response["cache_key_header"],
            )
            require(
                type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
                "invalid latency",
            )
            require(
                type(http) is int
                and 100 <= http <= 599
                or http in ("transport_error", "cancelled_in_flight"),
                "invalid HTTP status",
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
                "invalid cache-header diagnostic",
            )
            if http == 200:
                body = response["response"]
                case = {
                    "payload": request["payload"],
                    "source": {"from_style": request["from_style"]},
                }
                answer, parsed = parse_choice(body, case)
                mode = returned_mode(body)
                compatible = (
                    answer == "valid_linked_answer" and mode == "no_returned_reasoning_observed"
                )
                status = "compatible" if compatible else "invalid_answer_or_mode"
            else:
                require(
                    response["response"] is None, "failed transport must not claim parsed response"
                )
                status = "transport_failed"
        rows.append(
            {
                **identity(request),
                "status": status,
                "compatible": compatible,
                "answer_status": answer,
                "returned_mode": mode,
                "parsed_choice": parsed,
                "choice": parsed if compatible else None,
                "strong_signature": signature(body),
                "cache_key_header": header,
                "body_sha256": materials.digest(body) if body is not None else None,
            }
        )
    cells, signatures = defaultdict(list), defaultdict(list)
    for row in rows:
        cells[row["agent_id"], row["condition"]].append(row)
        if row["strong_signature"] is not None:
            signatures[materials.digest(row["strong_signature"])].append(row["slot_id"])
    summaries = []
    for (agent_id, condition), group in cells.items():
        valid = [r for r in group if r["compatible"]]
        counts = {
            style: sum(r["choice"]["target_style"] == style for r in valid)
            for style in ("fundamental", "technical")
        }
        summaries.append(
            {
                "agent_id": agent_id,
                "condition": condition,
                "planned": 3,
                "dispatched": sum(r["slot_id"] in dispatched for r in group),
                "valid": len(valid),
                "invalid": sum(r["slot_id"] in received and not r["compatible"] for r in group),
                "unsubmitted": sum(r["status"] == "unsubmitted" for r in group),
                "pending_records": sum(
                    r["status"] in ("request_record_only", "awaiting_response") for r in group
                ),
                "target_counts": counts,
                "fundamental_fraction_descriptive": counts["fundamental"] / len(valid)
                if valid
                else None,
                "stay_count": sum(r["choice"]["decision"] == "stay" for r in valid),
                "switch_count": sum(r["choice"]["decision"] == "switch" for r in valid),
                "all_three_valid": len(valid) == 3,
                "different_targets_on_same_input": len({r["choice"]["target_style"] for r in valid})
                > 1,
            }
        )
    by_cell = {(r["agent_id"], r["condition"]): r for r in summaries}
    contrasts = []
    for agent_id in ("A000", "A001"):
        for left, right in (
            ("fundamental_majority", "technical_majority"),
            ("fundamental_majority", "hidden"),
            ("technical_majority", "hidden"),
        ):
            a, b = by_cell[agent_id, left], by_cell[agent_id, right]
            complete = a["all_three_valid"] and b["all_three_valid"]
            contrasts.append(
                {
                    "agent_id": agent_id,
                    "left": left,
                    "right": right,
                    "complete": complete,
                    "fundamental_fraction_difference": a["fundamental_fraction_descriptive"]
                    - b["fundamental_fraction_descriptive"]
                    if complete
                    else None,
                }
            )
    return {
        "evidence_kind": packet["evidence_kind"],
        "local_logs_match_frozen_packet": True,
        "planned": 18,
        "logged_dispatch_attempts": len(dispatched),
        "protocol_complete": all(r["compatible"] for r in rows),
        "rows": rows,
        "cells": summaries,
        "contrasts": contrasts,
        "repeated_strong_signature_groups": [g for g in signatures.values() if len(g) > 1],
        "unknown_signature_slots": [
            r["slot_id"] for r in rows if r["slot_id"] in received and r["strong_signature"] is None
        ],
        "independent_generation_count": None,
        "external_model_provenance_authenticated": False,
        "limits": "Local logs can match planned inputs without proving external provenance, cache "
        "avoidance or independent generations. Targets are hypothetical choices, "
        "not applied trades. No human alignment or market fidelity is measured.",
    }


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def prepare(destination):
    packet = make_packet()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "packet.json", packet)
    (destination / "expected_requests.jsonl").write_text(
        "".join(materials.canonical(r) + "\n" for r in packet["requests"])
    )
    save(destination / "pending_evaluation.json", evaluate(packet, [], [], []))
    save(
        destination / "authorization_request.json",
        {
            "status": "awaiting_explicit_user_grant",
            "question": APPROVAL_QUESTION,
            "approved_reply": APPROVAL_REPLY,
            "scope": grant_scope(packet),
        },
    )
    (destination / "说明.txt").write_text(
        "这是可检查的18次执行计划，尚未发送。expected文件只代表计划，不是实际日志。\n"
        "两个假设账户、三种群体信息、每格3次；主要比较须两侧各3条有效回答，缺失保留未知。\n"
        "逐条发送，不用本地缓存、不重试、不修复、不补问；首次异常立即停止。"
        "回答全相同也原样保留；不同返回编号和缓存开关不证明独立生成。\n"
        "当前115次已用完。真实执行须先收到并记录这份计划对应的新明确授权，"
        "最多新增18次、累计最多133次。代码在读取密钥和联网前核对授权。\n"
    )
    return packet


def load_packet(folder):
    packet = strict_json((folder / "packet.json").read_text())
    check_packet(packet)
    require(
        materials.canonical(read_jsonl(folder / "expected_requests.jsonl"))
        == materials.canonical(packet["requests"]),
        "expected wire records differ",
    )
    return packet


def grant_scope(packet):
    return {
        "packet_sha256": packet["packet_sha256"],
        "run_id": RUN_ID,
        "endpoint": packet["endpoint"],
        "model": packet["model"],
        "prior_POSTs": 115,
        "additional_POST_limit": 18,
        "global_cap": 133,
        "transport_retries": 0,
        "format_repairs": 0,
        "no_resume": True,
    }


def now_utc():
    return datetime.now(UTC)


def seconds_remaining(expires_at):
    if expires_at is None:
        return None
    remaining = (aware_time(expires_at) - now_utc()).total_seconds()
    require(remaining > 0, "one-hour authorization expired")
    return remaining


def deadline_hook(expires_at):
    async def guard(request):
        seconds_remaining(expires_at)

    return guard


def verify_authorization(packet, grant):
    check_packet(packet)
    require(packet["evidence_kind"] == "recorded_model_logs", "synthetic test cannot execute live")
    require(
        isinstance(grant, dict) and set(grant) == {"decision", "scope", "evidence"},
        "explicit scoped user grant required",
    )
    require(grant["decision"] == "approved_by_user", "user approval is pending")
    require(
        materials.canonical(grant["scope"]) == materials.canonical(grant_scope(packet)),
        "grant scope differs; need all18 calls",
    )
    evidence = grant["evidence"]
    expires_at = None
    if isinstance(evidence, dict) and evidence.get("kind") == "explicit_user_message":
        require(
            set(evidence) == {"kind", "verbatim_reply", "received_at_utc", "expires_at_utc"}
            and evidence["verbatim_reply"] == HOUR_APPROVAL_REPLY,
            "actual one-hour user message required",
        )
        received = aware_time(evidence["received_at_utc"])
        expires = aware_time(evidence["expires_at_utc"])
        require(
            expires == received + timedelta(hours=1), "authorization must last exactly one hour"
        )
        require(received <= now_utc() < expires, "one-hour authorization is future or expired")
        expires_at = evidence["expires_at_utc"]
    else:
        verify_specific_reply(evidence)
    # A receipt records an actual human reply; it cannot create permission by itself.
    return {
        "prior_POSTs": 115,
        "global_cap": 133,
        "grant_source": str(GRANT_PATH),
        "expires_at_utc": expires_at,
    }


def verify_specific_reply(evidence):
    require(
        isinstance(evidence, dict)
        and set(evidence)
        == {"kind", "question_item_id", "question", "verbatim_reply", "received_at_utc"},
        "user reply evidence required",
    )
    require(
        evidence["kind"] == "explicit_user_reply"
        and all(
            isinstance(evidence[k], str) and evidence[k].strip()
            for k in ("question_item_id", "question", "verbatim_reply")
        ),
        "numeric cap alone is not a user grant",
    )
    require(
        evidence["question"] == APPROVAL_QUESTION
        and evidence["verbatim_reply"].strip() == APPROVAL_REPLY,
        "receipt must record the concrete question and explicit affirmative reply",
    )
    aware_time(evidence["received_at_utc"])


async def execute(folder, authorization_path):
    packet = load_packet(folder)
    require(
        authorization_path is not None and Path(authorization_path).resolve() == GRANT_PATH,
        "fixed new authorization receipt required",
    )
    authorization = verify_authorization(packet, strict_json(GRANT_PATH.read_text()))
    require(not LIVE_OUTPUT.exists(), "execution directory exists; never resume")
    expires_at = authorization["expires_at_utc"]
    seconds_remaining(expires_at)
    from dotenv import dotenv_values

    key = dotenv_values(materials.REPO / ".env").get("AIGC_API_KEY")
    require(isinstance(key, str) and bool(key), "AIGC_API_KEY missing")
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        timeout=120,
        trust_env=False,
        follow_redirects=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
        event_hooks={"request": [deadline_hook(expires_at)]},
    ) as client:
        async with asyncio.timeout(seconds_remaining(expires_at)):
            return await collect(
                packet, LIVE_OUTPUT, client, None, evaluate, authorization, secret=key
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "evaluate"))
    parser.add_argument("--package", type=Path, default=STAGE / "execution_package_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        packet = prepare(args.output or args.package)
        print(
            materials.canonical(
                {"status": "prepared_not_sent", "packet_sha256": packet["packet_sha256"]}
            )
        )
    elif args.action == "execute":
        result = asyncio.run(execute(args.package, args.authorization))
        print(
            materials.canonical(
                {k: result[k] for k in ("protocol_complete", "logged_dispatch_attempts")}
            )
        )
    else:
        packet = load_packet(args.package)
        logs = [read_jsonl(LIVE_OUTPUT / n) if (LIVE_OUTPUT / n).exists() else [] for n in LOGS]
        result = evaluate(packet, *logs)
        require(args.output is not None, "evaluation output file required")
        save(args.output, result)
        print(
            materials.canonical(
                {k: result[k] for k in ("protocol_complete", "logged_dispatch_attempts")}
            )
        )


if __name__ == "__main__":
    main()
