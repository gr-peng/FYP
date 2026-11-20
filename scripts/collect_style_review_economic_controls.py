"""Prepare/collect the fixed12 economic controls. Offline actions never read credentials."""

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
from scripts import build_style_review_economic_controls as materials

WORKSPACE = Path(__file__).resolve().parents[2]
STAGE = WORKSPACE / "iterations/style_economic_collection_20261106"
MATERIAL_STAGE = WORKSPACE / "iterations/style_review_economic_controls_20261024"
ACCEPTED_FREEZE_SHA = "f407c2cfee7aee9fddc4ec76836561a03181b50b7b9525d1ebefbb8e394e0c9e"
PROTOCOL = "economic-style-collection-v1"
RUN_ID = "economic-style-12-20261106-v1"
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
LIVE_OUTPUT = STAGE / "live_execution"
GRANT_PATH = STAGE / "execution_authorization.json"
EVIDENCE_KINDS = ("recorded_model_logs", "synthetic_tool_test")
APPROVAL_REPLY = "允许新增最多12次，累计最多145次"
HOUR_APPROVAL_REPLY = "同意之后一个小时你的所有操作"
APPROVAL_QUESTION = (
    "是否允许执行已验收的12次经济信息对照，将累计上限从133提高到145？"
    "材料是4份假设情境：原基本面、原技术账户各比较替代策略历史较好与较差，"
    "每份问3次，固定隐藏群体信息。发送假设账户、本人记忆、构造行情、假设估值、"
    "逐期收益计算和策略选择说明至学校既有接口 "
    "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions 的DeepSeek-V4-Flash。"
    "不含真人记录、密钥或本地规则答案；逐条发送，不重试、不修复、不补问，"
    "首次传输、身份、格式、模式或策略一致性异常即停。相同选择和重复返回如实保留。"
    "此前133次额度已用完，原一小时授权已过期，因此需要本次具体的新额度。"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def checked_materials():
    freeze_path = MATERIAL_STAGE / "final_freeze.json"
    require(fingerprint(freeze_path)["sha256"] == ACCEPTED_FREEZE_SHA, "accepted freeze differs")
    freeze = strict_json(freeze_path.read_text())
    for name, expected in freeze["sources"].items():
        require(fingerprint(WORKSPACE / name) == expected, "accepted source differs")
    for name in (
        "materials_v1/materials.json",
        "materials_v1/artifact_manifest.json",
        "budget_ledger.json",
    ):
        require(
            fingerprint(MATERIAL_STAGE / name) == freeze["files"][name], "material/budget differs"
        )
    package = strict_json((MATERIAL_STAGE / "materials_v1/materials.json").read_text())
    require(
        materials.canonical(package) == materials.canonical(materials.build_materials()),
        "material reconstruction differs",
    )
    ledger = strict_json((MATERIAL_STAGE / "budget_ledger.json").read_text())
    require(
        ledger["used_completion_POSTs"] == ledger["global_authorized_completion_POST_cap"] == 133
        and ledger["reserved_next_completion_POSTs"] == 0,
        "expected exhausted133 allowance",
    )
    return package, freeze["sources"], freeze["files"]["budget_ledger.json"]


def make_packet(evidence_kind="recorded_model_logs"):
    require(evidence_kind in EVIDENCE_KINDS, "unknown evidence kind")
    package, sources, budget = checked_materials()
    cases = {case["case_id"]: case for case in package["cases"]}
    requests = []
    for round_number in (1, 2, 3):
        first = ("positive", "negative") if round_number != 2 else ("negative", "positive")
        order = ("A000", "A001") if round_number != 2 else ("A001", "A000")
        for position in (1, 2):
            for agent_id in order:
                conditions = first if agent_id == "A000" else first[::-1]
                condition = conditions[position - 1]
                case = cases[f"{agent_id}.{condition}"]
                origin = {
                    "kind": PROTOCOL,
                    "run_id": RUN_ID,
                    "evidence_kind": evidence_kind,
                    "slot_id": f"{RUN_ID}.{agent_id}.r{round_number}.{condition}",
                    "position": len(requests) + 1,
                    "case_id": case["case_id"],
                    "agent_id": agent_id,
                    "from_style": case["original_style"],
                    "condition": condition,
                    "round": round_number,
                    "within_account_round_position": position,
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
    core = {
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "evidence_kind": evidence_kind,
        "endpoint": ENDPOINT,
        "model": requests[0]["payload"]["model"],
        "accepted_materials_sha256": fingerprint(MATERIAL_STAGE / "materials_v1/materials.json")[
            "sha256"
        ],
        "source_receipt": sources,
        "adapter_sources": {
            name: fingerprint(materials.REPO / name)
            for name in (
                "scripts/collect_style_review_economic_controls.py",
                "tests/test_style_review_economic_collection.py",
            )
        },
        "budget_basis": budget,
        "requests": requests,
        "transport_retries": 0,
        "format_repairs": 0,
        "expected_records_only": True,
        "order_note": "Three rounds cannot balance both positions within one account exactly: "
        "A000 P/N,N/P,P/N; A001 reversed. Each condition3 and aggregate position counts balance.",
    }
    return {**core, "packet_sha256": materials.digest(core)}


def check_packet(packet):
    require(isinstance(packet, dict), "packet must be an object")
    require(
        materials.canonical(packet)
        == materials.canonical(make_packet(packet.get("evidence_kind"))),
        "packet differs from pinned sources/materials/schedule",
    )


def identity(request):
    return {key: value for key, value in request.items() if key != "payload"}


def evaluate(packet, requests, attempts, responses, targets=None, *, packet_validator=None):
    """Rebuild outcomes from raw prefixes, never from execution-state or summary flags."""
    require(targets is None, "A/B answer targets are inapplicable")
    (check_packet if packet_validator is None else packet_validator)(packet)
    expected = {row["slot_id"]: row for row in packet["requests"]}
    maps = []
    for name, records in zip(LOGS, (requests, attempts, responses), strict=True):
        require(isinstance(records, list), "records must be lists")
        mapped = {}
        for row in records:
            require(isinstance(row, dict), "log row must be an object")
            sid = row.get("slot_id")
            require(
                isinstance(sid, str) and sid in expected and sid not in mapped,
                "foreign/duplicate slot",
            )
            origin = identity(expected[sid])
            require(
                all(type(row.get(k)) is type(v) and row[k] == v for k, v in origin.items()),
                "identity/hash differs",
            )
            if name == LOGS[0]:
                require(
                    materials.canonical(row) == materials.canonical(expected[sid]), "wire differs"
                )
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
                require(set(row) == set(origin) | fields, "unexpected log fields")
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
            require(previous_finish is None or start >= previous_finish, "overlapping dispatch")
            status = "awaiting_response"
        if sid in received:
            response = received[sid]
            require(response["cached"] is False, "local replay forbidden")
            start, finish = (
                aware_time(response["requested_at"]),
                aware_time(response["response_received_at"]),
            )
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
                answer, parsed = parse_choice(
                    body,
                    {
                        "payload": request["payload"],
                        "source": {"from_style": request["from_style"]},
                    },
                )
                mode = returned_mode(body)
                compatible = (
                    answer == "valid_linked_answer" and mode == "no_returned_reasoning_observed"
                )
                status = "compatible" if compatible else "invalid_answer_or_mode"
            else:
                require(response["response"] is None, "failed transport must not claim parsed body")
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
                "body_sha256": materials.digest(body) if body is not None else None,
                "cache_key_header": header,
            }
        )
    cells, signatures = defaultdict(list), defaultdict(list)
    for row in rows:
        cells[row["agent_id"], row["condition"]].append(row)
        if row["strong_signature"] is not None:
            signatures[materials.digest(row["strong_signature"])].append(row["slot_id"])
    summaries = []
    for (agent_id, condition), group in cells.items():
        valid = [row for row in group if row["compatible"]]
        original = group[0]["from_style"]
        alternative = "technical" if original == "fundamental" else "fundamental"
        counts = {
            style: sum(row["choice"]["target_style"] == style for row in valid)
            for style in ("fundamental", "technical")
        }
        summaries.append(
            {
                "agent_id": agent_id,
                "original_style": original,
                "alternative_style": alternative,
                "condition": condition,
                "planned": 3,
                "dispatched": sum(row["slot_id"] in dispatched for row in group),
                "valid": len(valid),
                "invalid": sum(
                    row["slot_id"] in received and not row["compatible"] for row in group
                ),
                "unsubmitted": sum(row["status"] == "unsubmitted" for row in group),
                "pending_records": sum(
                    row["status"] in ("request_record_only", "awaiting_response") for row in group
                ),
                "target_counts": counts,
                "alternative_count": counts[alternative],
                "stay_count": sum(row["choice"]["decision"] == "stay" for row in valid),
                "switch_count": sum(row["choice"]["decision"] == "switch" for row in valid),
                "alternative_fraction_descriptive": counts[alternative] / len(valid)
                if valid
                else None,
                "fundamental_fraction_descriptive": counts["fundamental"] / len(valid)
                if valid
                else None,
                "all_three_valid": len(valid) == 3,
                "different_targets_on_same_input": len({r["choice"]["target_style"] for r in valid})
                > 1,
            }
        )
    by_cell = {(row["agent_id"], row["condition"]): row for row in summaries}
    contrasts = []
    for agent_id in ("A000", "A001"):
        positive, negative = by_cell[agent_id, "positive"], by_cell[agent_id, "negative"]
        complete = positive["all_three_valid"] and negative["all_three_valid"]
        contrasts.append(
            {
                "agent_id": agent_id,
                "original_style": positive["original_style"],
                "alternative_style": positive["alternative_style"],
                "complete": complete,
                "alternative_fraction_positive_minus_negative": positive[
                    "alternative_fraction_descriptive"
                ]
                - negative["alternative_fraction_descriptive"]
                if complete
                else None,
                "fundamental_fraction_positive_minus_negative": positive[
                    "fundamental_fraction_descriptive"
                ]
                - negative["fundamental_fraction_descriptive"]
                if complete
                else None,
            }
        )
    return {
        "evidence_kind": packet["evidence_kind"],
        "local_logs_match_frozen_packet": True,
        "planned": 12,
        "logged_dispatch_attempts": len(dispatched),
        "protocol_complete": all(row["compatible"] for row in rows),
        "rows": rows,
        "cells": summaries,
        "contrasts": contrasts,
        "repeated_strong_signature_groups": [
            group for group in signatures.values() if len(group) > 1
        ],
        "unknown_signature_slots": [
            row["slot_id"]
            for row in rows
            if row["slot_id"] in received and row["strong_signature"] is None
        ],
        "independent_generation_count": None,
        "external_model_provenance_authenticated": False,
        "limits": "Economic-evidence sensitivity only, no mandatory correct target or human/market "
        "validity claim. No comparison against old18 as a wording-only effect. Local logs, cache "
        "controls and distinct signatures do not prove provenance or independent generations.",
    }


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def describe(result):
    lines = [
        "经济信息对照：仅按本组记录报告",
        "",
        "本地假回答检查，不能当成真实模型结果。"
        if result["evidence_kind"] == "synthetic_tool_test"
        else "计划12次；未发送或未合格的记录不算保持，也不算成功。",
        f"已记录尝试：{result['logged_dispatch_attempts']}；全部完成：{result['protocol_complete']}",
        "",
    ]
    for row in result["cells"]:
        lines.append(
            f"{row['agent_id']} / {row['condition']}：计划3，尝试{row['dispatched']}，"
            f"有效{row['valid']}，失败{row['invalid']}，未发送{row['unsubmitted']}，"
            f"待返回/仅请求{row['pending_records']}；选基本面{row['target_counts']['fundamental']}，"
            f"选技术{row['target_counts']['technical']}，改选{row['switch_count']}。"
        )
    lines.append("")
    for row in result["contrasts"]:
        difference = row["alternative_fraction_positive_minus_negative"]
        value = (
            "未知（两边须各3条有效）" if difference is None else f"{difference * 100:.1f}个百分点"
        )
        lines.append(f"{row['agent_id']} 改选替代策略的比例：正经济证据减负经济证据 = {value}")
    lines += [
        "",
        "每个账户单独比较，不合并成真人样本。历史收益占优不是必须选择的正确答案。",
        "原18次与本组提示、经济信息和解释线索都不同，不能把两组差异归因于澄清句。",
        "重复返回和缺失标识原样保留；没有独立生成数量或真人相似度的证明。",
        "",
    ]
    return "\n".join(lines)


def grant_scope(packet):
    return {
        "packet_sha256": packet["packet_sha256"],
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "endpoint": ENDPOINT,
        "model": packet["model"],
        "prior_POSTs": 133,
        "additional_POST_limit": 12,
        "global_cap": 145,
        "transport_retries": 0,
        "format_repairs": 0,
        "no_resume": True,
    }


def prepare(destination):
    packet = make_packet()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "packet.json", packet)
    (destination / "expected_requests.jsonl").write_text(
        "".join(materials.canonical(row) + "\n" for row in packet["requests"]), encoding="utf-8"
    )
    pending = evaluate(packet, [], [], [])
    save(destination / "pending_evaluation.json", pending)
    save(
        destination / "authorization_request.json",
        {
            "status": "awaiting_fresh_user_grant",
            "question": APPROVAL_QUESTION,
            "approved_reply": APPROVAL_REPLY,
            "scope": grant_scope(packet),
        },
    )
    (destination / "说明.txt").write_text(
        "12次计划，4份情境各3次，尚未发送。expected文件是计划，不是实际日志。\n"
        "原基本面、原技术账户分开比较正负经济证据；固定基础提示、隐藏群体信息。\n"
        "三轮无法让单账户两个位置完全等量；原基本面顺序为正负/负正/正负，原技术反序，整体平衡。\n"
        "策略选择本身不下单或自动收费；历史赢家不规定为正确答案。主要差值须两边各3条有效。\n"
        "逐条发送，首次异常即停，不重试、修复或补问。原样保留相同选择、波动及重复返回。\n"
        "当前133次额度已用完，旧一小时授权已过期。须有对应本包的新明确授权，最多新增12次、累计145次。\n",
        encoding="utf-8",
    )
    return packet


def load_packet(folder):
    packet = strict_json((folder / "packet.json").read_text())
    check_packet(packet)
    require(
        materials.canonical(read_jsonl(folder / "expected_requests.jsonl"))
        == materials.canonical(packet["requests"]),
        "expected requests differ",
    )
    return packet


def now_utc():
    return datetime.now(UTC)


def seconds_remaining(expires_at):
    if expires_at is None:
        return None
    remaining = (aware_time(expires_at) - now_utc()).total_seconds()
    require(remaining > 0, "hour authorization expired")
    return remaining


def deadline_hook(expires_at):
    async def guard(request):
        seconds_remaining(expires_at)

    return guard


def verify_authorization(packet, grant):
    check_packet(packet)
    require(
        packet["evidence_kind"] == "recorded_model_logs", "synthetic packet cannot execute live"
    )
    require(
        isinstance(grant, dict) and set(grant) == {"decision", "scope", "evidence"},
        "fresh explicit user grant required",
    )
    require(grant["decision"] == "approved_by_user", "approval pending")
    require(
        materials.canonical(grant["scope"]) == materials.canonical(grant_scope(packet)),
        "grant differs from complete12/145 scope",
    )
    evidence = grant["evidence"]
    require(isinstance(evidence, dict), "actual user-reply evidence required")
    expires_at = None
    if evidence.get("kind") == "explicit_user_hour":
        require(
            set(evidence)
            == {"kind", "source_message_ref", "verbatim_reply", "received_at_utc", "expires_at_utc"}
            and evidence["verbatim_reply"] == HOUR_APPROVAL_REPLY,
            "actual one-hour reply required",
        )
        received, expires = (
            aware_time(evidence["received_at_utc"]),
            aware_time(evidence["expires_at_utc"]),
        )
        require(expires == received + timedelta(hours=1), "hour grant cannot be extended")
        require(received <= now_utc() < expires, "hour grant is future or expired")
        expires_at = evidence["expires_at_utc"]
    else:
        require(
            set(evidence)
            == {"kind", "source_message_ref", "question", "verbatim_reply", "received_at_utc"},
            "specific reply evidence required",
        )
        require(
            evidence["kind"] == "explicit_user_reply" and evidence["question"] == APPROVAL_QUESTION,
            "unrelated approval question",
        )
        require(
            isinstance(evidence["verbatim_reply"], str)
            and evidence["verbatim_reply"].strip().casefold()
            in {APPROVAL_REPLY, "同意", "允许", "可以", "yes"},
            "explicit affirmative reply required",
        )
        require(
            aware_time(evidence["received_at_utc"]) <= now_utc(), "reply cannot be future dated"
        )
    require(
        isinstance(evidence["source_message_ref"], str)
        and bool(evidence["source_message_ref"].strip()),
        "actual user-message reference required",
    )
    # The operator must record a real user reply. A local JSON receipt cannot create permission.
    return {
        "prior_POSTs": 133,
        "global_cap": 145,
        "grant_source": str(GRANT_PATH),
        "expires_at_utc": expires_at,
    }


def authorized_http_client(key, expires_at):
    # HTTPX's default transports use zero retries. Supplying a custom transport
    # disables its environment proxy routing, so retain the installed defaults.
    return httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        timeout=120,
        trust_env=True,
        verify=True,
        follow_redirects=False,
        event_hooks={"request": [deadline_hook(expires_at)]},
    )


async def execute(folder, authorization_path):
    packet = load_packet(folder)
    require(
        authorization_path is not None and Path(authorization_path).resolve() == GRANT_PATH,
        "fixed fresh authorization receipt required",
    )
    authorization = verify_authorization(packet, strict_json(GRANT_PATH.read_text()))
    require(not LIVE_OUTPUT.exists(), "execution exists; never resume")
    expires_at = authorization["expires_at_utc"]
    seconds_remaining(expires_at)
    from dotenv import dotenv_values

    key = dotenv_values(materials.REPO / ".env").get("AIGC_API_KEY")
    require(isinstance(key, str) and bool(key), "AIGC_API_KEY missing")
    async with authorized_http_client(key, expires_at) as client:
        async with asyncio.timeout(seconds_remaining(expires_at)):
            result = await collect(
                packet, LIVE_OUTPUT, client, None, evaluate, authorization, secret=key
            )
    (LIVE_OUTPUT / "REPORT.txt").write_text(describe(result), encoding="utf-8")
    return result


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
                {key: result[key] for key in ("protocol_complete", "logged_dispatch_attempts")}
            )
        )
    else:
        packet = load_packet(args.package)
        logs = [
            read_jsonl(LIVE_OUTPUT / name) if (LIVE_OUTPUT / name).exists() else [] for name in LOGS
        ]
        result = evaluate(packet, *logs)
        require(args.output is not None, "evaluation output folder required")
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
        print(
            materials.canonical(
                {key: result[key] for key in ("protocol_complete", "logged_dispatch_attempts")}
            )
        )


if __name__ == "__main__":
    main()
