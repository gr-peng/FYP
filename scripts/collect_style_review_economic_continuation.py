"""Run separately frozen economic batches under the user's cumulative500 grant."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import httpx

from behavioral_market.llm.transport_diagnostics import COMPLETION, DiagnosticClient
from scripts import collect_style_review_economic_controls as original

WORKSPACE = original.WORKSPACE
REPO = original.materials.REPO
AUTO_ROOT = WORKSPACE / "iterations/style_economic_auto_20261106"
GRANT_PATH = WORKSPACE / "iterations/network_recovery_20261106/cumulative_authorization.json"
BUDGET_PATH = WORKSPACE / "iterations/cumulative_model_budget_500.json"
ROUTE_CONFIRMATION = (
    WORKSPACE / "iterations/network_recovery_20261106/school_route_confirmation.json"
)
REGISTRY_EVIDENCE = (
    WORKSPACE / "iterations/network_recovery_20261106/public_model_ids_observation.json"
)
MODEL = "DeepSeek-V4.1-Flash"
PROTOCOL = "economic-style-cumulative500-v2"
USER_REPLY = "持续迭代不用停，500次之内的调用不用经我确认"
ADAPTER_SOURCES = (
    "scripts/collect_style_review_economic_controls.py",
    "tests/test_style_review_economic_collection.py",
    "src/behavioral_market/llm/transport_diagnostics.py",
    "tests/test_transport_diagnostics.py",
    "scripts/collect_style_review_economic_continuation.py",
    "tests/test_style_economic_continuation.py",
)
require = original.require
save = original.save
canonical = original.materials.canonical
digest = original.materials.digest


def read(path):
    return original.strict_json(path.read_text())


def validate_plan(plan):
    require(
        isinstance(plan, dict) and set(plan) == {"run_id", "prior_POSTs", "route", "model"},
        "plan schema",
    )
    require(
        isinstance(plan["run_id"], str)
        and re.fullmatch(r"economic-auto-[a-zA-Z0-9._-]{1,80}", plan["run_id"]),
        "new run identity",
    )
    require(
        type(plan["prior_POSTs"]) is int and 133 <= plan["prior_POSTs"] <= 488,
        "whole12 batch must fit500",
    )
    require(plan["route"] in ("environment_proxy", "school_direct"), "unknown school route")
    require(plan["model"] == MODEL, "this study must explicitly name the available Flash version")


def check_route_confirmation(plan):
    if plan["route"] == "environment_proxy":
        return None
    evidence = read(ROUTE_CONFIRMATION)
    require(
        evidence["kind"] == "explicit_user_network_path_reply"
        and evidence["verbatim_reply"] == "校园网或校园VPN直连"
        and bool(evidence["question_item_id"]),
        "confirmed school network path required",
    )
    return original.fingerprint(ROUTE_CONFIRMATION)


def check_user_grant():
    grant = read(GRANT_PATH)
    require(
        grant["kind"] == "explicit_user_cumulative_authorization"
        and grant["verbatim_reply"] == USER_REPLY
        and type(grant["global_authorized_completion_POST_cap"]) is int
        and grant["global_authorized_completion_POST_cap"] == 500
        and bool(grant["source_message_ref"]),
        "actual cumulative500 user grant required",
    )
    # This receipt records the real user message; creating a JSON file never grants permission.
    return original.fingerprint(GRANT_PATH)


def make_packet(plan, evidence_kind="recorded_model_logs"):
    validate_plan(plan)
    grant_fingerprint = check_user_grant()
    registry = read(REGISTRY_EVIDENCE)
    require(
        registry["endpoint"] == "https://aigc-api.hkust-gz.edu.cn/api/v2/models"
        and MODEL in registry["model_ids"],
        "model registry evidence differs",
    )
    base = original.make_packet(evidence_kind)
    requests = []
    for request in base["requests"]:
        origin = {k: v for k, v in request.items() if k not in ("payload", "request_hash")}
        origin.update(
            kind=PROTOCOL,
            run_id=plan["run_id"],
            slot_id=f"{plan['run_id']}.{request['agent_id']}.r{request['round']}.{request['condition']}",
        )
        payload = deepcopy(request["payload"])
        payload["model"] = plan["model"]
        requests.append(
            {**origin, "request_hash": digest([payload, origin, PROTOCOL]), "payload": payload}
        )
    core = {k: v for k, v in base.items() if k != "packet_sha256"}
    core.update(
        protocol=PROTOCOL,
        run_id=plan["run_id"],
        model=plan["model"],
        requests=requests,
        study_plan=deepcopy(plan),
        cumulative_authorization=grant_fingerprint,
        network_route_confirmation=check_route_confirmation(plan),
        model_migration={
            "base_material_model": base["model"],
            "requested_model": plan["model"],
            "public_registry_evidence": original.fingerprint(REGISTRY_EVIDENCE),
            "messages_and_other_payload_options_unchanged": True,
            "must_report_separately_from_old_model_studies": True,
        },
        adapter_sources={name: original.fingerprint(REPO / name) for name in ADAPTER_SOURCES},
        budget_basis={"prior_POSTs": plan["prior_POSTs"], "global_cap": 500},
    )
    return {**core, "packet_sha256": digest(core)}


def check_packet(packet, plan):
    require(isinstance(packet, dict), "packet schema")
    require(
        canonical(packet) == canonical(make_packet(plan, packet.get("evidence_kind"))),
        "packet differs from frozen plan/sources/materials/grant",
    )


def evaluator(plan):
    def evaluate(packet, requests, attempts, responses, targets=None):
        result = original.evaluate(
            packet,
            requests,
            attempts,
            responses,
            targets,
            packet_validator=lambda candidate: check_packet(candidate, plan),
        )
        return {
            **result,
            "requested_model": packet["model"],
            "base_material_model": packet["model_migration"]["base_material_model"],
            "model_version_changed_from_old18": True,
        }

    return evaluate


def describe(result):
    return (
        f"本组请求模型：{result['requested_model']}。原18次使用的Flash版本不同，本组单独报告。\n"
        + original.describe(result)
    )


def prepare(folder, plan, evidence_kind="recorded_model_logs"):
    packet = make_packet(plan, evidence_kind)
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", plan)
    save(folder / "packet.json", packet)
    (folder / "expected_requests.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in packet["requests"]), encoding="utf-8"
    )
    save(folder / "pending_evaluation.json", evaluator(plan)(packet, [], [], []))
    save(
        folder / "scope.json",
        {
            "status": "prepared_not_sent",
            "planned_POSTs": 12,
            "cumulative_authorized_cap": 500,
            "prior_POSTs": plan["prior_POSTs"],
            "authorization": str(GRANT_PATH),
            "new_run_no_retry_no_repair_no_resume": True,
        },
    )
    return packet


def load_packet(folder):
    plan, packet = read(folder / "plan.json"), read(folder / "packet.json")
    check_packet(packet, plan)
    require(
        canonical(original.read_jsonl(folder / "expected_requests.jsonl"))
        == canonical(packet["requests"]),
        "expected wire differs",
    )
    return plan, packet


def check_budget(ledger, plan):
    require(
        ledger["global_authorized_completion_POST_cap"] == 500
        and type(ledger["global_authorized_completion_POST_cap"]) is int,
        "budget cap differs",
    )
    require(
        type(ledger["used_completion_POSTs"]) is int
        and ledger["used_completion_POSTs"] == plan["prior_POSTs"],
        "prior call count differs",
    )
    require(
        ledger["active_run_id"] is None and ledger["reserved_next_completion_POSTs"] == 0,
        "another run or unresolved reservation exists",
    )
    require(
        ledger["used_completion_POSTs"] + 12 <= 500
        and ledger["available_unreserved_POSTs"] == 500 - ledger["used_completion_POSTs"],
        "whole batch exceeds remaining allowance",
    )


def save_budget(path, ledger):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_budget(path):
    with path.with_name(path.name + ".lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def terminal_budget(ledger, plan, output):
    # Count raw dispatch records, including cancelled/uncertain transmissions, conservatively.
    attempt_file = output / original.LOGS[1]
    attempts = original.read_jsonl(attempt_file) if attempt_file.exists() else []
    sent = len(attempts)
    require(0 <= sent <= 12, "unexpected attempt count")
    return {
        **ledger,
        "used_completion_POSTs": plan["prior_POSTs"] + sent,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 500 - plan["prior_POSTs"] - sent,
        "active_run_id": None,
        "status": "terminal_pending_independent_audit",
        "last_run_id": plan["run_id"],
        "last_output": str(output),
        "version": ledger["version"] + 1,
    }


def school_client(key, route):
    if route == "environment_proxy":
        return original.authorized_http_client(key, None)
    require(route == "school_direct", "unknown school route")
    # User confirmed this API uses campus/VPN direct access. Keep global proxy
    # settings untouched and retain inherited CA trust. Only the two school URLs
    # are admitted by DiagnosticClient; there is no automatic route fallback.
    transport = httpx.AsyncHTTPTransport(proxy=None, verify=True, trust_env=True, retries=0)
    return httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        transport=transport,
        trust_env=True,
        timeout=120,
        follow_redirects=False,
    )


async def execute(folder):
    plan, packet = load_packet(folder)
    require(
        folder.resolve() == (AUTO_ROOT / plan["run_id"]).resolve(),
        "execution must use the canonical folder for this run identity",
    )
    require(
        packet["evidence_kind"] == "recorded_model_logs", "synthetic packet cannot execute live"
    )
    review = read(folder / "independent_preflight.json")
    require(
        review["accepted_for_execution"] is True
        and review["mustfix"] == []
        and review["packet_sha256"] == packet["packet_sha256"],
        "independent preflight required",
    )
    output, diagnostic_dir = folder / "live_execution", folder / "transport_diagnostics"
    started = folder / "execution_started.json"
    require(
        not output.exists() and not diagnostic_dir.exists() and not started.exists(),
        "never resume an existing batch",
    )
    with exclusive_budget(BUDGET_PATH):
        ledger = read(BUDGET_PATH)
        check_budget(ledger, plan)
        with started.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": plan["run_id"],
                    "packet_sha256": packet["packet_sha256"],
                    "started_utc": original.now_utc().isoformat(),
                    "no_resume": True,
                },
                handle,
            )
            handle.write("\n")
        save_budget(
            BUDGET_PATH,
            {
                **ledger,
                "reserved_next_completion_POSTs": 12,
                "available_unreserved_POSTs": 500 - plan["prior_POSTs"] - 12,
                "active_run_id": plan["run_id"],
                "status": "running",
            },
        )
        try:
            from dotenv import dotenv_values

            key = dotenv_values(REPO / ".env").get("AIGC_API_KEY")
            require(isinstance(key, str) and bool(key), "AIGC_API_KEY missing")
            async with school_client(key, plan["route"]) as client:
                route = (
                    "environment_proxy"
                    if client._transport_for_url(original.httpx.URL(COMPLETION))
                    is not client._transport
                    else "default_direct"
                )
                expected_route = (
                    "default_direct" if plan["route"] == "school_direct" else "environment_proxy"
                )
                require(route == expected_route, "selected network route differs from frozen plan")
                logged_client = DiagnosticClient(client, diagnostic_dir, route=route)
                result = await original.collect(
                    packet,
                    output,
                    logged_client,
                    None,
                    evaluator(plan),
                    {
                        "prior_POSTs": plan["prior_POSTs"],
                        "global_cap": 500,
                        "grant_source": str(GRANT_PATH),
                    },
                    secret=key,
                )
            (output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
            return result
        finally:
            save_budget(BUDGET_PATH, terminal_budget(ledger, plan, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "evaluate"))
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.plan is not None, "plan file required")
        packet = prepare(args.folder, read(args.plan))
        print(canonical({"status": "prepared_not_sent", "packet_sha256": packet["packet_sha256"]}))
    elif args.action == "execute":
        result = asyncio.run(execute(args.folder))
        print(
            canonical(
                {key: result[key] for key in ("logged_dispatch_attempts", "protocol_complete")}
            )
        )
    else:
        plan, packet = load_packet(args.folder)
        raw = args.folder / "live_execution"
        logs = [original.read_jsonl(raw / n) if (raw / n).exists() else [] for n in original.LOGS]
        result = evaluator(plan)(packet, *logs)
        require(args.output is not None, "new evaluation folder required")
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.txt").write_text(describe(result), encoding="utf-8")


if __name__ == "__main__":
    main()
