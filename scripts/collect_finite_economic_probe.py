"""Collect a separately frozen12 with finite-account economic materials, under cap500."""

from __future__ import annotations

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

from behavioral_market.llm.transport_diagnostics import COMPLETION, DiagnosticClient
from scripts import build_finite_economic_controls as builder
from scripts import collect_style_review_economic_continuation as shared

original = shared.original
require, save, read = shared.require, shared.save, shared.read
canonical, digest = shared.canonical, shared.digest
MATERIALS = shared.WORKSPACE / "iterations/finite_economic_materials_20261106"
AUTO_ROOT = shared.WORKSPACE / "iterations/finite_economic_auto_20261106"
PROTOCOL = "finite-account-style-collection-v1"
SOURCES = ("scripts/collect_finite_economic_probe.py", "tests/test_finite_economic_collection.py")


def checked_materials():
    freeze = read(MATERIALS / "freeze_v1.json")
    for name, fp in freeze["sources"].items():
        require(
            original.fingerprint(shared.WORKSPACE / name) == fp, "finite material source differs"
        )
    material = MATERIALS / "materials_v1/materials.json"
    require(original.fingerprint(material) == freeze["materials"], "finite materials differ")
    package = read(material)
    require(canonical(package) == canonical(builder.build_materials()), "finite rebuild differs")
    review = read(MATERIALS / "independent_review_v1.json")
    require(
        review["accepted_materials"] is True and review["mustfix"] == [],
        "finite materials require independent acceptance",
    )
    return package


def make_packet(plan, evidence_kind="recorded_model_logs"):
    shared.validate_plan(plan)
    require(plan["run_id"].startswith("economic-auto-finite-"), "new finite study identity")
    require(plan["prior_POSTs"] >= 181, "new study follows completed181")
    base = shared.make_packet(plan, evidence_kind)
    cases = {c["case_id"]: c for c in checked_materials()["cases"]}
    requests = []
    for row in base["requests"]:
        origin = original.identity(row)
        origin.update(kind=PROTOCOL)
        origin.pop("request_hash")
        payload = deepcopy(cases[row["case_id"]]["payload"])
        requests.append(
            {**origin, "request_hash": digest([payload, origin, PROTOCOL]), "payload": payload}
        )
    core = {k: v for k, v in base.items() if k != "packet_sha256"}
    core.update(
        protocol=PROTOCOL,
        requests=requests,
        accepted_materials_sha256=original.fingerprint(MATERIALS / "materials_v1/materials.json")[
            "sha256"
        ],
        schedule_base_materials_sha256=base["accepted_materials_sha256"],
        finite_materials_receipts={
            str(p.relative_to(shared.WORKSPACE)): original.fingerprint(p)
            for p in (
                MATERIALS / "freeze_v1.json",
                MATERIALS / "materials_v1/materials.json",
                MATERIALS / "independent_review_v1.json",
            )
        },
        finite_sources=read(MATERIALS / "freeze_v1.json")["sources"],
        adapter_sources={
            **base["adapter_sources"],
            **{name: original.fingerprint(shared.REPO / name) for name in SOURCES},
        },
        model_migration={
            **base["model_migration"],
            "messages_and_other_payload_options_unchanged": False,
        },
        order_note=base["order_note"] + " New finite-account material bundle,all hidden; "
        "do not pool old12/36or interpret as a benchmark-only causal intervention.",
    )
    return {**core, "packet_sha256": digest(core)}


def check_packet(packet, plan):
    require(
        canonical(packet) == canonical(make_packet(plan, packet.get("evidence_kind"))),
        "finite packet differs from frozen materials/sources/grant/schedule",
    )


def evaluator(plan):
    def evaluate(packet, requests, attempts, responses, targets=None):
        result = original.evaluate(
            packet,
            requests,
            attempts,
            responses,
            targets,
            packet_validator=lambda p: check_packet(p, plan),
        )
        result.update(
            requested_model=packet["model"],
            counterfactual_method=builder.METHOD,
            limits="Four new finite-account hypothetical cases,not oldproxy cases. Compare each "
            "original style's complete economic bundle,not oneisolated return field. Three "
            "repeats are not people;historical winner is not a mandatory answer. No benchmark-only "
            "comparison with old12/36,no human/market/independent-generation proof.",
        )
        return result

    return evaluate


def describe(result):
    return (
        "本组使用同起始账户、同约束和明确成交假设下的策略复算收益。\n"
        f"模型{result['requested_model']}；与旧12/36分开报告。\n" + original.describe(result)
    )


def prepare(folder, plan, evidence_kind="recorded_model_logs"):
    packet = make_packet(plan, evidence_kind)
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", plan)
    save(folder / "packet.json", packet)
    (folder / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in packet["requests"]), encoding="utf-8"
    )
    save(folder / "pending_evaluation.json", evaluator(plan)(packet, [], [], []))
    save(
        folder / "scope.json",
        {
            "status": "prepared_not_sent",
            "planned_POSTs": 12,
            "prior_POSTs": plan["prior_POSTs"],
            "cap": 500,
            "no_resume": True,
            "authorization": str(shared.GRANT_PATH),
        },
    )
    return packet


def load_packet(folder):
    plan, packet = read(folder / "plan.json"), read(folder / "packet.json")
    check_packet(packet, plan)
    require(
        canonical(original.read_jsonl(folder / "expected_requests.jsonl"))
        == canonical(packet["requests"]),
        "wire differs",
    )
    return plan, packet


async def execute(folder):
    plan, packet = load_packet(folder)
    require(folder.resolve() == (AUTO_ROOT / plan["run_id"]).resolve(), "canonical folder required")
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
    output, diagnostics = folder / "live_execution", folder / "transport_diagnostics"
    started = folder / "execution_started.json"
    require(not any(p.exists() for p in (output, diagnostics, started)), "never resume a batch")
    with shared.exclusive_budget(shared.BUDGET_PATH):
        ledger = read(shared.BUDGET_PATH)
        shared.check_budget(ledger, plan)
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
        shared.save_budget(
            shared.BUDGET_PATH,
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

            key = dotenv_values(shared.REPO / ".env").get("AIGC_API_KEY")
            require(isinstance(key, str) and bool(key), "AIGC_API_KEY missing")
            async with shared.school_client(key, plan["route"]) as client:
                route = (
                    "environment_proxy"
                    if client._transport_for_url(original.httpx.URL(COMPLETION))
                    is not client._transport
                    else "default_direct"
                )
                expected = (
                    "default_direct" if plan["route"] == "school_direct" else "environment_proxy"
                )
                require(route == expected, "selected route differs from frozen plan")
                result = await original.collect(
                    packet,
                    output,
                    DiagnosticClient(client, diagnostics, route=route),
                    None,
                    evaluator(plan),
                    {
                        "prior_POSTs": plan["prior_POSTs"],
                        "global_cap": 500,
                        "grant_source": str(shared.GRANT_PATH),
                    },
                    secret=key,
                )
            (output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
            return result
        finally:
            shared.save_budget(shared.BUDGET_PATH, shared.terminal_budget(ledger, plan, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.plan is not None, "plan required")
        packet = prepare(args.folder, read(args.plan))
        print(canonical({"status": "prepared_not_sent", "packet_sha256": packet["packet_sha256"]}))
    else:
        result = asyncio.run(execute(args.folder))
        print(canonical({k: result[k] for k in ("logged_dispatch_attempts", "protocol_complete")}))


if __name__ == "__main__":
    main()
