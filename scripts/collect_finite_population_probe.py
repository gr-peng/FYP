"""New finite-account population study, using accepted historical material bytes."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from copy import deepcopy
from pathlib import Path

from behavioral_market.llm.transport_diagnostics import COMPLETION, DiagnosticClient
from scripts import collect_population_economic_probe as population

shared = population.shared
original = shared.original
require, save, read = shared.require, shared.save, shared.read
canonical, digest = shared.canonical, shared.digest
COUNT, GROUPS = population.COUNT, population.GROUPS
PROTOCOL = "finite-account-population-information-v1"
METHOD = "finite_rule_order_replay_with_assigned_quotes"
MATERIALS = shared.WORKSPACE / "iterations/finite_economic_materials_20261106"
DESIGN = shared.WORKSPACE / "iterations/finite_population_design_20261106/design_v1.json"
AUTO_ROOT = shared.WORKSPACE / "iterations/finite_population_auto_20261106"
ACCEPTED_FREEZE_SHA = "fdce9e07ea6101d8a7c0096fea1179cd957c6e2274a051b46085f78c97a365fd"
ACCEPTED_MATERIAL_SHA = "4abf154f3e710b673cb686a104a7cb233242a47d7c18d1217c4637baeaaa82b7"
ADAPTER_FILES = (
    *shared.ADAPTER_SOURCES,
    *population.SOURCE_FILES,
    "scripts/collect_finite_population_probe.py",
    "tests/test_finite_population_probe.py",
    "scripts/build_style_review_economic_controls.py",
    "scripts/build_population_information_probe.py",
    "scripts/collect_population_information_probe.py",
    "scripts/style_cache_control.py",
)


def validate_plan(plan):
    shared.validate_plan(plan)
    require(plan["run_id"].startswith("economic-auto-finite-population-"), "new finite identity")
    require(193 <= plan["prior_POSTs"] <= 500 - COUNT, "whole36 must follow193 and fit500")
    require(plan["route"] == "school_direct", "use user-confirmed campus direct route")


def checked_materials():
    """Authenticate old files against their seal, never rebuild them with new code."""
    final = MATERIALS / "final_freeze.json"
    require(original.fingerprint(final)["sha256"] == ACCEPTED_FREEZE_SHA, "historical seal differs")
    seal = read(final)
    for name, expected in seal["files"].items():
        require(original.fingerprint(MATERIALS / name) == expected, "historical evidence differs")
    freeze = read(MATERIALS / "freeze_v1.json")
    for name, expected in freeze["sources"].items():
        require(
            original.fingerprint(MATERIALS / "sources_v1" / name) == expected,
            "historical source archive differs",
        )
    path = MATERIALS / "materials_v1/materials.json"
    require(original.fingerprint(path)["sha256"] == ACCEPTED_MATERIAL_SHA, "finite bytes differ")
    review = read(MATERIALS / "independent_review_v1.json")
    require(review["accepted_materials"] is True and review["mustfix"] == [], "material review")
    package = read(path)
    require(package["counterfactual_method"] == METHOD, "finite method differs")
    return package


def source_fingerprints():
    names = {str(p.relative_to(shared.REPO)) for p in (shared.REPO / "src").rglob("*.py")}
    names.update(ADAPTER_FILES)
    return {name: original.fingerprint(shared.REPO / name) for name in sorted(names)}


def input_paths():
    names = ["final_freeze.json", *read(MATERIALS / "final_freeze.json")["files"]]
    return {
        **{f"accepted_finite_materials/{name}": MATERIALS / name for name in names},
        "design.json": DESIGN,
        "authorization.json": shared.GRANT_PATH,
        "route.json": shared.ROUTE_CONFIRMATION,
        "registry.json": shared.REGISTRY_EVIDENCE,
    }


def make_packet(plan, evidence_kind="recorded_model_logs"):
    validate_plan(plan)
    require(evidence_kind in original.EVIDENCE_KINDS, "unknown evidence kind")
    grant = shared.check_user_grant()
    route = shared.check_route_confirmation(plan)
    registry = read(shared.REGISTRY_EVIDENCE)
    require(
        registry["endpoint"] == "https://aigc-api.hkust-gz.edu.cn/api/v2/models"
        and plan["model"] in registry["model_ids"],
        "model registry evidence differs",
    )
    design = read(DESIGN)
    require(
        design["planned_completion_POSTs"] == COUNT
        and design["material_sha256"] == ACCEPTED_MATERIAL_SHA,
        "new study design differs",
    )
    cases = {c["case_id"]: c for c in checked_materials()["cases"]}
    require(set(cases) == set(design["cases"]), "case set differs")
    payloads, ledgers = {}, {}
    for case_id, case in cases.items():
        require(case["payload"]["model"] == plan["model"], "preserve finite material model")
        for group in GROUPS:
            payload = deepcopy(case["payload"])
            if group != "hidden":
                field, ledger = population.population_snapshot(case, group)
                user = original.strict_json(payload["messages"][1]["content"])
                user["observation"].pop("mean_field_availability")
                user["observation"]["mean_field"] = field
                payload["messages"][1]["content"] = canonical(user)
                ledgers[f"{case_id}.{group}"] = ledger
            payloads[case_id, group] = payload
    requests = []
    for repeat in (1, 2, 3):
        first = ("positive", "negative") if repeat != 2 else ("negative", "positive")
        agents = ("A000", "A001") if repeat != 2 else ("A001", "A000")
        groups = GROUPS[repeat - 1 :] + GROUPS[: repeat - 1]
        for position in (1, 2):
            for agent_id in agents:
                condition = (first if agent_id == "A000" else first[::-1])[position - 1]
                case_id = f"{agent_id}.{condition}"
                for group_position, group in enumerate(groups, 1):
                    origin = {
                        "kind": PROTOCOL,
                        "run_id": plan["run_id"],
                        "evidence_kind": evidence_kind,
                        "slot_id": f"{plan['run_id']}.{case_id}.{group}.r{repeat}",
                        "position": len(requests) + 1,
                        "case_id": case_id,
                        "agent_id": agent_id,
                        "from_style": cases[case_id]["original_style"],
                        "condition": condition,
                        "round": repeat,
                        "within_account_round_position": position,
                        "population_condition": group,
                        "within_group_position": group_position,
                        "repair": 0,
                    }
                    payload = deepcopy(payloads[case_id, group])
                    requests.append(
                        {
                            **origin,
                            "request_hash": digest([payload, origin, PROTOCOL]),
                            "payload": payload,
                        }
                    )
    core = {
        "protocol": PROTOCOL,
        "run_id": plan["run_id"],
        "evidence_kind": evidence_kind,
        "endpoint": COMPLETION,
        "model": plan["model"],
        "study_plan": deepcopy(plan),
        "design_receipt": original.fingerprint(DESIGN),
        "accepted_historical_materials": {
            "final_freeze_sha256": ACCEPTED_FREEZE_SHA,
            "material_sha256": ACCEPTED_MATERIAL_SHA,
            "source_policy": "Frozen historical bytes and source copies verified; "
            "not rebuilt with current production code.",
        },
        "current_sources": source_fingerprints(),
        "input_receipts": {
            name: original.fingerprint(path) for name, path in input_paths().items()
        },
        "cumulative_authorization": grant,
        "network_route_confirmation": route,
        "public_registry_evidence": original.fingerprint(shared.REGISTRY_EVIDENCE),
        "budget_basis": {"prior_POSTs": plan["prior_POSTs"], "global_cap": 500},
        "construction_not_sent": ledgers,
        "requests": requests,
        "transport_retries": 0,
        "format_repairs": 0,
        "expected_records_only": True,
        "order_note": "Four accepted finite economic cases, three group-order rotations. "
        "Fresh hidden responses, four primary population and six secondary economic "
        "contrasts. No pooling with old studies.",
    }
    return {**core, "packet_sha256": digest(core)}


def check_packet(packet, plan):
    require(
        canonical(packet) == canonical(make_packet(plan, packet.get("evidence_kind"))),
        "finite population packet differs from materials/source/grant/schedule",
    )


def evaluator(plan):
    evaluate = population.evaluator(plan, packet_validator=lambda p: check_packet(p, plan))

    def wrapped(packet, requests, attempts, responses, targets=None):
        result = evaluate(packet, requests, attempts, responses, targets)
        result["counterfactual_method"] = METHOD
        result["limits"] += (
            " Accepted finite-account material bytes, not legacy proxy materials. "
            "A difference from the older36 cannot identify a return-method-only effect."
        )
        return result

    return wrapped


def describe(result):
    return "本组使用已验收的有限账户经济材料，与旧简化收益组分开报告。\n" + population.describe(
        result
    )


def prepare(folder, plan, evidence_kind="recorded_model_logs"):
    packet = make_packet(plan, evidence_kind)
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", plan)
    save(folder / "packet.json", packet)
    (folder / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in packet["requests"])
    )
    save(folder / "pending_evaluation.json", evaluator(plan)(packet, [], [], []))
    for name in packet["current_sources"]:
        target = folder / "executed_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shared.REPO / name, target)
    for name, path in input_paths().items():
        target = folder / "execution_inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    save(
        folder / "scope.json",
        {
            "status": "prepared_not_sent",
            "planned_POSTs": COUNT,
            "prior_POSTs": plan["prior_POSTs"],
            "global_cap": 500,
            "authorization": str(shared.GRANT_PATH),
            "no_resume": True,
            "personal_records_sent": False,
            "answers_or_private_cohort_ledger_sent": False,
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
    for name, expected in packet["current_sources"].items():
        require(
            original.fingerprint(folder / "executed_sources" / name) == expected,
            "execution source archive differs",
        )
    for name, expected in packet["input_receipts"].items():
        require(
            original.fingerprint(folder / "execution_inputs" / name) == expected,
            "execution input archive differs",
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
        population.check_budget(ledger, plan)
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
                "reserved_next_completion_POSTs": COUNT,
                "available_unreserved_POSTs": 500 - plan["prior_POSTs"] - COUNT,
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
                require(route == "default_direct", "selected route differs from frozen plan")
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
            shared.save_budget(shared.BUDGET_PATH, population.terminal_budget(ledger, plan, output))


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
        print(canonical({k: result[k] for k in ("protocol_complete", "logged_dispatch_attempts")}))


if __name__ == "__main__":
    main()
