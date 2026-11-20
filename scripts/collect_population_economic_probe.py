"""Frozen36: population information crossed with four fictional economic cases."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from behavioral_market.agents.state import AgentPersona, AgentState
from behavioral_market.llm.schemas import OrderDecision
from behavioral_market.llm.transport_diagnostics import COMPLETION, DiagnosticClient
from behavioral_market.market.portfolio import Portfolio
from behavioral_market.population.mean_field import compute_mean_field
from scripts import collect_style_review_economic_continuation as shared

original = shared.original
require, save, read = shared.require, shared.save, shared.read
canonical, digest = shared.canonical, shared.digest
COUNT = 36
PROTOCOL = "population-economic-information-v1"
GROUPS = ("fundamental_majority", "technical_majority", "hidden")
AUTO_ROOT = shared.WORKSPACE / "iterations/population_economic_auto_20261106"
SOURCE_FILES = (
    "scripts/collect_population_economic_probe.py",
    "tests/test_population_economic_probe.py",
)


def validate_plan(plan):
    shared.validate_plan(plan)
    require(plan["run_id"].startswith("economic-auto-population-"), "new population identity")
    require(145 <= plan["prior_POSTs"] <= 500 - COUNT, "whole36 must fit500")


def population_snapshot(case, group):
    """Lagged accounts and pending intentions, with no claimed fills or policy derivation."""
    require(group in GROUPS[:2], "visible group required")
    context = original.strict_json(case["payload"]["messages"][1]["content"])["agent_state"]
    peer_f = 7 if group == GROUPS[0] else 1
    styles = ["fundamental", "technical"] + ["fundamental"] * peer_f
    styles += ["technical"] * (10 - len(styles))
    agents, decisions, participants = [], [], []
    for i, style in enumerate(styles):
        agent_id = f"A{i:03d}"
        portfolio = Portfolio(1000, 2, 100)
        portfolio.mark_to_market(100)
        memory = deepcopy(context["memory"][:-1])
        for item in memory:
            item.update(owner=agent_id, style=style)
        state = AgentState(
            AgentPersona(agent_id, 0.5, 0.5, 0.5, 0.5, 0.5, style),
            portfolio,
            style,
            initial_equity=1200,
            block_start_equity=1200,
            memory=memory,
            peak_equity=1220,
        )
        decision = OrderDecision(
            action="hold" if i < 2 else "buy" if i < 6 else "sell",
            quantity=0 if i < 2 else 1,
            limit_price=None if i < 2 else 99 if i < 6 else 101,
            rationale="Constructed lagged pending intention; no fill is asserted.",
            sentiment="neutral",
        )
        participants.append(
            {
                "agent_id": agent_id,
                "current_style": style,
                "cash": 1000,
                "position": 2,
                "equity": 1200,
                "initial_equity": 1200,
                "memory": memory,
                "pending_order": decision.model_dump(),
            }
        )
        agents.append(state)
        decisions.append(decision)
    field = compute_mean_field(agents, decisions, as_of_session="2026-01-08")
    return field, {
        "data_origin": "constructed_fictional_population",
        "as_of_session": "2026-01-08",
        "participants": participants,
        "n_participants": 10,
        "includes_fixed_focal_accounts": ["A000", "A001"],
        "mean_field": field,
        "limits": "Other participants' HOLD histories are constructed, not derived policy "
        "decisions. Pending intentions are not accepted orders or executions. Only the "
        "aggregate mean_field is sent; this ledger and other participants' memories are local.",
    }


def make_packet(plan, evidence_kind="recorded_model_logs"):
    validate_plan(plan)
    base = shared.make_packet(plan, evidence_kind)
    package, _, _ = original.checked_materials()
    cases = {c["case_id"]: c for c in package["cases"]}
    payloads, ledgers = {}, {}
    for case_id, case in cases.items():
        for group in GROUPS:
            payload = deepcopy(case["payload"])
            payload["model"] = plan["model"]
            if group != "hidden":
                field, ledger = population_snapshot(case, group)
                user = original.strict_json(payload["messages"][1]["content"])
                user["observation"].pop("mean_field_availability")
                user["observation"]["mean_field"] = field
                payload["messages"][1]["content"] = canonical(user)
                ledgers[f"{case_id}.{group}"] = ledger
            payloads[case_id, group] = payload
    requests = []
    for base_request in base["requests"]:
        rotation = base_request["round"] - 1
        groups = GROUPS[rotation:] + GROUPS[:rotation]
        for group_position, group in enumerate(groups, 1):
            origin = original.identity(base_request)
            origin.update(
                kind=PROTOCOL,
                slot_id=f"{plan['run_id']}.{base_request['case_id']}.{group}"
                f".r{base_request['round']}",
                position=len(requests) + 1,
                population_condition=group,
                within_group_position=group_position,
            )
            origin.pop("request_hash")
            payload = deepcopy(payloads[base_request["case_id"], group])
            requests.append(
                {**origin, "request_hash": digest([payload, origin, PROTOCOL]), "payload": payload}
            )
    core = {k: v for k, v in base.items() if k != "packet_sha256"}
    core.update(
        protocol=PROTOCOL,
        requests=requests,
        adapter_sources={
            **base["adapter_sources"],
            **{name: original.fingerprint(shared.REPO / name) for name in SOURCE_FILES},
        },
        model_migration={
            **base["model_migration"],
            "messages_and_other_payload_options_unchanged": False,
        },
        construction_not_sent=ledgers,
        order_note="Original economic-case order, three Latin rotations of group display; "
        "each case/group appears once in each of three within-group positions. No old "
        "hidden answers are reused. Group effects are external information effects, not "
        "endogenous population feedback. Four primary and six secondary contrasts fixed.",
    )
    return {**core, "packet_sha256": digest(core)}


def check_packet(packet, plan):
    require(isinstance(packet, dict), "packet schema")
    require(
        canonical(packet) == canonical(make_packet(plan, packet.get("evidence_kind"))),
        "packet differs from frozen population/economic sources/materials/grant",
    )


def evaluator(plan, *, packet_validator=None):
    def evaluate(packet, requests, attempts, responses, targets=None):
        # Reuse only strict raw-log authentication/parsing and signature diagnostics.
        # The old12 summaries group by economic condition only and are deliberately discarded.
        raw = original.evaluate(
            packet,
            requests,
            attempts,
            responses,
            targets,
            packet_validator=(
                (lambda p: check_packet(p, plan)) if packet_validator is None else packet_validator
            ),
        )
        groups = defaultdict(list)
        for row in raw["rows"]:
            groups[row["agent_id"], row["condition"], row["population_condition"]].append(row)
        dispatched = {r["slot_id"] for r in attempts}
        received = {r["slot_id"] for r in responses}
        cells = []
        for (agent, economic, population), rows in groups.items():
            valid = [r for r in rows if r["compatible"]]
            counts = {
                s: sum(r["choice"]["target_style"] == s for r in valid)
                for s in ("fundamental", "technical")
            }
            alternative = "technical" if rows[0]["from_style"] == "fundamental" else "fundamental"
            cells.append(
                {
                    "agent_id": agent,
                    "original_style": rows[0]["from_style"],
                    "condition": economic,
                    "population_condition": population,
                    "planned": 3,
                    "dispatched": sum(r["slot_id"] in dispatched for r in rows),
                    "valid": len(valid),
                    "invalid": sum(r["slot_id"] in received and not r["compatible"] for r in rows),
                    "unsubmitted": sum(r["status"] == "unsubmitted" for r in rows),
                    "pending_records": sum(
                        r["status"] in ("request_record_only", "awaiting_response") for r in rows
                    ),
                    "target_counts": counts,
                    "stay_count": sum(r["choice"]["decision"] == "stay" for r in valid),
                    "switch_count": sum(r["choice"]["decision"] == "switch" for r in valid),
                    "fundamental_fraction_descriptive": counts["fundamental"] / len(valid)
                    if valid
                    else None,
                    "alternative_fraction_descriptive": counts[alternative] / len(valid)
                    if valid
                    else None,
                    "all_three_valid": len(valid) == 3,
                    "different_targets_on_same_input": len(
                        {r["choice"]["target_style"] for r in valid}
                    )
                    > 1,
                }
            )
        lookup = {(c["agent_id"], c["condition"], c["population_condition"]): c for c in cells}

        def contrast(left, right, field):
            complete = left["all_three_valid"] and right["all_three_valid"]
            return {
                "complete": complete,
                "difference": left[field] - right[field] if complete else None,
                "left_valid": left["valid"],
                "right_valid": right["valid"],
            }

        primary, secondary = [], []
        for agent in ("A000", "A001"):
            for economic in ("positive", "negative"):
                primary.append(
                    {
                        "agent_id": agent,
                        "condition": economic,
                        "definition": "raw fundamental fraction: F-majority minus T-majority",
                        **contrast(
                            lookup[agent, economic, GROUPS[0]],
                            lookup[agent, economic, GROUPS[1]],
                            "fundamental_fraction_descriptive",
                        ),
                    }
                )
            for population in GROUPS:
                secondary.append(
                    {
                        "agent_id": agent,
                        "population_condition": population,
                        "definition": "alternative fraction: economic positive minus negative",
                        **contrast(
                            lookup[agent, "positive", population],
                            lookup[agent, "negative", population],
                            "alternative_fraction_descriptive",
                        ),
                    }
                )
        return {
            **{
                k: raw[k]
                for k in (
                    "evidence_kind",
                    "local_logs_match_frozen_packet",
                    "logged_dispatch_attempts",
                    "protocol_complete",
                    "rows",
                    "repeated_strong_signature_groups",
                    "unknown_signature_slots",
                    "independent_generation_count",
                    "external_model_provenance_authenticated",
                )
            },
            "planned": COUNT,
            "requested_model": packet["model"],
            "cells": cells,
            "primary_population_contrasts": primary,
            "secondary_economic_contrasts": secondary,
            "limits": "Two fixed fictional accounts, four economic bundles, three repeats each. "
            "No mandatory correct style, no inference from old-model or prior hidden answers. "
            "Assigned information effects are not endogenous feedback, human alignment or "
            "market realism. Repeated/distinct signatures do not establish "
            "independent generations.",
        }

    return evaluate


def describe(result):
    lines = [
        f"群体信息与经济信息检查：计划36，尝试{result['logged_dispatch_attempts']}，"
        f"全部合格{result['protocol_complete']}。模型{result['requested_model']}。"
    ]
    for cell in result["cells"]:
        lines.append(
            f"{cell['agent_id']} / {cell['condition']} / {cell['population_condition']}："
            f"有效{cell['valid']}/3，基本面{cell['target_counts']['fundamental']}，"
            f"技术{cell['target_counts']['technical']}，改选{cell['switch_count']}，"
            f"无效{cell['invalid']}，未发{cell['unsubmitted']}，待定{cell['pending_records']}。"
        )
    for row in result["primary_population_contrasts"]:
        value = "未知" if row["difference"] is None else f"{row['difference'] * 100:.1f}个百分点"
        lines.append(
            f"{row['agent_id']} / {row['condition']} 基本面选择比例："
            f"多数人偏基本面 减 多数人偏技术 = {value}。"
        )
    lines += [
        "仅是两份假设账户对所给信息的反应，不代表真人、统计显著或市场真实性。",
        "隐藏信息也在本组重新问三次，旧回答没有混入本组。",
        "",
    ]
    return "\n".join(lines)


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
            "planned_POSTs": COUNT,
            "prior_POSTs": plan["prior_POSTs"],
            "global_cap": 500,
            "authorization": str(shared.GRANT_PATH),
            "no_resume": True,
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
    # Includes strict cap/prior/no-reservation checks, then strengthens twelve to thirty-six.
    shared.check_budget(ledger, plan)
    require(plan["prior_POSTs"] + COUNT <= 500, "whole36 exceeds remaining allowance")


def terminal_budget(ledger, plan, output):
    path = output / original.LOGS[1]
    sent = len(original.read_jsonl(path)) if path.exists() else 0
    require(0 <= sent <= COUNT, "unexpected attempt count")
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
            shared.save_budget(shared.BUDGET_PATH, terminal_budget(ledger, plan, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "evaluate"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.plan is not None, "plan file required")
        packet = prepare(args.folder, read(args.plan))
        print(canonical({"status": "prepared_not_sent", "packet_sha256": packet["packet_sha256"]}))
    elif args.action == "execute":
        result = asyncio.run(execute(args.folder))
        print(canonical({k: result[k] for k in ("logged_dispatch_attempts", "protocol_complete")}))
    else:
        plan, packet = load_packet(args.folder)
        raw = args.folder / "live_execution"
        logs = [original.read_jsonl(raw / n) if (raw / n).exists() else [] for n in original.LOGS]
        result = evaluator(plan)(packet, *logs)
        require(args.output is not None, "new output folder required")
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.txt").write_text(describe(result), encoding="utf-8")


if __name__ == "__main__":
    main()
