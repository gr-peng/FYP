"""Precommit sixteen calls and reuse the accepted four-cell executor by block."""

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from behavioral_market.evaluation import style_bank_square as square
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from scripts import style_bank_square as cli
from scripts.style_fact_check import fingerprint, read_jsonl, save

BASE = ("synthetic_0000", "synthetic_0100", "synthetic_1000", "synthetic_1100")
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
CONFIG_KEYS = (
    "base_case_ids",
    "block_orders",
    "frozen_run_id",
    "execution_root",
    "evidence_kind",
    "prior_used_completion_POSTs",
    "budget_sources",
    "historical_anchor",
    "source_files",
)


def child_plan(master, index, prior_descriptor):
    c, b = master["config"], master["blocks"][index]
    return {
        "source_files": c["source_files"],
        "selected_case_ids": b["order"],
        "frozen_run_id": b["run_id"],
        "fixed_execution_output": b["execution_output"],
        "evidence_kind": c["evidence_kind"],
        "prior_used_completion_POSTs": c["prior_used_completion_POSTs"] + 4 * index,
        "budget_sources": {"grant": c["budget_sources"]["grant"], "prior_ledger": prior_descriptor},
        "historical_anchor": c["historical_anchor"],
    }


def make_master(bank, plan):
    c = {k: deepcopy(plan[k]) for k in CONFIG_KEYS}
    square.require(c["base_case_ids"] == list(BASE), "fixed four reference/origin cells required")
    orders = [[BASE[i] for i in order] for order in ORDERS]
    square.require(c["block_orders"] == orders, "fixed balanced orders differ")
    prior = c["prior_used_completion_POSTs"]
    square.require(type(prior) is int and 0 <= prior <= 84, "sixteen calls exceed global100")
    root = Path(c["execution_root"])
    square.require(root.is_absolute(), "fixed absolute execution root required")
    by_id = {r["case_id"]: r for r in bank}
    blocks = []
    for i, order in enumerate(orders):
        run_id = c["frozen_run_id"] + f"-b{i + 1}"
        cases = [by_id[cid] for cid in order]
        stub = {"cases": cases, "run_id": run_id, "evidence_kind": c["evidence_kind"]}
        blocks.append(
            {
                "index": i + 1,
                "run_id": run_id,
                "order": order,
                "execution_output": str(root / f"block_{i + 1}" / "live_execution"),
                "requests": [square.expected_request(case, stub) for case in cases],
            }
        )
    core = {
        "protocol": "position-balanced-square-repeat-v1",
        "config": c,
        "blocks": blocks,
        "max_new_POSTs": 16,
        "global_cap": 100,
        "allocation_stop_at": prior + 16,
        "late_binding_only": (
            "Previous actual terminal local ledger path/fingerprint and resulting child "
            "package/auth hashes; no future ledger fabricated."
        ),
        "stop_policy": (
            "Any protocol failure stops entire plan; no resume/retry/repair/salt. "
            "Valid choices and rationale errors retained."
        ),
        "analysis": (
            "Sixteen new opportunities; per-case F/T/unknown, per-block contrasts and "
            "cross-block reuse; no old pooling, p values or independent-generation claim."
        ),
    }
    # Validate the actual bank, square, model envelope and descriptors using the existing engine.
    square.make_package(bank, child_plan(core, 0, c["budget_sources"]["prior_ledger"]))
    return {**core, "master_sha256": digest(core)}


def checked_master(master, workspace):
    root = Path(master["config"]["execution_root"]).resolve()
    square.require(root.is_relative_to(workspace.resolve()), "execution root outside workspace")
    square.require(
        root.relative_to(workspace.resolve()).parts[0] == "iterations",
        "execution root must be in iterations for later budget sources",
    )
    for p, fp in master["config"]["source_files"].items():
        square.require(fingerprint(workspace / p) == fp, "bank source bytes differ")
    bank = read_jsonl(workspace / BANK_PATH)
    square.require(
        canonical(master) == canonical(make_master(bank, master["config"])),
        "master/request/order differs",
    )
    return bank


def prior_descriptor(master, workspace, index):
    if index == 0:
        return master["config"]["budget_sources"]["prior_ledger"]
    p = Path(master["config"]["execution_root"]) / f"block_{index}" / "budget_ledger.json"
    square.require(p.resolve().is_relative_to(workspace.resolve()), "ledger outside workspace")
    return {"path": str(p.relative_to(workspace)), "fingerprint": fingerprint(p)}


def audit(master, workspace):
    """Rebuild all sixteen statuses from raw logs; never trust master/child state flags."""
    bank = checked_master(master, workspace)
    by_id = {c["case_id"]: c for c in bank}
    rows, blocks, dispatched = [], [], 0
    previous_complete, previous_finish = True, None
    root = Path(master["config"]["execution_root"])
    allowed_logs = set()
    for i, block in enumerate(master["blocks"]):
        folder = root / f"block_{i + 1}"
        if (folder / "package").exists():
            square.require(previous_complete, "block started after protocol failure")
            package = cli.load_package(folder / "package")
            wanted = square.make_package(
                bank, child_plan(master, i, prior_descriptor(master, workspace, i))
            )
            square.require(canonical(package) == canonical(wanted), "late-bound package differs")
            square.require(
                canonical([square.expected_request(c, package) for c in package["cases"]])
                == canonical(block["requests"]),
                "requests differ from frozen sixteen",
            )
            cli.verify_budget_sources(workspace, package)
            live = Path(block["execution_output"])
            result = cli.scored(package, workspace, live)
            attempts, responses = (
                read_jsonl(live / "llm_attempts.jsonl"),
                read_jsonl(live / "llm_responses.jsonl"),
            )
            allowed_logs.add(live / "llm_attempts.jsonl")
            if attempts and previous_finish is not None:
                square.require(
                    aware_time(attempts[0]["dispatched_at"]) >= previous_finish,
                    "overlap between blocks",
                )
            if result["complete"]:
                previous_finish = aware_time(responses[-1]["response_received_at"])
            dispatched += result["logged_dispatch_attempts"]
            previous_complete = result["complete"]
            blocks.append(
                {
                    "block_index": i + 1,
                    "complete": result["complete"],
                    "contrasts": result["contrasts"],
                    "historical_anchor_diagnostics": result["historical_anchor_diagnostics"],
                }
            )
            current = result["rows"]
        else:
            previous_complete = False
            first = square.make_package(
                bank, child_plan(master, 0, master["config"]["budget_sources"]["prior_ledger"])
            )
            contrasts = [
                {**e, "comparable": False, "target_fundamental_difference": None}
                for e in first["contrasts"]
            ]
            blocks.append({"block_index": i + 1, "complete": False, "contrasts": contrasts})
            current = [
                {
                    "case_id": cid,
                    "factors": by_id[cid]["factors"],
                    "status": "unsubmitted",
                    "compatible": False,
                    "choice": None,
                    "provider_id": None,
                    "created": None,
                    "body_sha256": None,
                    "text_sha256": None,
                }
                for cid in block["order"]
            ]
        for j, row in enumerate(current):
            rows.append(
                {
                    **row,
                    "block_index": i + 1,
                    "global_position": i * 4 + j + 1,
                    "instance_id": f"block_{i + 1}:{row['case_id']}",
                }
            )
    square.require(set(root.rglob("llm_attempts.jsonl")) <= allowed_logs, "unplanned attempt log")
    summary = {}
    for cid in BASE:
        subset = [r for r in rows if r["case_id"] == cid]
        counts = Counter()
        for row in subset:
            if row["compatible"]:
                counts[row["choice"]["target_style"]] += 1
            elif row["status"] == "unsubmitted":
                counts["unsubmitted"] += 1
            elif row["status"] in {"request_record_only", "awaiting_response"}:
                counts["missing_response"] += 1
            else:
                counts["invalid_or_transport"] += 1
        counts = {
            key: counts[key]
            for key in (
                "fundamental",
                "technical",
                "unsubmitted",
                "missing_response",
                "invalid_or_transport",
            )
        }
        valid = counts["fundamental"] + counts["technical"]
        summary[cid] = {
            "counts": counts,
            "four_new_returns_valid": valid == 4,
            "observed_different_new_targets": bool(counts["fundamental"] and counts["technical"])
            if valid
            else None,
        }
    repeats = {}
    for field in ("provider_id", "created", "body_sha256", "text_sha256", "same_id_created_text"):
        groups = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if field == "same_id_created_text":
                value = (
                    [row["provider_id"], row["created"], row["text_sha256"]]
                    if all(row[k] is not None for k in ("provider_id", "created", "text_sha256"))
                    else None
                )
            if value is not None:
                groups[digest(value)].append(row["instance_id"])
        repeats[field] = [ids for ids in groups.values() if len(ids) > 1]
    return {
        "evidence_kind": master["config"]["evidence_kind"],
        "rows": rows,
        "blocks": blocks,
        "per_case": summary,
        "planned_opportunities": 16,
        "logged_dispatch_attempts": dispatched,
        "execution_complete": all(b["complete"] for b in blocks),
        "cross_block_return_diagnostics": repeats,
        "old_returns_in_new_denominator": False,
        "independent_generation_count": None,
        "stability_probability": None,
        "p_value": None,
        "human_alignment_established": False,
        "limit": (
            "Finite position-balanced service returns, not independent samples or stable "
            "psychological/market mechanisms."
        ),
    }


def describe(result):
    text = ["# 十六条固定重复检查", ""]
    if result["evidence_kind"] == "synthetic_tool_test":
        text += ["**本地工具测试，不能算模型回答。**", ""]
    text += [
        f"计划16个机会，记录到{result['logged_dispatch_attempts']}次发送尝试。",
        "",
        "| 输入 | 基本面 | 技术 | 未提交 | 缺返回 | 无效或传输异常 | 本轮观察到不同目标 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, item in result["per_case"].items():
        c = item["counts"]
        change = item["observed_different_new_targets"]
        text.append(
            f"| {cid} | {c['fundamental']} | {c['technical']} | {c['unsubmitted']} | "
            f"{c['missing_response']} | {c['invalid_or_transport']} | "
            f"{'未知' if change is None else '是' if change else '尚未观察到'} |"
        )
    text += ["", "| 块 | 四个单因素差值（选择基本面） |", "|---|---|"]
    for block in result["blocks"]:
        diffs = [
            str(e["target_fundamental_difference"]) if e["comparable"] else "未知"
            for e in block["contrasts"]
        ]
        text.append(f"| {block['block_index']} | {', '.join(diffs)} |")
    text += [
        "",
        "全部16条跨块返回重复组：",
        canonical(result["cross_block_return_diagnostics"]),
        "",
        "旧的基本面/技术两次发现记录不计入新16条分母。没有重复不证明独立生成，相同选择不证明稳定机制。",
        "未完成、无效与不同选择分开保留；不补发、不改提示、不加随机文字、不做显著性检验。",
        "",
    ]
    return "\n".join(text)


def prepare(args):
    plan = strict_json(args.plan.read_text())
    bank = read_jsonl(args.workspace / BANK_PATH)
    master = make_master(bank, plan)
    checked_master(master, args.workspace)
    first = square.make_package(
        bank, child_plan(master, 0, master["config"]["budget_sources"]["prior_ledger"])
    )
    grant = cli.verify_budget_sources(args.workspace, first)
    old = cli.load_history(args.workspace, first)
    square.historical_diagnostics(
        square.evaluate(first, [], [], []), old["body"], old["payload_sha256"], first
    )
    square.require(
        not Path(master["config"]["execution_root"]).exists(), "execution root already exists"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "master.json", master)
    frozen = [
        {"block_index": b["index"], "global_position": i * 4 + j + 1, "request": request}
        for i, b in enumerate(master["blocks"])
        for j, request in enumerate(b["requests"])
    ]
    (args.output / "frozen_requests.jsonl").write_text("".join(canonical(r) + "\n" for r in frozen))
    save(
        args.output / "global_allocation.json",
        {
            "master_sha256": master["master_sha256"],
            "decision": "allocated_under_existing_global_grant",
            "prior_POSTs": first["prior_POSTs"],
            "additional_limit": 16,
            "cumulative_stop": master["allocation_stop_at"],
            "global_cap": 100,
            "specific_user_reply": grant["specific_user_reply"],
            "budget_sources": first["budget_sources"],
        },
    )
    pending = audit(master, args.workspace)
    save(args.output / "pending_evaluation.json", pending)
    (args.output / "REPORT.md").write_text(describe(pending))
    return {"status": "prepared_not_sent", "opportunities": 16}


async def run(master, workspace, block_executor=None):
    bank = checked_master(master, workspace)
    first = square.make_package(
        bank, child_plan(master, 0, master["config"]["budget_sources"]["prior_ledger"])
    )
    cli.verify_budget_sources(workspace, first)
    if block_executor is None:
        square.require(
            master["config"]["evidence_kind"] == "recorded_model_logs", "tool test cannot call live"
        )
        block_executor = cli.execute
    root = Path(master["config"]["execution_root"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "master.json", master)
    state = {"status": "running", "completed_blocks": 0, "stop_reason": None}

    def checkpoint():
        save(root / "master_state.json", state)

    checkpoint()
    try:
        for i, block in enumerate(master["blocks"]):
            folder = root / f"block_{i + 1}"
            folder.mkdir()
            plan = child_plan(master, i, prior_descriptor(master, workspace, i))
            save(folder / "plan.json", plan)
            cli.prepare(
                SimpleNamespace(
                    workspace=workspace, plan=folder / "plan.json", output=folder / "package"
                )
            )
            package = cli.load_package(folder / "package")
            square.require(
                canonical([square.expected_request(c, package) for c in package["cases"]])
                == canonical(block["requests"]),
                "late binding changed frozen requests",
            )
            await block_executor(
                SimpleNamespace(
                    workspace=workspace,
                    package=folder / "package",
                    authorization=folder / "package/allocated_authorization.json",
                )
            )
            result = cli.scored(package, workspace, Path(block["execution_output"]))
            # Validate the complete observed prefix before accepting this block or
            # dispatching another one; per-block scoring cannot detect overlap.
            audit(master, workspace)
            used = plan["prior_used_completion_POSTs"] + result["logged_dispatch_attempts"]
            ledger = {
                "global_authorization_source": str(
                    (workspace / master["config"]["budget_sources"]["grant"]["path"]).resolve()
                ),
                "global_authorized_completion_POST_cap": 100,
                "used_completion_POSTs": used,
                "reserved_next_completion_POSTs": 0,
                "active_run_id": None,
                "status": "four_completed_locally_accepted"
                if result["complete"]
                else "stopped_not_accepted",
                "independent_review_performed": False,
            }
            save(folder / "budget_ledger.json", ledger)
            if not result["complete"]:
                state.update(status="stopped", stop_reason="block protocol failure")
                break
            state["completed_blocks"] = i + 1
            checkpoint()
        else:
            state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(
            status="cancelled", stop_reason="cancelled; in-flight billing unknown; no restart"
        )
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        checkpoint()
        attempts = sum(len(p.read_text().splitlines()) for p in root.rglob("llm_attempts.jsonl"))
        used = master["config"]["prior_used_completion_POSTs"] + attempts
        save(
            root / "budget_ledger.json",
            {
                "global_authorization_source": str(
                    (workspace / master["config"]["budget_sources"]["grant"]["path"]).resolve()
                ),
                "global_authorized_completion_POST_cap": 100,
                "used_completion_POSTs": used,
                "reserved_next_completion_POSTs": 0,
                "available_unreserved_POSTs": 100 - used,
                "active_run_id": None,
                "status": "completed_pending_independent_audit"
                if state["status"] == "complete"
                else "stopped_pending_audit",
                "new_logged_dispatches": attempts,
                "no_resume": True,
            },
        )
    result = audit(master, workspace)
    save(root / "evaluation.json", result)
    (root / "REPORT.md").write_text(describe(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "execute", "evaluate"])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--package", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    else:
        master = strict_json((args.package / "master.json").read_text())
        frozen = [
            {"block_index": b["index"], "global_position": i * 4 + j + 1, "request": request}
            for i, b in enumerate(master["blocks"])
            for j, request in enumerate(b["requests"])
        ]
        square.require(
            canonical(read_jsonl(args.package / "frozen_requests.jsonl")) == canonical(frozen),
            "frozen request artifact differs",
        )
        if args.action == "execute":
            result = asyncio.run(run(master, args.workspace))
        else:
            result = audit(master, args.workspace)
            args.output.mkdir(parents=True, exist_ok=False)
            save(args.output / "evaluation.json", result)
            (args.output / "REPORT.md").write_text(describe(result))
        result = {
            "execution_complete": result["execution_complete"],
            "logged_dispatch_attempts": result["logged_dispatch_attempts"],
        }
    print(canonical(result))


if __name__ == "__main__":
    main()
