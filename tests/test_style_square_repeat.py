"""Sixteen-call plan, real shared client and MockTransport; no credentials/network."""

import asyncio
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_style_bank_square import answer

from behavioral_market.evaluation import style_bank_square as square
from behavioral_market.evaluation.style_mechanism_design import canonical, contrast_edges, digest
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from behavioral_market.llm.client import OpenAICompatibleClient
from scripts import style_bank_square as cli
from scripts.style_fact_check import fingerprint, read_jsonl, save
from scripts.style_square_repeat import BASE, ORDERS, audit, make_master, prepare, run

pytest_plugins = ["test_style_bank_square"]


@pytest.fixture
def repeated(context):
    bank = context.workspace / BANK_PATH
    bank.parent.mkdir(parents=True)
    bank.write_text("".join(canonical(c) + "\n" for c in context.bank))
    edges = bank.parent / "contrasts.json"
    save(edges, contrast_edges(context.bank))
    plan = {
        "base_case_ids": list(BASE),
        "block_orders": [[BASE[i] for i in o] for o in ORDERS],
        "frozen_run_id": "repeat-in-memory",
        "execution_root": str(context.workspace / "iterations/repeat_execution"),
        "evidence_kind": "synthetic_tool_test",
        "prior_used_completion_POSTs": 35,
        "budget_sources": context.plan["budget_sources"],
        "historical_anchor": context.plan["historical_anchor"],
        "source_files": {
            str(p.relative_to(context.workspace)): fingerprint(p) for p in [bank, edges]
        },
    }
    return SimpleNamespace(context=context, plan=plan, master=make_master(context.bank, plan))


def executor(calls, bad_at=None):
    async def execute(args):
        package = cli.load_package(args.package)
        auth = json.loads(args.authorization.read_text())
        square.check_authorization(package, auth)
        cli.verify_budget_sources(args.workspace, package)
        output = Path(package["execution_output"])
        client = OpenAICompatibleClient(
            square.execution_config(package),
            "IN_MEMORY_TEST_ONLY",
            output,
            output / "fresh_cache",
            cache_namespace=square.NAMESPACE,
        )
        save(output / "package.json", package)
        save(output / "authorization_receipt.json", auth)
        await client.http.aclose()

        def handler(request):
            calls.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
            i = sum(r.method == "POST" for r in calls) - 1
            if i == bad_at:
                value = answer()
                value["choices"][0]["message"]["content"] = "{}"
            else:
                case = package["cases"][i % 4]
                value = answer(
                    current=case["factors"]["current_style"],
                    target="fundamental" if i // 4 % 2 == 0 else "technical",
                    identifier="same-fixture-id",
                )
            return httpx.Response(200, json=value)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await square.collect(package, auth, client, output)
        finally:
            await client.close()
        result = cli.scored(package, args.workspace, output)
        save(output / "evaluation.json", result)
        return result

    return execute


def test_all16_fixed_payloads_position_balance_and_late_binding(repeated):
    calls = []
    result = asyncio.run(run(repeated.master, repeated.context.workspace, executor(calls)))
    assert result["execution_complete"] and result["logged_dispatch_attempts"] == 16
    assert sum(r.method == "POST" for r in calls) == 16
    assert all(
        v["counts"]
        == {
            "fundamental": 2,
            "technical": 2,
            "unsubmitted": 0,
            "missing_response": 0,
            "invalid_or_transport": 0,
        }
        and v["observed_different_new_targets"]
        for v in result["per_case"].values()
    )
    positions = {cid: Counter() for cid in BASE}
    for b in repeated.master["blocks"]:
        for j, cid in enumerate(b["order"]):
            positions[cid][j + 1] += 1
    assert all(c == {1: 1, 2: 1, 3: 1, 4: 1} for c in positions.values())
    for request, frozen in zip(
        (r for r in calls if r.method == "POST"),
        (q for b in repeated.master["blocks"] for q in b["requests"]),
        strict=True,
    ):
        assert json.loads(request.content) == frozen["payload"]
        assert "budget_sources" not in request.content.decode()
        assert "source_bank_not_sent" not in request.content.decode()
    root = Path(repeated.plan["execution_root"])
    for i in range(4):
        package = cli.load_package(root / f"block_{i + 1}/package")
        assert package["prior_POSTs"] == 35 + i * 4
        ledger = json.loads((root / f"block_{i + 1}/budget_ledger.json").read_text())
        assert ledger["independent_review_performed"] is False
        assert ledger["status"] == "four_completed_locally_accepted"
    final = json.loads((root / "budget_ledger.json").read_text())
    assert final["used_completion_POSTs"] == 51 and final["new_logged_dispatches"] == 16
    assert result["cross_block_return_diagnostics"]["provider_id"] == [
        [r["instance_id"] for r in result["rows"]]
    ]
    assert result["independent_generation_count"] is result["p_value"] is None
    assert result["old_returns_in_new_denominator"] is False


def test_one_failure_stops_entire16_and_state_flags_cannot_hide_unknowns(repeated):
    calls = []
    result = asyncio.run(
        run(repeated.master, repeated.context.workspace, executor(calls, bad_at=5))
    )
    root = Path(repeated.plan["execution_root"])
    assert not result["execution_complete"] and result["logged_dispatch_attempts"] == 6
    assert sum(r.method == "POST" for r in calls) == 6
    assert not (root / "block_3").exists() and not (root / "block_4").exists()
    assert sum(r["compatible"] for r in result["rows"]) == 5
    assert sum(v["counts"]["invalid_or_transport"] for v in result["per_case"].values()) == 1
    assert sum(v["counts"]["unsubmitted"] for v in result["per_case"].values()) == 10
    save(root / "master_state.json", {"status": "complete", "completed_blocks": 4})
    assert audit(repeated.master, repeated.context.workspace) == result
    assert json.loads((root / "budget_ledger.json").read_text())["used_completion_POSTs"] == 41


@pytest.mark.parametrize("mutation", ["payload", "hash", "order", "position"])
def test_resigned_master_cannot_change_precommitted_sixteen(repeated, mutation):
    master = deepcopy(repeated.master)
    r = master["blocks"][0]["requests"][0]
    if mutation == "payload":
        r["payload"]["messages"][1]["content"] = '{"answer":"fundamental"}'
    elif mutation == "hash":
        r["request_hash"] = "wrong"
    elif mutation == "order":
        master["blocks"][1]["order"].reverse()
    else:
        r["position"] = 4
    master["master_sha256"] = digest({k: v for k, v in master.items() if k != "master_sha256"})
    calls = []
    with pytest.raises(ValueError):
        asyncio.run(run(master, repeated.context.workspace, executor(calls)))
    assert calls == [] and not Path(repeated.plan["execution_root"]).exists()


def test_no_resume_and_synthetic_package_cannot_use_live_executor(repeated):
    with pytest.raises(ValueError, match="tool test cannot call live"):
        asyncio.run(run(repeated.master, repeated.context.workspace))
    calls = []
    asyncio.run(run(repeated.master, repeated.context.workspace, executor(calls)))
    again = []
    with pytest.raises(FileExistsError):
        asyncio.run(run(repeated.master, repeated.context.workspace, executor(again)))
    assert again == []


def test_cross_block_clock_overlap_rejected_even_if_each_block_is_self_consistent(repeated):
    asyncio.run(run(repeated.master, repeated.context.workspace, executor([])))
    root = Path(repeated.plan["execution_root"])
    first = read_jsonl(root / "block_1/live_execution/llm_attempts.jsonl")[0]["dispatched_at"]
    attempts = read_jsonl(root / "block_2/live_execution/llm_attempts.jsonl")
    responses = read_jsonl(root / "block_2/live_execution/llm_responses.jsonl")
    attempts[0]["dispatched_at"], responses[0]["requested_at"] = first, first
    for name, rows in [("llm_attempts.jsonl", attempts), ("llm_responses.jsonl", responses)]:
        (root / f"block_2/live_execution/{name}").write_text(
            "".join(canonical(r) + "\n" for r in rows)
        )
    with pytest.raises(ValueError, match="overlap between blocks"):
        audit(repeated.master, repeated.context.workspace)


def test_cross_block_failure_stops_before_accepting_or_dispatching_next_block(repeated):
    calls = []
    execute = executor(calls)
    root = Path(repeated.plan["execution_root"])

    async def overlapping(args):
        await execute(args)
        if args.package.parent.name != "block_2":
            return
        first = read_jsonl(root / "block_1/live_execution/llm_attempts.jsonl")[0]["dispatched_at"]
        attempts = read_jsonl(root / "block_2/live_execution/llm_attempts.jsonl")
        responses = read_jsonl(root / "block_2/live_execution/llm_responses.jsonl")
        attempts[0]["dispatched_at"], responses[0]["requested_at"] = first, first
        for name, rows in [("llm_attempts.jsonl", attempts), ("llm_responses.jsonl", responses)]:
            (root / f"block_2/live_execution/{name}").write_text(
                "".join(canonical(row) + "\n" for row in rows)
            )

    with pytest.raises(ValueError, match="overlap between blocks"):
        asyncio.run(run(repeated.master, repeated.context.workspace, overlapping))
    assert sum(request.method == "POST" for request in calls) == 8
    assert not (root / "block_3").exists() and not (root / "block_4").exists()
    assert not (root / "block_2/budget_ledger.json").exists()
    state = json.loads((root / "master_state.json").read_text())
    assert state["status"] == "stopped" and state["completed_blocks"] == 1
    ledger = json.loads((root / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 43
    assert ledger["new_logged_dispatches"] == 8
    assert ledger["status"] == "stopped_pending_audit"


def test_prepare_has16_requests_but_no_fabricated_future_packages(repeated):
    workspace = repeated.context.workspace
    path = workspace / "plan.json"
    save(path, repeated.plan)
    folder = workspace / "master_package"
    assert (
        prepare(SimpleNamespace(workspace=workspace, plan=path, output=folder))["opportunities"]
        == 16
    )
    assert len(read_jsonl(folder / "frozen_requests.jsonl")) == 16
    assert not Path(repeated.plan["execution_root"]).exists()
    pending = json.loads((folder / "pending_evaluation.json").read_text())
    assert pending["logged_dispatch_attempts"] == 0
    assert all(v["counts"]["unsubmitted"] == 4 for v in pending["per_case"].values())


def test_incompatible_future_budget_source_layout_rejected_before_any_call(repeated):
    plan = deepcopy(repeated.plan)
    plan["execution_root"] = str(repeated.context.workspace / "outside_iterations")
    master = make_master(repeated.context.bank, plan)
    calls = []
    with pytest.raises(ValueError, match="execution root must be in iterations"):
        asyncio.run(run(master, repeated.context.workspace, executor(calls)))
    assert calls == [] and not Path(plan["execution_root"]).exists()
