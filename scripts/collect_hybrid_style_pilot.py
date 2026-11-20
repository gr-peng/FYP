"""A frozen 20-account/30-session hybrid pilot, with at most 120 style POSTs.

Future prompts depend on earlier choices. Freeze the generator and schedule,
then retain each exact prompt before dispatch; never claim all wires pre-existed.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from behavioral_market.evaluation.choice_presentation import signature
from behavioral_market.evaluation.style_probe_intake import aware_time, parse_choice, strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode
from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.llm.transport_diagnostics import COMPLETION, REGISTRY, DiagnosticClient
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.finite_rule_pair import METHOD, PROMPT_VERSION
from behavioral_market.simulation.hybrid_review import PROTOCOL
from scripts import collect_style_review_economic_continuation as shared
from scripts.audit_finite_lifecycle import audit
from scripts.run_hybrid_style_demo import compare, setup
from scripts.run_multiagent_rule_comparison import NoModelClient

REPO, WORKSPACE = shared.REPO, shared.WORKSPACE
AUTO_ROOT = WORKSPACE / "iterations/hybrid_style_auto_20261106"
IMPLEMENTATION = WORKSPACE / "iterations/hybrid_style_lifecycle_20261106"
COUNT, SEED = 120, 20261106
RUN_ID = "hybrid-style-20261106-001"
PRIOR = 229
LOGS = ("requests.jsonl", "attempts.jsonl", "responses.jsonl")
require, save, read, digest, canonical = (
    shared.require,
    shared.save,
    shared.read,
    shared.digest,
    shared.canonical,
)
OPTIONS = {
    "model": shared.MODEL,
    "temperature": 0.7,
    "max_tokens": 1200,
    "enable_thinking": False,
    "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    "cache": {"no-cache": True, "no-store": True},
    "response_format": {"type": "json_object"},
}


def now():
    return datetime.now(UTC).isoformat()


def records(path):
    return [strict_json(line) for line in path.read_text().splitlines()]


def append(path, row, secret=""):
    value = canonical(row)
    if secret:
        value = value.replace(secret, "[REDACTED]")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identities():
    return [
        {
            "agent_id": f"A{agent:03d}",
            "session_date": f"2026-01-{day:02d}",
            "treatment": "hybrid_style_llm",
            "kind": "style",
            "mode": "endogenous",
            "seed": SEED,
            "shock": 0,
            "event_condition": "no_event",
            "decision_batch": None,
            "mean_field_visibility": "hidden",
            "strategy_return_method": METHOD,
            "prompt_version": PROMPT_VERSION,
            "order_source": "rule",
            "style_source": "model_interface",
        }
        for day in (5, 10, 15, 20, 25, 30)
        for agent in range(20)
    ]


def payload(system, user):
    return {
        **copy.deepcopy(OPTIONS),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def parsed_response(row, request):
    require(
        row["http_status"] == 200
        and row["cached"] is False
        and row["retry_count"] == 0
        and row["cache_key_header"]["present"] is False,
        "transport or cache evidence incompatible",
    )
    choice, parsed = parse_choice(
        row["response"],
        {
            "payload": request["payload"],
            "source": {"from_style": request["from_style"]},
        },
    )
    require(
        choice == "valid_linked_answer"
        and returned_mode(row["response"]) == "no_returned_reasoning_observed",
        "response identity, format, mode or strategy inconsistency",
    )
    return StyleDecision.model_validate(parsed)


class StrictAdapter:
    hybrid_review_protocol = PROTOCOL
    review_provenance = "live_strict"

    def __init__(self, client, directory, plan, secret=""):
        directory.mkdir(parents=True, exist_ok=False)
        self.client, self.directory, self.plan, self.secret = client, directory, plan, secret
        self.ready, self.stopped, self.count = False, False, 0
        for name in LOGS:
            (directory / name).touch()

    async def verify(self):
        require(not self.ready and not self.stopped, "registry is single-attempt")
        self.stopped = True
        response = await self.client.get(REGISTRY)
        require(response.status_code == 200, "registry HTTP failure")
        ids = [row["id"] for row in strict_json(response.text)["data"]]
        save(self.directory / "model_registry.json", {"model_ids": ids})
        require(OPTIONS["model"] in ids, "required model missing")
        self.ready, self.stopped = True, False

    async def decide(self, system, user, schema, identity):
        require(self.ready and not self.stopped and self.count < COUNT, "adapter unavailable")
        self.stopped = True  # Remains stopped on every exception, including cancellation.
        require(
            schema is StyleDecision
            and canonical(identity) == canonical(self.plan["identities"][self.count]),
            "unexpected style identity/order",
        )
        require(not self.secret or self.secret not in system + user, "secret in prompt")
        state = strict_json(user)["agent_state"]
        require(set(state["persona"]) == {"agent_id"}, "unexpected persona disclosure")
        wire = payload(system, user)
        request = {
            "run_id": self.plan["run_id"],
            "position": self.count + 1,
            "identity": copy.deepcopy(identity),
            "from_style": state["current_style"],
            "payload": wire,
            "request_hash": digest({"identity": identity, "payload": wire}),
        }
        append(self.directory / LOGS[0], request, self.secret)
        start = now()
        base = {key: request[key] for key in ("run_id", "position", "identity", "request_hash")}
        append(self.directory / LOGS[1], {**base, "dispatched_at": start, "retry_count": 0})
        self.count += 1  # Conservatively consumes a slot even if transport outcome is unknown.
        save(
            self.directory / "budget_progress.json",
            {
                "used_completion_POSTs": self.plan["prior_POSTs"] + self.count,
                "new_attempts": self.count,
                "reserved_remaining": COUNT - self.count,
                "cap": 500,
                "no_resume": True,
            },
        )
        begun, status, body, cancelled = time.perf_counter(), "transport_error", None, False
        header = {"present": False, "value_sha256": None}
        try:
            response = await self.client.post(COMPLETION, json=wire)
            status = response.status_code
            value = response.headers.get("x-litellm-cache-key")
            header = {
                "present": value is not None,
                "value_sha256": hashlib.sha256(value.encode()).hexdigest() if value else None,
            }
            if status == 200:
                try:
                    body = strict_json(response.text)
                except ValueError:
                    body = {"_invalid_json_text": response.text}
        except asyncio.CancelledError:
            status, cancelled = "cancelled_in_flight", True
        except (httpx.HTTPError, ValueError):
            pass
        row = {
            **base,
            "requested_at": start,
            "response_received_at": now(),
            "retry_count": 0,
            "cached": False,
            "http_status": status,
            "latency_seconds": time.perf_counter() - begun,
            "response": body,
            "cache_key_header": header,
        }
        append(self.directory / LOGS[2], row, self.secret)  # Before parsing or applying any choice.
        if cancelled:
            raise asyncio.CancelledError
        decision = parsed_response(row, request)
        self.stopped = False
        return decision, {
            "status": "ok",
            "fallback": False,
            "cache_hit": False,
            "repair_count": 0,
            "request_hash": request["request_hash"],
        }


def check_logs(directory, plan):
    requests, attempts, responses = [records(directory / name) for name in LOGS]
    require(len(responses) <= len(attempts) <= len(requests) <= COUNT, "invalid log prefix")
    rows, previous_finish = [], None
    for index, request in enumerate(requests):
        require(
            set(request)
            == {"run_id", "position", "identity", "from_style", "payload", "request_hash"},
            "unexpected request fields",
        )
        require(index == 0 or rows[-1]["compatible"], "dispatch after invalid/pending response")
        require(
            request["run_id"] == plan["run_id"]
            and request["position"] == index + 1
            and type(request["position"]) is int
            and canonical(request["identity"]) == canonical(plan["identities"][index]),
            "log identity mismatch",
        )
        messages = request["payload"]["messages"]
        require(
            request["payload"] == payload(messages[0]["content"], messages[1]["content"])
            and request["request_hash"]
            == digest({"identity": request["identity"], "payload": request["payload"]}),
            "wire/options/hash mismatch",
        )
        require(
            strict_json(messages[1]["content"])["agent_state"]["current_style"]
            == request["from_style"],
            "request current style mismatch",
        )
        base = {key: request[key] for key in ("run_id", "position", "identity", "request_hash")}
        row = {
            "position": index + 1,
            "compatible": False,
            "parsed": None,
            "strong_signature": None,
            "status": "pending",
        }
        if index < len(attempts):
            attempt = attempts[index]
            require(
                set(attempt) == set(base) | {"dispatched_at", "retry_count"}
                and all(canonical(attempt.get(k)) == canonical(v) for k, v in base.items())
                and type(attempt["retry_count"]) is int
                and attempt["retry_count"] == 0,
                "attempt identity/retry mismatch",
            )
            start = aware_time(attempt["dispatched_at"])
            require(previous_finish is None or start >= previous_finish, "overlapping requests")
        if index < len(responses):
            response = responses[index]
            require(
                set(response)
                == set(base)
                | {
                    "requested_at",
                    "response_received_at",
                    "retry_count",
                    "cached",
                    "http_status",
                    "latency_seconds",
                    "response",
                    "cache_key_header",
                }
                and all(canonical(response.get(k)) == canonical(v) for k, v in base.items())
                and type(response["retry_count"]) is int
                and response["retry_count"] == 0
                and response["requested_at"] == attempts[index]["dispatched_at"],
                "response identity/time mismatch",
            )
            previous_finish = aware_time(response["response_received_at"])
            require(
                previous_finish >= start
                and type(response["latency_seconds"]) in (int, float)
                and math.isfinite(response["latency_seconds"])
                and response["latency_seconds"] >= 0,
                "invalid response timing",
            )
            try:
                parsed = parsed_response(response, request)
            except ValueError:
                row["status"] = "incompatible_response"
            else:
                row.update(compatible=True, parsed=parsed.model_dump(), status="compatible")
            row["strong_signature"] = signature(response["response"])
        rows.append(row)
    signatures = {}
    for row in rows:
        if row["strong_signature"] is not None:
            signatures.setdefault(digest(row["strong_signature"]), []).append(row["position"])
    return {
        "attempts": len(attempts),
        "compatible_responses": sum(r["compatible"] for r in rows),
        "complete": len(rows) == COUNT and all(r["compatible"] for r in rows),
        "rows": rows,
        "repeated_signature_groups": [group for group in signatures.values() if len(group) > 1],
        "unknown_signature_positions": [
            r["position"] for r in rows if r["strong_signature"] is None
        ],
        "independent_generation_count": None,
    }


class RecordedAdapter:
    """Offline replay verifies the full dynamic wire before supplying recorded answers."""

    hybrid_review_protocol = PROTOCOL
    review_provenance = "scripted"

    def __init__(self, directory):
        self.requests = records(directory / LOGS[0])
        self.responses = records(directory / LOGS[2])
        self.index = 0

    async def decide(self, system, user, schema, identity):
        require(
            schema is StyleDecision and self.index < len(self.requests), "replay schedule overflow"
        )
        request = self.requests[self.index]
        require(
            canonical(identity) == canonical(request["identity"])
            and canonical(payload(system, user)) == canonical(request["payload"]),
            "recomputed dynamic prompt differs from actual wire",
        )
        decision = parsed_response(self.responses[self.index], request)
        self.index += 1
        return decision, {
            "status": "ok",
            "fallback": False,
            "cache_hit": False,
            "repair_count": 0,
            "request_hash": request["request_hash"],
        }


async def replay_complete_run(folder, destination):
    plan = checked_plan(folder)
    scored = check_logs(folder / "live_logs", plan)
    require(scored["complete"], "only a complete valid run can be replayed for comparison")
    client = RecordedAdapter(folder / "live_logs")
    await event_runner.run_experiment(
        read(folder / "hybrid_config.json"),
        "endogenous",
        "hybrid_style_llm",
        "mock",
        destination,
        bars=pd.read_csv(folder / "warmup.csv"),
        events=[],
        client=client,
    )
    require(client.index == COUNT, "replay omitted responses")
    audit(destination)
    compare(folder / "hybrid", destination)
    require(
        (folder / "hybrid/finite_rule_replay.jsonl").read_bytes()
        == (destination / "finite_rule_replay.jsonl").read_bytes(),
        "paired-account replay differs",
    )
    frames = [
        pd.read_parquet(path / "style_switches.parquet").drop(columns="review_provenance")
        for path in (folder / "hybrid", destination)
    ]
    pd.testing.assert_frame_equal(*frames)
    return {
        "new_model_POSTs": 0,
        "exact_dynamic_prompts_checked": client.index,
        "trading_and_review_tables_match": True,
        "replay_provenance": "recorded_answers",
    }


def summarize_market(folder, response_audit):
    hybrid_reviews = pd.read_parquet(folder / "hybrid/style_switches.parquet")
    switched = hybrid_reviews.loc[hybrid_reviews.decision == "switch", "session_date"]
    first_switch = str(switched.min()) if len(switched) else None
    last_unchanged_day = first_switch or "2026-01-30"
    compare(folder / "fixed", folder / "hybrid", last_unchanged_day)
    summary = {
        "seed": SEED,
        "accounts": 20,
        "sessions": 30,
        "arms": {},
        "model_POSTs": response_audit["attempts"],
        "first5_trading_days_equal": True,
        "first_actual_switch_session": first_switch,
        "all_trading_equal_through": last_unchanged_day,
        "terminal_review_has_no_later_market": True,
        "independent_generation_count": None,
        "limits": "One-seed mechanism pilot, rules submit orders, model chooses styles. "
        "No human calibration, majority-effect, superiority or market-realism conclusion. "
        "Cash1000/shares2 capacity asymmetry is shared by both arms.",
    }
    for arm in ("fixed", "hybrid"):
        market = pd.read_parquet(folder / arm / "market.parquet")
        agents = pd.read_parquet(folder / arm / "agents.parquet")
        reviews = pd.read_parquet(folder / arm / "style_switches.parquet")
        totals = agents.groupby("session_date")[["cash", "position", "total_equity"]].sum()
        require(
            (totals.cash - 20000).abs().max() <= 0.00002 and (totals.position == 40).all(),
            "aggregate cash or shares not conserved",
        )
        reviews_by_date = [
            {
                "session_date": date,
                "switches": int((group.decision == "switch").sum()),
                "fundamental_targets": int((group.target_style == "fundamental").sum()),
                "accounts": len(group),
            }
            for date, group in reviews.groupby("session_date")
        ]
        last = float(market.close.iloc[-1])
        summary["arms"][arm] = {
            "last_close": last,
            "single_side_traded_quantity": int(market.simulated_volume.sum()),
            "trades": int(market.trade_count.sum()),
            "cash_total": float(totals.cash.iloc[-1]),
            "shares_total": int(totals.position.iloc[-1]),
            "wealth_gain_from_marking": float(totals.total_equity.iloc[-1]) - 24000,
            "daily_closes": market[["session_date", "close"]].to_dict("records"),
            "reviews": reviews_by_date,
        }
        require(
            abs(summary["arms"][arm]["wealth_gain_from_marking"] - 40 * (last - 100)) <= 0.00002,
            "wealth change differs from marking fixed stock supply",
        )
    summary["last_close_difference_hybrid_minus_fixed"] = (
        summary["arms"]["hybrid"]["last_close"] - summary["arms"]["fixed"]["last_close"]
    )
    save(folder / "market_comparison.json", summary)
    fixed, hybrid = summary["arms"]["fixed"], summary["arms"]["hybrid"]
    report = (
        "20个模拟账户、30天：规则下单，模型每5天选一次策略。\n\n"
        f"新增模型请求{summary['model_POSTs']}次。前5天两组交易完全一致。\n"
        f"固定策略组最后价格{fixed['last_close']:.2f}；模型选策略组{hybrid['last_close']:.2f}。\n"
        f"两组单边成交股数分别为{fixed['single_side_traded_quantity']}和{hybrid['single_side_traded_quantity']}。\n"
        "模型组每轮换策略人数："
        + "、".join(str(row["switches"]) for row in hybrid["reviews"])
        + "。\n"
        "最后一轮发生在30天全部交易结束后，不能用来解释之前的价格。\n"
        "全体现金与股票数量保持不变，合计财富变化来自股票按收盘价重新估值。\n"
        "用保存的回答重新生成全部120份请求和30天交易记录，已逐条核对一致；重放没有调用模型。\n\n"
        "这只是一组预先固定起点的机制演示。初始卖出能力受持仓限制；没有证明模型比规则好、"
        "选择像真人，或价格像真实市场。群体信息被隐藏，也没有检验多数人是否带动他人。\n"
    )
    (folder / "REPORT.txt").write_text(report, encoding="utf-8")
    return summary


def prepare(folder):
    # Do not guess an acceptance: implementation review and seal must already exist.
    implementation = read(IMPLEMENTATION / "independent_implementation_review.json")
    require(
        implementation["accepted_implementation"] is True and implementation["mustfix"] == [],
        "independently accepted offline implementation required",
    )
    seal = read(IMPLEMENTATION / "final_freeze.json")
    for name, info in seal["files"].items():
        path = IMPLEMENTATION / name
        require(shared.original.fingerprint(path) == info, "implementation evidence changed")
    shared.check_user_grant()
    shared.check_route_confirmation({"route": "school_direct"})
    folder.mkdir(parents=True, exist_ok=False)
    fixed, bars = setup()
    hybrid = copy.deepcopy(fixed)
    hybrid["behavior"].pop("rule_style_adaptation")
    hybrid["llm"] = {"providers": {"aigc_relay": copy.deepcopy(OPTIONS)}}
    save(folder / "fixed_config.json", fixed)
    save(folder / "hybrid_config.json", hybrid)
    bars.to_csv(folder / "warmup.csv", index=False)
    sources = (
        sorted((REPO / "src").rglob("*.py"))
        + sorted((REPO / "scripts").rglob("*.py"))
        + [
            REPO / path
            for path in (
                "tests/test_hybrid_style_collection.py",
                "tests/test_hybrid_style_lifecycle.py",
                "pyproject.toml",
            )
        ]
    )
    source_records = {}
    for path in sources:
        relative = path.relative_to(REPO)
        target = folder / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_records[str(relative)] = shared.original.fingerprint(path)
    inputs = ("fixed_config.json", "hybrid_config.json", "warmup.csv")
    plan = {
        "protocol": "hybrid-style-dynamic-pilot-v1",
        "evidence_kind": "recorded_model_logs",
        "run_id": RUN_ID,
        "prior_POSTs": PRIOR,
        "maximum_new_POSTs": COUNT,
        "maximum_cumulative_POSTs": PRIOR + COUNT,
        "route": "school_direct",
        "model": OPTIONS["model"],
        "options": OPTIONS,
        "identities": identities(),
        "sources": source_records,
        "inputs": {name: shared.original.fingerprint(folder / name) for name in inputs},
        "implementation_seal": shared.original.fingerprint(IMPLEMENTATION / "final_freeze.json"),
        "grant": shared.original.fingerprint(shared.GRANT_PATH),
        "route_confirmation": shared.original.fingerprint(shared.ROUTE_CONFIRMATION),
        "design": {
            "seed": SEED,
            "seed_origin": "task date; fixed before hybrid outcomes",
            "accounts": 20,
            "sessions": 30,
            "review_every": 5,
            "hidden_population": True,
            "primary": "paired fixed versus hybrid price/order paths after first review; "
            "all first5 days must match; describe switch counts and cash/share conservation",
            "terminal_boundary": "last20 choices have no subsequent trading window",
            "dynamic_wire": "source/config/schedule frozen; exact prompts depend on prior outcomes "
            "and all20 are saved at each review before its first dispatch",
            "known_limit": "1000cash/2shares initial inventory asymmetry retained in both arms; "
            "one seed, no human behavior or realistic market/superiority inference",
        },
    }
    save(folder / "plan.json", plan)
    (folder / "plan.sha256").write_text(digest(plan) + "\n")
    return plan


def checked_plan(folder):
    plan = read(folder / "plan.json")
    require(digest(plan) == (folder / "plan.sha256").read_text().strip(), "plan checksum mismatch")
    require(
        plan["run_id"] == RUN_ID
        and plan["evidence_kind"] == "recorded_model_logs"
        and plan["prior_POSTs"] == PRIOR
        and plan["maximum_new_POSTs"] == COUNT
        and plan["maximum_cumulative_POSTs"] == PRIOR + COUNT
        and plan["identities"] == identities()
        and plan["options"] == OPTIONS
        and plan["route"] == "school_direct",
        "plan differs from fixed protocol",
    )
    for name, info in plan["sources"].items():
        require(
            shared.original.fingerprint(REPO / name) == info
            and shared.original.fingerprint(folder / "sources" / name) == info,
            "source changed after freeze",
        )
    for name, info in plan["inputs"].items():
        require(shared.original.fingerprint(folder / name) == info, "input changed after freeze")
    require(
        shared.original.fingerprint(IMPLEMENTATION / "final_freeze.json")
        == plan["implementation_seal"],
        "implementation seal changed",
    )
    shared.check_user_grant()
    shared.check_route_confirmation(plan)
    require(
        shared.original.fingerprint(shared.GRANT_PATH) == plan["grant"]
        and shared.original.fingerprint(shared.ROUTE_CONFIRMATION) == plan["route_confirmation"],
        "grant or network route receipt changed",
    )
    return plan


async def execute(folder):
    plan = checked_plan(folder)
    require(folder.resolve() == (AUTO_ROOT / RUN_ID).resolve(), "canonical run directory required")
    review = read(folder / "independent_preflight.json")
    require(
        review["accepted_for_execution"] is True
        and review["mustfix"] == []
        and review["plan_sha256"] == digest(plan),
        "independent preflight required",
    )

    def require_unused():
        require(
            not any(
                (folder / name).exists()
                for name in (
                    "execution_started.json",
                    "fixed",
                    "hybrid",
                    "live_logs",
                    "transport_diagnostics",
                )
            ),
            "never resume a started pilot",
        )

    require_unused()
    with shared.exclusive_budget(shared.BUDGET_PATH):
        # A zero-POST failure can leave used quota unchanged. Recheck inside
        # the lock, and atomically create the start marker without overwriting it.
        require_unused()
        ledger = read(shared.BUDGET_PATH)
        shared.check_budget(ledger, {"prior_POSTs": PRIOR})
        require(
            PRIOR + COUNT <= ledger["global_authorized_completion_POST_cap"],
            "full quota unavailable",
        )
        with (folder / "execution_started.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "started_utc": now(),
                    "run_id": RUN_ID,
                    "plan_sha256": digest(plan),
                    "no_resume": True,
                },
                handle,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        shared.save_budget(
            shared.BUDGET_PATH,
            {
                **ledger,
                "reserved_next_completion_POSTs": COUNT,
                "available_unreserved_POSTs": 500 - PRIOR - COUNT,
                "active_run_id": RUN_ID,
                "status": "running",
            },
        )
        state = {"status": "running", "new_attempts": 0}
        try:
            bars = pd.read_csv(folder / "warmup.csv")
            await event_runner.run_experiment(
                read(folder / "fixed_config.json"),
                "endogenous",
                "rule_abm",
                "mock",
                folder / "fixed",
                bars=bars,
                events=[],
                client=NoModelClient(),
            )
            save(folder / "fixed_accounting_audit.json", audit(folder / "fixed"))
            from dotenv import dotenv_values

            key = dotenv_values(REPO / ".env").get("AIGC_API_KEY")
            require(isinstance(key, str) and bool(key), "configured school API key missing")
            async with shared.school_client(key, "school_direct") as http:
                require(
                    http._transport_for_url(httpx.URL(COMPLETION)) is http._transport,
                    "school route differs from direct plan",
                )
                client = StrictAdapter(
                    DiagnosticClient(
                        http, folder / "transport_diagnostics", route="default_direct"
                    ),
                    folder / "live_logs",
                    plan,
                    secret=key,
                )
                await client.verify()
                await event_runner.run_experiment(
                    read(folder / "hybrid_config.json"),
                    "endogenous",
                    "hybrid_style_llm",
                    "aigc_relay",
                    folder / "hybrid",
                    bars=bars,
                    events=[],
                    client=client,
                    require_clean=False,
                )
            save(folder / "hybrid_accounting_audit.json", audit(folder / "hybrid"))
            compare(folder / "fixed", folder / "hybrid", "2026-01-05")
            scored = check_logs(folder / "live_logs", plan)
            require(scored["complete"], "incomplete response collection")
            save(folder / "response_audit.json", scored)
            replay = await replay_complete_run(folder, folder / "recorded_replay")
            save(folder / "recorded_replay_audit.json", replay)
            summarize_market(folder, scored)
            state["status"] = "complete_pending_independent_audit"
        except (Exception, asyncio.CancelledError) as exc:
            state.update(status="stopped_incomplete", safe_stop_reason=type(exc).__name__)
            raise
        finally:
            attempt_path = folder / "live_logs" / LOGS[1]
            sent = len(records(attempt_path)) if attempt_path.exists() else 0
            require(0 <= sent <= COUNT, "attempt count exceeded frozen quota")
            state["new_attempts"] = sent
            state["finished_utc"] = now()
            save(folder / "execution_state.json", state)
            shared.save_budget(
                shared.BUDGET_PATH,
                {
                    **ledger,
                    "used_completion_POSTs": PRIOR + sent,
                    "reserved_next_completion_POSTs": 0,
                    "available_unreserved_POSTs": 500 - PRIOR - sent,
                    "active_run_id": None,
                    "status": "terminal_pending_independent_audit",
                    "last_run_id": RUN_ID,
                    "last_output": str(folder),
                    "version": ledger["version"] + 1,
                },
            )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.folder) if args.action == "prepare" else asyncio.run(execute(args.folder))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key in ("run_id", "status", "new_attempts")
            }
        )
    )


if __name__ == "__main__":
    main()
