"""Bounded live dynamic pilot and zero-call replay under the existing500 authorization."""

import argparse
import asyncio
import hashlib
import math
import os
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import httpx

from behavioral_market.llm.transport_diagnostics import COMPLETION, REGISTRY, DiagnosticClient
from experiments import conditional_market_pilot as study
from scripts import collect_hypothetical_memory_probe as old
from scripts import collect_style_review_economic_continuation as shared
from scripts.validate_memory_information import save

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / "iterations/midterm_boundary_20261115"
FOLDER = STAGE / study.RUN_ID
read, rows, append = old.read, old.rows, old.append
canonical, require, digest = study.canonical, study.require, study.digest


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return sorted(
        str(p.relative_to(ROOT))
        for directory in ("src", "scripts", "experiments")
        for p in (ROOT / directory).rglob("*.py")
    ) + ["tests/test_conditional_market_pilot.py", "tests/test_conditional_market_collection.py"]


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in sources():
        data = (ROOT / name).read_bytes()
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_hashes[name] = sha(target)
    for name in ("GOAL.txt", "design.json"):
        (folder / name).write_bytes((STAGE / name).read_bytes())
    plan = {
        "run_id": study.RUN_ID,
        "model": study.MODEL,
        "route": "school_direct",
        "prior_POSTs": study.PRIOR,
        "maximum_new_POSTs": study.MAX_POSTS,
        "maximum_cumulative_POSTs": 469,
        "cap": 500,
        "sources": source_hashes,
        "inputs": {name: sha(folder / name) for name in ("GOAL.txt", "design.json")},
        "grant": shared.check_user_grant(),
        "route_confirmation": shared.check_route_confirmation({"route": "school_direct"}),
        "data_origin": "invented_accounts_and_simulated_history_only",
    }
    save(folder / "plan.json", plan)
    (folder / "plan.sha256").write_text(sha(folder / "plan.json") + "\n")
    return plan


def checked_plan(folder):
    plan = read(folder / "plan.json")
    require(
        sha(folder / "plan.json") == (folder / "plan.sha256").read_text().strip(), "plan changed"
    )
    require(
        plan["run_id"] == study.RUN_ID
        and plan["model"] == study.MODEL
        and plan["route"] == "school_direct"
        and plan["prior_POSTs"] == study.PRIOR
        and plan["maximum_new_POSTs"] == study.MAX_POSTS
        and plan["maximum_cumulative_POSTs"] == 469
        and plan["cap"] == 500
        and plan["data_origin"] == "invented_accounts_and_simulated_history_only"
        and set(plan["sources"]) == set(sources()),
        "scope changed",
    )
    for name, expected in plan["sources"].items():
        require(
            sha(ROOT / name) == expected and sha(folder / "sources" / name) == expected,
            "source changed: " + name,
        )
    for name, expected in plan["inputs"].items():
        require(sha(folder / name) == expected, "design input changed")
    require(
        shared.check_user_grant() == plan["grant"]
        and shared.check_route_confirmation(plan) == plan["route_confirmation"],
        "authority changed",
    )
    return plan


def audit_logs(output):
    requests, attempts, responses = [
        rows(output / name) for name in ("requests.jsonl", "attempts.jsonl", "responses.jsonl")
    ]
    require(len(requests) == len(attempts) == len(responses) <= study.MAX_POSTS, "incomplete logs")
    previous_end = None
    for index, (request, attempt, response) in enumerate(
        zip(requests, attempts, responses, strict=True)
    ):
        require(
            type(request["position"]) is int and request["position"] == index + 1,
            "request position changed",
        )
        identity = {k: request[k] for k in ("position", "request_hash")}
        require(
            all(
                type(attempt[k]) is type(v)
                and attempt[k] == v
                and type(response[k]) is type(v)
                and response[k] == v
                for k, v in identity.items()
            )
            and type(attempt["retry_count"]) is int
            and attempt["retry_count"] == 0
            and response["requested_at"] == attempt["dispatched_at"],
            "attempt identity changed",
        )
        start, end = (
            old.aware_time(attempt["dispatched_at"]),
            old.aware_time(response["response_received_at"]),
        )
        require(
            end >= start
            and (previous_end is None or start >= previous_end)
            and type(response["latency_seconds"]) in (int, float)
            and math.isfinite(response["latency_seconds"])
            and response["latency_seconds"] >= 0,
            "overlapping request times",
        )
        previous_end = end
        old.parsed_response(response, request)
    results = study.replay(requests, responses)
    for arm, result in results.items():
        require(canonical(read(output / (arm + ".json"))) == canonical(result), "replay differs")
        for tick, event in enumerate(result["events"]):
            require(
                canonical(read(output / f"event_{tick}_{arm}.json")) == canonical(event),
                "recorded event differs",
            )
    expected_pairs = []
    for index in range(0, len(requests), 2):
        request = requests[index]
        values = [
            old.parsed_response(responses[j], requests[j])["probability_buy"]
            for j in (index, index + 1)
        ]
        expected_pairs.append(
            {
                **{k: request[k] for k in ("arm", "agent_id", "tick")},
                "probabilities_buy": values,
                "mean_probability_buy": study.pair_probability(values),
            }
        )
    require(
        canonical(rows(output / "pairs.jsonl")) == canonical(expected_pairs),
        "pair evidence differs",
    )
    return results, study.summarize(results, requests, responses)


async def collect(client, output, secret=""):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("requests.jsonl", "attempts.jsonl", "responses.jsonl", "pairs.jsonl"):
        (output / name).touch()
    registry = await client.get(REGISTRY)
    require(registry.status_code == 200, "registry request failed")
    ids = [r["id"] for r in old.strict_json(registry.text)["data"]]
    save(output / "model_registry.json", {"model_ids": ids})
    require(study.MODEL in ids, "frozen model not available")
    count = 0
    demos = {}

    async def query_pair(messages, arm, agent, tick):
        nonlocal count
        values = []
        for material in study.request_pair(messages, arm, agent, tick):
            require(count < study.MAX_POSTS, "maximum POST bound reached")
            request = {**material, "position": count + 1}
            require(not secret or secret not in canonical(request["payload"]), "secret in request")
            append(output / "requests.jsonl", request, secret)
            start = now()
            identity = {k: request[k] for k in ("position", "request_hash")}
            append(
                output / "attempts.jsonl", {**identity, "dispatched_at": start, "retry_count": 0}
            )
            count += 1
            save(
                output / "budget_progress.json",
                {
                    "new_attempts": count,
                    "used_completion_POSTs": study.PRIOR + count,
                    "reserved_remaining": study.MAX_POSTS - count,
                    "cap": 500,
                    "no_resume": True,
                },
            )
            begun, status, body = time.perf_counter(), "transport_error", None
            header = {"present": False, "value_sha256": None}
            try:
                response = await client.post(COMPLETION, json=request["payload"])
                status = response.status_code
                value = response.headers.get("x-litellm-cache-key")
                header = {
                    "present": value is not None,
                    "value_sha256": hashlib.sha256(value.encode()).hexdigest() if value else None,
                }
                if status == 200:
                    try:
                        body = old.strict_json(response.text)
                    except ValueError:
                        body = {"_invalid_json_text": response.text}
            except (httpx.HTTPError, ValueError):
                pass
            row = {
                **identity,
                "requested_at": start,
                "response_received_at": now(),
                "retry_count": 0,
                "cached": False,
                "http_status": status,
                "latency_seconds": time.perf_counter() - begun,
                "cache_key_header": header,
                "response": body,
            }
            append(output / "responses.jsonl", row, secret)
            values.append(old.parsed_response(row, request)["probability_buy"])
        mean = study.pair_probability(values)
        append(
            output / "pairs.jsonl",
            {
                "arm": arm,
                "agent_id": agent,
                "tick": tick,
                "probabilities_buy": values,
                "mean_probability_buy": mean,
            },
        )
        return mean

    def unprepared(_messages):
        raise ValueError("probabilities must be prepared before executing a tick")

    for arm in study.ARMS:
        demos[arm] = study.execution.ConditionalExecutionDemo(
            dict.fromkeys(study.execution.AGENTS, unprepared), arm, study.CONFIG
        )
    for tick in range(6):
        for arm, demo in demos.items():
            # Freeze all eligible messages before any query or order in this batch.
            period, second = study.execution.TICKS[tick]
            prepared = [
                (
                    agent,
                    study.execution.prompt.build_question(
                        demo.observation(agent, period, second), arm, "buy_first"
                    )["messages"],
                )
                for i, agent in enumerate(study.execution.AGENTS)
                if study.CONFIG.opportunities[tick * 2 + i]
                and not demo.gate(agent, study.CONFIG.quantities[i])
            ]
            queue = []
            for agent, messages in prepared:
                queue.append((deepcopy(messages), await query_pair(messages, arm, agent, tick)))

            def callback(messages, *, queue=queue):
                require(bool(queue), "unprepared callback")
                expected_messages, value = queue.pop(0)
                require(
                    canonical(messages) == canonical(expected_messages), "prepared state changed"
                )
                return value

            demo.predictors = dict.fromkeys(study.execution.AGENTS, callback)
            event = demo.step()
            require(not queue, "eligible callback coverage differs")
            save(output / f"event_{tick}_{arm}.json", event)
    for arm, demo in demos.items():
        save(output / (arm + ".json"), demo.result())
    results, summary = audit_logs(output)
    save(output / "summary.json", summary)
    return results, summary


async def execute(folder):
    checked_plan(folder)
    require(folder.resolve() == FOLDER.resolve(), "canonical run folder required")
    review = read(folder / "independent_preflight.json")
    require(
        review.get("accepted_for_execution") is True
        and review.get("mustfix") == []
        and review.get("plan_sha256") == sha(folder / "plan.json"),
        "independent review required",
    )
    output, marker = folder / "live", folder / "execution_started.json"

    def unused():
        require(
            not any(p.exists() for p in (output, marker, folder / "transport_diagnostics")),
            "never resume started run",
        )

    unused()
    with shared.exclusive_budget(shared.BUDGET_PATH):
        unused()
        ledger = read(shared.BUDGET_PATH)
        shared.check_budget(ledger, {"prior_POSTs": study.PRIOR})
        require(study.PRIOR + study.MAX_POSTS <= 500, "full reservation exceeds authority")
        with marker.open("x") as handle:
            handle.write(
                canonical(
                    {
                        "run_id": study.RUN_ID,
                        "plan_sha256": sha(folder / "plan.json"),
                        "started_utc": now(),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        save(
            shared.BUDGET_PATH,
            {
                **ledger,
                "reserved_next_completion_POSTs": study.MAX_POSTS,
                "available_unreserved_POSTs": 500 - study.PRIOR - study.MAX_POSTS,
                "active_run_id": study.RUN_ID,
                "status": "running",
            },
        )
        state = {"status": "running"}
        try:
            from dotenv import dotenv_values

            key = dotenv_values(ROOT / ".env").get("AIGC_API_KEY")
            require(isinstance(key, str) and bool(key), "school key unavailable")
            async with shared.school_client(key, "school_direct") as http:
                require(
                    http._transport_for_url(httpx.URL(COMPLETION)) is http._transport,
                    "unexpected network route",
                )
                client = DiagnosticClient(
                    http, folder / "transport_diagnostics", route="default_direct"
                )
                _, summary = await collect(client, output, key)
            checked_plan(folder)
            save(folder / "summary.json", summary)
            state["status"] = "complete_pending_independent_audit"
        except BaseException as exc:
            state.update(status="stopped_incomplete", safe_stop_reason=type(exc).__name__)
            raise
        finally:
            sent = len(rows(output / "attempts.jsonl"))
            require(0 <= sent <= study.MAX_POSTS, "unexpected POST count")
            terminal = {
                **ledger,
                "used_completion_POSTs": study.PRIOR + sent,
                "reserved_next_completion_POSTs": 0,
                "available_unreserved_POSTs": 500 - study.PRIOR - sent,
                "active_run_id": None,
                "status": "terminal_pending_independent_audit",
                "last_run_id": study.RUN_ID,
                "last_output": str(folder),
                "version": ledger["version"] + 1,
            }
            persisted = False
            try:
                try:
                    save(
                        folder / "execution_state.json",
                        {**state, "new_attempts": sent, "finished_utc": now()},
                    )
                finally:
                    save(shared.BUDGET_PATH, terminal)
                persisted = True
            finally:
                # A completion must include a durable complete terminal receipt.
                if not persisted or (
                    read(shared.BUDGET_PATH)["active_run_id"] is not None
                    or state["status"] != "complete_pending_independent_audit"
                ):
                    if (folder / "summary.json").exists():
                        (folder / "summary.json").rename(folder / "incomplete_summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "replay"))
    parser.add_argument("--folder", type=Path, default=FOLDER)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.folder)
    elif args.action == "execute":
        asyncio.run(execute(args.folder))
    else:
        _, summary = audit_logs(args.folder / "live")
        require(
            canonical(summary) == canonical(read(args.folder / "summary.json")),
            "saved summary differs",
        )
        print(
            canonical({"replayed": True, "POSTs": 0, "recorded_POSTs": summary["new_model_POSTs"]})
        )


if __name__ == "__main__":
    main()
