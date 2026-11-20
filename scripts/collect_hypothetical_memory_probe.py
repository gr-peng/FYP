"""Run 48 frozen hypothetical memory questions under the existing cumulative500 grant."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from behavioral_market.evaluation.choice_presentation import signature
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from behavioral_market.llm.transport_diagnostics import COMPLETION, REGISTRY, DiagnosticClient
from experiments import hypothetical_memory_probe as study
from scripts import collect_style_review_economic_continuation as shared
from scripts.validate_memory_information import save

ROOT = shared.REPO
STAGE = ROOT.parent / "iterations/hypothetical_memory_probe_20261107"
FOLDER = STAGE / study.RUN_ID
BRIDGE = ROOT.parent / "iterations/conditional_memory_prompt_20261107"
BRIDGE_SEAL = "eeacf5aa05233e1aeb6e932a0d5dc3aff42f3782413bfbddb9594e1b74cb246f"
DESIGN_SHA = "d21bf26ef469af18e2d3552347796c96f7cc1862aa230e34a3857df43638281c"
LOGS = ("requests.jsonl", "attempts.jsonl", "responses.jsonl")
require, canonical, digest = study.require, study.canonical, study.digest


def read(path):
    return strict_json(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def append(path, value, secret=""):
    raw = canonical(value)
    if secret:
        raw = raw.replace(secret, "[REDACTED]")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(raw + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rows(path):
    return [strict_json(line) for line in path.read_text().splitlines()] if path.exists() else []


def sources():
    return sorted(
        {
            str(p.relative_to(ROOT))
            for name in ("src", "experiments", "scripts")
            for p in (ROOT / name).rglob("*.py")
        }
        | {
            "tests/test_hypothetical_memory_probe.py",
            "tests/test_hypothetical_memory_collection.py",
            "tests/test_conditional_memory_prompt.py",
            "tests/test_conditional_order_memory.py",
            "tests/test_replication_quote_memory.py",
            "tests/test_replication_quote_states.py",
            "pyproject.toml",
        }
    )


def prepare(folder):
    require(sha(STAGE / "design_v1.json") == DESIGN_SHA, "design changed")
    require(sha(BRIDGE / "final_freeze.json") == BRIDGE_SEAL, "accepted bridge changed")
    review = read(BRIDGE / "independent_engineering_review_v2.json")
    require(
        review["accepted_synthetic_material_builder"] is True and review["mustfix"] == [],
        "bridge not independently accepted",
    )
    grant = shared.check_user_grant()
    route = shared.check_route_confirmation({"route": "school_direct"})
    folder.mkdir(parents=True, exist_ok=False)
    inputs = {
        "design.json": STAGE / "design_v1.json",
        "bridge_review.json": BRIDGE / "independent_engineering_review_v2.json",
        "bridge_seal.json": BRIDGE / "final_freeze.json",
    }
    for name, path in inputs.items():
        shutil.copyfile(path, folder / name)
    source_hashes = {}
    for name in sources():
        dest = folder / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        source_hashes[name] = sha(dest)
    save(folder / "materials.json", study.materials())
    plan = {
        "protocol": study.PROTOCOL,
        "run_id": study.RUN_ID,
        "route": "school_direct",
        "prior_POSTs": study.PRIOR,
        "maximum_new_POSTs": study.COUNT,
        "maximum_cumulative_POSTs": study.PRIOR + study.COUNT,
        "cap": 500,
        "sources": source_hashes,
        "inputs": {name: sha(folder / name) for name in (*inputs, "materials.json")},
        "grant": grant,
        "route_confirmation": route,
        "bridge_seal_sha256": BRIDGE_SEAL,
        "data_origin": "entirely_invented_no_human_rows",
    }
    save(folder / "plan.json", plan)
    (folder / "plan.sha256").write_text(sha(folder / "plan.json") + "\n")
    return plan


def checked_plan(folder):
    plan = read(folder / "plan.json")
    require(
        sha(folder / "plan.json") == (folder / "plan.sha256").read_text().strip(),
        "plan fingerprint differs",
    )
    require(
        plan["protocol"] == study.PROTOCOL
        and plan["run_id"] == study.RUN_ID
        and plan["route"] == "school_direct"
        and plan["prior_POSTs"] == study.PRIOR
        and plan["maximum_new_POSTs"] == study.COUNT
        and plan["maximum_cumulative_POSTs"] == 397
        and plan["cap"] == 500
        and plan["data_origin"] == "entirely_invented_no_human_rows"
        and set(plan["sources"]) == set(sources()),
        "scope differs",
    )
    for name, expected in plan["sources"].items():
        require(
            sha(ROOT / name) == expected and sha(folder / "sources" / name) == expected,
            "frozen source changed: " + name,
        )
    for name, expected in plan["inputs"].items():
        require(sha(folder / name) == expected, "frozen input changed: " + name)
    require(
        sha(BRIDGE / "final_freeze.json") == plan["bridge_seal_sha256"] == BRIDGE_SEAL,
        "bridge seal changed",
    )
    require(
        shared.check_user_grant() == plan["grant"]
        and shared.check_route_confirmation(plan) == plan["route_confirmation"],
        "authority or route receipt changed",
    )
    require(
        canonical(read(folder / "materials.json")) == canonical(study.materials()),
        "wire package differs",
    )
    return plan


def parsed_response(row, request):
    require(
        row["http_status"] == 200
        and type(row["http_status"]) is int
        and type(row["retry_count"]) is int
        and row["retry_count"] == 0
        and row["cached"] is False
        and row["cache_key_header"].get("present") is False
        and row["cache_key_header"] == {"present": False, "value_sha256": None},
        "incompatible transport/cache",
    )
    return study.parse_response(row["response"], request)


def check_logs(output, package):
    require(canonical(package) == canonical(study.materials()), "unfrozen materials")
    requests, attempts, responses = [rows(output / name) for name in LOGS]
    require(
        0 <= len(responses) <= len(attempts) <= len(requests) <= study.COUNT,
        "log prefix coverage differs",
    )
    require(
        len(requests) - len(attempts) <= 1 and len(attempts) - len(responses) <= 1,
        "too many pending requests",
    )
    expected, parsed, observations = package["requests"], [], []
    previous_end = None
    for index, request in enumerate(requests):
        require(canonical(request) == canonical(expected[index]), "logged request differs")
        base = {k: request[k] for k in ("position", "request_hash")}
        if index >= len(attempts):
            break
        attempt = attempts[index]
        require(
            set(attempt) == {*base, "dispatched_at", "retry_count"}
            and all(type(attempt[k]) is type(v) and attempt[k] == v for k, v in base.items())
            and type(attempt["retry_count"]) is int
            and attempt["retry_count"] == 0,
            "attempt identity differs",
        )
        start = aware_time(attempt["dispatched_at"])
        require(previous_end is None or start >= previous_end, "overlapping requests")
        if index >= len(responses):
            break
        row = responses[index]
        require(
            set(row)
            == {
                *base,
                "requested_at",
                "response_received_at",
                "retry_count",
                "cached",
                "http_status",
                "latency_seconds",
                "cache_key_header",
                "response",
            }
            and all(type(row[k]) is type(v) and row[k] == v for k, v in base.items())
            and row["requested_at"] == attempt["dispatched_at"],
            "response identity differs",
        )
        previous_end = aware_time(row["response_received_at"])
        require(
            previous_end >= start
            and type(row["latency_seconds"]) in (int, float)
            and math.isfinite(row["latency_seconds"])
            and row["latency_seconds"] >= 0,
            "response timing differs",
        )
        try:
            parsed.append({**base, **parsed_response(row, request)})
        except (ValueError, KeyError, TypeError):
            require(index == len(attempts) - 1, "continued after incompatible response")
        observations.append({**base, "signature": signature(row["response"])})
    groups = defaultdict_signatures(observations)
    return {
        "logged_attempts": len(attempts),
        "compatible_responses": len(parsed),
        "complete": len(parsed) == study.COUNT,
        "rows": parsed,
        "repeated_signature_groups": groups,
        "unknown_signature_positions": [
            r["position"] for r in observations if r["signature"] is None
        ],
        "independent_generation_count": None,
    }


def defaultdict_signatures(observations):
    grouped = {}
    for row in observations:
        if row["signature"] is not None:
            grouped.setdefault(digest(row["signature"]), []).append(row["position"])
    return [positions for positions in grouped.values() if len(positions) > 1]


async def collect(client, output, package, secret=""):
    require(canonical(package) == canonical(study.materials()), "unfrozen materials")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*LOGS, "parsed.jsonl"):
        (output / name).touch()
    registry = await client.get(REGISTRY)
    require(registry.status_code == 200, "registry HTTP failure")
    ids = [r["id"] for r in strict_json(registry.text)["data"]]
    save(output / "model_registry.json", {"model_ids": ids})
    require(study.MODEL in ids, "requested model unavailable")
    for request in package["requests"]:
        require(not secret or secret not in canonical(request["payload"]), "secret in payload")
        append(output / LOGS[0], request, secret)
        start = now()
        base = {k: request[k] for k in ("position", "request_hash")}
        append(output / LOGS[1], {**base, "dispatched_at": start, "retry_count": 0})
        save(
            output / "budget_progress.json",
            {
                "used_completion_POSTs": study.PRIOR + request["position"],
                "new_attempts": request["position"],
                "reserved_remaining": study.COUNT - request["position"],
                "cap": 500,
                "no_resume": True,
            },
        )
        begun, status, body, cancelled = time.perf_counter(), "transport_error", None, False
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
            "cache_key_header": header,
            "response": body,
        }
        append(output / LOGS[2], row, secret)
        if cancelled:
            raise asyncio.CancelledError
        append(output / "parsed.jsonl", {**base, **parsed_response(row, request)})
    return check_logs(output, package)


def describe(result):
    lines = [
        "假设的个人买卖历史会怎样改变模型的概率预测？",
        "",
        f"本组模型：{result['requested_model']}；运行编号：{result['run_id']}。",
        "四份完全假设情境、三种历史范围、两种选项顺序，每种问两次，共48次。没有真人答案。",
        "下面的主差：显示该段历史时的偏买/偏卖差异，减去隐藏该段历史时的差异。",
        "正负只描述模型反应，不代表正确或错误。数值单位是概率百分点。",
        "",
    ]
    names = {
        "current_visible_minus_hidden_cue_effect": "本期本人历史",
        "completed_visible_minus_hidden_cue_effect": "上一期本人历史",
    }
    for key, values in result["primary_descriptive_differences"].items():
        lines.append(
            f"{names[key]}：两种显示顺序平均{100 * values['equal_menu_average']:+.2f}；"
            f"买在前{100 * values['buy_first']:+.2f}，卖在前{100 * values['sell_first']:+.2f}。"
        )
    lines += [
        "",
        "各情境原始平均买入概率全部保留在summary.json；全部48个原值、"
        "12项显示顺序差异和14组相同输入的变化范围也保留。",
        "本组只检查当前模型对假设历史线索的反应。历史次数及相应派生数值一起改变，"
        "不能单独归因于某一种文字或数值。两次回答不是两个人，独立生成次数仍未知。",
        "没有计算真人一致率、正确率或市场成绩，不能据此说模型更像真人或市场更真实。",
        "",
    ]
    return "\n".join(lines)


async def execute(folder):
    checked_plan(folder)
    require(folder.resolve() == FOLDER.resolve(), "canonical run directory required")
    approval = read(folder / "independent_preflight.json")
    require(
        approval.get("accepted_for_execution") is True
        and approval.get("mustfix") == []
        and approval.get("plan_sha256") == sha(folder / "plan.json"),
        "execution package not accepted",
    )
    output, diagnostic = folder / "live_logs", folder / "transport_diagnostics"
    marker = folder / "execution_started.json"

    def unused():
        require(
            not any(p.exists() for p in (output, diagnostic, marker)), "never resume a started run"
        )

    unused()
    with shared.exclusive_budget(shared.BUDGET_PATH):
        unused()
        ledger = read(shared.BUDGET_PATH)
        shared.check_budget(ledger, {"prior_POSTs": study.PRIOR})
        require(study.PRIOR + study.COUNT <= 500, "full batch exceeds cap")
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(
                canonical(
                    {
                        "run_id": study.RUN_ID,
                        "plan_sha256": sha(folder / "plan.json"),
                        "started_utc": now(),
                        "no_resume": True,
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
                "reserved_next_completion_POSTs": study.COUNT,
                "available_unreserved_POSTs": 500 - study.PRIOR - study.COUNT,
                "active_run_id": study.RUN_ID,
                "status": "running",
            },
        )
        state = {"status": "running", "new_attempts": 0}
        try:
            from dotenv import dotenv_values

            key = dotenv_values(ROOT / ".env").get("AIGC_API_KEY")
            require(isinstance(key, str) and bool(key), "configured school key missing")
            async with shared.school_client(key, "school_direct") as http:
                require(
                    http._transport_for_url(httpx.URL(COMPLETION)) is http._transport,
                    "school route differs from direct plan",
                )
                client = DiagnosticClient(http, diagnostic, route="default_direct")
                audited = await collect(client, output, read(folder / "materials.json"), key)
            require(audited["complete"], "incomplete response collection")
            checked_plan(folder)
            save(folder / "response_audit.json", audited)
            result = study.summarize(audited["rows"], read(folder / "materials.json"))
            result.update(requested_model=study.MODEL, run_id=study.RUN_ID)
            result["new_model_POSTs"] = audited["logged_attempts"]
            result["repeated_signature_groups"] = audited["repeated_signature_groups"]
            save(folder / "summary.json", result)
            (folder / "REPORT.txt").write_text(describe(result), encoding="utf-8")
            state["status"] = "complete_pending_independent_audit"
            return result
        except BaseException as exc:
            state.update(status="stopped_incomplete", safe_stop_reason=type(exc).__name__)
            for name in ("summary.json", "REPORT.txt"):
                if (folder / name).exists():
                    (folder / name).rename(folder / ("incomplete_" + name))
            raise
        finally:
            sent = len(rows(output / LOGS[1]))
            require(0 <= sent <= study.COUNT, "attempt count exceeded quota")
            state.update(new_attempts=sent, finished_utc=now())
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
            try:
                try:
                    save(folder / "execution_state.json", state)
                finally:
                    save(shared.BUDGET_PATH, terminal)
            except BaseException:
                for name in ("summary.json", "REPORT.txt"):
                    if (folder / name).exists():
                        (folder / name).rename(folder / ("incomplete_" + name))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--folder", type=Path, default=FOLDER)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.folder)
        print(json.dumps({"prepared": True, "planned_POSTs": study.COUNT, "sent": 0}))
    else:
        result = asyncio.run(execute(args.folder))
        print(
            json.dumps(
                {
                    "valid_responses": result["valid_responses"],
                    "new_model_POSTs": result["new_model_POSTs"],
                }
            )
        )


if __name__ == "__main__":
    main()
