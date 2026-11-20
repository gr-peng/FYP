"""Prepare and execute sixteen fixed hypothetical-choice presentation requests."""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from behavioral_market.evaluation import choice_presentation as study
from behavioral_market.evaluation.style_mechanism_design import canonical
from behavioral_market.evaluation.style_probe_intake import strict_json
from scripts.audit_choices13k_p0_reuse import BASE, prefix
from scripts.style_bank_square import checked_path, verify_budget_sources
from scripts.style_fact_check import fingerprint, read_jsonl, save

ITERATION = "iterations/choices13k_presentation_20261022/"
ORIGIN = "iterations/choices13k_p0_human_comparison_20261022/next_iteration_plan.json"
RAW = (
    "behavioral-llm-market/outputs/human_benchmark/choices13k_deepseek_pilot/"
    "calibration/llm_requests.jsonl"
)


def packet_for(workspace, plan):
    study.require(plan["origin_plan"]["path"] == ORIGIN, "fixed origin required")
    study.require(
        plan["source_offsets"]["path"] == ITERATION + "source_offsets.json",
        "fixed offsets file required",
    )
    origin = strict_json(checked_path(workspace, plan["origin_plan"]).read_text())
    offsets = strict_json(checked_path(workspace, plan["source_offsets"]).read_text())
    study.require(plan["schedule"] == origin["schedule"] == study.SCHEDULE, "schedule differs")
    study.require(
        plan["budget_sources"]
        == {
            "grant": origin["authorization_source"],
            "prior_ledger": origin["latest_accepted_ledger"],
        },
        "budget sources differ",
    )
    study.require(
        plan["fixed_execution_output"] == str((workspace / ITERATION / "live_execution").resolve()),
        "fixed output required",
    )
    scope = origin["material_source_scope"]
    study.require(scope["raw_request_file"]["path"] == RAW, "only fixed raw request source")
    study.require(
        offsets["raw_request_file"] == scope["raw_request_file"]
        and offsets["frozen_index"] == scope["frozen_index"]
        and offsets["allowed_request_payload_bodies"] == origin["allowed_request_payload_bodies"]
        and len(offsets["entries"]) == 4,
        "four fixed offsets required",
    )
    for key in ("accepted_selection", "original_prompt_source"):
        checked_path(workspace, scope[key])
    selected = strict_json(checked_path(workspace, scope["accepted_selection"]).read_text())
    study.require(selected["selected_item_ids"][:2] == [166, 211], "lowest two IDs differ")
    index = strict_json(checked_path(workspace, scope["frozen_index"]).read_text())
    raw_path = checked_path(workspace, scope["raw_request_file"])
    bases = {"166": {}, "211": {}}
    with raw_path.open("rb") as handle:
        for wanted, entry in zip(
            origin["allowed_request_payload_bodies"], offsets["entries"], strict=True
        ):
            candidates = [
                e
                for e in index["files"][RAW]["selected_entries"]
                if all(e["metadata"].get(k) == v for k, v in wanted.items())
                and e["metadata"]["repair"] == 0
                and e["metadata"]["prompt_id"] == "P0"
            ]
            study.require(
                len(candidates) == 1 and canonical(candidates[0]) == canonical(entry),
                "selected offset differs from frozen index",
            )
            handle.seek(entry["offset"])
            blob = handle.read(entry["length"])
            study.require(hashlib.sha256(blob).hexdigest() == entry["sha256"], "row bytes differ")
            meta = prefix(blob, "llm_requests.jsonl")
            study.require(
                meta == entry["metadata"]
                and meta["run_id"] == "choices13k_deepseek_pilot"
                and meta["split"] == "calibration"
                and meta["prompt_id"] == "P0"
                and type(meta["repair"]) is int
                and meta["repair"] == 0,
                "selected identity differs before body decode",
            )
            row = strict_json(blob.decode())
            study.require(set(row) == BASE | {"payload"}, "old request fields differ")
            ident = {k: row[k] for k in BASE - {"request_hash"}}
            legacy_hash = hashlib.sha256(
                json.dumps([row["payload"], ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            study.require(legacy_hash == row["request_hash"], "legacy request hash differs")
            bases[str(row["item_id"])][row["display_order"]] = row["payload"]
    study.require(
        fingerprint(raw_path) == scope["raw_request_file"]["fingerprint"], "source changed"
    )
    result = study.packet(plan, bases)
    verify_budget_sources(workspace, result)
    return result


def checked_packet(workspace, package):
    study.require(
        canonical(package) == canonical(packet_for(workspace, package["plan"])),
        "frozen package differs from sources",
    )


def scored(package, output):
    return study.evaluate(package, *(read_jsonl(output / n) for n in study.LOGS))


def describe(result):
    lines = [
        "# 两题字母与位置检查",
        "",
        f"记录到{result['logged_dispatch_attempts']}次发送尝试，有效{result['valid_returns']}条。",
        "",
        "| 题号 | 从上到下的原选项 | 从上到下的标签 | 两次显示选择 | 两次原选项 | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["cells"]:
        lines.append(
            f"| {row['item_id']} | {row['content_order']} | {row['letter_order']} | "
            f"{row['display_choices']} | {row['original_choices']} | {row['valid_returns']}/2 |"
        )
    lines += ["", "| 题号 | 比较 | 固定条件 | 选择原B比例差 |", "|---|---|---|---|"]
    for edge in result["contrasts"]:
        position = edge["factor"] == "position_first_minus_second"
        name = "原B放前面减放后面" if position else "原B标B减标A"
        fixed = (
            "原B字母=" + edge["fixed_original_B_letter"]
            if position
            else "原B位置=" + str(edge["fixed_original_B_position"])
        )
        value = (
            f"{edge['difference'] * 100:+.0f}个百分点" if edge["difference"] is not None else "未知"
        )
        lines.append(f"| {edge['item_id']} | {name} | {fixed} | {value} |")
    lines += [
        "",
        "编号、时间、正文重复组：" + canonical(result["repeated_id_created_text_groups"]),
        "每格两个收到的答案都保留；任一端缺失，比较保留未知。重复不补发，也不当独立样本。",
        "两道题均曾查看，本轮使用新的模式与缓存设置；不证明控制生效、人的偏好或市场真实性。",
        "",
        "这里把字母与位置分别变化，仅描述这两题的返回；不作显著性、一般因果或真人相似度结论。",
        "",
    ]
    return "\n".join(lines)


async def collect(package, workspace, client, secret=""):
    checked_packet(workspace, package)
    output = Path(package["plan"]["fixed_execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    for name in study.LOGS:
        (output / name).touch()
    save(output / "packet.json", package)
    state = {"status": "running", "stop_reason": None, "run_id": package["plan"]["frozen_run_id"]}

    def log(name, row):
        text = canonical(row)
        if secret:
            text = text.replace(secret, "[REDACTED]")
        with (output / name).open("a") as handle:
            handle.write(text + "\n")

    def ledger(running):
        attempts = len(read_jsonl(output / study.LOGS[1]))
        reserved = 16 - attempts if running else 0
        save(
            output / "budget_ledger.json",
            {
                "global_authorization_source": str(
                    (workspace / package["budget_sources"]["grant"]["path"]).resolve()
                ),
                "global_authorized_completion_POST_cap": 100,
                "used_completion_POSTs": 59 + attempts,
                "reserved_next_completion_POSTs": reserved,
                "available_unreserved_POSTs": 41 - attempts - reserved,
                "active_run_id": state["run_id"] if running else None,
                "new_logged_dispatches": attempts,
                "status": "running" if running else "terminal_pending_audit",
                "no_resume": True,
            },
        )

    save(output / "execution_state.json", state)
    ledger(True)
    try:
        response = await client.get("https://aigc-api.hkust-gz.edu.cn/api/v2/models")
        study.require(response.status_code == 200, "registry HTTP failure")
        ids = [x["id"] for x in strict_json(response.text)["data"]]
        study.require(study.MODEL in ids, "model absent")
        save(output / "model_registry.json", {"model_ids": ids})
        for request in package["requests"]:
            # Reconstruct the whole logged prefix before recording the next dispatch.
            before = scored(package, output)
            study.require(
                before["logged_dispatch_attempts"] == request["position"] - 1,
                "dispatch prefix differs",
            )
            study.require(
                all(r["compatible"] for r in before["rows"][: request["position"] - 1]),
                "prior response became invalid",
            )
            study.require(
                all(
                    r["status"] == "unsubmitted" for r in before["rows"][request["position"] - 1 :]
                ),
                "unexpected pending request before next dispatch",
            )
            log(study.LOGS[0], request)
            base, start = study.identity(request), datetime.now(UTC).isoformat()
            log(study.LOGS[1], {**base, "dispatched_at": start, "retry_count": 0})
            ledger(True)
            begun, http, body, cancelled = time.perf_counter(), "transport_error", None, False
            header = {"present": False, "value_sha256": None}
            try:
                response = await client.post(study.ENDPOINT, json=request["payload"])
                http = response.status_code
                key = response.headers.get("x-litellm-cache-key")
                header = {
                    "present": key is not None,
                    "value_sha256": hashlib.sha256(key.encode()).hexdigest()
                    if key is not None
                    else None,
                }
                if http == 200:
                    try:
                        body = strict_json(response.text)
                    except ValueError:
                        body = {"_invalid_json_text": response.text}
            except asyncio.CancelledError:
                http, cancelled = "cancelled_in_flight", True
            except (httpx.HTTPError, ValueError):
                pass
            log(
                study.LOGS[2],
                {
                    **base,
                    "requested_at": start,
                    "response_received_at": datetime.now(UTC).isoformat(),
                    "cached": False,
                    "http_status": http,
                    "retry_count": 0,
                    "latency_seconds": time.perf_counter() - begun,
                    "response": body,
                    "cache_key_header": header,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            result = scored(package, output)
            save(output / "evaluation.json", result)
            if not result["rows"][request["position"] - 1]["compatible"]:
                state.update(status="stopped", stop_reason="protocol failure")
                break
            print(
                canonical(
                    {"completed_slots": request["position"], "used_POSTs": 59 + request["position"]}
                ),
                flush=True,
            )
        else:
            state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="in-flight accounting uncertain;never resume")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save(output / "execution_state.json", state)
        ledger(False)
    result = scored(package, output)
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(describe(result))
    return result


def prepare(args):
    package = packet_for(args.workspace, strict_json(args.plan.read_text()))
    study.require(not Path(package["plan"]["fixed_execution_output"]).exists(), "never resume")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "packet.json", package)
    (args.output / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in package["requests"])
    )
    save(args.output / "pending_evaluation.json", study.evaluate(package, [], [], []))
    return {"prepared_not_sent": True, "maximum_new_POSTs": 16}


def load_packet(workspace, folder):
    package = strict_json((folder / "packet.json").read_text())
    checked_packet(workspace, package)
    study.require(
        canonical(read_jsonl(folder / "expected_requests.jsonl")) == canonical(package["requests"]),
        "expected wire requests differ",
    )
    return package


async def execute(args):
    package = load_packet(args.workspace, args.package)
    study.require(
        package["plan"]["evidence_kind"] == "recorded_model_logs", "synthetic cannot execute live"
    )
    study.require(not Path(package["plan"]["fixed_execution_output"]).exists(), "never resume")
    from dotenv import dotenv_values

    key = dotenv_values(args.workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    study.require(bool(key), "configured school credential missing")
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        timeout=120,
        trust_env=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as client:
        return await collect(package, args.workspace, client, secret=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "execute", "evaluate"])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "execute":
        result = asyncio.run(execute(args))
    else:
        package = load_packet(args.workspace, args.package)
        result = scored(package, Path(package["plan"]["fixed_execution_output"]))
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(describe(result))
    print(canonical({k: v for k, v in result.items() if k not in {"rows", "cells", "contrasts"}}))


if __name__ == "__main__":
    main()
