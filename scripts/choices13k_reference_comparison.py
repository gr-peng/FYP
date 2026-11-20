"""Prepare and collect one fixed aggregate-reference comparison, at most 16 POSTs."""

import argparse
import asyncio
import hashlib
import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import httpx

from behavioral_market.benchmarks import reference_comparison as study
from behavioral_market.evaluation.choice_presentation import split_user
from behavioral_market.evaluation.style_mechanism_design import canonical
from behavioral_market.evaluation.style_probe_intake import strict_json
from scripts.audit_choices13k_p0_reuse import BASE, prefix
from scripts.style_bank_square import checked_path, verify_budget_sources
from scripts.style_fact_check import fingerprint, read_jsonl, save

ITERATION = "iterations/choices13k_reference_comparison_20261022/"
SOURCE_NAMES = {
    "origin_plan",
    "existing_presentation_packet",
    "source_offsets",
    "raw_requests",
    "frozen_index",
    "human_calibration",
    "prior_ledger",
    "grant",
}
SOURCE_PATHS = {
    "origin_plan": "iterations/presentation_comparison_20261022/next_iteration_plan.json",
    "existing_presentation_packet": "iterations/choices13k_presentation_20261022/"
    "package/packet.json",
    "source_offsets": ITERATION + "source_offsets.json",
    "raw_requests": "behavioral-llm-market/outputs/human_benchmark/"
    "choices13k_deepseek_pilot/calibration/llm_requests.jsonl",
    "frozen_index": "iterations/choices13k_p0_full_calibration_reuse_20261022/index/index.json",
    "human_calibration": "behavioral-llm-market/data/processed/"
    "human_benchmarks/choices13k/calibration_items.parquet",
    "prior_ledger": "iterations/presentation_comparison_20261022/budget_ledger.json",
    "grant": "iterations/style_fact_execution_20261022/user_budget_authorization.json",
}


def checked_plan(workspace):
    plan = strict_json((workspace / ITERATION / "plan.json").read_text())
    study.require(set(plan["sources"]) == SOURCE_NAMES, "fixed source allowlist")
    study.require(
        {name: desc["path"] for name, desc in plan["sources"].items()} == SOURCE_PATHS,
        "fixed source paths",
    )
    for descriptor in plan["sources"].values():
        checked_path(workspace, descriptor)
    study.require(
        plan["fixed_target_item_id"] == study.TARGET
        and plan["fixed_reference_item_ids"] == list(study.REFERENCES)
        and plan["human_read"]
        == {
            "columns": ["item_id", "human_b_rate"],
            "filter_item_ids": [166, 211, 384, 542],
            "rows_expected": 4,
        }
        and plan["prior_POSTs"] == 75
        and plan["maximum_new_POSTs"] == 16
        and plan["batch_cumulative_stop"] == 91
        and plan["global_cap"] == 100
        and plan["endpoint"] == study.ENDPOINT
        and plan["model"] == study.MODEL
        and plan["fixed_execution_output"]
        == str((workspace / ITERATION / "live_execution").resolve())
        and canonical(plan["controls"])
        == canonical({k: v for k, v in study.CONTROLS.items() if k != "model"}),
        "fixed protocol differs",
    )
    origin = strict_json(checked_path(workspace, plan["sources"]["origin_plan"]).read_text())
    study.require(
        origin["target_item_id"] == 384
        and origin["reference_item_ids"] == [166, 211, 542]
        and {method: origin["reference_pairing"][method] for method in study.METHODS}
        == plan["reference_pairing"],
        "accepted direction differs",
    )
    ledger = strict_json(checked_path(workspace, plan["sources"]["prior_ledger"]).read_text())
    study.require(
        ledger["used_completion_POSTs"] == 75
        and ledger["reserved_next_completion_POSTs"] == 0
        and ledger["active_run_id"] is None
        and ledger["global_authorized_completion_POST_cap"] == 100,
        "budget unavailable",
    )
    verify_budget_sources(
        workspace,
        {
            "budget_sources": {k: plan["sources"][k] for k in ("grant", "prior_ledger")},
            "endpoint": study.ENDPOINT,
            "model": study.MODEL,
            "prior_POSTs": 75,
            "batch_cumulative_stop": 91,
        },
    )
    return plan


def _options(pair):
    study.require(set(pair) == {"A_first", "B_first"}, "both source orientations")
    parsed = {}
    for order, payload in pair.items():
        study.require(
            set(payload)
            == {
                "model",
                "messages",
                "temperature",
                "max_tokens",
                "enable_thinking",
                "response_format",
            },
            "legacy payload schema",
        )
        expected = {
            k: v for k, v in study.CONTROLS.items() if k not in {"chat_template_kwargs", "cache"}
        }
        expected["messages"] = deepcopy(payload["messages"])
        study.require(canonical(payload) == canonical(expected), "legacy settings")
        messages = payload["messages"]
        study.require(
            isinstance(messages, list)
            and len(messages) == 2
            and messages[0] == {"role": "system", "content": study.panel_tool.SYSTEM}
            and set(messages[1]) == {"role", "content"}
            and messages[1]["role"] == "user",
            "legacy messages",
        )
        parsed[order] = split_user(messages[1]["content"])
    a, b = parsed["A_first"], parsed["B_first"]
    study.require(
        a == (b[0], b[2], b[1]) and a[0] == "Choose the option you prefer.", "source reversal"
    )
    return {"option_a_text": a[1], "option_b_text": a[2]}


def read_sources(workspace, plan):
    """Reconstruct from the same four frozen request rows and two columns/four IDs."""
    sources = plan["sources"]
    offsets = strict_json(checked_path(workspace, sources["source_offsets"]).read_text())
    index = strict_json(checked_path(workspace, sources["frozen_index"]).read_text())
    raw = sources["raw_requests"]["path"]
    study.require(
        offsets["raw_request_file"] == sources["raw_requests"]
        and offsets["frozen_index"] == sources["frozen_index"]
        and len(offsets["entries"]) == 4,
        "offset provenance",
    )
    old = strict_json(checked_path(workspace, sources["existing_presentation_packet"]).read_text())
    pairs = {i: old["bases"][str(i)] for i in (166, 211)}
    pairs.update({384: {}, 542: {}})
    wanted = [(i, order) for i in (384, 542) for order in ("A_first", "B_first")]
    with checked_path(workspace, sources["raw_requests"]).open("rb") as handle:
        for (item, order), entry in zip(wanted, offsets["entries"], strict=True):
            candidates = [
                e
                for e in index["files"][raw]["selected_entries"]
                if e["metadata"]["item_id"] == item
                and e["metadata"]["display_order"] == order
                and e["metadata"]["repair"] == 0
                and e["metadata"]["prompt_id"] == "P0"
            ]
            study.require(
                candidates and entry == min(candidates, key=lambda e: e["metadata"]["repeat_id"]),
                "source offset selection",
            )
            handle.seek(entry["offset"])
            blob = handle.read(entry["length"])
            study.require(hashlib.sha256(blob).hexdigest() == entry["sha256"], "source row bytes")
            meta = prefix(blob, "llm_requests.jsonl")
            study.require(
                meta == entry["metadata"]
                and meta["run_id"] == "choices13k_deepseek_pilot"
                and meta["split"] == "calibration",
                "source identity before decode",
            )
            row = strict_json(blob.decode())
            study.require(set(row) == BASE | {"payload"}, "source request schema")
            ident = {k: row[k] for k in BASE - {"request_hash"}}
            original_hash = hashlib.sha256(
                json.dumps([row["payload"], ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            study.require(original_hash == row["request_hash"], "source request hash")
            pairs[item][order] = row["payload"]
    # The only table projection is declared before opening the table.
    import pyarrow.parquet as pq

    spec = plan["human_read"]
    table = pq.read_table(
        checked_path(workspace, sources["human_calibration"]),
        columns=spec["columns"],
        filters=[("item_id", "in", spec["filter_item_ids"])],
    )
    study.require(table.column_names == spec["columns"], "unexpected projected columns")
    rate_rows = table.to_pylist()
    study.require(
        len(rate_rows) == 4
        and all(type(r["item_id"]) is int for r in rate_rows)
        and sorted(r["item_id"] for r in rate_rows) == spec["filter_item_ids"],
        "four human rows",
    )
    rates = {r["item_id"]: r["human_b_rate"] for r in rate_rows}
    text = {i: {"item_id": i, **_options(pair)} for i, pair in pairs.items()}
    materials = {
        "target": text[384],
        "references": [{**text[i], "human_b_rate": rates[i]} for i in study.REFERENCES],
    }
    targets = {384: rates[384]}
    checked_plan(workspace)  # Reject changed source bytes after bounded reads.
    return materials, targets


def prepare(workspace, output):
    plan = checked_plan(workspace)
    study.require(
        output.resolve() == (workspace / ITERATION / "package").resolve(), "fixed package path"
    )
    study.require(not output.exists(), "package cannot be overwritten")
    materials, targets = read_sources(workspace, plan)
    packet = study.make_packet(materials, plan["frozen_run_id"], "recorded_model_logs")
    pending = study.evaluate(packet, [], [], [], targets)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "source_materials.json", materials)
    save(output / "local_target.json", {"item_id": 384, "human_b_rate": targets[384]})
    save(output / "packet.json", packet)
    (output / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in packet["requests"])
    )
    save(output / "pending_evaluation.json", pending)
    save(
        output / "source_read_receipt.json",
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "selected_request_body_decodes": 4,
            "selected_request_keys": [
                (i, order) for i in (384, 542) for order in ("A_first", "B_first")
            ],
            "projected_columns": plan["human_read"]["columns"],
            "human_item_ids": plan["human_read"]["filter_item_ids"],
            "human_rows_materialized": 4,
            "old_response_bodies_decoded": 0,
            "heldout_opened": False,
            "source_plan": {
                "path": ITERATION + "plan.json",
                "fingerprint": fingerprint(workspace / ITERATION / "plan.json"),
            },
            "sources": plan["sources"],
            "derived_files": {
                name: fingerprint(output / name)
                for name in (
                    "source_materials.json",
                    "local_target.json",
                    "packet.json",
                    "expected_requests.jsonl",
                )
            },
            "target_rate_not_in_packet_builder": True,
            "new_model_POSTs": 0,
        },
    )
    return {"prepared_not_sent": True, "planned_POSTs": 16, "used_POSTs": 75}


def load_packet(workspace, package):
    study.require(
        package.resolve() == (workspace / ITERATION / "package").resolve(), "fixed package"
    )
    plan = checked_plan(workspace)
    receipt = strict_json((package / "source_read_receipt.json").read_text())
    study.require(receipt["sources"] == plan["sources"], "source receipt provenance")
    study.require(
        receipt["source_plan"]["fingerprint"] == fingerprint(workspace / ITERATION / "plan.json"),
        "frozen source plan",
    )
    study.require(
        set(receipt["derived_files"])
        == {"source_materials.json", "local_target.json", "packet.json", "expected_requests.jsonl"},
        "fixed derived file names",
    )
    for name, expected in receipt["derived_files"].items():
        study.require(fingerprint(package / name) == expected, "derived source fingerprint")
    materials = strict_json((package / "source_materials.json").read_text())
    source_materials, source_targets = read_sources(workspace, plan)
    study.require(
        canonical(materials) == canonical(source_materials),
        "materials differ from original sources",
    )
    packet = strict_json((package / "packet.json").read_text())
    rebuilt = study.make_packet(materials, plan["frozen_run_id"], "recorded_model_logs")
    study.require(
        canonical(packet) == canonical(rebuilt), "packet not reconstructed from source materials"
    )
    study.require(
        read_jsonl(package / "expected_requests.jsonl") == packet["requests"], "expected wire"
    )
    target = strict_json((package / "local_target.json").read_text())
    study.require(
        set(target) == {"item_id", "human_b_rate"}
        and type(target["item_id"]) is int
        and target["item_id"] == 384,
        "local target schema",
    )
    study.require(
        canonical({"item_id": 384, "human_b_rate": source_targets[384]}) == canonical(target),
        "local target differs from original source",
    )
    return packet, {384: target["human_b_rate"]}


def describe(result):
    comp = result["comparison"]
    lines = [
        "# 单题人群参考信息对照",
        "",
        f"计划16次，实际发送{result['logged_dispatch_attempts']}次；累计{result['used_completion_POSTs']}/100次。",
        "",
        "这是真实返回的单题原型检查，不证明模型整体更像真人。",
        "",
        "| 参考方式 | 原题顺序 | 显示字母顺序 | 两次选择对应的原方案 |",
        "|---|---|---|---|",
    ]
    labels = {study.METHODS[0]: "正确配对", study.METHODS[1]: "打乱配对"}
    for method in comp["item_methods"]:
        for cell in method["cells"]:
            lines.append(
                f"| {labels[method['method_id']]} | {cell['content_order']} | "
                f"{cell['letter_order']} | {cell['original_choices']} |"
            )
    lines.extend(
        [
            "",
            f"本地核对的人群原B选择比例：{comp['item_pairs'][0]['human_b_rate']}。该值没有给模型看。",
            "",
        ]
    )
    for method in comp["item_methods"]:
        lines.append(
            f"{labels[method['method_id']]}：原B比例{method['pooled_original_B_rate']}，与人群比例差距{method['absolute_error']}。"
        )
    lines.extend(
        [
            "",
            f"正确配对减打乱配对的误差差值：{comp['primary']['left_minus_right_MAE']}；负数表示本次正确配对更接近。",
            "",
            "不足16份有效回答时，完整主比较保持未知。相同输入的相同或相反回答均保留，不补发，也不声明独立生成。",
            "",
            "两组只改变三道参考题与人群比例的对应关系。使用的是群体参考资料，不是个人记忆学习。一道题不能支持总体改进、原因解释或市场真实性结论。",
            "",
        ]
    )
    return "\n".join(lines)


async def collect(packet, output, client, targets, grant_source, secret=""):
    study.validate_packet(packet)
    output.mkdir(parents=True, exist_ok=False)
    for name in study.LOGS:
        (output / name).touch()
    save(output / "packet.json", packet)
    state = {"status": "running", "stop_reason": None, "run_id": packet["run_id"]}

    def log(name, row):
        text = canonical(row)
        if secret:
            text = text.replace(secret, "[REDACTED]")
        with (output / name).open("a") as handle:
            handle.write(text + "\n")

    def scored():
        return study.evaluate(packet, *(read_jsonl(output / n) for n in study.LOGS), targets)

    def ledger(running):
        sent = len(read_jsonl(output / study.LOGS[1]))
        reserved = 16 - sent if running else 0
        save(
            output / "budget_ledger.json",
            {
                "global_authorization_source": str(grant_source),
                "global_authorized_completion_POST_cap": 100,
                "used_completion_POSTs": 75 + sent,
                "reserved_next_completion_POSTs": reserved,
                "available_unreserved_POSTs": 25 - sent - reserved,
                "active_run_id": state["run_id"] if running else None,
                "new_logged_dispatches": sent,
                "status": "running" if running else "terminal_pending_audit",
                "no_resume": True,
            },
        )

    save(output / "execution_state.json", state)
    ledger(True)
    try:
        registry = await client.get("https://aigc-api.hkust-gz.edu.cn/api/v2/models")
        study.require(registry.status_code == 200, "registry HTTP")
        ids = [r["id"] for r in strict_json(registry.text)["data"]]
        study.require(study.MODEL in ids, "model absent")
        save(output / "model_registry.json", {"model_ids": ids})
        for request in packet["requests"]:
            before = scored()
            study.require(
                before["logged_dispatch_attempts"] == request["position"] - 1, "dispatch prefix"
            )
            study.require(
                all(r["compatible"] for r in before["rows"][: request["position"] - 1]),
                "prior invalid",
            )
            study.require(
                all(
                    r["status"] == "unsubmitted" for r in before["rows"][request["position"] - 1 :]
                ),
                "unexpected pending",
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
                cache_key = response.headers.get("x-litellm-cache-key")
                header = {
                    "present": cache_key is not None,
                    "value_sha256": hashlib.sha256(cache_key.encode()).hexdigest()
                    if cache_key is not None
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
            result = scored()
            save(output / "evaluation.json", result)
            if not result["rows"][request["position"] - 1]["compatible"]:
                state.update(status="stopped", stop_reason="protocol failure")
                break
            print(
                canonical(
                    {"completed_slots": request["position"], "used_POSTs": 75 + request["position"]}
                ),
                flush=True,
            )
        else:
            state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="cancelled in flight; no resume")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save(output / "execution_state.json", state)
        ledger(False)
    result = scored()
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(describe(result))
    return result


async def execute(workspace, package):
    freeze = strict_json((workspace / ITERATION / "pre_execution_freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        study.require(
            fingerprint(workspace / relative) == expected, "preflight frozen file changed"
        )
    review = strict_json((workspace / ITERATION / "pre_execution_review.json").read_text())
    study.require(review["status"] == "accepted", "independent preflight required")
    packet, targets = load_packet(workspace, package)
    study.require(packet["evidence_kind"] == "recorded_model_logs", "synthetic cannot execute live")
    study.require(
        review["packet_fingerprint"] == fingerprint(package / "packet.json"), "reviewed packet"
    )
    plan = checked_plan(workspace)
    output = Path(plan["fixed_execution_output"])
    study.require(not output.exists(), "no retry or resume")
    from dotenv import dotenv_values

    key = dotenv_values(workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    study.require(bool(key), "configured school credential missing")
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + key},
        timeout=120,
        trust_env=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as client:
        return await collect(
            packet,
            output,
            client,
            targets,
            checked_path(workspace, plan["sources"]["grant"]),
            secret=key,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "execute", "evaluate"])
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    package = workspace / ITERATION / "package"
    if args.action == "prepare":
        result = prepare(workspace, package)
    elif args.action == "execute":
        result = asyncio.run(execute(workspace, package))
        result = {
            "protocol_complete": result["protocol_complete"],
            "used_POSTs": result["used_completion_POSTs"],
        }
    else:
        packet, targets = load_packet(workspace, package)
        output = workspace / ITERATION / "live_execution"
        result = study.evaluate(packet, *(read_jsonl(output / n) for n in study.LOGS), targets)
        save(output / "evaluation.json", result)
        (output / "REPORT.md").write_text(describe(result))
        result = {
            "protocol_complete": result["protocol_complete"],
            "used_POSTs": result["used_completion_POSTs"],
        }
    print(canonical(result))


if __name__ == "__main__":
    main()
