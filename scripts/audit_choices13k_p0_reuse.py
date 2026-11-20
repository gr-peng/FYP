"""Offline, bounded P0 calibration audit; no dataset labels or model clients."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from behavioral_market.benchmarks.sampling import orientations
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from scripts.style_fact_check import fingerprint, save

ROOT = "behavioral-llm-market/"
CAL = ROOT + "outputs/human_benchmark/choices13k_deepseek_pilot/calibration/"
NAMES = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
BASE = {
    "run_id",
    "split",
    "prompt_id",
    "item_id",
    "problem_id",
    "repeat_id",
    "display_order",
    "model",
    "temperature",
    "repair",
    "request_hash",
}
META = BASE | {
    "retry_count",
    "dispatched_at",
    "cached",
    "status",
    "requested_at",
    "response_received_at",
    "http_status",
    "trace_id",
    "latency_seconds",
    "retry_after_seconds",
}
SOURCES = {CAL + name for name in (*NAMES, "responses.jsonl")} | {
    ROOT + "data/manifests/choices13k_manifest.json",
    ROOT + "data/processed/human_benchmarks/choices13k/calibration_items.parquet",
    ROOT + "outputs/human_benchmark/choices13k_deepseek_pilot/manifest.json",
    ROOT + "outputs/human_benchmark/choices13k_deepseek_pilot/config_snapshot.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def skip_value(blob, start):
    """Find a JSON value boundary without decoding strings or body objects."""
    if blob[start] == 34:
        i, escape = start + 1, False
        while i < len(blob):
            c = blob[i]
            if escape:
                escape = False
            elif c == 92:
                escape = True
            elif c == 34:
                return i + 1
            i += 1
        raise ValueError("unterminated string")
    if blob[start] in (123, 91):
        stack, i = [125 if blob[start] == 123 else 93], start + 1
        while i < len(blob):
            c = blob[i]
            if c == 34:
                i = skip_value(blob, i)
                continue
            if c in (123, 91):
                require(len(stack) < 256, "body nesting too deep")
                stack.append(125 if c == 123 else 93)
            elif c in (125, 93):
                require(bool(stack) and stack.pop() == c, "mismatched body delimiters")
                if not stack:
                    return i + 1
            i += 1
        raise ValueError("unterminated body")
    i = start
    while i < len(blob) and blob[i] not in b",} \t\r\n":
        i += 1
    require(i > start, "missing scalar")
    return i


def prefix(blob, name):
    """Only decode allowed scalar metadata; confirm body-last before selection."""
    data, i, meta = blob.strip(), 1, {}
    require(data.startswith(b"{"), "JSON object required")
    while i < len(data):
        while data[i] in b" \t\r\n":
            i += 1
        if data[i] == 125:
            require(data[i:] == b"}" and name == NAMES[1], "missing body or trailing content")
            break
        require(data[i] == 34, "metadata key required")
        end = skip_value(data, i)
        key = strict_json(data[i:end].decode())
        require(key not in meta, "duplicate prefix key")
        i = end
        while data[i] in b" \t\r\n":
            i += 1
        require(data[i] == 58, "colon required")
        i += 1
        while data[i] in b" \t\r\n":
            i += 1
        if key in {"payload", "response"}:
            require(
                key == ("payload" if name == NAMES[0] else "response") and name != NAMES[1],
                "wrong body field",
            )
            end = skip_value(data, i)
            require(data[end:].strip() == b"}", "body must be last; no post-body identity")
            break
        require(key in META and data[i] not in (123, 91), "unknown or nonscalar metadata")
        end = skip_value(data, i)
        meta[key] = strict_json(data[i:end].decode())
        i = end
        while data[i] in b" \t\r\n":
            i += 1
        if data[i] == 44:
            i += 1
        else:
            require(data[i] == 125, "comma or object end required")
    require(BASE <= set(meta), "identity incomplete before body")
    require(
        meta["split"] == "calibration" and meta["prompt_id"] in {"P0", "P1", "P2", "P3"},
        "unexpected stage or prompt identity",
    )
    require(type(meta["item_id"]) is int and meta["item_id"] >= 0, "CSV-row identity required")
    return meta


def check_sources(workspace, selection):
    require(set(selection["source_files"]) == SOURCES, "source allowlist differs")
    require(
        len(selection["selected_item_ids"]) in (20, 200)
        and all(type(i) is int and i >= 0 for i in selection["selected_item_ids"])
        and selection["selected_item_ids"] == sorted(set(selection["selected_item_ids"]))
        and selection["projected_columns"] == ["item_id"],
        "fixed20 or full200 ID-only selection required",
    )
    for path, fp in selection["source_files"].items():
        require(fingerprint(workspace / path) == fp, "frozen source bytes differ")
    if len(selection["selected_item_ids"]) == 200:
        import pyarrow.parquet as pq

        path = (
            workspace
            / ROOT
            / "data/processed/human_benchmarks/choices13k/calibration_items.parquet"
        )
        manifest = strict_json(
            (workspace / ROOT / "data/manifests/choices13k_manifest.json").read_text()
        )
        require(
            fingerprint(path)["sha256"] == manifest["splits"]["calibration_sha256"],
            "calibration split fingerprint differs",
        )
        ids = pq.read_table(path, columns=["item_id"]).column("item_id").to_pylist()
        require(
            len(ids) == len(set(ids)) == 200
            and all(type(i) is int and i >= 0 for i in ids)
            and sorted(ids) == selection["selected_item_ids"],
            "full calibration IDs differ",
        )
        require(
            selection.get("expansion_classification") == "coverage_expansion_after_observing20",
            "full coverage must disclose prior20 observation",
        )
        previous = []
        for key in ("prior_inspected_selection", "prior_inspected_result"):
            descriptor = selection[key]
            source = (workspace / descriptor["path"]).resolve()
            require(
                source.is_relative_to((workspace / "iterations").resolve()),
                "prior outside iterations",
            )
            require(fingerprint(source) == descriptor["fingerprint"], "prior snapshot differs")
            if key == "prior_inspected_selection":
                prior = strict_json(source.read_text())
                require(
                    prior["source_files"] == selection["source_files"], "prior data sources differ"
                )
                previous = prior["selected_item_ids"]
        require(
            previous == sorted(ids)[:20]
            and selection["previously_inspected_item_ids"] == previous
            and selection["additional_item_ids"] == sorted(set(ids) - set(previous)),
            "known20/additional180 coverage differs",
        )
    expected = [
        {"item_id": i, "repeat_id": j, "display_order": orientation}
        for i in selection["selected_item_ids"]
        for j, orientation in enumerate(orientations(i, 10, 20260916))
    ]
    require(selection["opportunities"] == expected, "frozen opportunities differ")


def make_index(workspace, selection):
    check_sources(workspace, selection)
    selected = set(selection["selected_item_ids"])
    files = {}
    for name in NAMES:
        entries, skipped, offset = [], 0, 0
        with (workspace / CAL / name).open("rb") as handle:
            for line_number, blob in enumerate(handle, 1):
                meta = prefix(blob, name)
                if meta["prompt_id"] == "P0" and meta["item_id"] in selected:
                    entries.append(
                        {
                            "line_number": line_number,
                            "offset": offset,
                            "length": len(blob),
                            "sha256": hashlib.sha256(blob).hexdigest(),
                            "metadata": meta,
                        }
                    )
                else:
                    skipped += 1
                offset += len(blob)
        files[CAL + name] = {"selected_entries": entries, "bodies_not_decoded": skipped}
    return {
        "selection_sha256": digest(selection),
        "source_files": selection["source_files"],
        "files": files,
        "selected_body_decodes": 0,
        "heldout_opened": False,
        "human_labels_loaded": False,
        "new_model_POSTs": 0,
    }


def decode_selected(blob):
    return strict_json(blob.decode())


def read_index(workspace, selection, index):
    # Rebuild before decoding to reject forged offsets or smuggled out-of-scope rows.
    require(canonical(index) == canonical(make_index(workspace, selection)), "index differs")
    records = {}
    for path, info in index["files"].items():
        rows = []
        with (workspace / path).open("rb") as handle:
            for entry in info["selected_entries"]:
                handle.seek(entry["offset"])
                blob = handle.read(entry["length"])
                require(hashlib.sha256(blob).hexdigest() == entry["sha256"], "selected row differs")
                meta = prefix(blob, Path(path).name)
                require(meta == entry["metadata"], "selected metadata differs")
                require(
                    meta["prompt_id"] == "P0" and meta["item_id"] in selection["selected_item_ids"],
                    "selected row outside scope",
                )
                rows.append({"line_number": entry["line_number"], "record": decode_selected(blob)})
        records[Path(path).name] = rows
    return records


def logical(row):
    return row["item_id"], row["repeat_id"]


def answer(body):
    if not isinstance(body, dict) or body.get("model") != "DeepSeek-V4-Flash":
        return None
    try:
        choices = body["choices"]
        require(len(choices) == 1 and choices[0]["finish_reason"] == "stop", "envelope")
        message = choices[0]["message"]
        require(message["role"] == "assistant", "role")
        value = strict_json(message["content"])
        require(set(value) == {"choice"} and value["choice"] in ("A", "B"), "schema")
        return value["choice"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def response_signature(body):
    if not isinstance(body, dict):
        return None
    identifier, created = body.get("id"), body.get("created")
    if not (
        isinstance(identifier, str) and identifier.strip() and type(created) is int and created >= 0
    ):
        return None
    try:
        text = body["choices"][0]["message"]["content"]
        return [identifier, created, digest(text)] if isinstance(text, str) else None
    except (KeyError, IndexError, TypeError):
        return None


def audit(selection, records):
    by = {name: defaultdict(list) for name in NAMES}
    planned = {
        (x["item_id"], x["repeat_id"]): x["display_order"] for x in selection["opportunities"]
    }
    for name, rows in records.items():
        for entry in rows:
            row = entry["record"]
            require(logical(row) in planned, "record outside frozen opportunities")
            require(type(row["repair"]) is int and row["repair"] in (0, 1), "repair index")
            by[name][logical(row)].append(entry)
    rows, auxiliary = [], []
    for key, order in planned.items():
        issues, payload_hash, terminal = [], None, None
        requested, attempts, responses = (by[n][key] for n in NAMES)
        expected = {
            "run_id": "choices13k_deepseek_pilot",
            "split": "calibration",
            "prompt_id": "P0",
            "item_id": key[0],
            "repeat_id": key[1],
            "display_order": order,
            "model": "DeepSeek-V4-Flash",
            "temperature": 0.7,
        }
        for entry in requested + attempts + responses:
            row = entry["record"]
            if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
                issues.append("identity_mismatch")
        requests_by_hash = defaultdict(list)
        for entry in requested:
            row = entry["record"]
            ident = {k: row[k] for k in BASE - {"request_hash"}}
            calculated = hashlib.sha256(
                json.dumps([row["payload"], ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            if calculated != row["request_hash"]:
                issues.append("request_hash_mismatch")
            requests_by_hash[row["request_hash"]].append(entry)
        for entry in attempts + responses:
            if entry["record"]["request_hash"] not in requests_by_hash:
                issues.append("orphan_attempt_or_response")
        primary = [e for e in requested if e["record"]["repair"] == 0]
        if len(primary) != 1:
            issues.append("missing_or_duplicate_primary_request")
        for hashed, request_entries in requests_by_hash.items():
            request = request_entries[0]["record"]
            local_issues = []
            if len(request_entries) != 1:
                local_issues.append("duplicate_request_record")
            aa = [e["record"] for e in attempts if e["record"]["request_hash"] == hashed]
            rr = [e["record"] for e in responses if e["record"]["request_hash"] == hashed]
            for r in aa + rr:
                if any(r.get(k) != request.get(k) for k in BASE):
                    local_issues.append("transport_identity_mismatch")
            cached = [r for r in rr if r.get("cached") is True]
            live = [r for r in rr if r.get("cached") is False]
            if len(cached) + len(live) != len(rr) or cached and (live or aa or len(cached) != 1):
                local_issues.append("ambiguous_cache_or_live_records")
            if any(r.get("status") != "ok" for r in cached):
                local_issues.append("invalid_local_cache_record")
            for r in live:
                status, latency = r.get("http_status"), r.get("latency_seconds")
                if not (
                    type(status) is int
                    and 100 <= status <= 599
                    or status in ("transport_error", "cancelled_in_flight")
                ):
                    local_issues.append("invalid_transport_status")
                if not (type(latency) in (int, float) and 0 <= latency < float("inf")):
                    local_issues.append("invalid_latency")
                if status != 200 and r.get("response") is not None:
                    local_issues.append("failed_transport_carries_body")
            attempt_map, response_map = defaultdict(list), defaultdict(list)
            for values, mapped in ((aa, attempt_map), (live, response_map)):
                for record in values:
                    retry = record.get("retry_count")
                    if type(retry) is not int or not 0 <= retry <= 5:
                        local_issues.append("invalid_retry_index")
                    else:
                        mapped[retry].append(record)
            indices = sorted(attempt_map)
            if indices != list(range(len(indices))):
                local_issues.append("retry_sequence_gap")
            if set(response_map) != set(attempt_map):
                local_issues.append("missing_or_orphan_transport_return")
            previous = None
            for i in indices:
                if len(attempt_map[i]) != 1 or len(response_map.get(i, [])) != 1:
                    local_issues.append("duplicate_or_missing_attempt_return")
                    continue
                a, r = attempt_map[i][0], response_map[i][0]
                try:
                    start, finish = (
                        aware_time(r["requested_at"]),
                        aware_time(r["response_received_at"]),
                    )
                    require(start == aware_time(a["dispatched_at"]) and finish >= start, "clock")
                    require(previous is None or start >= previous["finish"], "retry clock")
                    if previous is not None:
                        require(previous["returned_body"] is False, "retry after returned body")
                    previous = {
                        "finish": finish,
                        "returned_body": (
                            r.get("http_status") == 200 and isinstance(r.get("response"), dict)
                        ),
                    }
                except (KeyError, TypeError, ValueError):
                    local_issues.append("invalid_transport_sequence")
            candidates = cached + [
                r
                for r in live
                if r.get("http_status") == 200 and isinstance(r.get("response"), dict)
            ]
            if len(candidates) != 1:
                local_issues.append("missing_or_multiple_terminal_bodies")
            body_record = candidates[0] if len(candidates) == 1 else None
            if request["repair"] == 0:
                payload_hash = digest(request["payload"]) if len(primary) == 1 else None
                terminal = body_record if not local_issues and len(primary) == 1 else None
                issues.extend(local_issues)
            else:
                auxiliary.append(
                    {
                        "item_id": key[0],
                        "repeat_id": key[1],
                        "repair": 1,
                        "request_hash": hashed,
                        "attempts": len(aa),
                        "returns": len(rr),
                        "choice": answer(body_record["response"])
                        if body_record and not local_issues and not issues
                        else None,
                        "issues": sorted(set(local_issues)),
                    }
                )
        body = terminal["response"] if terminal and not issues else None
        choice = answer(body)
        if choice and any(e["record"]["repair"] == 1 for e in requested):
            issues.append("repair_after_valid_primary")
            body, choice = None, None
        sig = response_signature(body) if choice else None
        is_cached = terminal["cached"] if body is not None else None
        rows.append(
            {
                "item_id": key[0],
                "repeat_id": key[1],
                "display_order": order,
                "status": "valid_primary_return" if choice else "unresolved_or_invalid_primary",
                "issues": sorted(set(issues)),
                "payload_sha256": payload_hash,
                "choice": choice,
                "local_cache": is_cached,
                "strong_signature": sig,
                "text_sha256": digest(body["choices"][0]["message"]["content"]) if choice else None,
                "body_sha256": digest(body) if body is not None else None,
                "recorded_attempts": len([e for e in attempts if e["record"]["repair"] == 0]),
            }
        )
    payloads, signatures, texts = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        ref = {"item_id": row["item_id"], "repeat_id": row["repeat_id"]}
        if row["payload_sha256"]:
            payloads[row["payload_sha256"]].append(row)
        if row["strong_signature"] is not None and row["local_cache"] is False:
            signatures[digest(row["strong_signature"])].append(ref)
        if row["text_sha256"] is not None:
            texts[row["text_sha256"]].append(ref)
    payload_groups = [
        {
            "payload_sha256": key,
            "opportunities": len(group),
            "items": sorted({r["item_id"] for r in group}),
            "display_orders": sorted({r["display_order"] for r in group}),
            "valid_primary_returns": sum(r["choice"] is not None for r in group),
            "local_replays": sum(r["local_cache"] is True for r in group),
            "observed_noncached_strong_signatures": len(
                {
                    digest(r["strong_signature"])
                    for r in group
                    if r["strong_signature"] is not None and r["local_cache"] is False
                }
            ),
        }
        for key, group in payloads.items()
    ]
    return {
        "rows": rows,
        "payload_groups": payload_groups,
        "repair_requests": auxiliary,
        "summary": {
            "items": len(selection["selected_item_ids"]),
            "planned_opportunities": len(planned),
            "valid_primary_returns": sum(r["choice"] is not None for r in rows),
            "unresolved_primary_opportunities": sum(r["choice"] is None for r in rows),
            "local_replay_opportunities": sum(r["local_cache"] is True for r in rows),
            "recorded_primary_attempts": sum(r["recorded_attempts"] for r in rows),
            "repair_request_groups": len(auxiliary),
            "full_payload_groups": len(payload_groups),
            "repeated_noncached_id_created_text_groups": sum(
                len(g) > 1 for g in signatures.values()
            ),
            "unknown_primary_strong_signatures": sum(r["strong_signature"] is None for r in rows),
        },
        "repeated_noncached_id_created_text_groups": [g for g in signatures.values() if len(g) > 1],
        "text_only_repetition_groups": [g for g in texts.values() if len(g) > 1],
        "orientation_counts": dict(Counter(r["display_order"] for r in rows)),
        "independent_generation_count": None,
        "deduplicated_choice_probability": None,
        "human_labels_loaded": False,
        "heldout_opened": False,
        "new_model_POSTs": 0,
        "limit": f"Only frozen{len(selection['selected_item_ids'])} P0 calibration items. "
        "Binary answer text repetition is expected "
        "and is not cache proof. Unique IDs are not proof of independent generation. "
        "No human-alignment metrics, prompt selection, or old results are recomputed.",
    }


def describe(result):
    s = result["summary"]
    return "\n".join(
        [
            f"# 旧风险选择返回核对（{'完整200' if s['items'] == 200 else '限定20'}题）",
            "",
            (
                "看过前20题后的完整覆盖扩展：已看20题＋新增180题，属于同一批旧记录的核对。"
                if s["items"] == 200
                else "预先固定20题。"
            ),
            f"本次{s['items']}题、{s['planned_opportunities']}个原始机会；"
            f"可唯一关联的有效主返回{s['valid_primary_returns']}，"
            f"未确认或无效{s['unresolved_primary_opportunities']}。",
            f"本地回放机会{s['local_replay_opportunities']}；"
            f"记录到主请求发送尝试{s['recorded_primary_attempts']}次。",
            "",
            f"实际完整输入分为{s['full_payload_groups']}组；"
            f"跨不同逻辑机会的非本地回放“编号、时间、正文”重复组"
            f"{s['repeated_noncached_id_created_text_groups']}个。",
            "",
            "二选一的答案文字相同很常见，单靠相同文字不能判断缓存。缺元数据保留未知。",
            "不估计实际独立生成次数，不计算去重后的选择概率，不读取真人答案或封存测试。",
            (
                "覆盖原P0校准200题，不推广到其他提示、封存测试或其他实验。新增模型请求0。"
                if s["items"] == 200
                else "仅覆盖预选20题，不推广到其余校准题、其他提示或实验。新增模型请求0。"
            ),
            "",
        ]
    )


def load_selection(workspace, plan_path):
    plan = strict_json(plan_path.read_text())
    path = (workspace / plan["selection_file"]).resolve()
    require(
        path.is_relative_to((workspace / "iterations").resolve()), "selection outside iterations"
    )
    require(fingerprint(path) == plan["selection_fingerprint"], "frozen selection differs")
    return strict_json(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "audit"])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = load_selection(args.workspace, args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.action == "index":
        result = make_index(args.workspace, selection)
        save(args.output / "index.json", result)
        print(canonical({"selected_body_decodes": 0, "new_model_POSTs": 0}))
    else:
        index = strict_json(args.index.read_text())
        result = audit(selection, read_index(args.workspace, selection, index))
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(describe(result))
        print(canonical(result["summary"]))


if __name__ == "__main__":
    main()
