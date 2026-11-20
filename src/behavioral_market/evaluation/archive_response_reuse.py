"""Describe archived request/response lineage without inferring independent generations."""

from collections import Counter, defaultdict

from behavioral_market.evaluation.style_probe_intake import parse_choice, strict_json
from behavioral_market.evaluation.style_probe_materials import payload_digest

ORIGIN_FIELDS = (
    "kind",
    "agent_id",
    "session_date",
    "treatment",
    "mode",
    "seed",
    "shock",
    "include_secondary",
    "repair",
)


def text_value(value):
    return value if isinstance(value, str) and value.strip() else None


def index_requests(batches):
    index = defaultdict(list)
    total = 0
    for batch in batches:
        for row in batch["requests"]:
            total += 1
            request_hash = text_value(row.get("request_hash"))
            if request_hash is not None:
                index[batch["run_id"], request_hash].append(row)
    links = {}
    for key, rows in index.items():
        digests = {payload_digest(r["payload"]) for r in rows if isinstance(r.get("payload"), dict)}
        complete = all(isinstance(r.get("payload"), dict) for r in rows)
        origins = {payload_digest({k: r.get(k) for k in ORIGIN_FIELDS}) for r in rows}
        known = complete and len(digests) == 1
        links[key] = {
            "request_rows": len(rows),
            "payload_sha256": next(iter(digests)) if known else None,
            "payload_relation": "unique_payload" if known else "ambiguous_or_missing_payload",
            "origin_consistent": len(origins) == 1,
            "request": rows[0] if known and len(origins) == 1 else None,
        }
    return links, total


def normalize_response(run_id, line, row, link):
    body = row.get("response")
    http200 = type(row.get("http_status")) is int and row["http_status"] == 200
    body = body if (http200 or row.get("cached") is True) and isinstance(body, dict) else None
    choice = None
    if body is not None:
        choices = body.get("choices")
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            choice = choices[0]
    message = choice.get("message") if choice else None
    text = message.get("content") if isinstance(message, dict) else None
    text = text if isinstance(text, str) else None
    origin_matches = None
    if link and link["request"] is not None:
        origin_matches = all(row.get(k) == link["request"].get(k) for k in ORIGIN_FIELDS)
    cached = row.get("cached")
    cache_state = (
        "local_cache" if cached is True else "nonlocal_cache" if cached is False else "unknown"
    )
    created = body.get("created") if body else None
    core = (
        None
        if body is None
        else {
            "id": body.get("id"),
            "created": created,
            "model": body.get("model"),
            "finish_reason": choice.get("finish_reason") if choice else None,
            "role": message.get("role") if isinstance(message, dict) else None,
            "content": text,
        }
    )
    return {
        "run_id": run_id,
        "response_line": line,
        "kind": row.get("kind"),
        "agent_id": row.get("agent_id"),
        "session_date": row.get("session_date"),
        "request_hash": text_value(row.get("request_hash")),
        "retry_index": row.get("retry_count") if type(row.get("retry_count")) is int else None,
        "repair_index": row.get("repair") if type(row.get("repair")) is int else None,
        "http_status": row.get("http_status"),
        "http200": http200,
        "cache_state": cache_state,
        "declared_noncached_http200": http200 and cached is False,
        "response_id": text_value(body.get("id")) if body else None,
        "created": created if type(created) is int and created >= 0 else None,
        "answer_text_sha256": payload_digest(text) if text is not None else None,
        "core_envelope_sha256": payload_digest(core) if core is not None else None,
        "full_body_sha256": payload_digest(body) if body is not None else None,
        "payload_sha256": link["payload_sha256"] if link and origin_matches is True else None,
        "payload_link_status": (
            "origin_mismatch_or_unknown"
            if link and origin_matches is not True
            else link["payload_relation"]
            if link
            else "missing_request"
        ),
        "request_log_rows": link["request_rows"] if link else 0,
        "request_origin_matches": origin_matches,
    }


def summarize_repeated_ids(records):
    groups = defaultdict(list)
    for row in records:
        if row["response_id"] is not None:
            groups[row["response_id"]].append(row)
    result = []
    for rid, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        new = [row for row in rows if row["declared_noncached_http200"]]
        known = {row["payload_sha256"] for row in rows if row["payload_sha256"] is not None}
        complete = all(row["payload_sha256"] is not None for row in rows)
        result.append(
            {
                "response_id": rid,
                "response_rows": len(rows),
                "logical_requests": len({(row["run_id"], row["request_hash"]) for row in rows}),
                "declared_noncached_http200_rows": len(new),
                "separate_noncached_logical_requests": len(
                    {(row["run_id"], row["request_hash"]) for row in new}
                ),
                "local_cache_rows": sum(row["cache_state"] == "local_cache" for row in rows),
                "unknown_cache_rows": sum(row["cache_state"] == "unknown" for row in rows),
                "known_distinct_payloads": len(known),
                "payload_relationship": "unknown"
                if not complete
                else "same"
                if len(known) == 1
                else "different",
                "distinct_nonmissing_answer_texts": len(
                    {
                        row["answer_text_sha256"]
                        for row in rows
                        if row["answer_text_sha256"] is not None
                    }
                ),
                "distinct_core_envelopes": len(
                    {
                        row["core_envelope_sha256"]
                        for row in rows
                        if row["core_envelope_sha256"] is not None
                    }
                ),
                "distinct_full_bodies": len(
                    {row["full_body_sha256"] for row in rows if row["full_body_sha256"] is not None}
                ),
                "record_refs": [
                    {"run_id": row["run_id"], "response_line": row["response_line"]} for row in rows
                ],
            }
        )
    return result


def mapping_status(review, links, response_index, record_index):
    if review.get("authority") == "programmed_rule":
        return "excluded_programmed_rule", None
    if review.get("authority") != "archived_llm":
        return "unknown_authority", None
    if review.get("valid_choice") is not True:
        return "not_previously_valid_ai_choice", None
    if text_value(review.get("raw_answer_request_hash")) is None:
        return "missing_answer_request_hash", None
    key = review.get("run_id"), text_value(review.get("raw_answer_request_hash"))
    link = links.get(key)
    if not link or link["request"] is None:
        return "missing_or_ambiguous_request", None
    request = link["request"]
    if (request.get("kind"), request.get("agent_id"), request.get("session_date")) != (
        "style",
        review.get("agent_id"),
        review.get("review_date"),
    ):
        return "request_identity_mismatch", None
    try:
        user = next(m["content"] for m in request["payload"]["messages"] if m["role"] == "user")
        context = strict_json(user)
        state, observation = context["agent_state"], context["observation"]
        if (state["current_style"], state["persona"]["agent_id"], observation["session_date"]) != (
            review.get("from_style"),
            review.get("agent_id"),
            review.get("review_date"),
        ):
            return "prompt_context_mismatch", None
    except (KeyError, ValueError, TypeError, StopIteration):
        return "prompt_context_unavailable", None
    candidates = []
    for line, response in response_index.get(key, []):
        record = record_index[review["run_id"], line]
        if (
            not (record["http200"] or record["cache_state"] == "local_cache")
            or record["request_origin_matches"] is not True
        ):
            continue
        status, choice = parse_choice(
            response.get("response"),
            {
                "payload": request["payload"],
                "source": {"from_style": review["from_style"]},
            },
        )
        if status == "valid_linked_answer":
            candidates.append((line, choice))
    if len(candidates) != 1:
        return "missing_or_ambiguous_matching_response", None
    line, choice = candidates[0]
    if any(choice[k] != review.get(k) for k in ("decision", "target_style")):
        return "published_choice_mismatch", None
    return "matched", record_index[review["run_id"], line]


def audit_archives(batches, reviews):
    if len({batch["run_id"] for batch in batches}) != len(batches):
        raise ValueError("Duplicate archive run identity")
    if any(
        not isinstance(row, dict)
        for batch in batches
        for kind in ("requests", "responses")
        for row in batch[kind]
    ):
        raise ValueError("Archive records must be JSON objects")
    links, request_rows = index_requests(batches)
    records = []
    response_index = defaultdict(list)
    for batch in batches:
        for line, row in enumerate(batch["responses"], start=1):
            key = batch["run_id"], text_value(row.get("request_hash"))
            records.append(normalize_response(batch["run_id"], line, row, links.get(key)))
            response_index[key].append((line, row))
    record_index = {(row["run_id"], row["response_line"]): row for row in records}
    id_groups = summarize_repeated_ids(records)
    reused_refs = {
        (ref["run_id"], ref["response_line"]) for group in id_groups for ref in group["record_refs"]
    }
    new_reused_refs = {
        (ref["run_id"], ref["response_line"])
        for group in id_groups
        if group["separate_noncached_logical_requests"] > 1
        for ref in group["record_refs"]
    }
    binding_keys = Counter(
        (r.get("run_id"), text_value(r.get("raw_answer_request_hash")))
        for r in reviews
        if r.get("authority") == "archived_llm"
        and text_value(r.get("raw_answer_request_hash")) is not None
    )
    decisions = []
    for row in reviews:
        key = row.get("run_id"), text_value(row.get("raw_answer_request_hash"))
        if row.get("authority") == "archived_llm" and binding_keys[key] > 1:
            status, matched = "ambiguous_decision_mapping_key", None
        else:
            status, matched = mapping_status(row, links, response_index, record_index)
        ref = None if matched is None else (matched["run_id"], matched["response_line"])
        decisions.append(
            {
                **{
                    k: row.get(k)
                    for k in (
                        "run_id",
                        "agent_id",
                        "review_date",
                        "treatment",
                        "authority",
                        "raw_answer_request_hash",
                    )
                },
                "status": status,
                "response_line": matched["response_line"] if matched else None,
                "cache_state": matched["cache_state"] if matched else None,
                "response_id": matched["response_id"] if matched else None,
                "provider_id_repeated_in_scope": ref in reused_refs if matched else None,
                "provider_id_repeated_across_noncached_logical_requests": ref in new_reused_refs
                if matched
                else None,
            }
        )
    payload_groups = defaultdict(list)
    for key, link in links.items():
        if link["payload_sha256"] is not None:
            payload_groups[link["payload_sha256"]].append(key)
    success_keys = {
        (row["run_id"], row["request_hash"])
        for row in records
        if row["declared_noncached_http200"] and row["payload_sha256"] is not None
    }
    repeated_payloads = [
        {
            "payload_sha256": h,
            "logical_requests": len(keys),
            "logical_requests_with_noncached_http200": sum(key in success_keys for key in keys),
            "request_refs": [{"run_id": key[0], "request_hash": key[1]} for key in keys],
        }
        for h, keys in sorted(payload_groups.items())
        if len(keys) > 1
    ]
    texts = defaultdict(list)
    for row in records:
        if row["answer_text_sha256"] is not None:
            texts[row["answer_text_sha256"]].append(row)
    repeated_texts = [
        {
            "answer_text_sha256": h,
            "response_rows": len(rows),
            "distinct_present_response_ids": len(
                {row["response_id"] for row in rows if row["response_id"] is not None}
            ),
            "missing_response_id_rows": sum(row["response_id"] is None for row in rows),
            "record_refs": [
                {"run_id": row["run_id"], "response_line": row["response_line"]} for row in rows
            ],
        }
        for h, rows in sorted(texts.items())
        if len(rows) > 1
    ]
    per_run = []
    for batch in batches:
        rows = [row for row in records if row["run_id"] == batch["run_id"]]
        per_run.append(
            {
                "run_id": batch["run_id"],
                "request_rows": len(batch["requests"]),
                "response_rows": len(rows),
                "http200_rows": sum(row["http200"] for row in rows),
                "cache_states": dict(Counter(row["cache_state"] for row in rows)),
                "response_kinds": dict(Counter(str(row["kind"]) for row in rows)),
                "format_repair_request_rows": sum(
                    type(row.get("repair")) is int and row["repair"] > 0
                    for row in batch["requests"]
                ),
                "retry_response_rows": sum(
                    row["retry_index"] is not None and row["retry_index"] > 0 for row in rows
                ),
                "successful_rows_without_response_id": sum(
                    row["http200"] and row["response_id"] is None for row in rows
                ),
            }
        )
    summary = {
        "analysis_status": "retrospective_fixed_archive_audit",
        "run_count": len(batches),
        "request_log_rows": request_rows,
        "response_log_rows": len(records),
        "http200_response_rows": sum(row["http200"] for row in records),
        "http_non200_or_unknown_rows": sum(not row["http200"] for row in records),
        "format_repair_request_rows": sum(row["format_repair_request_rows"] for row in per_run),
        "retry_response_rows": sum(row["retry_response_rows"] for row in per_run),
        "cache_state_counts": dict(Counter(row["cache_state"] for row in records)),
        "duplicate_request_log_rows": sum(link["request_rows"] - 1 for link in links.values()),
        "logical_request_keys_with_unknown_payload": sum(
            link["payload_sha256"] is None for link in links.values()
        ),
        "response_rows_with_unknown_payload": sum(row["payload_sha256"] is None for row in records),
        "response_rows_with_origin_mismatch_or_unknown": sum(
            row["request_origin_matches"] is not True for row in records
        ),
        "http200_rows_without_provider_id": sum(
            row["http200"] and row["response_id"] is None for row in records
        ),
        "distinct_present_provider_ids": len(
            {row["response_id"] for row in records if row["response_id"] is not None}
        ),
        "repeated_provider_id_groups": len(id_groups),
        "repeated_provider_ids_across_noncached_logical_requests": sum(
            g["separate_noncached_logical_requests"] > 1 for g in id_groups
        ),
        "distinct_uniquely_linked_payloads": len(payload_groups),
        "repeated_payload_groups_across_logical_requests": len(repeated_payloads),
        "same_input_opportunity_groups_with_multiple_noncached_successes": sum(
            g["logical_requests_with_noncached_http200"] > 1 for g in repeated_payloads
        ),
        "repeated_answer_text_groups": len(repeated_texts),
        "review_mapping_rows": len(reviews),
        "decision_mapping_status_counts": dict(Counter(row["status"] for row in decisions)),
        "matched_style_choices_with_repeated_provider_id": sum(
            row["provider_id_repeated_in_scope"] is True for row in decisions
        ),
        "matched_style_choices_with_id_repeated_across_noncached_requests": sum(
            row["provider_id_repeated_across_noncached_logical_requests"] is True
            for row in decisions
        ),
        "independent_generation_count": None,
        "independent_sampling_established": False,
        "provider_cache_mechanism_confirmed": False,
        "human_alignment_established": False,
        "causal_effect_established": False,
        "order_execution_or_fills_established": False,
        "new_model_calls": 0,
        "per_run": per_run,
    }
    return {
        "summary": summary,
        "responses": records,
        "decisions": decisions,
        "repeated_provider_ids": id_groups,
        "repeated_payloads": repeated_payloads,
        "repeated_answer_texts": repeated_texts,
    }
