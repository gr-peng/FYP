"""Six fixed fact-only diagnostics; answer keys never belong to a model payload."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from behavioral_market.evaluation.style_mechanism_design import (
    ENVELOPE,
    canonical,
    changed_paths,
    digest,
    semantic_payload,
    validate_case,
)
from behavioral_market.evaluation.style_probe_intake import aware_time, strict_json
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode

SOURCE = "iterations/style_conflict_package_20261022/package/scheduled_inputs.jsonl"
SOURCE_BYTES = 23170
SOURCE_SHA256 = "f61a18344416eb4325993458234df89ef77a890f1bdae7ba34cd1580fbd32e58"
BASE_IDS = tuple(f"synthetic_00{i:02b}" for i in range(4))
CASE_IDS = tuple(f"facts_{i:04b}" for i in range(4)) + ("facts_control_90", "facts_control_105")
NAMESPACE = "fact-reading-six-v1"
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
FIELDS = {
    "price_reference_relation": ("above", "equal", "below"),
    "benchmark_better_style": ("fundamental", "technical"),
    "majority_style": ("fundamental", "technical"),
    "population_as_of_session": None,
    "population_size_excluding_self": None,
}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        key: {"type": "string", "enum": list(values)}
        if values
        else {"type": "integer", "minimum": 0}
        for key, values in FIELDS.items()
    },
    "required": list(FIELDS),
}
SYSTEM = (
    "Read the supplied hypothetical market context and return the requested facts. "
    "Compare last_close with reference.value: above/equal/below describes the price "
    "relative to that supplied reference. For benchmark_better_style choose the named "
    "style with the larger supplied benchmark return, including when both are negative. "
    "For majority_style use the supplied population counts. Copy the population snapshot "
    "session and size excluding yourself. Return exactly the five fields in output_schema. "
    "Do not choose a trading strategy or add a rationale."
)
INSTRUCTION = (
    "Fill only the requested facts from observation. The reference is an assigned "
    "hypothetical input, and named benchmark returns are historical supplied examples, "
    "not personal counterfactual returns or forecasts. The population is a snapshot "
    "at its own as_of_session and excludes the agent."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_materials(base):
    require([r["case_id"] for r in base] == list(BASE_IDS), "exact four source cases required")
    require(all(not validate_case(r) for r in base), "invalid original synthetic context")
    rows = []
    for i, case_id in enumerate(CASE_IDS):
        source = base[i] if i < 4 else base[0]
        payload = deepcopy(source["payload"])
        user = strict_json(payload["messages"][1]["content"])
        user["decision_instruction"], user["output_schema"] = INSTRUCTION, deepcopy(SCHEMA)
        control = None if i < 4 else (90 if i == 4 else 105)
        if control is not None:
            user["observation"]["reference"]["value"] = control
        payload["messages"][0]["content"] = SYSTEM
        payload["messages"][1]["content"] = canonical(user)
        rows.append(
            {
                "case_id": case_id,
                "source_case_id": source["case_id"],
                "source_row_sha256": digest(source),
                "control_reference_value": control,
                "material_type": "new_fact_task_from_synthetic_context"
                if i < 4
                else "new_synthetic_reference_control",
                "payload": payload,
                "payload_sha256": digest(payload),
            }
        )
    return rows


def validate_materials(cases, base):
    require(
        canonical(cases) == canonical(build_materials(base)), "materials or source linkage differ"
    )


def gold_for_case(case):
    """Compute from model-visible numbers only, never from condition labels."""
    d = strict_json(case["payload"]["messages"][1]["content"])
    o = d["observation"]
    close, reference = o["last_close"], o["reference"]["value"]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in (close, reference)),
        "nonfinite price",
    )
    returns, population = o["style_benchmarks"]["returns"], o["mean_field"]
    f, t = returns["fundamental"], returns["technical"]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in (f, t)) and f != t,
        "undefined benchmark winner",
    )
    fc, tc = population["fundamental_count"], population["technical_count"]
    size, session = population["population_size_excluding_self"], population["as_of_session"]
    require(
        all(type(v) is int and v >= 0 for v in (fc, tc, size, session)),
        "invalid population integers",
    )
    require(size > 0 and fc + tc == size and fc != tc, "undefined population majority")
    require(session < o["session"], "population is not lagged")
    require(
        math.isclose(population["fundamental_share"], fc / size)
        and math.isclose(population["technical_share"], tc / size),
        "inconsistent population shares",
    )
    facts = {
        "price_reference_relation": "above"
        if close > reference
        else "below"
        if close < reference
        else "equal",
        "benchmark_better_style": "fundamental" if f > t else "technical",
        "majority_style": "fundamental" if fc > tc else "technical",
        "population_as_of_session": session,
        "population_size_excluding_self": size,
    }
    return {
        "facts": facts,
        "evidence": {
            "observation.last_close": close,
            "observation.reference.value": reference,
            "observation.style_benchmarks.returns": deepcopy(returns),
            "observation.mean_field": deepcopy(population),
            "observation.session": o["session"],
        },
    }


def make_gold(cases):
    return {
        "materials_sha256": digest(cases),
        "by_case": {c["case_id"]: gold_for_case(c) for c in cases},
        "scope": (
            "Private computed key for these six inputs, not a general understanding benchmark."
        ),
    }


def make_package(cases, gold, run_id, output, *, source_rows, evidence_kind="recorded_model_logs"):
    require(
        isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id),
        "invalid run ID",
    )
    require(Path(output).is_absolute(), "fixed absolute output required")
    require(
        evidence_kind in {"recorded_model_logs", "synthetic_tool_test"}, "invalid evidence kind"
    )
    require(canonical(gold) == canonical(make_gold(cases)), "answer key differs")
    core = {
        "protocol": NAMESPACE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "cases": deepcopy(cases),
        "source_snapshot_not_sent": {
            "path": SOURCE,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "canonical_jsonl_rows": deepcopy(source_rows),
        },
        "gold_sha256": digest(gold),
        "endpoint": ENDPOINT,
        "model": ENVELOPE["model"],
        "execution_output": str(output),
        "currently_authorized_new_POSTs": 0,
        "used_prior_POSTs": 25,
        "planned_max_POSTs_not_authorized": 6,
        "transport_retries": 0,
        "format_repairs": 0,
        "max_concurrency": 1,
        "stop_policy": (
            "Stop transport/identity/cache/schema/mode failure; a schema-valid wrong fact "
            "is an observed outcome and does not stop the remaining cases."
        ),
        "interpretation": (
            "New fact-only task; no strategy choice, old-rationale validation, general "
            "understanding or human/market-mechanism proof."
        ),
    }
    package = {**core, "package_sha256": digest(core)}
    validate_package(package, gold)
    return package


def validate_package(package, gold=None):
    core = {k: v for k, v in package.items() if k != "package_sha256"}
    require(
        digest(core) == package["package_sha256"] and package["protocol"] == NAMESPACE,
        "package hash/protocol differs",
    )
    snapshot = package["source_snapshot_not_sent"]
    require(
        set(snapshot) == {"path", "bytes", "sha256", "canonical_jsonl_rows"},
        "source snapshot fields differ",
    )
    require(
        snapshot["path"] == SOURCE
        and type(snapshot["bytes"]) is int
        and snapshot["bytes"] == SOURCE_BYTES
        and snapshot["sha256"] == SOURCE_SHA256,
        "declared source identity differs",
    )
    source_raw = "".join(canonical(row) + "\n" for row in snapshot["canonical_jsonl_rows"]).encode()
    require(
        len(source_raw) == SOURCE_BYTES and hashlib.sha256(source_raw).hexdigest() == SOURCE_SHA256,
        "frozen source content differs",
    )
    validate_materials(package["cases"], snapshot["canonical_jsonl_rows"])
    require(
        [c["case_id"] for c in package["cases"]] == list(CASE_IDS), "six fixed cases/order required"
    )
    require(
        package["evidence_kind"] in {"recorded_model_logs", "synthetic_tool_test"},
        "invalid evidence kind",
    )
    require(
        isinstance(package["run_id"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", package["run_id"]),
        "invalid run ID",
    )
    require(
        package["endpoint"] == ENDPOINT and package["model"] == ENVELOPE["model"], "route differs"
    )
    for key, value in (
        ("currently_authorized_new_POSTs", 0),
        ("used_prior_POSTs", 25),
        ("planned_max_POSTs_not_authorized", 6),
        ("transport_retries", 0),
        ("format_repairs", 0),
        ("max_concurrency", 1),
    ):
        require(type(package.get(key)) is int and package[key] == value, "prepared limits differ")
    for c in package["cases"]:
        p = c["payload"]
        require(c["payload_sha256"] == digest(p), "payload hash differs")
        require(
            canonical({k: v for k, v in p.items() if k != "messages"}) == canonical(ENVELOPE),
            "model envelope differs",
        )
        require(
            len(p["messages"]) == 2
            and p["messages"][0] == {"role": "system", "content": SYSTEM}
            and p["messages"][1]["role"] == "user",
            "fact-only messages differ",
        )
        d = strict_json(p["messages"][1]["content"])
        require(
            d["decision_instruction"] == INSTRUCTION
            and canonical(d["output_schema"]) == canonical(SCHEMA),
            "fact-only instruction/schema differs",
        )
    for i, case in enumerate(package["cases"]):
        require(
            set(case)
            == {
                "case_id",
                "source_case_id",
                "source_row_sha256",
                "control_reference_value",
                "material_type",
                "payload",
                "payload_sha256",
            },
            "unexpected case metadata",
        )
        require(
            case["source_case_id"] == (BASE_IDS[i] if i < 4 else BASE_IDS[0]), "source case differs"
        )
        if i < 4:
            require(case["control_reference_value"] is None, "base marked as control")
        else:
            reference = 90 if i == 4 else 105
            require(
                type(case["control_reference_value"]) is int
                and case["control_reference_value"] == reference,
                "control label differs",
            )
            require(
                changed_paths(
                    semantic_payload(package["cases"][0]["payload"]),
                    semantic_payload(case["payload"]),
                )
                == ["messages.1.content.observation.reference.value"],
                "control changes other information",
            )
            require(
                strict_json(case["payload"]["messages"][1]["content"])["observation"]["reference"][
                    "value"
                ]
                == reference,
                "control value differs",
            )
    require(Path(package["execution_output"]).is_absolute(), "fixed output differs")
    expected = make_gold(package["cases"])
    require(package["gold_sha256"] == digest(expected), "computed key binding differs")
    if gold is not None:
        require(canonical(gold) == canonical(expected), "hidden key altered")


def authorization_template(package):
    return {
        "decision": "not_authorized",
        "package_sha256": package["package_sha256"],
        "run_id": package["run_id"],
        "endpoint": ENDPOINT,
        "model": package["model"],
        "prior_POSTs": 25,
        "additional_POST_limit": 0,
        "cumulative_POST_cap": 25,
        "specific_user_reply": None,
    }


def check_authorization(package, authorization):
    validate_package(package)
    require(authorization.get("decision") == "explicit_user_granted", "new calls not authorized")
    for key in ("package_sha256", "run_id", "endpoint", "model"):
        require(authorization.get(key) == package[key], "authorization binding differs")
    for key, value in (
        ("prior_POSTs", 25),
        ("additional_POST_limit", 6),
        ("cumulative_POST_cap", 31),
    ):
        require(
            type(authorization.get(key)) is int and authorization[key] == value,
            "specific cap differs",
        )
    require(
        isinstance(authorization.get("specific_user_reply"), str)
        and authorization["specific_user_reply"].strip(),
        "specific user reply required",
    )


def identity(case, package):
    return {
        "kind": "fact_reading_probe",
        "probe_namespace": NAMESPACE,
        "probe_run_id": package["run_id"],
        "evidence_kind": package["evidence_kind"],
        "case_id": case["case_id"],
        "position": list(CASE_IDS).index(case["case_id"]) + 1,
        "repair": 0,
    }


def expected_request(case, package):
    origin = identity(case, package)
    h = hashlib.sha256(
        json.dumps([case["payload"], origin, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    return {**origin, "request_hash": h, "payload": deepcopy(case["payload"])}


def execution_config(package):
    validate_package(package)
    return {
        **{k: deepcopy(v) for k, v in ENVELOPE.items() if k != "response_format"},
        "allowed_host": "aigc-api.hkust-gz.edu.cn",
        "chat_completions_url": ENDPOINT,
        "models_url": "https://aigc-api.hkust-gz.edu.cn/api/v2/models",
        "api_key_env": "AIGC_API_KEY",
        "trust_env": False,
        "max_concurrency": 1,
        "transport_retries": 0,
        "timeout_seconds": 120,
    }


def parse_answer(body):
    if not isinstance(body, dict) or body.get("model") != ENVELOPE["model"]:
        return "invalid_model_or_envelope", None
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return "invalid_envelope", None
    c = choices[0]
    message = c.get("message")
    if (
        c.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        return "invalid_envelope", None
    try:
        value = strict_json(message["content"])
        require(isinstance(value, dict) and set(value) == set(FIELDS), "five fields required")
        for key, options in FIELDS.items():
            if options:
                require(type(value[key]) is str and value[key] in options, "invalid enum")
            else:
                require(
                    type(value[key]) is int and value[key] >= 0,
                    "nonnegative strict integer required",
                )
    except (ValueError, TypeError):
        return "invalid_schema", None
    return "valid_schema", value


async def collect(package, authorization, client, output):
    check_authorization(package, authorization)
    require(output.resolve() == Path(package["execution_output"]).resolve(), "output differs")
    require(not (output / "execution_state.json").exists(), "no resume")
    require(
        canonical(client.config) == canonical(execution_config(package))
        and client.cache_namespace == NAMESPACE,
        "client differs",
    )
    require(not list(client.cache.iterdir()), "empty local cache required")
    names = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
    require(
        not any((output / n).exists() and (output / n).stat().st_size for n in names),
        "no existing logs",
    )
    for name in names:
        (output / name).touch(exist_ok=True)
    state = {
        "status": "running",
        "stop_reason": None,
        "slots": [{"case_id": k, "status": "unsubmitted"} for k in CASE_IDS],
    }

    def save():
        (output / "execution_state.json").write_text(canonical(state) + "\n")

    save()
    active = None
    try:
        await client.verify_model()
        for i, case in enumerate(package["cases"]):
            active = i
            state["slots"][i]["status"] = "request_in_progress"
            save()
            body, meta = await client.completion(
                case["payload"]["messages"], identity(case, package)
            )
            require(
                meta.get("request_hash") == expected_request(case, package)["request_hash"],
                "request hash differs",
            )
            require(meta.get("cache_hit") is False, "local cache replay")
            status, _ = parse_answer(body)
            require(status == "valid_schema", status)
            require(returned_mode(body) == "no_returned_reasoning_observed", "mode failure")
            state["slots"][i]["status"] = "schema_valid_not_fact_scored"
            save()
        state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="in-flight billing unknown; no retry")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        if active is not None and state["slots"][active]["status"] == "request_in_progress":
            state["slots"][active]["status"] = "failed_or_unknown"
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save()
    return state


def linked_rows(package, requests, attempts, responses):
    """Reconstruct protocol status from raw logs, without any execution-state flag."""
    validate_package(package)
    wanted = {c["case_id"]: expected_request(c, package) for c in package["cases"]}
    maps = []
    for kind, records in (("request", requests), ("attempt", attempts), ("response", responses)):
        mapped = {}
        for row in records:
            cid = row.get("case_id")
            require(cid in wanted and cid not in mapped, "foreign or duplicate row")
            expected = wanted[cid]
            require(
                all(row.get(k) == v for k, v in expected.items() if k != "payload"),
                "identity/hash differs",
            )
            require(
                type(row.get("repair")) is int and type(row.get("position")) is int,
                "identity types differ",
            )
            base = set(expected) - {"payload"}
            if kind == "request":
                require(canonical(row) == canonical(expected), "request payload differs")
            elif kind == "attempt":
                require(
                    set(row) == base | {"retry_count", "dispatched_at"}, "attempt fields differ"
                )
            elif row.get("cached") is True:
                require(
                    set(row) == base | {"cached", "response", "status"} and row["status"] == "ok",
                    "cache fields differ",
                )
            else:
                fields = base | {
                    "cached",
                    "response",
                    "http_status",
                    "retry_count",
                    "requested_at",
                    "response_received_at",
                    "latency_seconds",
                }
                require(
                    fields <= set(row) <= fields | {"trace_id", "retry_after_seconds"},
                    "response fields differ",
                )
                latency = row["latency_seconds"]
                require(
                    type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
                    "invalid latency",
                )
            if kind == "attempt" or kind == "response" and row.get("cached") is not True:
                require(
                    type(row.get("retry_count")) is int and row["retry_count"] == 0,
                    "retries forbidden",
                )
            mapped[cid] = row
        require(list(mapped) == list(CASE_IDS[: len(mapped)]), "non-prefix log order")
        maps.append(mapped)
    submitted, dispatched, received = maps
    require(set(dispatched) <= set(submitted) and set(received) <= set(submitted), "orphan log")
    rows = []
    for i, case in enumerate(package["cases"]):
        cid = case["case_id"]
        status, mode, answer, compatible, body = "unsubmitted", "unobserved", None, False, None
        if cid in submitted:
            require(not i or rows[-1]["protocol_compatible"], "request after protocol failure")
            status = "request_record_only"
        if cid in dispatched:
            start = aware_time(dispatched[cid]["dispatched_at"])
            if i:
                require(
                    start >= aware_time(received[CASE_IDS[i - 1]]["response_received_at"]),
                    "overlapping requests",
                )
            status = "awaiting_response"
        if cid in received:
            response = received[cid]
            require(type(response.get("cached")) is bool, "explicit cache status required")
            if response["cached"]:
                require(cid not in dispatched, "cache cannot claim dispatch")
                status = "local_cache_replay"
            else:
                require(cid in dispatched, "response lacks dispatch")
                start, finish = (
                    aware_time(response["requested_at"]),
                    aware_time(response["response_received_at"]),
                )
                require(
                    start == aware_time(dispatched[cid]["dispatched_at"]) and finish >= start,
                    "response clock differs",
                )
                http = response["http_status"]
                require(
                    type(http) is int
                    and 100 <= http <= 599
                    or http in ("transport_error", "cancelled_in_flight"),
                    "invalid transport status",
                )
                if http == 200:
                    body = response["response"]
                    status, answer = parse_answer(body)
                    mode = returned_mode(body)
                    compatible = (
                        status == "valid_schema" and mode == "no_returned_reasoning_observed"
                    )
                else:
                    require(response["response"] is None, "failed transport contains answer")
                    status = "transport_failed"
        text = body["choices"][0]["message"]["content"] if answer is not None else None
        rows.append(
            {
                "case_id": cid,
                "request_hash": wanted[cid]["request_hash"],
                "format_status": status,
                "mode_status": mode,
                "protocol_compatible": compatible,
                "parsed_answer": answer,
                "provider_id": body.get("id") if isinstance(body, dict) else None,
                "created": body.get("created") if isinstance(body, dict) else None,
                "body_sha256": digest(body) if body is not None else None,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest()
                if text is not None
                else None,
            }
        )
    return rows, len(dispatched)


def evaluate(package, gold, requests, attempts, responses):
    validate_package(package, gold)
    rows, dispatched = linked_rows(package, requests, attempts, responses)
    for row in rows:
        key = gold["by_case"][row["case_id"]]["facts"]
        checked = {
            f: row["parsed_answer"][f] == key[f] if row["protocol_compatible"] else None
            for f in FIELDS
        }
        row["field_correct"] = checked
        row["fact_status"] = (
            "unknown"
            if not row["protocol_compatible"]
            else "correct"
            if all(checked.values())
            else "wrong"
        )
    repeated = {}
    for field in ("provider_id", "body_sha256", "text_sha256", "same_id_created_text"):
        groups = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if field == "same_id_created_text":
                value = (
                    digest([row["provider_id"], row["created"], row["text_sha256"]])
                    if row["provider_id"] and row["created"] is not None and row["text_sha256"]
                    else None
                )
            if isinstance(value, str) and value:
                groups[value].append(row["case_id"])
        repeated[field] = [group for group in groups.values() if len(group) > 1]
    for row in rows:
        row["reuse_flags"] = [
            field for field, groups in repeated.items() if any(row["case_id"] in g for g in groups)
        ]
    counts = {
        f: {
            "correct": sum(r["field_correct"][f] is True for r in rows),
            "wrong": sum(r["field_correct"][f] is False for r in rows),
            "unknown": sum(r["field_correct"][f] is None for r in rows),
        }
        for f in FIELDS
    }
    complete = all(r["protocol_compatible"] for r in rows)
    return {
        "evidence_kind": package["evidence_kind"],
        "rows": rows,
        "execution_complete": complete,
        "schema_valid_returns": sum(r["format_status"] == "valid_schema" for r in rows),
        "protocol_valid_returns": sum(r["protocol_compatible"] for r in rows),
        "fact_status_counts": dict(Counter(r["fact_status"] for r in rows)),
        "all_fields_correct": all(r["fact_status"] == "correct" for r in rows)
        if complete
        else None,
        "per_field_counts": counts,
        "logged_dispatch_attempts": dispatched,
        "return_identity_diagnostics": repeated,
        "independent_generation_count": None,
        "general_understanding_score": None,
        "real_external_execution_authenticated": False,
        "old_rationale_correctness_established": False,
        "human_alignment_established": False,
        "limit": (
            "Six specified fact-only service returns; not a strategy task, general accuracy, "
            "old-rationale correctness or fresh independent-generation proof."
        ),
    }
