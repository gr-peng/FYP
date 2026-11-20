"""Four fixed service-return checks, with no authority to make new calls by default."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from behavioral_market.evaluation.style_mechanism_design import (
    ENVELOPE,
    allowed_paths,
    canonical,
    changed_paths,
    digest,
    semantic_payload,
    validate_case,
)
from behavioral_market.evaluation.style_probe_intake import aware_time, parse_choice
from behavioral_market.evaluation.style_probe_mode_v2 import returned_mode

NAMESPACE = "style-conflict-four-v1"
SELECTED = tuple(f"synthetic_00{i:02b}" for i in range(4))
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
FIXED_EDGES = {
    ("synthetic_0000", "synthetic_0001"): "majority_style",
    ("synthetic_0010", "synthetic_0011"): "majority_style",
    ("synthetic_0000", "synthetic_0010"): "benchmark_favored_style",
    ("synthetic_0001", "synthetic_0011"): "benchmark_favored_style",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_package(bank, edges, run_id, execution_output, evidence_kind="recorded_model_logs"):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id), "invalid run identity")
    require(
        evidence_kind in {"recorded_model_logs", "synthetic_tool_test"}, "invalid evidence kind"
    )
    require(Path(execution_output).is_absolute(), "fixed absolute execution output required")
    by_id = {r["case_id"]: r for r in bank}
    require(len(bank) == len(by_id) == 16, "retain the sixteen source cells")
    cases = [deepcopy(by_id[k]) for k in SELECTED]
    require(all(not validate_case(r) for r in cases), "ineligible selected material")
    require(
        all(
            r["factors"]["current_style"] == "fundamental"
            and r["factors"]["reference_condition"] == "neutral"
            for r in cases
        ),
        "fixed factors differ",
    )
    contrasts = [
        deepcopy(e) for e in edges if e["left_case"] in SELECTED and e["right_case"] in SELECTED
    ]
    expected_pairs = {
        ("synthetic_0000", "synthetic_0001"),
        ("synthetic_0010", "synthetic_0011"),
        ("synthetic_0000", "synthetic_0010"),
        ("synthetic_0001", "synthetic_0011"),
    }
    require(
        len(contrasts) == 4
        and {(e["left_case"], e["right_case"]) for e in contrasts} == expected_pairs,
        "require all four fixed contrasts",
    )
    core = {
        "protocol": NAMESPACE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "cases": cases,
        "contrasts": contrasts,
        "not_scheduled_case_ids": [r["case_id"] for r in bank if r["case_id"] not in SELECTED],
        "execution_output": str(execution_output),
        "endpoint": ENDPOINT,
        "model": ENVELOPE["model"],
        "transport_retries": 0,
        "format_repairs": 0,
        "max_concurrency": 1,
        "planned_max_POSTs_not_authorized": 4,
        "currently_authorized_new_POSTs": 0,
        "used_prior_POSTs": 21,
        "prior_cumulative_cap": 21,
        "ordering_limit": "One fixed sequential return per condition; order/time effect"
        "s are not identified.",
        "reuse_policy": "Local replay fails; provider-ID/text/core repeats are diagno"
        "stic only, without retries or fresh-generation claims.",
        "stop_policy": "Stop on first transport,identity,payload,answer,mode or loca"
        "l-cache failure; never resume automatically.",
    }
    package = {**core, "package_sha256": digest(core)}
    validate_package(package)
    return package


def validate_package(package):
    core = {k: v for k, v in package.items() if k != "package_sha256"}
    require(package["package_sha256"] == digest(core), "package fingerprint differs")
    require(package["protocol"] == NAMESPACE, "wrong protocol")
    require(
        isinstance(package["run_id"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", package["run_id"]),
        "invalid run identity",
    )
    require(
        package["evidence_kind"] in {"recorded_model_logs", "synthetic_tool_test"},
        "invalid evidence kind",
    )
    require([r["case_id"] for r in package["cases"]] == list(SELECTED), "fixed order differs")
    require(all(not validate_case(r) for r in package["cases"]), "invalid material")
    for key, value in (
        ("currently_authorized_new_POSTs", 0),
        ("used_prior_POSTs", 21),
        ("prior_cumulative_cap", 21),
        ("planned_max_POSTs_not_authorized", 4),
        ("transport_retries", 0),
        ("format_repairs", 0),
        ("max_concurrency", 1),
    ):
        require(type(package.get(key)) is int and package[key] == value, "prepared limits differ")
    require(
        all(
            r["factors"]["current_style"] == "fundamental"
            and r["factors"]["reference_condition"] == "neutral"
            for r in package["cases"]
        ),
        "fixed factors differ",
    )
    require(
        len(package["not_scheduled_case_ids"]) == 12
        and set(package["not_scheduled_case_ids"])
        == {f"synthetic_{i:04b}" for i in range(16)} - set(SELECTED),
        "excluded cells differ",
    )
    require(
        package["endpoint"] == ENDPOINT and package["model"] == ENVELOPE["model"], "route differs"
    )
    require(Path(package["execution_output"]).is_absolute(), "fixed absolute output required")
    by_id = {r["case_id"]: r for r in package["cases"]}
    edges = package["contrasts"]
    require(
        len(edges) == 4 and {(e["left_case"], e["right_case"]) for e in edges} == set(FIXED_EDGES),
        "contrast pairs differ",
    )
    for edge in edges:
        left, right = edge["left_case"], edge["right_case"]
        factor = FIXED_EDGES[left, right]
        require(
            edge["factor"] == factor
            and edge["positive_case"] == left
            and edge["negative_case"] == right,
            "contrast direction differs",
        )
        require(
            edge["changed_paths"]
            == allowed_paths()[factor]
            == changed_paths(
                semantic_payload(by_id[left]["payload"]), semantic_payload(by_id[right]["payload"])
            ),
            "contrast fields differ",
        )
        require(
            edge["conditioning"]
            == {k: v for k, v in by_id[left]["factors"].items() if k != factor},
            "contrast conditioning differs",
        )


def authorization_template(package):
    return {
        "decision": "not_authorized",
        "package_sha256": package["package_sha256"],
        "run_id": package["run_id"],
        "endpoint": ENDPOINT,
        "model": package["model"],
        "prior_POSTs": 21,
        "additional_POST_limit": 0,
        "cumulative_POST_cap": 21,
        "specific_user_reply": None,
        "limit": "Local consistency receipt only; a file or label does not aut"
        "henticate user approval.",
    }


def check_authorization(package, authorization):
    validate_package(package)
    require(
        authorization.get("decision") == "explicit_user_granted", "new calls are not authorized"
    )
    for key in ("package_sha256", "run_id", "endpoint", "model"):
        require(authorization.get(key) == package[key], "authorization does not match package")
    for key, value in (
        ("prior_POSTs", 21),
        ("additional_POST_limit", 4),
        ("cumulative_POST_cap", 25),
    ):
        require(
            type(authorization.get(key)) is int and authorization[key] == value,
            "specific call cap differs",
        )
    require(
        isinstance(authorization.get("specific_user_reply"), str)
        and authorization["specific_user_reply"].strip(),
        "specific user reply must be recorded",
    )


def request_identity(case, package):
    return {
        "kind": "style_conflict_check",
        "probe_namespace": NAMESPACE,
        "probe_run_id": package["run_id"],
        "evidence_kind": package["evidence_kind"],
        "case_id": case["case_id"],
        "position": list(SELECTED).index(case["case_id"]) + 1,
        "repair": 0,
    }


def expected_request(case, package):
    origin = request_identity(case, package)
    key = hashlib.sha256(
        json.dumps([case["payload"], origin, NAMESPACE], sort_keys=True).encode()
    ).hexdigest()
    return {**origin, "request_hash": key, "payload": deepcopy(case["payload"])}


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


def choice_status(body, case):
    adapter = {
        "payload": case["payload"],
        "source": {"from_style": case["factors"]["current_style"]},
    }
    answer, choice = parse_choice(body, adapter)
    mode = returned_mode(body)
    return answer, mode, choice


async def collect(package, authorization, client, output):
    check_authorization(package, authorization)
    require(
        output.resolve() == Path(package["execution_output"]).resolve(),
        "fixed execution output differs",
    )
    require(not (output / "execution_state.json").exists(), "never resume an execution state")
    require(
        canonical(client.config) == canonical(execution_config(package)),
        "client configuration differs",
    )
    require(
        client.cache_namespace == NAMESPACE and not list(client.cache.iterdir()),
        "fresh local namespace/cache required",
    )
    names = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
    require(
        not any((output / n).exists() and (output / n).stat().st_size for n in names),
        "never resume prior calls",
    )
    for name in names:
        (output / name).touch(exist_ok=True)
    state = {
        "status": "running",
        "stop_reason": None,
        "slots": [{"case_id": k, "status": "unsubmitted"} for k in SELECTED],
        "new_market_runs": 0,
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
                case["payload"]["messages"], request_identity(case, package)
            )
            require(
                meta.get("request_hash") == expected_request(case, package)["request_hash"],
                "dispatched payload/hash differs",
            )
            answer, mode, _ = choice_status(body, case)
            require(meta.get("cache_hit") is False, "local cache replay")
            require(answer == "valid_linked_answer", answer)
            require(mode == "no_returned_reasoning_observed", mode)
            state["slots"][i]["status"] = "valid_return"
            save()
        state["status"] = "complete"
    except asyncio.CancelledError:
        state.update(status="cancelled", stop_reason="cancelled; dispatched billing unknown")
        raise
    except Exception as exc:
        state.update(status="stopped", stop_reason=type(exc).__name__)
    finally:
        if active is not None and state["slots"][active]["status"] == "request_in_progress":
            state["slots"][active]["status"] = "failed_or_unknown"
        state["finished_utc"] = datetime.now(UTC).isoformat()
        save()
    return state


def evaluate(package, requests, attempts, responses):
    """Validate independent raw logs; execution_state.json is never evidence here."""
    validate_package(package)
    expected = {r["case_id"]: expected_request(r, package) for r in package["cases"]}
    maps = []
    for kind, records in (("request", requests), ("attempt", attempts), ("response", responses)):
        mapped = {}
        for row in records:
            case_id = row.get("case_id")
            require(case_id in expected and case_id not in mapped, "foreign or duplicate log row")
            wanted = expected[case_id]
            require(
                all(row.get(k) == v for k, v in wanted.items() if k != "payload"),
                "identity/hash differs",
            )
            require(
                type(row.get("position")) is int and type(row.get("repair")) is int,
                "identity types differ",
            )
            if kind == "request":
                require(canonical(row) == canonical(wanted), "request payload or fields differ")
            else:
                base = set(wanted) - {"payload"}
                if kind == "attempt":
                    require(
                        set(row) == base | {"retry_count", "dispatched_at"},
                        "unexpected attempt fields",
                    )
                elif row.get("cached") is True:
                    require(
                        set(row) == base | {"cached", "response", "status"}
                        and row["status"] == "ok",
                        "unexpected replay fields",
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
                        "unexpected response fields",
                    )
                    latency = row["latency_seconds"]
                    require(
                        type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
                        "invalid response latency",
                    )
            if kind == "attempt" or kind == "response" and row.get("cached") is not True:
                require(
                    type(row.get("retry_count")) is int and row["retry_count"] == 0,
                    "retries forbidden",
                )
            mapped[case_id] = row
        require(
            list(mapped) == list(SELECTED[: len(mapped)]), "logs must follow fixed prefix order"
        )
        maps.append(mapped)
    submitted, dispatched, received = maps
    require(
        set(dispatched) <= set(submitted) and set(received) <= set(submitted),
        "orphan attempt/response",
    )
    rows = []
    for i, case in enumerate(package["cases"]):
        cid = case["case_id"]
        status, mode, choice, compatible = "unsubmitted", "unobserved", None, False
        body = None
        if cid in submitted:
            require(not i or rows[-1]["compatible"], "request after failure violates stop rule")
            status = "request_record_only"
        if cid in dispatched:
            start = aware_time(dispatched[cid]["dispatched_at"])
            if i:
                require(
                    start >= aware_time(received[SELECTED[i - 1]]["response_received_at"]),
                    "overlapping calls",
                )
            status = "awaiting_response"
        if cid in received:
            response = received[cid]
            require(type(response.get("cached")) is bool, "explicit cache status required")
            if response["cached"]:
                require(cid not in dispatched, "replay cannot claim new dispatch")
                status = "local_cache_replay"
            else:
                require(cid in dispatched, "response lacks dispatch")
                start, finish = (
                    aware_time(response["requested_at"]),
                    aware_time(response["response_received_at"]),
                )
                require(
                    start == aware_time(dispatched[cid]["dispatched_at"]) and finish >= start,
                    "response clocks differ",
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
                    status, mode, parsed = choice_status(body, case)
                    compatible = (
                        status == "valid_linked_answer" and mode == "no_returned_reasoning_observed"
                    )
                    choice = parsed if compatible else None
                else:
                    require(response["response"] is None, "failed transport carries answer")
                    status = "transport_failed"
        message = body["choices"][0]["message"]["content"] if compatible else None
        rows.append(
            {
                "case_id": cid,
                "factors": case["factors"],
                "status": status,
                "mode": mode,
                "compatible": compatible,
                "choice": choice,
                "request_hash": expected[cid]["request_hash"],
                "provider_id": body.get("id") if isinstance(body, dict) else None,
                "created": body.get("created") if isinstance(body, dict) else None,
                "body_sha256": digest(body) if body is not None else None,
                "text_sha256": hashlib.sha256(message.encode()).hexdigest()
                if message is not None
                else None,
            }
        )
    diagnostics = {}
    for field in ("provider_id", "body_sha256", "text_sha256"):
        groups = defaultdict(list)
        for row in rows:
            value = row[field]
            if isinstance(value, str) and value:
                groups[value].append(row["case_id"])
        diagnostics[field] = [ids for ids in groups.values() if len(ids) > 1]
    signatures = defaultdict(list)
    for row in rows:
        if row["provider_id"] and row["created"] is not None and row["text_sha256"]:
            signatures[digest([row["provider_id"], row["created"], row["text_sha256"]])].append(
                row["case_id"]
            )
    diagnostics["same_id_created_text"] = [ids for ids in signatures.values() if len(ids) > 1]
    by_id = {r["case_id"]: r for r in rows}
    contrasts = []
    for edge in package["contrasts"]:
        a, b = by_id[edge["positive_case"]], by_id[edge["negative_case"]]
        complete = a["compatible"] and b["compatible"]
        flags = [
            key
            for key, groups in diagnostics.items()
            if any(a["case_id"] in g and b["case_id"] in g for g in groups)
        ]
        elsewhere = [
            key
            for key, groups in diagnostics.items()
            if any(
                ({a["case_id"], b["case_id"]} & set(group))
                and (set(group) - {a["case_id"], b["case_id"]})
                for group in groups
            )
        ]
        contrasts.append(
            {
                **edge,
                "comparable": complete,
                "target_fundamental_difference": int(a["choice"]["target_style"] == "fundamental")
                - int(b["choice"]["target_style"] == "fundamental")
                if complete
                else None,
                "return_identity_flags": flags,
                "endpoint_reused_elsewhere_flags": elsewhere,
            }
        )
    valid = [r for r in rows if r["compatible"]]
    return {
        "evidence_kind": package["evidence_kind"],
        "rows": rows,
        "contrasts": contrasts,
        "valid_returns": len(valid),
        "scheduled_returns": 4,
        "logged_dispatch_attempts": len(dispatched),
        "complete": len(valid) == 4,
        "all_same_target": len({r["choice"]["target_style"] for r in valid}) == 1
        if len(valid) == 4
        else None,
        "return_identity_diagnostics": diagnostics,
        "independent_generation_count": None,
        "real_external_execution_authenticated": False,
        "human_alignment_established": False,
        "causal_mechanism_established": False,
        "not_scheduled_case_ids": package["not_scheduled_case_ids"],
        "limit": "Describes locally linked service returns only. Repeated cont"
        "ent is not cache proof; flagged zero differences are not abs"
        "ence-of-sensitivity evidence. No state flag authenticates ex"
        "ecution.",
    }
