"""Reusable four-cell squares from the one immutable synthetic strategy bank."""

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

from behavioral_market.evaluation.style_conflict_package import choice_status, require
from behavioral_market.evaluation.style_mechanism_design import (
    ENVELOPE,
    canonical,
    contrast_edges,
    digest,
)
from behavioral_market.evaluation.style_probe_intake import aware_time
from behavioral_market.evaluation.style_reference_origin import BANK_BYTES, BANK_SHA256

NAMESPACE = "frozen-style-square-v1"
ENDPOINT = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"


def selected_ids(package):
    return tuple(c["case_id"] for c in package["cases"])


def source_descriptor(descriptor):
    require(set(descriptor) == {"path", "fingerprint"}, "source descriptor differs")
    path, fp = Path(descriptor["path"]), descriptor["fingerprint"]
    require(
        not path.is_absolute()
        and path.parts
        and path.parts[0] == "iterations"
        and ".." not in path.parts,
        "source path outside iterations",
    )
    require(
        set(fp) == {"bytes", "sha256"}
        and type(fp["bytes"]) is int
        and fp["bytes"] > 0
        and isinstance(fp["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", fp["sha256"]),
        "source fingerprint differs",
    )


def _core(bank, selection, run_id, output, evidence_kind, prior, budget_sources, anchor):
    raw = "".join(canonical(row) + "\n" for row in bank).encode()
    require(
        len(raw) == BANK_BYTES and hashlib.sha256(raw).hexdigest() == BANK_SHA256,
        "frozen source bank differs",
    )
    by_id = {r["case_id"]: r for r in bank}
    require(
        isinstance(selection, list)
        and len(selection) == len(set(selection)) == 4
        and all(cid in by_id for cid in selection),
        "four unique bank cases required",
    )
    edges = [
        e
        for e in contrast_edges(bank)
        if e["left_case"] in selection and e["right_case"] in selection
    ]
    require(
        len(edges) == 4 and sorted(Counter(e["factor"] for e in edges).values()) == [2, 2],
        "selection is not a complete two-factor square",
    )
    require(
        isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id),
        "invalid run ID",
    )
    require(Path(output).is_absolute(), "fixed absolute output required")
    require(
        evidence_kind in {"recorded_model_logs", "synthetic_tool_test"}, "invalid evidence kind"
    )
    require(type(prior) is int and 0 <= prior <= 96, "four requests exceed global100 cap")
    require(set(budget_sources) == {"grant", "prior_ledger"}, "budget sources differ")
    for descriptor in budget_sources.values():
        source_descriptor(descriptor)
    require(
        set(anchor) == {"case_id", "sources"} and anchor["case_id"] in selection,
        "historical anchor differs",
    )
    require(
        len(anchor["sources"]) == 3
        and {Path(p).name for p in anchor["sources"]}
        == {"llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl"}
        and len({str(Path(p).parent) for p in anchor["sources"]}) == 1,
        "three colocated historical logs required",
    )
    for path, fp in anchor["sources"].items():
        source_descriptor({"path": path, "fingerprint": fp})
    return {
        "protocol": NAMESPACE,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "execution_output": str(output),
        "endpoint": ENDPOINT,
        "model": ENVELOPE["model"],
        "source_bank_not_sent": deepcopy(bank),
        "cases": [deepcopy(by_id[cid]) for cid in selection],
        "contrasts": edges,
        "not_scheduled_case_ids": [r["case_id"] for r in bank if r["case_id"] not in selection],
        "prior_POSTs": prior,
        "additional_POST_limit": 4,
        "batch_cumulative_stop": prior + 4,
        "global_user_authorized_cap": 100,
        "transport_retries": 0,
        "format_repairs": 0,
        "max_concurrency": 1,
        "budget_sources": deepcopy(budget_sources),
        "historical_anchor": deepcopy(anchor),
        "primary_outcome": "target_style",
        "history_policy": (
            "Historical returns are diagnostics only, never pooled. Reuse is flagged; "
            "no salt or resend."
        ),
        "limit": (
            "Exploratory one-return-per-cell square; shared contrasts, assigned reference "
            "and labels. No psychological mechanism, human alignment, stable law or "
            "independent-generation proof."
        ),
    }


def make_package(bank, plan):
    core = _core(
        bank,
        plan["selected_case_ids"],
        plan["frozen_run_id"],
        plan["fixed_execution_output"],
        plan["evidence_kind"],
        plan["prior_used_completion_POSTs"],
        plan["budget_sources"],
        plan["historical_anchor"],
    )
    return {**core, "package_sha256": digest(core)}


def validate_package(package):
    core = _core(
        package["source_bank_not_sent"],
        list(selected_ids(package)),
        package["run_id"],
        package["execution_output"],
        package["evidence_kind"],
        package["prior_POSTs"],
        package["budget_sources"],
        package["historical_anchor"],
    )
    require(
        canonical(package) == canonical({**core, "package_sha256": digest(core)}),
        "package/source/context differs",
    )


def authorization_template(package):
    return {
        "decision": "not_allocated",
        "package_sha256": package["package_sha256"],
        "run_id": package["run_id"],
        "endpoint": ENDPOINT,
        "model": package["model"],
        "prior_POSTs": package["prior_POSTs"],
        "additional_POST_limit": 0,
        "cumulative_POST_cap": package["prior_POSTs"],
        "global_user_authorized_cap": 100,
        "specific_user_reply": None,
    }


def check_authorization(package, authorization):
    validate_package(package)
    require(authorization.get("decision") == "explicit_user_granted", "batch not allocated")
    for key in ("package_sha256", "run_id", "endpoint", "model", "budget_sources"):
        require(
            canonical(authorization.get(key)) == canonical(package[key]),
            "authorization binding differs",
        )
    for key, expected in (
        ("prior_POSTs", package["prior_POSTs"]),
        ("additional_POST_limit", 4),
        ("cumulative_POST_cap", package["batch_cumulative_stop"]),
        ("global_user_authorized_cap", 100),
    ):
        require(
            type(authorization.get(key)) is int and authorization[key] == expected, "budget differs"
        )
    require(
        isinstance(authorization.get("specific_user_reply"), str)
        and authorization["specific_user_reply"].strip(),
        "actual user reply required",
    )


def historical_diagnostics(result, old_body, old_payload_sha256, package):
    cid = package["historical_anchor"]["case_id"]
    case = next(c for c in package["cases"] if c["case_id"] == cid)
    require(digest(case["payload"]) == old_payload_sha256, "historical anchor payload differs")
    status, mode, _ = choice_status(old_body, case)
    require(
        status == "valid_linked_answer" and mode == "no_returned_reasoning_observed",
        "historical anchor invalid",
    )
    old = {
        "provider_id": old_body.get("id"),
        "created": old_body.get("created"),
        "body_sha256": digest(old_body),
        "text_sha256": hashlib.sha256(
            old_body["choices"][0]["message"]["content"].encode()
        ).hexdigest(),
    }
    matches = {}
    for row in result["rows"]:
        flags = [
            k
            for k in ("provider_id", "body_sha256", "text_sha256")
            if isinstance(old[k], str) and old[k] and row[k] == old[k]
        ]
        if all(
            old[k] is not None and row[k] == old[k]
            for k in ("provider_id", "created", "text_sha256")
        ):
            flags.append("same_id_created_text")
        matches[row["case_id"]] = flags
    return {
        "old_anchor_not_in_denominator": True,
        "old_anchor_case_id": cid,
        "matches_by_new_case": matches,
        "independent_generation_established": False,
    }


def request_identity(case, package):
    return {
        "kind": "style_bank_square",
        "probe_namespace": NAMESPACE,
        "probe_run_id": package["run_id"],
        "evidence_kind": package["evidence_kind"],
        "case_id": case["case_id"],
        "position": list(selected_ids(package)).index(case["case_id"]) + 1,
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
        "slots": [{"case_id": k, "status": "unsubmitted"} for k in selected_ids(package)],
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
            list(mapped) == list(selected_ids(package)[: len(mapped)]),
            "logs must follow fixed prefix order",
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
                    start
                    >= aware_time(received[selected_ids(package)[i - 1]]["response_received_at"]),
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
