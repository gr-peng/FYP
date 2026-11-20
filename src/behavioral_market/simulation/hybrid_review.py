"""Sequential, all-or-nothing style reviews; never a model order interface.

The injected adapter owns raw transport evidence. This layer persists every
prepared prompt and returned parsed result, including unapplied partial batches.
Protocol declarations describe an interface, not proof of remote inference.
"""

import asyncio
import copy
import json
import re

from behavioral_market.llm.schemas import StyleDecision

PROTOCOL = "sequential_no_retry_v1"


def validate_client(client):
    if (
        client is None
        or getattr(client, "hybrid_review_protocol", None) != PROTOCOL
        or getattr(client, "review_provenance", None) not in {"scripted", "live_strict"}
    ):
        raise ValueError("hybrid_style_llm requires an injected strict review adapter")


def append_event(path, event):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def validate_result(result, current_style):
    decision, meta = result
    if not isinstance(decision, StyleDecision) or not isinstance(meta, dict):
        raise ValueError("hybrid review returned an invalid result type")
    StyleDecision.model_validate(decision.model_dump())
    coherent = (decision.decision == "stay") == (decision.target_style == current_style)
    if (
        not coherent
        or meta.get("status") != "ok"
        or meta.get("fallback") is not False
        or meta.get("cache_hit") is not False
        or type(meta.get("repair_count")) is not int
        or meta["repair_count"] != 0
        or not isinstance(meta.get("request_hash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", meta["request_hash"]) is None
    ):
        raise ValueError("hybrid review failed strict response checks")


async def review_batch(client, prepared, current_styles, path):
    # Each entry consists only of immutable strings plus a private identity dict.
    # All prompts are persisted before the first dispatch in this review.
    batch = prepared[0][2]["session_date"]
    append_event(
        path,
        {
            "phase": "prepared",
            "batch": batch,
            "requests": [
                {"system": system, "user": user, "identity": identity}
                for system, user, identity in prepared
            ],
        },
    )
    results = []
    attempted = 0
    try:
        for index, ((system, user, identity), style) in enumerate(
            zip(prepared, current_styles, strict=True)
        ):
            append_event(path, {"phase": "attempt", "batch": batch, "index": index})
            attempted += 1
            result = copy.deepcopy(await client.decide(system, user, StyleDecision, dict(identity)))
            decision, meta = result
            # Save parsed responses before validating; raw envelopes belong to
            # the adapter's separate log and must be written before parsing.
            append_event(
                path,
                {
                    "phase": "returned_unapplied",
                    "batch": batch,
                    "index": index,
                    "decision": decision.model_dump()
                    if isinstance(decision, StyleDecision)
                    else None,
                    "metadata": {
                        key: meta.get(key)
                        for key in (
                            "status",
                            "fallback",
                            "cache_hit",
                            "repair_count",
                            "request_hash",
                        )
                    }
                    if isinstance(meta, dict)
                    else None,
                },
            )
            validate_result(result, style)
            results.append(result)
        append_event(path, {"phase": "validated_unapplied", "batch": batch, "count": len(results)})
    except (Exception, asyncio.CancelledError) as exc:
        append_event(
            path,
            {
                "phase": "incomplete",
                "batch": batch,
                "attempted": attempted,
                "valid_unapplied": len(results),
                "applied": 0,
                "stop_reason": type(exc).__name__,
                "unsent_identities": [entry[2] for entry in prepared[attempted:]],
            },
        )
        raise
    return results
