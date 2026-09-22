"""A strict generative-LLM probability comparator for the Jev experiments.

Transport, cache isolation and model-version checks are supplied by the existing
OpenAICompatibleClient. This wrapper never calls its trading-specific fallback.
"""

import json
import math
import time


def validate_probabilities(raw: str, criteria: dict) -> tuple[dict, dict]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"probabilities"}:
        raise ValueError("expected exactly the probabilities field")
    probabilities = value["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
        raise ValueError("probability keys differ from available choices")
    if any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("probabilities must be finite numbers in [0,1]")
    mass = sum(probabilities.values())
    if abs(mass - 1) > 0.001:
        raise ValueError("probabilities do not sum to one")
    return {k: p / mass for k, p in probabilities.items()}, {
        "raw_probability_mass": mass,
        "normalized": mass != 1,
    }


class LLMProbabilityClient:
    def __init__(self, transport):
        self.transport = transport

    async def evaluate_choice(self, state, instructions, criteria, identity):
        if len(criteria) < 2 or any(not isinstance(key, str) for key in criteria):
            raise ValueError("at least two named criteria required")
        messages = [
            {
                "role": "system",
                "content": "Estimate a human participant's choice probabilities using only the "
                "supplied visible state and behavioral context. "
                "Follow the prediction instructions. "
                "Return only JSON with exactly one field, probabilities, mapping every supplied "
                "choice label to a finite number between 0 and 1. The numbers must sum to 1.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "state": state,
                        "instructions": instructions,
                        "criteria": criteria,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        started = time.perf_counter()
        calls, raw, status = [], None, "invalid_probability_schema"
        for repair in range(2):
            body, meta = await self.transport.completion(
                messages, {**identity, "repair": repair, "elicitation": "probability-v1"}
            )
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            calls.append({**meta, "usage": usage})
            if body is None:
                status = "transport_exhausted"
                break
            try:
                raw = body["choices"][0]["message"]["content"]
                probabilities, normalization = validate_probabilities(raw, criteria)
                return probabilities, {
                    "status": "ok",
                    "raw_response": raw,
                    "repair_count": repair,
                    "calls": calls,
                    **normalization,
                    "elapsed_including_queue_seconds": time.perf_counter() - started,
                }
            except (ValueError, KeyError, IndexError, TypeError):
                if repair == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "The response failed validation. Return only "
                            '{"probabilities": {...}} with these exact choice keys: '
                            + json.dumps(list(criteria))
                            + ". Use finite numeric probabilities in [0,1] summing to 1.",
                        }
                    )
        return None, {
            "status": status,
            "raw_response": raw,
            "repair_count": len(calls) - 1,
            "calls": calls,
            "elapsed_including_queue_seconds": time.perf_counter() - started,
        }
