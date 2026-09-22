"""Version-pinned TypeSafe Jev choice transport, independent of market accounting."""

import asyncio
import json
import math
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.client import RequestPacer


class JevContractError(RuntimeError):
    pass


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise JevContractError("probability must be a numeric JSON value")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise JevContractError("probability must be finite and in [0, 1]")
    return float(value)


def validate_answer(body, question_id, criteria, expected_model):
    if not isinstance(body, dict) or body.get("model") != expected_model:
        raise JevContractError("resolved model differs from the pinned model")
    try:
        answer = body["answers"][question_id]
        probabilities = answer["probabilities"]
        choice = answer["choice"]
        confidence = _probability(answer["confidence"])
    except (KeyError, TypeError) as exc:
        raise JevContractError("missing required choice response fields") from exc
    if answer.get("type") != "choice" or not isinstance(probabilities, dict):
        raise JevContractError("response is not a choice probability distribution")
    if set(probabilities) != set(criteria) or choice not in criteria:
        raise JevContractError("answer options differ from the requested action set")
    raw = {key: _probability(value) for key, value in probabilities.items()}
    mass = sum(raw.values())
    if abs(mass - 1) > 1e-3:
        raise JevContractError("choice probabilities do not sum approximately to one")
    if raw[choice] + 1e-8 < max(raw.values()):
        raise JevContractError("API choice is not a maximum-probability option")
    return {
        "choice": choice,
        "raw_probabilities": raw,
        "probabilities": {key: value / mass for key, value in raw.items()},
        "probability_mass_before_normalization": mass,
        "normalization_applied": mass != 1.0,
        "confidence": confidence,
        "resolved_model": body["model"],
        "usage": body.get("usage", {}),
    }


def sample_action(probabilities, uniform):
    """Explicit stochastic policy; does not call or masquerade as the API argmax."""
    if not math.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("sample must lie in [0, 1)")
    if not probabilities or abs(sum(probabilities.values()) - 1) > 1e-8:
        raise ValueError("normalized probability distribution required")
    probabilities = {key: _probability(value) for key, value in probabilities.items()}
    cumulative = 0.0
    for key in sorted(probabilities):
        cumulative += probabilities[key]
        if uniform < cumulative:
            return key
    return sorted(probabilities)[-1]


class JevClient:
    def __init__(
        self,
        api_key,
        output,
        cache,
        model="jev-1.13.0",
        max_rps=2.0,
        endpoint="https://api.typesafe.ai/v1/systemone",
        transport=None,
        cache_only=False,
        trust_env=False,
    ):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.typesafe.ai"
            or parsed.port not in (None, 443)
            or parsed.path != "/v1/systemone"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("TypeSafe credentials may only be sent to the official endpoint")
        if model in {"jev-latest", "jev-preview"}:
            raise ValueError("research runs require a version-pinned Jev model")
        if not api_key and not cache_only:
            raise RuntimeError("TypeSafe API credential is absent; no substitute model is used")
        self.key = api_key or ""
        self.output, self.cache = Path(output), Path(cache)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model, self.endpoint, self.cache_only = model, endpoint, cache_only
        self.pacer = RequestPacer(max_rps)
        self.http = httpx.AsyncClient(
            headers={"Authorization": "Bearer " + self.key} if self.key else {},
            transport=transport,
            timeout=60,
            trust_env=trust_env,
        )

    async def close(self):
        await self.http.aclose()

    def log(self, name, value):
        text = json.dumps(value, allow_nan=False)
        if self.key:
            text = text.replace(self.key, "[REDACTED]")
        with (self.output / name).open("a") as handle:
            handle.write(text + "\n")

    async def evaluate_choice(self, state, instructions, criteria, identity):
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError("choice comparison requires 2–255 options")
        question_id = "decision"
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                question_id: {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        }
        key = digest(json.dumps([payload, identity], sort_keys=True).encode())
        cache_path = self.cache / f"{key}.json"
        self.log("requests.jsonl", {"identity": identity, "request_hash": key, "payload": payload})
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            answer = validate_answer(raw, question_id, criteria, self.model)
            return {**answer, "cache_hit": True, "request_hash": key, "latency_seconds": None}
        if self.cache_only:
            raise RuntimeError("missing Jev response in cache-only replay")
        for attempt in range(4):
            await self.pacer.wait()
            started = time.perf_counter()
            status, body, request_id, retry_after = "transport_error", None, None, None
            try:
                response = await self.http.post(self.endpoint, json=payload)
                status = response.status_code
                request_id = response.headers.get("x-typesafe-request-id")
                retry_after = response.headers.get("retry-after")
                if status == 200:
                    body = response.json()
            except (httpx.HTTPError, ValueError):
                pass
            latency = time.perf_counter() - started
            self.log(
                "attempts.jsonl",
                {
                    "identity": identity,
                    "request_hash": key,
                    "attempt": attempt,
                    "status": status,
                    "request_id": request_id,
                    "latency_seconds": latency,
                    "response": body,
                },
            )
            if body is not None:
                answer = validate_answer(body, question_id, criteria, self.model)
                write_json(cache_path, body)
                return {
                    **answer,
                    "cache_hit": False,
                    "request_hash": key,
                    "latency_seconds": latency,
                    "retry_count": attempt,
                    "request_id": request_id,
                }
            if status in {400, 401, 402, 403, 404, 422}:
                raise RuntimeError(f"Jev API HTTP {status}; no automatic model substitution")
            if attempt < 3:
                delay = min(2**attempt, 8)
                if status == 429:
                    try:
                        delay = max(delay, min(float(retry_after), 60))
                    except (TypeError, ValueError):
                        delay = 10
                    await self.pacer.cooldown(delay)
                else:
                    await asyncio.sleep(delay)
        raise RuntimeError("Jev API retries exhausted")
