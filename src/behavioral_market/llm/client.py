import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from behavioral_market.data.archive import write_json

from .prompts import PROMPT_VERSION
from .schemas import OrderDecision, StyleDecision, hold


class FatalAPIError(RuntimeError):
    pass


class RequestPacer:
    """One shared, evenly spaced dispatch schedule across all concurrent runs."""

    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.next_dispatch = 0.0
        self.lock = asyncio.Lock()
        self.requests_per_second = requests_per_second

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            if self.next_dispatch > now:
                await asyncio.sleep(self.next_dispatch - now)
            self.next_dispatch = max(now, self.next_dispatch) + self.interval

    async def cooldown(self, seconds):
        async with self.lock:
            self.next_dispatch = max(self.next_dispatch, time.monotonic() + seconds)


class LLMClient(Protocol):
    async def decide(self, system: str, user: str, schema: type[BaseModel], identity: dict): ...


class MockLLMClient:
    """Deterministic JSON through the same schema contract; never claims live inference."""

    async def decide(self, system, user, schema, identity):
        context = json.loads(user)
        state = context["agent_state"]
        if schema is StyleDecision:
            gap = state["counterfactual"]["gap"]
            current = state["current_style"]
            switch = gap > 0.01
            raw = {
                "decision": "switch" if switch else "stay",
                "target_style": ("technical" if current == "fundamental" else "fundamental")
                if switch
                else current,
                "rationale": "deterministic mock",
            }
        else:
            number = int(hashlib.sha256((user + system).encode()).hexdigest()[:8], 16)
            side = "buy" if number % 2 else "sell"
            raw = {
                "action": side,
                "quantity": 1 + number % 3,
                "limit_price": round(
                    context["observation"]["last_close"] * (1.01 if side == "buy" else 0.99), 2
                ),
                "rationale": "deterministic mock",
                "sentiment": "bullish" if side == "buy" else "bearish",
            }
        return schema.model_validate(raw), {"status": "mock", "provider": "mock"}


class SiliconFlowClient:
    def __init__(
        self,
        config,
        api_key,
        output: Path,
        cache: Path,
        semaphore=None,
        pacer=None,
        cache_only=False,
    ):
        if not api_key and not cache_only:
            raise FatalAPIError("SILICONFLOW_API_KEY is missing")
        if config["base_url"].rstrip("/") != "https://api.siliconflow.cn/v1":
            raise FatalAPIError("SiliconFlow credentials may only be sent to api.siliconflow.cn")
        self.config, self.output, self.cache = config, output, cache
        self._key = api_key or ""
        self.semaphore = semaphore or asyncio.Semaphore(config["max_concurrency"])
        self.pacer = pacer
        self.cache_only = cache_only
        self.http = httpx.AsyncClient(
            base_url=config["base_url"].rstrip("/") + "/",
            headers={"Authorization": "Bearer " + api_key} if api_key else {},
            timeout=config["timeout_seconds"],
        )
        output.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)

    async def close(self):
        await self.http.aclose()

    def log(self, name, record):
        text = json.dumps(record, ensure_ascii=False, default=str, allow_nan=False)
        if self._key:
            text = text.replace(self._key, "[REDACTED]")
        with (self.output / name).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    async def verify_model(self):
        try:
            response = await self.http.get("models", params={"sub_type": "chat"})
        except httpx.HTTPError as exc:
            raise FatalAPIError(f"model registry transport failure: {type(exc).__name__}") from None
        if response.status_code != 200:
            raise FatalAPIError(f"model registry HTTP {response.status_code}")
        models = response.json()
        write_json(self.output / "model_registry.json", models)
        if self.config["model"] not in {row["id"] for row in models.get("data", [])}:
            raise FatalAPIError("requested model is absent from SiliconFlow model registry")
        return self.config["model"]

    async def completion(self, messages, identity):
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "enable_thinking": self.config["enable_thinking"],
            "response_format": {"type": "json_object"},
        }
        key = hashlib.sha256(
            json.dumps([payload, identity, PROMPT_VERSION], sort_keys=True).encode()
        ).hexdigest()
        cache_file = self.cache / (key + ".json")
        self.log("llm_requests.jsonl", {**identity, "request_hash": key, "payload": payload})
        if cache_file.exists():
            body = json.loads(cache_file.read_text())
            if body.get("model") != self.config["model"]:
                raise FatalAPIError("cached response model does not match configured model")
            self.log(
                "llm_responses.jsonl",
                {**identity, "request_hash": key, "cached": True, "status": "ok", "response": body},
            )
            return body, {"cache_hit": True, "request_hash": key}
        if self.cache_only:
            raise FatalAPIError(f"cache-only replay is missing response {key}")
        attempts = self.config.get("transport_retries", 5) + 1
        for attempt in range(attempts):
            started = time.perf_counter()
            status, trace, body = "transport_error", None, None
            dispatched_at = None
            retry_after = None
            try:
                async with self.semaphore:
                    if self.pacer:
                        await self.pacer.wait()
                    dispatched_at = datetime.now(UTC).isoformat()
                    self.log(
                        "llm_attempts.jsonl",
                        {
                            **identity,
                            "request_hash": key,
                            "retry_count": attempt,
                            "dispatched_at": dispatched_at,
                        },
                    )
                    response = await self.http.post("chat/completions", json=payload)
                status = response.status_code
                trace = response.headers.get("x-siliconcloud-trace-id")
                value = response.headers.get("retry-after")
                retry_after = (
                    float(value) if value and value.replace(".", "", 1).isdigit() else None
                )
                if status == 200:
                    body = response.json()
            except asyncio.CancelledError:
                if dispatched_at is not None:
                    self.log(
                        "llm_responses.jsonl",
                        {
                            **identity,
                            "request_hash": key,
                            "requested_at": dispatched_at,
                            "response_received_at": datetime.now(UTC).isoformat(),
                            "cached": False,
                            "http_status": "cancelled_in_flight",
                            "retry_count": attempt,
                            "latency_seconds": time.perf_counter() - started,
                            "response": None,
                        },
                    )
                raise
            except (httpx.HTTPError, ValueError):
                pass
            meta = {
                **identity,
                "request_hash": key,
                "requested_at": dispatched_at or datetime.now(UTC).isoformat(),
                "response_received_at": datetime.now(UTC).isoformat(),
                "cached": False,
                "http_status": status,
                "trace_id": trace,
                "retry_count": attempt,
                "latency_seconds": time.perf_counter() - started,
                "retry_after_seconds": retry_after,
            }
            self.log("llm_responses.jsonl", {**meta, "response": body})
            if body is not None:
                if body.get("model") != self.config["model"]:
                    raise FatalAPIError("response model does not match configured model")
                write_json(cache_file, body)
                return body, {
                    "cache_hit": False,
                    "request_hash": key,
                    "usage": body.get("usage", {}),
                }
            if status in (400, 401, 402, 403, 404, 422):
                raise FatalAPIError(
                    f"chat/completions HTTP {status}; no automatic model substitution"
                )
            if status == 429 and self.pacer:
                await self.pacer.cooldown(
                    retry_after if retry_after is not None else max(30, min(2**attempt, 16))
                )
            if attempt + 1 < attempts:
                if status != 429 or not self.pacer:
                    await asyncio.sleep(min(2**attempt, 16))
        return None, {"status": "transport_exhausted", "request_hash": key}

    async def decide(self, system, user, schema, identity):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        error = "invalid_json"
        for repair in range(2):
            body, meta = await self.completion(messages, {**identity, "repair": repair})
            if body is None:
                error = "transport_exhausted"
                break
            try:
                raw = body["choices"][0]["message"]["content"]
                decision = schema.model_validate_json(raw)
                return decision, {**meta, "status": "ok", "repair_count": repair}
            except (ValidationError, KeyError, IndexError, TypeError):
                error = "invalid_json_or_schema"
                if repair == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your response failed JSON/schema validation. "
                            "Return only the required JSON object with all required fields "
                            "and types. Do not add any other fields.",
                        }
                    )
        if schema is OrderDecision:
            decision = hold(error)
        else:
            current = json.loads(user)["agent_state"]["current_style"]
            decision = StyleDecision(decision="stay", target_style=current, rationale=error)
        return decision, {"status": error, "fallback": True}
