"""Record request stages without exception strings, credentials, URLs or bodies."""

from __future__ import annotations

import asyncio
import errno
import json
import socket
import ssl
from datetime import UTC, datetime

import httpx

REGISTRY = "https://aigc-api.hkust-gz.edu.cn/api/v2/models"
COMPLETION = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"


def safe_causes(error):
    """Bounded causal graph. Map only fixed categories; never serialize messages."""
    pending, seen, results = [(error, 0)], set(), []
    while pending and len(results) < 12:
        current, depth = pending.pop(0)
        if depth >= 4 or id(current) in seen:
            continue
        seen.add(id(current))
        category = "unknown_error"
        if isinstance(current, ssl.SSLCertVerificationError):
            category = "tls_certificate"
        elif isinstance(current, ssl.SSLError):
            category = "tls_protocol"
        elif isinstance(current, socket.gaierror):
            category = "dns_resolution"
        elif isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            category = "connection_refused"
        elif isinstance(current, OSError) and current.errno in (
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        ):
            category = "network_unreachable"
        elif isinstance(current, httpx.ProxyError):
            category = "proxy"
        elif isinstance(current, (httpx.TimeoutException, TimeoutError)):
            category = "timeout"
        elif isinstance(current, httpx.ConnectError):
            category = "connect_failure"
        elif isinstance(current, asyncio.CancelledError):
            category = "cancelled"
        elif isinstance(current, httpx.TransportError):
            category = "other_transport"
        results.append(category)
        children = []
        if current.__cause__ is not None:
            children.append(current.__cause__)
        elif current.__context__ is not None:
            children.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            children.extend(current.exceptions[:12])
        pending.extend((child, depth + 1) for child in children)
    return results


class DiagnosticClient:
    """Delegate once to an existing client, retaining its proxy/TLS configuration."""

    def __init__(self, client, output, *, route):
        if route not in ("environment_proxy", "default_direct", "synthetic_transport"):
            raise ValueError("unknown route category")
        output.mkdir(parents=True, exist_ok=False)
        self.client, self.output, self.route, self.sequence = client, output, route, 0

    def _record(self, number, phase, event, **fields):
        path = self.output / f"{number:02d}-{phase}-{event}.json"
        value = {
            "request_number": number,
            "phase": phase,
            "event": event,
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "route": self.route,
            "tls_verification_configured": True,
            "automatic_retries": 0,
            **fields,
        }
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")

    async def _request(self, method, url, **kwargs):
        phase = {("GET", REGISTRY): "registry_get", ("POST", COMPLETION): "completion_post"}.get(
            (method, str(url))
        )
        if phase is None:
            raise ValueError("unexpected request destination or method")
        self.sequence += 1
        number = self.sequence
        self._record(number, phase, "before")
        try:
            response = await getattr(self.client, method.lower())(url, **kwargs)
        except (Exception, asyncio.CancelledError) as error:
            self._record(number, phase, "after", outcome="exception", causes=safe_causes(error))
            raise
        self._record(
            number, phase, "after", outcome="http_response", http_status=response.status_code
        )
        return response

    async def get(self, url, **kwargs):
        return await self._request("GET", url, **kwargs)

    async def post(self, url, **kwargs):
        return await self._request("POST", url, **kwargs)
