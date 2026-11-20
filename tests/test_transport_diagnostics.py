"""Synthetic errors and transports only; no credentials or real networking."""

import asyncio
import errno
import json
import socket
import ssl

import httpx
import pytest

from behavioral_market.llm.transport_diagnostics import (
    COMPLETION,
    REGISTRY,
    DiagnosticClient,
    safe_causes,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No real networking in diagnostics tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)


@pytest.mark.parametrize(
    "error,category",
    [
        (ssl.SSLCertVerificationError(1, "fake"), "tls_certificate"),
        (ssl.SSLError(1, "fake"), "tls_protocol"),
        (socket.gaierror(-2, "fake"), "dns_resolution"),
        (OSError(errno.ECONNREFUSED, "fake"), "connection_refused"),
        (OSError(errno.ENETUNREACH, "fake"), "network_unreachable"),
        (httpx.ProxyError("fake"), "proxy"),
        (httpx.ConnectTimeout("fake"), "timeout"),
        (httpx.ConnectError("fake"), "connect_failure"),
        (asyncio.CancelledError(), "cancelled"),
        (httpx.ReadError("fake"), "other_transport"),
        (ValueError("fake"), "unknown_error"),
    ],
)
def test_fixed_categories(error, category):
    assert safe_causes(error) == [category]


def test_causal_graph_is_bounded_cycle_safe_and_never_serializes_fake_secrets():
    secret = "FAKE_KEY_https://user:password@host/path?token=NOT_REAL"
    error = httpx.ConnectError(secret)
    error.__cause__ = ExceptionGroup(
        secret, [socket.gaierror(-2, secret), ssl.SSLCertVerificationError(1, secret)]
    )
    assert safe_causes(error) == [
        "connect_failure",
        "unknown_error",
        "dns_resolution",
        "tls_certificate",
    ]
    error.__cause__ = error
    assert safe_causes(error) == ["connect_failure"]
    root = error
    for _ in range(20):
        child = ValueError(secret)
        child.__cause__ = root
        root = child
    assert len(safe_causes(root)) == 4
    assert secret not in json.dumps(safe_causes(root))


@pytest.mark.parametrize("method,url", [("GET", REGISTRY), ("POST", COMPLETION)])
def test_transport_failure_is_recorded_once_without_secret_or_payload(tmp_path, method, url):
    calls = []
    secret = "FICTIONAL_KEY_DO_NOT_RECORD"

    async def respond(request):
        calls.append(request.method)
        error = httpx.ConnectError(secret + " https://user:password@host")
        error.__cause__ = OSError(errno.ECONNREFUSED, secret)
        raise error

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            wrapped = DiagnosticClient(client, tmp_path / "events", route="synthetic_transport")
            with pytest.raises(httpx.ConnectError):
                await getattr(wrapped, method.lower())(url, headers={"Authorization": secret})

    asyncio.run(run())
    assert calls == [method]
    contents = [p.read_text() for p in sorted((tmp_path / "events").iterdir())]
    assert len(contents) == 2
    assert secret not in "".join(contents) and "password" not in "".join(contents)
    receipt = json.loads(next((tmp_path / "events").glob("*-after.json")).read_text())
    assert receipt["outcome"] == "exception"
    assert receipt["causes"] == ["connect_failure", "connection_refused"]


def test_status_only_no_headers_body_or_retry(tmp_path):
    calls = []

    async def respond(request):
        calls.append(request.method)
        return httpx.Response(503, headers={"Authorization": "FAKE_SECRET"}, text="FAKE_BODY")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            wrapped = DiagnosticClient(client, tmp_path / "events", route="synthetic_transport")
            response = await wrapped.get(REGISTRY)
            assert response.status_code == 503
            with pytest.raises(ValueError):
                await wrapped.get("https://unexpected.example/")

    asyncio.run(run())
    assert calls == ["GET"]
    contents = "".join(p.read_text() for p in (tmp_path / "events").iterdir())
    assert "FAKE_SECRET" not in contents and "FAKE_BODY" not in contents
    assert '"http_status": 503' in contents


@pytest.mark.parametrize("when,expected_calls", [("before", 0), ("after", 1)])
def test_receipt_failure_does_not_repeat_network_request(
    tmp_path, monkeypatch, when, expected_calls
):
    calls = []

    async def respond(request):
        calls.append(request.method)
        return httpx.Response(200, json={"data": []})

    original = DiagnosticClient._record

    def fail(self, number, phase, event, **fields):
        if event == when:
            raise OSError("FICTIONAL_DISK_FAILURE")
        return original(self, number, phase, event, **fields)

    monkeypatch.setattr(DiagnosticClient, "_record", fail)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            wrapped = DiagnosticClient(client, tmp_path / "events", route="synthetic_transport")
            with pytest.raises(OSError):
                await wrapped.get(REGISTRY)

    asyncio.run(run())
    assert len(calls) == expected_calls


def test_cancelled_request_has_a_single_preserved_receipt(tmp_path):
    calls = []

    async def respond(request):
        calls.append(request.method)
        raise asyncio.CancelledError

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            wrapped = DiagnosticClient(client, tmp_path / "events", route="synthetic_transport")
            with pytest.raises(asyncio.CancelledError):
                await wrapped.post(COMPLETION, json={"fictional": True})

    asyncio.run(run())
    assert calls == ["POST"]
    after = json.loads(next((tmp_path / "events").glob("*-after.json")).read_text())
    assert after["causes"] == ["cancelled"]
