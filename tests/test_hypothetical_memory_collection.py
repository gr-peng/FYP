"""Local fake transport, real durable logs, and isolated temporary budget ledgers."""

import asyncio
import socket
from pathlib import Path

import dotenv
import httpx
import pytest
from test_conditional_order_memory import no_external_access as no_external_access
from test_hypothetical_memory_probe import response

from scripts import collect_hypothetical_memory_probe as tool

ORIGINAL_SOCKET = socket.socket


@pytest.fixture(autouse=True)
def allow_asyncio_self_pipe(request, monkeypatch):
    request.getfixturevalue("no_external_access")

    def local_socket(family=socket.AF_INET, *args, **kwargs):
        if family != socket.AF_UNIX:
            pytest.fail("Internet sockets forbidden in fake transport tests")
        return ORIGINAL_SOCKET(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", local_socket)


class FakeHTTP:
    def __init__(self, failure=None):
        self.failure, self.posts, self.gets = failure, 0, 0
        self._transport = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def _transport_for_url(self, _url):
        return self._transport

    async def get(self, url):
        assert url == tool.REGISTRY
        self.gets += 1
        return httpx.Response(
            200, json={"data": [] if self.failure == "registry" else [{"id": tool.study.MODEL}]}
        )

    async def post(self, url, json):
        assert url == tool.COMPLETION
        assert json["model"] == tool.study.MODEL
        self.posts += 1
        body, headers = response(), {}
        if self.posts == 3:
            if self.failure == "network":
                raise httpx.ConnectError("synthetic failure")
            if self.failure == "cancel":
                raise asyncio.CancelledError
            if self.failure == "format":
                body["choices"][0]["message"]["content"] = '{"A":0.8,"B":0.8}'
            if self.failure == "identity":
                body["model"] = "different-model"
            if self.failure == "cache":
                headers["x-litellm-cache-key"] = "synthetic-cache-evidence"
            if self.failure == "mode":
                body["choices"][0]["message"]["reasoning_content"] = "unexpected"
        # Intentionally identical signatures; the collector must retain, not retry.
        return httpx.Response(200, json=body, headers=headers)


def test_complete_fake_collection_keeps_repeated_signatures_and_all_records(tmp_path):
    client = FakeHTTP()
    result = asyncio.run(tool.collect(client, tmp_path / "logs", tool.study.materials()))
    assert client.posts == 48 and client.gets == 1
    assert result["complete"] and result["compatible_responses"] == 48
    assert result["independent_generation_count"] is None
    assert result["repeated_signature_groups"] == [list(range(1, 49))]
    assert len(tool.rows(tmp_path / "logs/parsed.jsonl")) == 48
    assert tool.read(tmp_path / "logs/budget_progress.json")["used_completion_POSTs"] == 397
    assert result == tool.check_logs(tmp_path / "logs", tool.study.materials())


@pytest.mark.parametrize("failure", ["network", "cancel", "format", "identity", "cache", "mode"])
def test_first_failure_stops_without_retry_or_repair_and_keeps_raw_response(tmp_path, failure):
    client = FakeHTTP(failure)
    with pytest.raises((ValueError, asyncio.CancelledError)):
        asyncio.run(tool.collect(client, tmp_path / "logs", tool.study.materials()))
    assert client.posts == 3
    assert len(tool.rows(tmp_path / "logs/attempts.jsonl")) == 3
    assert len(tool.rows(tmp_path / "logs/responses.jsonl")) == 3
    assert len(tool.rows(tmp_path / "logs/parsed.jsonl")) == 2
    result = tool.check_logs(tmp_path / "logs", tool.study.materials())
    assert not result["complete"] and result["compatible_responses"] == 2
    assert tool.read(tmp_path / "logs/budget_progress.json")["used_completion_POSTs"] == 352


def test_failed_registry_never_posts(tmp_path):
    client = FakeHTTP("registry")
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(tool.collect(client, tmp_path / "logs", tool.study.materials()))
    assert client.posts == 0
    assert tool.rows(tmp_path / "logs/attempts.jsonl") == []


def test_durable_attempt_failure_prevents_network_dispatch(tmp_path, monkeypatch):
    original = tool.os.fsync

    def failing(fd):
        if str(Path(f"/proc/self/fd/{fd}").resolve()).endswith("attempts.jsonl"):
            raise OSError("synthetic durable attempt failure")
        return original(fd)

    monkeypatch.setattr(tool.os, "fsync", failing)
    client = FakeHTTP()
    with pytest.raises(OSError):
        asyncio.run(tool.collect(client, tmp_path / "logs", tool.study.materials()))
    assert client.posts == 0
    # The attempted durable record may exist: conservatively consume it if uncertain.
    assert len(tool.rows(tmp_path / "logs/attempts.jsonl")) == 1


def executable(tmp_path, monkeypatch, failure=None):
    folder = tmp_path / "synthetic_run"
    plan = tool.prepare(folder)
    tool.save(
        folder / "independent_preflight.json",
        {
            "accepted_for_execution": True,
            "mustfix": [],
            "plan_sha256": tool.sha(folder / "plan.json"),
            "evidence_kind": "synthetic_test_fixture_not_actual_authorization",
        },
    )
    budget = tmp_path / "budget.json"
    ledger = {
        "global_authorized_completion_POST_cap": 500,
        "used_completion_POSTs": 349,
        "reserved_next_completion_POSTs": 0,
        "available_unreserved_POSTs": 151,
        "active_run_id": None,
        "version": 10,
        "status": "synthetic_ledger",
    }
    tool.save(budget, ledger)
    monkeypatch.setattr(tool.shared, "BUDGET_PATH", budget)
    monkeypatch.setattr(tool, "FOLDER", folder)
    client = FakeHTTP(failure)
    monkeypatch.setattr(tool.shared, "school_client", lambda *_: client)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_: {"AIGC_API_KEY": "synthetic-not-a-key"})
    return folder, budget, client, plan


def test_full_fake_execution_updates_only_temporary_budget_and_rejects_resume(
    tmp_path, monkeypatch
):
    folder, budget, client, _ = executable(tmp_path, monkeypatch)
    result = asyncio.run(tool.execute(folder))
    assert result["new_model_POSTs"] == 48 and client.posts == 48
    assert result["requested_model"] == tool.study.MODEL
    assert result["run_id"] == tool.study.RUN_ID
    assert tool.study.MODEL in (folder / "REPORT.txt").read_text()
    ledger = tool.read(budget)
    assert ledger["used_completion_POSTs"] == 397
    assert ledger["available_unreserved_POSTs"] == 103
    assert ledger["active_run_id"] is None and ledger["version"] == 11
    assert (
        tool.read(folder / "execution_state.json")["status"] == "complete_pending_independent_audit"
    )
    assert (folder / "REPORT.txt").exists()
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    assert client.posts == 48


@pytest.mark.parametrize(
    "failure", ["approval", "quota", "format", "source", "terminal_state", "terminal_budget"]
)
def test_execution_failure_cannot_publish_success_or_exceed_budget(tmp_path, monkeypatch, failure):
    folder, budget, client, _ = executable(tmp_path, monkeypatch, failure)
    original_check, original_save = tool.checked_plan, tool.save
    checked = []
    if failure == "approval":
        tool.save(folder / "independent_preflight.json", {"accepted_for_execution": False})
    elif failure == "quota":
        ledger = tool.read(budget)
        ledger["used_completion_POSTs"] = 480
        ledger["available_unreserved_POSTs"] = 20
        tool.save(budget, ledger)
    elif failure == "source":

        def check(path):
            checked.append(True)
            if len(checked) == 2:
                raise ValueError("synthetic source drift")
            return original_check(path)

        monkeypatch.setattr(tool, "checked_plan", check)
    elif failure in {"terminal_state", "terminal_budget"}:

        def failed_save(path, value):
            if failure == "terminal_state" and path.name == "execution_state.json":
                raise OSError("synthetic status persistence failure")
            if failure == "terminal_budget" and path == budget and value["active_run_id"] is None:
                raise OSError("synthetic budget persistence failure")
            return original_save(path, value)

        monkeypatch.setattr(tool, "save", failed_save)
    with pytest.raises((ValueError, OSError)):
        asyncio.run(tool.execute(folder))
    expected = 0 if failure in {"approval", "quota"} else 3 if failure == "format" else 48
    assert client.posts == expected
    assert not (folder / "summary.json").exists()
    assert not (folder / "REPORT.txt").exists()
    ledger = tool.read(budget)
    if failure == "terminal_budget":
        assert ledger["active_run_id"] == tool.study.RUN_ID
        assert ledger["reserved_next_completion_POSTs"] == 48
        assert ledger["available_unreserved_POSTs"] == 103
    elif failure != "quota":
        assert ledger["used_completion_POSTs"] == 349 + expected


def test_tampered_frozen_input_rejected_before_any_key_or_model_access(tmp_path, monkeypatch):
    folder, budget, client, _ = executable(tmp_path, monkeypatch)
    path = folder / "materials.json"
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(
        dotenv, "dotenv_values", lambda *_: pytest.fail("key access before validation")
    )
    with pytest.raises(ValueError, match="frozen input"):
        asyncio.run(tool.execute(folder))
    assert client.posts == client.gets == 0
    assert tool.read(budget)["used_completion_POSTs"] == 349
