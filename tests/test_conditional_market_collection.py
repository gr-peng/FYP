"""Fake-only dynamic collection, quota, durability and exact replay."""

import asyncio
from copy import deepcopy

import dotenv
import httpx
import pytest
from test_conditional_order_memory import no_external_access as no_external_access
from test_hypothetical_memory_collection import allow_asyncio_self_pipe as allow_asyncio_self_pipe

from scripts import collect_conditional_market_pilot as tool


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
        self.posts += 1
        if self.posts == 2 and self.failure == "network":
            raise httpx.ConnectError("invented failure")
        body = {
            "id": "fake-" + str(self.posts),
            "created": 1,
            "model": tool.study.MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"A":0.5,"B":0.5}'},
                }
            ],
        }
        headers = {}
        if self.posts == 2:
            if self.failure == "format":
                body["choices"][0]["message"]["content"] = '{"A":0.8,"B":0.8}'
            if self.failure == "identity":
                body["model"] = "wrong"
            if self.failure == "mode":
                body["choices"][0]["message"]["reasoning_content"] = "unexpected"
            if self.failure == "cache":
                headers["x-litellm-cache-key"] = "synthetic"
        return httpx.Response(200, json=body, headers=headers)


def test_dynamic_fake_collection_roundtrips_all_states_with_zero_call_replay(tmp_path):
    client = FakeHTTP()
    output = tmp_path / "live"
    results, summary = asyncio.run(tool.collect(client, output))
    assert client.gets == 1 and 12 <= client.posts <= 72
    assert tool.audit_logs(output) == (results, summary)
    branches = list(results.values())
    for key in ("orders", "trades", "accounts_final", "denominators", "preallocated_uniforms"):
        assert branches[0][key] == branches[1][key] == branches[2][key]
    assert len(summary["menu_pairs"]) * 2 == client.posts
    for pair in summary["menu_pairs"]:
        assert pair["mean_buy_probability"] == 0.5
        assert not pair["menu_directions_disagree_at_observed_uniform"]


@pytest.mark.parametrize("failure", ["network", "format", "identity", "mode", "cache"])
def test_bad_second_menu_stops_first_batch_without_orders_or_retry(tmp_path, failure):
    client = FakeHTTP(failure)
    out = tmp_path / "live"
    with pytest.raises(ValueError):
        asyncio.run(tool.collect(client, out))
    assert client.posts == 2
    assert len(tool.rows(out / "attempts.jsonl")) == len(tool.rows(out / "responses.jsonl")) == 2
    assert tool.rows(out / "pairs.jsonl") == []
    assert not list(out.glob("event_*.json"))
    assert not (out / "summary.json").exists()


def test_registry_missing_model_stops_before_any_post(tmp_path):
    client = FakeHTTP("registry")
    with pytest.raises(ValueError, match="not available"):
        asyncio.run(tool.collect(client, tmp_path / "live"))
    assert client.posts == 0


@pytest.mark.parametrize("what", ["wire", "event", "pair", "identity", "time", "extra", "latency"])
def test_tampered_evidence_cannot_pass_replay(tmp_path, what):
    out = tmp_path / "live"
    asyncio.run(tool.collect(FakeHTTP(), out))
    if what == "event":
        path = next(out.glob("event_*.json"))
        obj = tool.read(path)
        obj["accounts_before"][tool.study.execution.AGENTS[0]]["cash"] = "999"
        tool.save(path, obj)
    else:
        name = {
            "wire": "requests",
            "pair": "pairs",
            "identity": "responses",
            "time": "attempts",
            "extra": "responses",
            "latency": "responses",
        }[what]
        path = out / (name + ".jsonl")
        items = tool.rows(path)
        if what == "wire":
            items[0]["payload"]["temperature"] = 0.9
        elif what == "pair":
            items[0]["mean_probability_buy"] = 0.9
        elif what == "identity":
            items[0]["position"] = True
        elif what == "time":
            items[0]["dispatched_at"] = "2099-01-01T00:00:00+00:00"
        elif what == "extra":
            items.append(deepcopy(items[-1]))
        elif what == "latency":
            items[0]["latency_seconds"] = -1
        path.write_text("".join(tool.canonical(row) + "\n" for row in items))
    with pytest.raises(ValueError):
        tool.audit_logs(out)


def setup_execute(tmp_path, monkeypatch, failure=None):
    folder = tmp_path / "canonical"
    folder.mkdir()
    plan = folder / "plan.json"
    tool.save(plan, {"test": True})
    tool.save(
        folder / "independent_preflight.json",
        {"accepted_for_execution": True, "mustfix": [], "plan_sha256": tool.sha(plan)},
    )
    budget = tmp_path / "budget.json"
    tool.save(
        budget,
        {
            "global_authorized_completion_POST_cap": 500,
            "used_completion_POSTs": 397,
            "reserved_next_completion_POSTs": 0,
            "available_unreserved_POSTs": 103,
            "active_run_id": None,
            "status": "test",
            "version": 12,
        },
    )
    client = FakeHTTP(failure)
    monkeypatch.setattr(tool, "FOLDER", folder)
    monkeypatch.setattr(tool, "checked_plan", lambda _: {})
    monkeypatch.setattr(tool.shared, "BUDGET_PATH", budget)
    monkeypatch.setattr(tool.shared, "school_client", lambda *_: client)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda _: {"AIGC_API_KEY": "synthetic-secret"})
    return folder, budget, client


@pytest.mark.parametrize("failure", [None, "format", "registry"])
def test_execute_reserves_counts_releases_and_never_resumes(tmp_path, monkeypatch, failure):
    folder, budget, client = setup_execute(tmp_path, monkeypatch, failure)
    if failure:
        with pytest.raises(ValueError):
            asyncio.run(tool.execute(folder))
    else:
        asyncio.run(tool.execute(folder))
    ledger = tool.read(budget)
    assert ledger["used_completion_POSTs"] == 397 + client.posts
    assert ledger["active_run_id"] is None and ledger["reserved_next_completion_POSTs"] == 0
    assert ledger["available_unreserved_POSTs"] == 103 - client.posts
    assert ledger["version"] == 13
    with pytest.raises(ValueError, match="never resume"):
        asyncio.run(tool.execute(folder))
    assert ledger == tool.read(budget)


def test_failing_durable_response_stops_after_counted_post(tmp_path, monkeypatch):
    folder, budget, client = setup_execute(tmp_path, monkeypatch)
    original = tool.append

    def failing(path, value, secret=""):
        if path.name == "responses.jsonl":
            raise OSError("synthetic disk error")
        return original(path, value, secret)

    monkeypatch.setattr(tool, "append", failing)
    with pytest.raises(OSError):
        asyncio.run(tool.execute(folder))
    assert client.posts == 1 and tool.read(budget)["used_completion_POSTs"] == 398
    assert not (folder / "summary.json").exists()


def test_failed_attempt_write_never_dispatches_network_and_counts_conservatively(
    tmp_path, monkeypatch
):
    folder, budget, client = setup_execute(tmp_path, monkeypatch)
    original = tool.append

    def failing(path, value, secret=""):
        original(path, value, secret)
        if path.name == "attempts.jsonl":
            raise OSError("synthetic fsync failure")

    monkeypatch.setattr(tool, "append", failing)
    with pytest.raises(OSError):
        asyncio.run(tool.execute(folder))
    assert client.posts == 0 and tool.read(budget)["used_completion_POSTs"] == 398


def test_wrong_preflight_hash_prevents_credentials_budget_and_client(tmp_path, monkeypatch):
    folder, budget, client = setup_execute(tmp_path, monkeypatch)
    prior = budget.read_bytes()
    review = tool.read(folder / "independent_preflight.json")
    review["plan_sha256"] = "wrong"
    tool.save(folder / "independent_preflight.json", review)
    with pytest.raises(ValueError, match="review required"):
        asyncio.run(tool.execute(folder))
    assert prior == budget.read_bytes() and client.posts == client.gets == 0
    assert not (folder / "execution_started.json").exists()
