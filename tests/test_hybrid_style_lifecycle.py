"""No network: model interface reviews only; rules execute all actual orders."""

import asyncio
import copy
import hashlib
import json

import pandas as pd
import pytest
from test_finite_lifecycle_audit import rehash
from test_style_path_integration import fictional_market
from test_style_path_integration import offline_boundary as offline_boundary

from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.hybrid_review import PROTOCOL
from scripts.audit_finite_lifecycle import audit
from scripts.run_multiagent_rule_comparison import NoModelClient, config_for


class ScriptedReview:
    hybrid_review_protocol = PROTOCOL
    review_provenance = "scripted"

    def __init__(self, directory, *, switch_day=None, fail_at=None, anomaly=None, mutate=False):
        self.directory = directory
        self.switch_day = switch_day
        self.fail_at = fail_at
        self.anomaly = anomaly
        self.mutate = mutate
        self.calls = []
        self.last_result = None

    async def decide(self, system, user, schema, identity):
        assert schema is StyleDecision and identity["kind"] == "style"
        saved = [
            json.loads(line)
            for line in (self.directory / "hybrid_review_batches.jsonl").read_text().splitlines()
        ]
        batch = [row for row in saved if row["phase"] == "prepared"][-1]
        assert len(batch["requests"]) == 20  # All immutable prompts saved before dispatch.
        index = len(self.calls) % 20
        assert batch["requests"][index] == {"system": system, "user": user, "identity": identity}
        payload = json.loads(user)
        assert payload["agent_state"]["persona"] == {"agent_id": identity["agent_id"]}
        assert "mean_field" not in payload["observation"]
        self.calls.append(copy.deepcopy(identity))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("fictional interrupted response")
        if self.mutate and self.last_result:
            self.last_result[0].target_style = "corrupted later"
            self.last_result[1]["fallback"] = True
        current = payload["agent_state"]["current_style"]
        switch = identity["session_date"] == self.switch_day
        target = ("technical" if current == "fundamental" else "fundamental") if switch else current
        decision = StyleDecision(
            decision="switch" if switch else "stay",
            target_style=target,
            rationale="scripted test response; not model evidence",
        )
        meta = {
            "status": "ok",
            "fallback": False,
            "cache_hit": False,
            "repair_count": 0,
            "request_hash": hashlib.sha256((system + user).encode()).hexdigest(),
        }
        if self.anomaly:
            key, value = self.anomaly
            if key == "decision":
                decision.decision = value
            else:
                meta[key] = value
        self.last_result = decision, meta
        identity["agent_id"] = "mutated adapter copy"
        return self.last_result


def setup(days=30):
    config = config_for(20261106, "fixed")
    config["data"]["experiment_end"] = f"2026-01-{days:02d}"
    # Batch size cannot cause model order requests in this treatment.
    config["behavior"]["order_decision_batch_size"] = 2
    _, bars = fictional_market(days=days)
    bars = bars[bars.session_date < "2026-01-01"].copy()
    bars["symbol"] = "SYNTHETIC"
    return config, bars


def run(directory, monkeypatch, *, client=None, days=30):
    config, bars = setup(days)
    treatment = "rule_abm" if client is None else "hybrid_style_llm"
    if client is not None:
        del config["behavior"]["rule_style_adaptation"]
    monkeypatch.setattr(event_runner, "source_hash", lambda: "a" * 64)
    result = asyncio.run(
        event_runner.run_experiment(
            config,
            "endogenous",
            treatment,
            "mock",
            directory,
            bars=bars,
            events=[],
            client=client or NoModelClient(),
        )
    )
    assert audit(directory)["checked_reviews"] == days // 5 * 20
    return result


def frames(directory):
    return {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ("orders", "trades", "agents", "market")
    }


def assert_equal(left, right, through=None):
    for name in left:
        first, second = left[name], right[name]
        if through is not None:
            first = first[first.session_date <= through].reset_index(drop=True)
            second = second[second.session_date <= through].reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)


def test_thirty_days_all_stay_equals_fixed_and_freezes_120_reviews(tmp_path, monkeypatch):
    client = ScriptedReview(tmp_path / "hybrid", mutate=True)
    run(tmp_path / "fixed", monkeypatch)
    run(tmp_path / "hybrid", monkeypatch, client=client)
    assert_equal(frames(tmp_path / "fixed"), frames(tmp_path / "hybrid"))
    assert len(client.calls) == 120
    assert len({json.dumps(row, sort_keys=True) for row in client.calls}) == 120
    assert sorted({row["session_date"] for row in client.calls}) == [
        f"2026-01-{day:02d}" for day in (5, 10, 15, 20, 25, 30)
    ]
    assert (tmp_path / "fixed/finite_rule_replay.jsonl").read_bytes() == (
        tmp_path / "hybrid/finite_rule_replay.jsonl"
    ).read_bytes()
    meta = json.loads((tmp_path / "hybrid/run_metadata.json").read_text())
    assert meta["review_provenance"] == "scripted" and meta["quality"] == "engineering_mock"
    with pytest.raises(RuntimeError, match="no resume"):
        run(tmp_path / "hybrid", monkeypatch, client=client)
    assert len(client.calls) == 120


@pytest.mark.parametrize("switch_day", ["2026-01-05", "2026-01-30"])
def test_switch_only_affects_subsequent_days(tmp_path, monkeypatch, switch_day):
    run(tmp_path / "fixed", monkeypatch)
    client = ScriptedReview(tmp_path / "hybrid", switch_day=switch_day)
    run(tmp_path / "hybrid", monkeypatch, client=client)
    fixed, hybrid = frames(tmp_path / "fixed"), frames(tmp_path / "hybrid")
    assert_equal(fixed, hybrid, through=switch_day)
    reviews = pd.read_parquet(tmp_path / "hybrid/style_switches.parquet")
    assert (reviews[reviews.session_date == switch_day].decision == "switch").all()
    if switch_day == "2026-01-05":
        first = fixed["orders"].query("session_date == '2026-01-06'")
        second = hybrid["orders"].query("session_date == '2026-01-06'")
        assert (first["style"] != second["style"]).all()
        assert (first.action != second.action).any()


def test_third_response_failure_preserves_pending_and_complete_trading_days(tmp_path, monkeypatch):
    client = ScriptedReview(tmp_path, switch_day="2026-01-05", fail_at=3)
    with pytest.raises(RuntimeError, match="fictional interrupted"):
        run(tmp_path, monkeypatch, client=client, days=10)
    assert len(client.calls) == 3
    records = [
        json.loads(line)
        for line in (tmp_path / "hybrid_review_batches.jsonl").read_text().splitlines()
    ]
    end = records[-1]
    assert (end["attempted"], end["valid_unapplied"], end["applied"]) == (3, 2, 0)
    assert len(end["unsent_identities"]) == 17
    returned = [r for r in records if r["phase"] == "returned_unapplied"]
    assert len(returned) == 2 and all(r["decision"]["decision"] == "switch" for r in returned)
    saved = frames(tmp_path)
    assert len(saved["market"]) == 5 and len(saved["orders"]) == len(saved["agents"]) == 100
    assert pd.read_parquet(tmp_path / "style_switches.parquet").empty
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["status"] == "interrupted" and meta["completed_market_sessions"] == 5
    assert meta["last_applied_review"] is None
    before = {
        r["identity"]["agent_id"]: json.loads(r["user"])["agent_state"]["current_style"]
        for r in records[0]["requests"]
    }
    assert meta["current_styles"] == before
    with pytest.raises(ValueError, match="incomplete"):
        audit(tmp_path)


@pytest.mark.parametrize(
    "anomaly",
    [
        ("fallback", True),
        ("cache_hit", True),
        ("repair_count", 1),
        ("repair_count", False),
        ("status", "mock"),
        ("decision", "switch"),
    ],
)
def test_invalid_review_stops_before_next_request(tmp_path, monkeypatch, anomaly):
    client = ScriptedReview(tmp_path, anomaly=anomaly)
    with pytest.raises(ValueError, match="strict response"):
        run(tmp_path, monkeypatch, client=client, days=5)
    assert len(client.calls) == 1
    assert pd.read_parquet(tmp_path / "style_switches.parquet").empty


@pytest.mark.parametrize("invalid", ["finite", "client", "protocol", "mode", "cache", "old_flag"])
def test_invalid_hybrid_rejected_before_io(tmp_path, invalid):
    config, _ = setup()
    del config["behavior"]["rule_style_adaptation"]
    client = ScriptedReview(tmp_path)
    mode, cache = "endogenous", False
    if invalid == "finite":
        config["behavior"].pop("strategy_return_method")
        config["behavior"].pop("finite_rule_pair")
    elif invalid == "client":
        client = None
    elif invalid == "protocol":
        client = NoModelClient()
    elif invalid == "mode":
        mode = "historical"
    elif invalid == "cache":
        cache = True
    else:
        config["behavior"]["rule_style_adaptation"] = "fixed"
    with pytest.raises(ValueError, match="hybrid_style_llm|requires finite rule_abm"):
        asyncio.run(
            event_runner.run_experiment(
                config, mode, "hybrid_style_llm", "mock", tmp_path, client=client, cache_only=cache
            )
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "tamper",
    [
        "source",
        "phase",
        "response",
        "prompt",
        "seed",
        "cash",
        "holdings",
        "gap",
        "start_cash",
        "review_time",
    ],
)
def test_audit_rejects_mislabelled_or_partly_applied_review(tmp_path, monkeypatch, tamper):
    run(tmp_path, monkeypatch, client=ScriptedReview(tmp_path), days=5)
    if tamper == "source":
        path = tmp_path / "run_metadata.json"
        meta = json.loads(path.read_text())
        meta["order_source"] = "model"
        path.write_text(json.dumps(meta))
    else:
        path = tmp_path / "hybrid_review_batches.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if tamper == "phase":
            rows[-1]["phase"] = "validated_unapplied"
        elif tamper == "response":
            rows[2]["decision"]["rationale"] = "changed response"
        elif tamper == "seed":
            rows[0]["requests"][0]["identity"]["seed"] = 123456789
        else:
            payload = json.loads(rows[0]["requests"][0]["user"])
            if tamper == "prompt":
                payload["agent_state"]["persona"]["herding"] = 0.8
            elif tamper == "cash":
                payload["agent_state"]["portfolio"]["cash"] = 1
            elif tamper == "holdings":
                payload["agent_state"]["portfolio"]["position"] += 1
            elif tamper == "gap":
                payload["agent_state"]["counterfactual"]["gap"] += 0.5
            elif tamper == "start_cash":
                payload["agent_state"]["counterfactual"]["block_start_portfolio"]["cash"] = "1"
            else:
                payload["observation"]["decision_at_utc"] = "2026-01-07T00:00:00+00:00"
            rows[0]["requests"][0]["user"] = json.dumps(payload)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        rehash(tmp_path, path.name)
    with pytest.raises(ValueError, match="hybrid|accounting mismatch"):
        audit(tmp_path)


def test_second_reset_summary_failure_leaves_every_style_and_review_unapplied(
    tmp_path, monkeypatch
):
    original = event_runner.FiniteRulePair.summary

    def fail_second_reset(pair, observed):
        if pair.block_index == 1 and pair.agent_id == "A001":
            raise RuntimeError("scripted second reset summary failure")
        return original(pair, observed)

    monkeypatch.setattr(event_runner.FiniteRulePair, "summary", fail_second_reset)
    client = ScriptedReview(tmp_path, switch_day="2026-01-05")
    with pytest.raises(RuntimeError, match="second reset summary"):
        run(tmp_path, monkeypatch, client=client, days=5)
    entries = [
        json.loads(line)
        for line in (tmp_path / "hybrid_review_batches.jsonl").read_text().splitlines()
    ]
    expected = {
        r["identity"]["agent_id"]: json.loads(r["user"])["agent_state"]["current_style"]
        for r in entries[0]["requests"]
    }
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert len(client.calls) == 20 and entries[-1]["phase"] == "validated_unapplied"
    assert meta["current_styles"] == expected
    assert meta["applied_review_rows"] == 0 and meta["last_applied_review"] is None
    assert meta["failure_phase"] == "prepare_review_resets"
    assert pd.read_parquet(tmp_path / "style_switches.parquet").empty
