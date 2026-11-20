"""Evidence-only task checks: no human rows, credentials or model transport."""

import copy
import json
from pathlib import Path

import pytest

from scripts import check_behavior_targets as tool


def contract():
    return json.loads(tool.CONTRACT.read_text())


def request(source="replication_active_order", target="active_order_direction", **changes):
    row = next(row for row in contract()["targets"] if row["id"] == source)
    return {
        "source": source,
        "target_task": target,
        "conditioning": row["conditioning"],
        "input_scope": "declared_source_decision_state",
        "mapping": "identity",
        **changes,
    }


def test_interface_matrix_keeps_only_the_four_observed_task_matches():
    rows = tool.default_checks(tool.load_contract())
    assert len(rows) == 35
    assert {(r["source"], r["target_task"]) for r in rows if r["target_interface_supported"]} == {
        ("choices13k_risk", "risk_probability"),
        ("cpc18_risk", "risk_probability"),
        ("replication_active_order", "active_order_direction"),
        ("scopic_owned_sell", "owned_asset_sale"),
    }
    assert all(r["evaluation_ready"] is False for r in rows)
    assert all(r["performance_claim_established"] is False for r in rows)
    assert all(
        not r["target_interface_supported"]
        for r in rows
        if r["target_task"]
        in {
            "human_style_choice",
            "daily_trading_action",
            "full_financial_policy",
            "human_market_realism",
        }
    )


@pytest.mark.parametrize(
    "source,target,changes,reason",
    [
        (
            "replication_active_order",
            "human_style_choice",
            {"mapping": "buy_is_fundamental"},
            "action_or_hindsight_relabeling_not_supported",
        ),
        (
            "replication_active_order",
            "active_order_direction",
            {"conditioning": "every_day"},
            "eligibility_or_denominator_mismatch",
        ),
        (
            "replication_active_order",
            "active_order_direction",
            {"input_scope": "synthetic_daily_state"},
            "information_set_or_clock_bridge_not_verified",
        ),
        (
            "scopic_owned_sell",
            "daily_trading_action",
            {"mapping": "no_sale_means_hold"},
            "action_or_hindsight_relabeling_not_supported",
        ),
        ("hybrid_style_review", "human_style_choice", {}, "simulator_output_is_not_human_truth"),
        (
            "hybrid_style_review",
            "human_style_choice",
            {"mapping": "more_profitable_rule_is_truth"},
            "action_or_hindsight_relabeling_not_supported",
        ),
        (
            "choices13k_risk",
            "risk_probability",
            {"conditioning": "individual_trader_history"},
            "eligibility_or_denominator_mismatch",
        ),
        (
            "cpc18_risk",
            "risk_probability",
            {"conditioning": "feedback_trials"},
            "eligibility_or_denominator_mismatch",
        ),
    ],
)
def test_unsupported_inferences_are_rejected(source, target, changes, reason):
    result = tool.assess_claim(contract(), request(source, target, **changes))
    assert not result["target_interface_supported"] and reason in result["reasons"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("conditioning", "every_day"),
        ("source_refs", ["src/behavioral_market/benchmarks/schemas.py"]),
        ("field", "invented_human_style_label"),
        ("label_status", "verified_FT"),
        ("decision_time", "daily"),
        ("split", "development_is_unopened"),
    ],
)
def test_contract_cannot_self_certify_new_semantics(tmp_path, monkeypatch, field, value):
    changed = contract()
    row = next(r for r in changed["targets"] if r["id"] == "replication_active_order")
    row[field] = value
    candidate = tmp_path / "changed.json"
    candidate.write_text(json.dumps(changed))
    monkeypatch.setattr(tool, "fingerprint", lambda _path: pytest.fail("source bytes read"))
    with pytest.raises(ValueError, match="interpretation changed"):
        tool.load_contract(candidate)
    with pytest.raises(ValueError, match="interpretation changed"):
        tool.assess_claim(changed, request(conditioning=row["conditioning"]))


def test_unapproved_source_path_rejected_before_reading_any_source(tmp_path, monkeypatch):
    changed = contract()
    changed["sources"]["data/final_holdout.csv"] = {"bytes": 1, "sha256": "a" * 64}
    candidate = tmp_path / "bad-path.json"
    candidate.write_text(json.dumps(changed))
    monkeypatch.setattr(tool, "fingerprint", lambda _path: pytest.fail("source bytes read"))
    with pytest.raises(ValueError, match="interpretation changed"):
        tool.load_contract(candidate)


def test_changed_source_invalidates_even_an_unchanged_contract(monkeypatch):
    original = tool.fingerprint

    def altered(path):
        result = original(path)
        if path.name == "replication_quote_states.py":
            return {**result, "sha256": "0" * 64}
        return result

    monkeypatch.setattr(tool, "fingerprint", altered)
    with pytest.raises(
        ValueError, match="source evidence changed: experiments/replication_quote_states"
    ):
        tool.load_contract()


def test_generate_reads_only_the_declared_metadata_and_source_files(tmp_path, monkeypatch):
    reads = []
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def read_text(path, *args, **kwargs):
        reads.append(path.resolve())
        return original_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        reads.append(path.resolve())
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = tool.generate(tmp_path / "report")
    assert set(reads) == {tool.CONTRACT.resolve()} | {
        (tool.ROOT / name).resolve() for name in tool.SOURCE_ALLOWLIST
    }
    assert result["new_model_POSTs"] == 0 and result["numeric_human_accuracy_produced"] is False
    assert result["final_holdout_outcomes_read"] is False


def test_rejected_claim_is_reported_without_a_performance_score(tmp_path):
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps([request(target="human_style_choice")]))
    result = tool.generate(tmp_path / "output", claims_path=claims)
    assert result["checks"][0]["target_interface_supported"] is False
    assert result["checks"][0]["performance_claim_established"] is False
    assert result["evaluation_ready"] is False
    # A second invocation must not replace the first report.
    with pytest.raises(FileExistsError):
        tool.generate(tmp_path / "output", claims_path=claims)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"source": "unknown"},
        {**request(), "extra": "silently ignored"},
        {**request(), "source": []},
        {**request(), "target_task": "invented_task"},
    ],
)
def test_invalid_claim_creates_no_output(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([bad]))
    with pytest.raises(ValueError):
        tool.generate(tmp_path / "output", claims_path=path)
    assert not (tmp_path / "output").exists()


def test_assessment_does_not_modify_inputs():
    source, claim = contract(), request()
    before = copy.deepcopy((source, claim))
    tool.assess_claim(source, claim)
    assert (source, claim) == before
