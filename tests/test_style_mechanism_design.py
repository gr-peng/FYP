"""Check accounting consistency and prevent confounded or incomplete contrast banks."""

import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy

import pytest

from behavioral_market.evaluation.style_mechanism_design import (
    FACTORS,
    build_bank,
    canonical,
    changed_paths,
    contrast_edges,
    design_specification,
    digest,
    flatten,
    semantic_payload,
    validate_case,
)


def document(row):
    return json.loads(row["payload"]["messages"][1]["content"])


def edit(row, path, value):
    d = document(row)
    node = d
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    row["payload"]["messages"][1]["content"] = canonical(d)
    row["payload_sha256"] = digest(row["payload"])


def test_complete_design_has_shared_endpoints_and_all_conditioning_levels():
    bank = build_bank()
    before = deepcopy(bank)
    assert len(bank) == len({r["case_id"] for r in bank}) == 16
    assert len({r["payload_sha256"] for r in bank}) == 16
    assert all(validate_case(r) == [] for r in bank)
    edges = contrast_edges(bank)
    assert len(edges) == 32
    assert Counter(e["factor"] for e in edges) == dict.fromkeys(FACTORS, 8)
    assert len({e[k] for e in edges for k in ("left_case", "right_case")}) == 16
    for edge in edges:
        assert set(edge["conditioning"]) == set(FACTORS) - {edge["factor"]}
    assert bank == before


def test_current_style_contrast_keeps_ledger_benchmarks_and_market_identical():
    bank = {r["case_id"]: r for r in build_bank()}
    for edge in contrast_edges(list(bank.values())):
        if edge["factor"] != "current_style":
            continue
        left, right = (document(bank[edge[k]]) for k in ("left_case", "right_case"))
        assert left["observation"] == right["observation"]
        for d in (left, right):
            d["agent_state"].pop("current_style")
            for m in d["agent_state"]["memory"]:
                m.pop("style")
        assert left == right


def test_benchmark_contrast_exposes_only_returns_and_gap_not_exposure():
    bank = {r["case_id"]: r for r in build_bank()}
    for row in bank.values():
        assert "initial_stock_weight" not in row["payload"]["messages"][1]["content"]
        assert "private_construction_ledger" not in canonical(row["payload"])
        assert set(document(row)["observation"]["style_benchmarks"]["returns"]) == {
            "fundamental",
            "technical",
        }
    for edge in contrast_edges(list(bank.values())):
        if edge["factor"] == "benchmark_favored_style":
            assert len(edge["changed_paths"]) == 3
            assert all("style_benchmarks" in path for path in edge["changed_paths"])


def test_each_comparison_has_predeclared_direction_using_named_target():
    bank = {r["case_id"]: r for r in build_bank()}
    for edge in contrast_edges(list(bank.values())):
        factor = edge["factor"]
        expected = "above_price" if factor == "reference_condition" else "fundamental"
        assert bank[edge["positive_case"]]["factors"][factor] == expected
        assert "target_style=fundamental" in edge["outcome"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("agent_state", "portfolio", "total_equity"), 20000),
        (("agent_state", "portfolio", "unrealized_pnl"), 0),
        (("agent_state", "actual_block_return"), 0.025),
        (("agent_state", "drawdown"), 0),
        (("agent_state", "memory", 0, "style"), "technical"),
        (("agent_state", "memory", 1, "market_return"), -0.005),
        (("agent_state", "memory", 0, "filled_quantity"), 1),
        (("agent_state", "memory", 0, "pnl_change"), 0),
        (("observation", "reference", "method"), "pre_start_close_anchor_scenario"),
        (("observation", "reference", "value"), 105),
        (("observation", "style_benchmarks", "returns", "fundamental"), 0.005),
        (("observation", "style_benchmarks", "fundamental_minus_technical_return"), 0),
        (("observation", "mean_field", "technical_count"), 80),
        (("observation", "mean_field", "fundamental_share"), 0.5),
        (("observation", "mean_field", "as_of_session"), 10),
        (("observation", "return_5d"), -0.05),
        (("observation", "future_price"), 120),
    ],
)
def test_inconsistent_fields_are_rejected_even_after_rehash(path, value):
    row = build_bank()[0]
    edit(row, path, value)
    before = deepcopy(row)
    assert validate_case(row)
    assert row == before  # Validation must not repair or fill unknown values.


def test_missing_memory_and_unknown_factor_cannot_enter_complete_bank():
    row = build_bank()[0]
    edit(row, ("agent_state", "memory"), [])
    assert validate_case(row)
    row = build_bank()[0]
    row["factors"]["current_style"] = "unknown"
    assert validate_case(row)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_invalid_private_ledger_value_is_rejected(value):
    row = build_bank()[0]
    row["private_construction_ledger_not_sent"]["portfolios"]["fundamental"]["return"] = value
    assert validate_case(row)


def test_private_construction_must_reconcile_to_visible_returns():
    row = build_bank()[0]
    row["private_construction_ledger_not_sent"]["portfolios"]["fundamental"]["final_equity"] = 20000
    assert validate_case(row)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown"])
def test_bank_does_not_silently_drop_or_overwrite_ineligible_cells(mode):
    bank = build_bank()
    if mode == "missing":
        bank.pop()
    elif mode == "duplicate":
        bank[-1] = deepcopy(bank[0])
    else:
        bank[0]["factors"].pop("current_style")
    with pytest.raises(ValueError, match="incomplete|ineligible"):
        contrast_edges(bank)


def test_no_execution_authorization_or_observed_effect_from_material_validation():
    spec = design_specification()
    assert spec["execution_gate"]["new_completion_POST_limit"] == 0
    assert spec["execution_gate"]["used_authorized_completion_POSTs"] == 21
    assert spec["human_alignment_established"] is False
    assert spec["causal_effect_observed"] is False
    assert "unknown" in spec["unknown_handling"]
    assert "not learned habit" in spec["effect_scopes"]["current_style"]


def test_validation_still_rejects_incoherent_ledger_under_optimized_python():
    script = """
import json
from behavioral_market.evaluation.style_mechanism_design import (
    build_bank, canonical, digest, validate_case,
)
r = build_bank()[0]
d = json.loads(r['payload']['messages'][1]['content'])
d['agent_state']['portfolio']['total_equity'] = 1
r['payload']['messages'][1]['content'] = canonical(d)
r['payload_sha256'] = digest(r['payload'])
raise SystemExit(0 if validate_case(r) else 1)
"""
    result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '"current_style":"fundamental"',
            '"current_style":"technical","current_style":"fundamental"',
        ),
        (
            '"current_style":"fundamental"',
            '"current_style":"fundamental","current_style":"fundamental"',
        ),
        ('"reference":{', '"reference":{"value":105,'),
        ('"returns":{', '"returns":{"fundamental":0.5,'),
        ('"value":95.0', '"value":1e999'),
        ('"value":95.0', '"value":NaN'),
    ],
)
def test_duplicate_keys_and_nonfinite_inner_json_never_become_eligible(needle, replacement):
    row = build_bank()[0]
    original = row["payload"]["messages"][1]["content"]
    assert original.count(needle) == 1
    row["payload"]["messages"][1]["content"] = original.replace(needle, replacement)
    row["payload_sha256"] = digest(row["payload"])
    assert validate_case(row)
    with pytest.raises(ValueError, match="Duplicate|Nonfinite"):
        semantic_payload(row["payload"])


def test_dotted_extra_key_cannot_hide_a_conflicting_strategy():
    row = build_bank()[0]
    edit(row, ("agent_state.current_style",), "technical")
    assert validate_case(row)


def test_display_paths_cannot_merge_literal_dots_with_nested_fields():
    left = {"agent_state": {"current_style": "fundamental"}}
    right = {**left, "agent_state.current_style": "technical"}
    assert len(flatten(right)) == 2
    assert changed_paths(left, right) == ["agent_state~1current_style"]
