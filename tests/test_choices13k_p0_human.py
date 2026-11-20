"""Fictional data only; no real human targets, logs or heldout files."""

from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from behavioral_market.benchmarks.prompts import build_prompt
from scripts import compare_choices13k_p0_human as mod


@pytest.fixture
def sample(tmp_path, request):
    behavior, target = getattr(request, "param", ("original_B", 1.0))
    ids = list(range(200))
    opportunities = [
        {"item_id": i, "repeat_id": j, "display_order": "A_first" if j < 5 else "B_first"}
        for i in ids
        for j in range(10)
    ]
    records = []
    for op in opportunities:
        if behavior == "original_B":
            choice = "B" if op["display_order"] == "A_first" else "A"
        elif behavior == "original_A":
            choice = "A" if op["display_order"] == "A_first" else "B"
        elif behavior == "split_display":
            choice = "A" if op["item_id"] < 100 else "B"
        else:
            choice = behavior
        records.append(
            {
                **op,
                "choice": choice,
                "status": "valid_primary_return",
                "issues": [],
                "local_cache": False,
                "strong_signature": ["same-observed-id", 123, "same-text-hash"],
            }
        )
    source = tmp_path / mod.PARQUET
    source.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "item_id": ids,
                "human_b_rate": [target] * 200,
                "forbidden_other_column": ["DO_NOT_PROJECT"] * 200,
            }
        ),
        source,
    )
    (source.parent / "heldout_items.parquet").write_bytes(b"MUST_NOT_OPEN")
    state = SimpleNamespace(
        workspace=tmp_path,
        selection={"selected_item_ids": ids, "opportunities": opportunities},
        result={
            "rows": records,
            "summary": {
                "items": 200,
                "planned_opportunities": 2000,
                "valid_primary_returns": 2000,
                "unresolved_primary_opportunities": 0,
                "local_replay_opportunities": 0,
                "repair_request_groups": 0,
            },
            "human_labels_loaded": False,
            "heldout_opened": False,
            "new_model_POSTs": 0,
        },
        manifest={},
        prior={},
        review={
            "decision": "full200_P0_offline_audit_accepted",
            "primary_opportunities": 2000,
            "valid_primary_returns": 2000,
        },
        plan={},
        plan_path=tmp_path / "iterations/new/plan.json",
        output=tmp_path / "iterations/new/results",
    )
    for relative in mod.SOURCES:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.PROMPTS).write_text("fictional declared mapping source")
    state.manifest = {"splits": {"calibration_sha256": mod.fingerprint(source)["sha256"]}}
    rewrite(state)
    return state


def rewrite(state):
    """Refresh fictional descriptors, keeping negative cases about content, not hashes."""
    w = state.workspace
    state.selection["source_files"] = {mod.PARQUET: mod.fingerprint(w / mod.PARQUET)}
    mod.save(w / mod.SELECTION, state.selection)
    mod.save(w / mod.EVALUATION, state.result)
    mod.save(w / mod.MANIFEST, state.manifest)
    mod.save(w / mod.REVIEW, state.review)
    state.prior = {"projected_columns": mod.COLUMNS}
    for key, relative in (
        ("calibration_parquet", mod.PARQUET),
        ("accepted_selection", mod.SELECTION),
        ("accepted_result", mod.EVALUATION),
        ("display_mapping_source", mod.PROMPTS),
    ):
        state.prior[key] = {"path": relative, "fingerprint": mod.fingerprint(w / relative)}
    mod.save(w / mod.PRIOR_PLAN, state.prior)
    state.plan = {
        "input_files": {p: mod.fingerprint(w / p) for p in mod.SOURCES},
        "projected_columns": mod.COLUMNS,
        "analysis_spec": deepcopy(mod.SPEC),
        "selected_item_ids": state.selection["selected_item_ids"],
        "planned_opportunities": state.selection["opportunities"],
    }
    state.plan_path.parent.mkdir(parents=True, exist_ok=True)
    mod.save(state.plan_path, state.plan)


@pytest.mark.parametrize("order", ["A_first", "B_first"])
@pytest.mark.parametrize("choice", ["A", "B"])
def test_mapping_agrees_with_original_prompt(order, choice):
    fake = SimpleNamespace(option_a=[(1, 10)], option_b=[(1, 20)], ambiguity=False)
    _, _, display = build_prompt(fake, "fictional", order)
    assert mod.original_b({"choice": choice, "display_order": order}) == int(display[choice] == "B")


@pytest.mark.parametrize(
    ("sample", "rate", "mae", "baseline", "delta"),
    [
        (("original_B", 1.0), 1.0, 0, 0.5, 0),
        (("original_A", 0.0), 0.0, 0, 0.5, 0),
        (("A", 0.5), 0.5, 0, 0, 1),
        (("B", 0.5), 0.5, 0, 0, -1),
        (("original_B", 0.5), 1.0, 0.5, 0, 0),
    ],
    indirect=["sample"],
)
def test_full200_expected_error_and_first_display_counterexample(
    sample, rate, mae, baseline, delta
):
    result = mod.run(sample.workspace, sample.plan_path, sample.output)
    s = result["summary"]
    assert s["received_MAE"] == mae
    assert s["fixed_half_MAE"] == baseline
    assert s["received_minus_fixed_half_MAE"] == mae - baseline
    assert s["mean_signed_orientation_difference"] == delta
    assert s["mean_absolute_orientation_difference"] == abs(delta)
    assert s["received_records"] == 2000
    assert len(result["rows"]) == 200
    assert all(r["received_original_B_rate"] == rate for r in result["rows"])
    assert all(r["received_records"] == 10 for r in result["rows"])
    assert result["independent_generation_count"] is None
    assert "始终选显示A" in (sample.output / "REPORT.md").read_text()


@pytest.mark.parametrize("sample", [("split_display", 0.5)], indirect=True)
def test_signed_cancellation_cannot_hide_large_per_item_orientation_gap(sample):
    result = mod.run(sample.workspace, sample.plan_path, sample.output)
    s = result["summary"]
    assert s["received_MAE"] == 0
    assert s["mean_signed_orientation_difference"] == 0
    assert s["mean_absolute_orientation_difference"] == 1
    assert s["items_with_orientation_difference"] == 200


def test_only_exact_two_columns_are_projected_and_no_wide_loader(sample, monkeypatch):
    real, calls = pq.read_table, []

    def spy(path, *, columns):
        assert path == sample.workspace / mod.PARQUET
        assert columns == ["item_id", "human_b_rate"]
        calls.append((str(path), list(columns)))
        return real(path, columns=columns)

    monkeypatch.setattr(mod.pq, "read_table", spy)
    result = mod.run(sample.workspace, sample.plan_path, sample.output)
    assert len(calls) == 1
    assert result["human_columns_projected"] == ["item_id", "human_b_rate"]
    assert result["heldout_opened"] is False
    assert result["raw_logs_opened"] is False


def deny_projection(*args, **kwargs):
    pytest.fail("human column was projected before all metadata checks passed")


@pytest.mark.parametrize(
    "bad",
    [
        "missing_row",
        "duplicate_row",
        "bool_item",
        "bool_repeat",
        "unknown_repeat",
        "wrong_order",
        "wrong_choice",
        "invalid_status",
        "local_replay",
        "issues",
        "wrong_summary",
        "uneven_orientation",
        "wrong_selection",
        "not_accepted",
        "wrong_split_hash",
    ],
)
def test_metadata_failure_prevents_human_projection(sample, monkeypatch, bad):
    rows = sample.result["rows"]
    if bad == "missing_row":
        rows.pop()
    elif bad == "duplicate_row":
        rows[-1] = deepcopy(rows[0])
    elif bad == "bool_item":
        rows[0]["item_id"] = False
    elif bad == "bool_repeat":
        rows[0]["repeat_id"] = False
    elif bad == "unknown_repeat":
        rows[0]["repeat_id"] = 10
    elif bad == "wrong_order":
        rows[0]["display_order"] = "B_first"
    elif bad == "wrong_choice":
        rows[0]["choice"] = "C"
    elif bad == "invalid_status":
        rows[0]["status"] = "unresolved_or_invalid_primary"
    elif bad == "local_replay":
        rows[0]["local_cache"] = True
    elif bad == "issues":
        rows[0]["issues"] = ["orphan"]
    elif bad == "wrong_summary":
        sample.result["summary"]["valid_primary_returns"] = 1999
    elif bad == "uneven_orientation":
        sample.selection["opportunities"][0]["display_order"] = "B_first"
        rows[0]["display_order"] = "B_first"
    elif bad == "wrong_selection":
        sample.selection["selected_item_ids"][-1] = 200
    elif bad == "not_accepted":
        sample.review["decision"] = "pending"
    elif bad == "wrong_split_hash":
        sample.manifest["splits"]["calibration_sha256"] = "bad"
    rewrite(sample)
    monkeypatch.setattr(mod.pq, "read_table", deny_projection)
    with pytest.raises(ValueError):
        mod.run(sample.workspace, sample.plan_path, sample.output)
    assert not sample.output.exists()


@pytest.mark.parametrize("bad", ["extra_source", "extra_column", "wrong_spec", "bad_fingerprint"])
def test_scope_and_freeze_failures_prevent_human_projection(sample, monkeypatch, bad):
    if bad == "extra_source":
        sample.plan["input_files"]["sealed/heldout.parquet"] = {"sha256": "bad"}
    elif bad == "extra_column":
        sample.plan["projected_columns"] = [*mod.COLUMNS, "problem_id"]
    elif bad == "wrong_spec":
        sample.plan["analysis_spec"]["received_per_item"] = 9
    else:
        sample.plan["input_files"][mod.EVALUATION]["sha256"] = "bad"
    mod.save(sample.plan_path, sample.plan)
    monkeypatch.setattr(mod.pq, "read_table", deny_projection)
    with pytest.raises(ValueError):
        mod.run(sample.workspace, sample.plan_path, sample.output)
    assert not sample.output.exists()


@pytest.mark.parametrize("value", [True, "0.5", None, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_human_rates_are_rejected(value):
    records = [{"item_id": i, "human_b_rate": 0.5} for i in range(200)]
    records[0]["human_b_rate"] = value
    with pytest.raises(ValueError):
        mod.validate_targets(list(range(200)), records)


@pytest.mark.parametrize("bad", ["bool_ID", "duplicate_ID", "missing", "foreign_ID", "wide"])
def test_no_silent_target_exclusion_or_wrong_join(bad):
    records = [{"item_id": i, "human_b_rate": 0.5} for i in range(200)]
    if bad == "bool_ID":
        records[0]["item_id"] = False
    elif bad == "duplicate_ID":
        records[-1]["item_id"] = 0
    elif bad == "missing":
        records.pop()
    elif bad == "foreign_ID":
        records[-1]["item_id"] = 200
    else:
        records[0]["problem_id"] = 0
    with pytest.raises(ValueError):
        mod.validate_targets(list(range(200)), records)


def test_source_change_during_projection_refuses_output(sample, monkeypatch):
    real = pq.read_table

    def changing_reader(path, *, columns):
        table = real(path, columns=columns)
        (sample.workspace / mod.EVALUATION).write_text("changed after preflight")
        return table

    monkeypatch.setattr(mod.pq, "read_table", changing_reader)
    with pytest.raises(ValueError, match="fingerprint"):
        mod.run(sample.workspace, sample.plan_path, sample.output)
    assert not sample.output.exists()


def test_existing_output_is_rejected_before_any_target_read(sample, monkeypatch):
    sample.output.mkdir()
    monkeypatch.setattr(mod.pq, "read_table", deny_projection)
    with pytest.raises(ValueError, match="overwrite"):
        mod.run(sample.workspace, sample.plan_path, sample.output)
