"""Fictional accepted derived files only; never open provider logs or datasets."""

import csv
import json
from copy import deepcopy

import pytest

from behavioral_market.benchmarks import tradeoff_comparison as study
from behavioral_market.evaluation import matched_choice_collection as collection
from behavioral_market.evaluation.choice_format_sensitivity import analyze
from scripts import analyze_choice_format_sensitivity as cli
from scripts.style_fact_check import fingerprint, save


def option(*pairs):
    return "\n".join(f"- {p}% chance of receiving {x}" for p, x in pairs)


@pytest.fixture
def sample():
    materials = {
        "target": {
            "item_id": 166,
            "option_a_text": option((90, -5), (10, 40)),
            "option_b_text": option((50, 3), (50, 4)),
        },
        "references": [
            {
                "item_id": 211,
                "option_a_text": option((100, 4)),
                "option_b_text": option((50, 0), (50, 10)),
                "human_b_rate": 0.4,
            },
            {
                "item_id": 384,
                "option_a_text": option((100, 1)),
                "option_b_text": option((100, -1)),
                "human_b_rate": 0.1,
            },
            {
                "item_id": 542,
                "option_a_text": option((100, 6)),
                "option_b_text": option((100, 4)),
                "human_b_rate": 0.7,
            },
        ],
    }
    packet = study.make_packet(materials, "fictional-sensitivity")
    counts = {
        "plain": [2, 2, 1, 2],
        "true_reference": [1, 2, 0, 2],
        "shuffled_reference": [1, 1, 1, 0],
    }
    requests, attempts, responses = [], [], []
    for req in packet["requests"]:
        fmt = list(study.panel_tool.FORMATS).index((req["content_order"], req["letter_order"]))
        original = "B" if req["repeat_id"] < counts[req["method_id"]][fmt] else "A"
        letter = req["letter_order"][req["content_order"].index(original)]
        ident = collection.identity(req)
        requests.append(deepcopy(req))
        attempts.append({**ident, "dispatched_at": "2026-10-22T00:00:00+00:00", "retry_count": 0})
        responses.append(
            {
                **ident,
                "requested_at": "2026-10-22T00:00:00+00:00",
                "response_received_at": "2026-10-22T00:00:00+00:00",
                "retry_count": 0,
                "cached": False,
                "http_status": 200,
                "latency_seconds": 0,
                "response": {
                    "id": f"fictional-{req['position']}",
                    "created": req["position"],
                    "model": study.MODEL,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"choice": letter}),
                            },
                        }
                    ],
                },
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    target = {"item_id": 166, "human_b_rate": 0.65}
    evaluation = study.evaluate(packet, requests, attempts, responses, {166: 0.65})
    return packet, target, evaluation


def test_hand_arithmetic_and_primary_retained(sample):
    result = analyze(*sample)
    assert result["analysis_kind"] == "post_result_format_sensitivity"
    assert result["overall_human_B_rate_exact"] == "13/20"
    assert [m["original_B_count"] for m in result["pooled_methods"]] == [7, 5, 3]
    assert [m["absolute_error_exact"] for m in result["pooled_methods"]] == [
        "9/40",
        "1/40",
        "11/40",
    ]
    assert [p["absolute_error_difference_exact"] for p in result["primary_comparisons"]] == [
        "-1/5",
        "-1/4",
    ]
    assert result["same_input_disagreement_cells"] == 5
    assert all(
        c["planned_records"] == c["received_records"] == 2 for c in result["format_method_cells"]
    )
    plain = [
        p["absolute_error_difference_exact"]
        for p in result["posthoc_format_pairs"]
        if p["right_method"] == "plain"
    ]
    shuffled = [
        p["absolute_error_difference_exact"]
        for p in result["posthoc_format_pairs"]
        if p["right_method"] == "shuffled_reference"
    ]
    assert plain == ["-1/5", "0", "1/2", "0"]
    assert shuffled == ["0", "1/5", "1/2", "-3/10"]
    assert result["posthoc_format_counts"] == [
        {
            "left_method": "true_reference",
            "right_method": "plain",
            "closer": 1,
            "same": 2,
            "farther": 1,
        },
        {
            "left_method": "true_reference",
            "right_method": "shuffled_reference",
            "closer": 1,
            "same": 1,
            "farther": 2,
        },
    ]
    assert result["format_specific_human_rates_available"] is False
    assert all(c["format_specific_human_rate"] is None for c in result["format_method_cells"])
    assert result["independent_generation_count"] is None and result["new_model_POSTs"] == 0


def test_derives_from_choices_not_stored_method_rates(sample):
    packet, target, evaluation = sample
    before = deepcopy(sample)
    expected = analyze(*sample)
    for comp in evaluation["comparisons"].values():
        comp["item_methods"] = [{"pooled_original_B_rate": "MUST_NOT_USE", "human_b_rate": 999}]
        comp["rows"] = [{"selected_original": "MUST_NOT_USE"}]
    assert analyze(packet, target, evaluation) == expected
    packet0, target0, evaluation0 = before
    unchanged = deepcopy(before)
    analyze(packet0, target0, evaluation0)
    assert (packet0, target0, evaluation0) == unchanged


def test_order_invariant_and_primary_serialization_tolerance(sample):
    packet, target, evaluation = sample
    result = analyze(*sample)
    evaluation["rows"].reverse()
    assert analyze(packet, target, evaluation) == result
    evaluation["comparisons"]["true_reference_vs_plain"]["primary"]["left_MAE"] += 5e-13
    assert analyze(packet, target, evaluation) == result
    evaluation["comparisons"]["true_reference_vs_plain"]["primary"]["left_MAE"] += 1e-8
    with pytest.raises(ValueError, match="arithmetic differs"):
        analyze(packet, target, evaluation)


def test_same_rates_need_not_show_primary_improvement(sample):
    packet, target, evaluation = sample
    expected = {r["slot_id"]: r for r in packet["requests"]}
    for row in evaluation["rows"]:
        request = expected[row["slot_id"]]
        original = "B" if request["repeat_id"] == 0 else "A"
        row["choice"] = request["letter_order"][request["content_order"].index(original)]
    for comp in evaluation["comparisons"].values():
        comp["primary"].update(left_MAE=0.15, right_MAE=0.15, left_minus_right_MAE=0)
    result = analyze(packet, target, evaluation)
    assert all(p["absolute_error_difference_exact"] == "0" for p in result["primary_comparisons"])
    assert all(
        p["same"] == 4 and p["closer"] == p["farther"] == 0 for p in result["posthoc_format_counts"]
    )


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "duplicate",
        "foreign",
        "wrong_method",
        "bool_identity",
        "invalid_choice",
        "status",
        "mode",
        "top_evidence",
        "row_evidence",
        "fake_primary",
        "wrong_direction",
        "fake_count",
        "bool_count",
        "choice_tamper",
        "packet_mapping",
    ],
)
def test_malformed_complete_source_rejected_without_imputation(sample, attack):
    packet, target, evaluation = sample
    row = evaluation["rows"][0]
    if attack == "missing":
        evaluation["rows"].pop()
    elif attack == "duplicate":
        evaluation["rows"][-1] = deepcopy(row)
    elif attack == "foreign":
        row["slot_id"] = "foreign"
    elif attack == "wrong_method":
        row["method_id"] = "true_reference"
    elif attack == "bool_identity":
        row["position"] = True
    elif attack == "invalid_choice":
        row["choice"] = "C"
    elif attack == "status":
        row["compatible"] = False
    elif attack == "mode":
        row["returned_mode"] = "returned_reasoning_observed"
    elif attack == "top_evidence":
        evaluation["evidence_kind"] = "recorded_model_logs"
    elif attack == "row_evidence":
        row["evidence_kind"] = "recorded_model_logs"
    elif attack == "fake_primary":
        evaluation["comparisons"]["true_reference_vs_plain"]["primary"]["left_minus_right_MAE"] = 0
    elif attack == "wrong_direction":
        evaluation["comparisons"]["true_reference_vs_plain"]["difference_direction"] = (
            "right_minus_left"
        )
    elif attack == "fake_count":
        evaluation["logged_dispatch_attempts"] = 23
    elif attack == "bool_count":
        evaluation["comparisons"]["true_reference_vs_plain"]["primary"]["planned_items"] = True
    elif attack == "choice_tamper":
        row["choice"] = "B" if row["choice"] == "A" else "A"
    else:
        packet["panel"]["opportunities"][0]["presentation"]["mapping"]["A"]["original_option"] = "B"
    with pytest.raises(ValueError):
        analyze(packet, target, evaluation)


@pytest.mark.parametrize(
    "target",
    [
        {"item_id": True, "human_b_rate": 0.65},
        {"item_id": 166, "human_b_rate": True},
        {"item_id": 166, "human_b_rate": float("nan")},
        {"item_id": 166, "human_b_rate": 1.1},
        {"item_id": 166, "human_b_rate": 0.65, "format_specific": [0.65] * 4},
    ],
)
def test_target_schema_does_not_invent_format_human_rates(sample, target):
    packet, _, evaluation = sample
    with pytest.raises(ValueError):
        analyze(packet, target, evaluation)


@pytest.fixture
def workspace(tmp_path, monkeypatch, sample):
    w = tmp_path
    accepted = w / "accepted"
    accepted.mkdir()
    data = {}
    for name, value in zip(("packet", "local_target", "evaluation"), sample, strict=True):
        path = accepted / (name + ".json")
        save(path, value)
        data[name] = {"path": str(path.relative_to(w)), "fingerprint": fingerprint(path)}

    def descriptor(path):
        return {"path": str(path.relative_to(w)), "fingerprint": fingerprint(path)}

    manifest = accepted / "terminal_manifest.json"
    save(
        manifest,
        {
            "files": {data[k]["path"]: data[k]["fingerprint"] for k in ("packet", "evaluation")}
            | {"../MUST_NOT_OPEN_RAW": {"bytes": 0, "sha256": "f" * 64}}
        },
    )
    freeze = accepted / "prepared_freeze.json"
    save(
        freeze,
        {
            "files": {
                data["local_target"]["path"]: data["local_target"]["fingerprint"],
                "unrelated_prepared_packet.json": {"bytes": 0, "sha256": "a" * 64},
            }
        },
    )
    review = accepted / "review.json"
    save(review, {"status": "accepted"})
    budget = accepted / "budget.json"
    save(
        budget,
        {
            "used_completion_POSTs": 115,
            "global_authorized_completion_POST_cap": 115,
            "reserved_next_completion_POSTs": 0,
            "active_run_id": None,
            "status": "fictional_accepted",
        },
    )
    metadata = {
        name: descriptor(path)
        for name, path in {
            "terminal_manifest": manifest,
            "prepared_freeze": freeze,
            "independent_review": review,
            "accepted_budget": budget,
        }.items()
    }
    origin = accepted / "origin.json"
    save(origin, {"data_sources": data, "metadata_sources": metadata})
    origin_descriptor = descriptor(origin)
    monkeypatch.setattr(cli, "ORIGIN", deepcopy(origin_descriptor))
    folder = w / cli.ITERATION
    folder.mkdir(parents=True)
    save(
        folder / "plan.json",
        {
            "origin_plan": origin_descriptor,
            "data_sources": data,
            "metadata_sources": metadata,
            "budget": {
                "used_POSTs": 115,
                "cap": 115,
                "remaining": 0,
                "new_POSTs": 0,
                "reserved": 0,
            },
            "analysis_kind": "Post-result sensitivity diagnostic,not a replacement primary metric.",
            "no_new_human_format_rates": True,
            "serialization_tolerance": {
                "relative": 0,
                "absolute": 1e-12,
                "purpose": "Compare independently reconstructed rational results "
                "with prior JSON floats only;not a measured error bound.",
            },
        },
    )
    return w


def test_runnable_three_file_analysis_and_csvs(workspace, monkeypatch):
    w = workspace
    import dotenv
    import httpx

    def forbidden(*args, **kwargs):
        pytest.fail("no HTTP client or credentials")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    out = w / cli.ITERATION / "results"
    actual = cli.run(w, out)
    assert (
        actual["reconstructed_records"] == 24
        and actual["format_pairs"] == 8
        and actual["new_POSTs"] == 0
    )
    assert len(list(csv.DictReader((out / "cells.csv").open()))) == 12
    pairs = list(csv.DictReader((out / "pairs.csv").open()))
    assert len(pairs) == 8 and all(row["format_specific_human_rate"] == "" for row in pairs)
    report = (out / "REPORT.md").read_text()
    assert "没有四种显示方式各自的人群比例" in report and "没有改变原定平均指标" in report
    assert "重新核验HTTP" in report
    receipt = json.loads((out / "source_receipt.json").read_text())
    assert receipt["derived_files_decoded"] == 3 and receipt["raw_logs_or_parquet_opened"] is False
    with pytest.raises(ValueError, match="overwrite"):
        cli.run(w, out)


@pytest.mark.parametrize(
    "attack", ["data", "newplan_resign", "origin_newplan_resign", "manifest_wrong_entry"]
)
def test_frozen_source_bindings_survive_resigning(workspace, attack):
    w = workspace
    plan_path = w / cli.ITERATION / "plan.json"
    plan = json.loads(plan_path.read_text())
    if attack == "manifest_wrong_entry":
        manifest = w / plan["metadata_sources"]["terminal_manifest"]["path"]
        changed = json.loads(manifest.read_text())
        changed["files"][plan["data_sources"]["packet"]["path"]] = {"bytes": 0, "sha256": "f" * 64}
        save(manifest, changed)
    else:
        target_path = w / plan["data_sources"]["local_target"]["path"]
        save(target_path, {"item_id": 166, "human_b_rate": 0.99})
        if attack != "data":
            plan["data_sources"]["local_target"]["fingerprint"] = fingerprint(target_path)
            if attack == "origin_newplan_resign":
                origin_path = w / cli.ORIGIN["path"]
                changed = json.loads(origin_path.read_text())
                changed["data_sources"] = plan["data_sources"]
                save(origin_path, changed)
                plan["origin_plan"]["fingerprint"] = fingerprint(origin_path)
            save(plan_path, plan)
    with pytest.raises(ValueError):
        cli.run(w, w / cli.ITERATION / "results")
    assert not (w / cli.ITERATION / "results").exists()
