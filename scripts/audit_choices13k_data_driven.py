#!/usr/bin/env python3
"""Audit a completed Choices13k data-driven calibration experiment offline."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.choices13k import verify_manifest  # noqa: E402
from behavioral_market.benchmarks.data_driven import (  # noqa: E402
    development_methods,
    validate_data_driven_config,
    validate_selected_method,
)
from behavioral_market.benchmarks.evaluation import load_responses  # noqa: E402
from behavioral_market.benchmarks.prepare import load_split  # noqa: E402
from behavioral_market.benchmarks.retrieval import prompt_template_hash  # noqa: E402
from behavioral_market.benchmarks.runner import (  # noqa: E402
    config_hash,
    llm_config,
    output_dir,
    source_hash,
)
from behavioral_market.data.archive import digest, write_json  # noqa: E402


def _audit_matrix(responses, items, methods, repeats, calibration_ids):
    assert len(responses) == len(items) * len(methods) * repeats
    assert set(responses.prompt_id) == set(methods)
    assert set(responses.status) == {"ok"}
    assert set(responses.validated_choice) <= {"A", "B"}
    counts = responses.groupby(["prompt_id", "item_id"]).size()
    assert (counts == repeats).all()
    orders = responses.groupby(["prompt_id", "item_id", "display_order"]).size()
    assert set(orders.index.get_level_values("display_order")) == {"A_first", "B_first"}
    assert (orders == repeats // 2).all()
    for row in responses.itertuples():
        demo_ids = set(row.demonstration_item_ids)
        if row.prompt_id.startswith("B2_"):
            assert demo_ids and demo_ids <= calibration_ids
        else:
            assert not demo_ids


def audit(config, root):
    validate_data_driven_config(config, root)
    manifest_path = root / config["dataset"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest, root)
    out = output_dir(config, root)
    run_manifest = json.loads((out / "manifest.json").read_text())
    assert run_manifest["dataset_manifest_hash"] == digest(manifest_path.read_bytes())
    assert run_manifest["config_hash"] == config_hash(config)
    assert run_manifest["source_hash"] == source_hash()
    assert run_manifest["data_driven_prompt_template_hash"] == prompt_template_hash(
        config["few_shot"]["pseudo_participants_per_item"]
    )
    assert run_manifest["dirty_worktree"] is False
    assert "final_test_git_commit" in run_manifest

    calibration, _ = load_split(config, root, "calibration_pool")
    development, _ = load_split(config, root, "development_validation")
    final, _ = load_split(config, root, "final_test")
    split_ids = [set(frame.item_id) for frame in (calibration, development, final)]
    overlaps = [
        split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)
    ]
    assert all(not ids for ids in overlaps)
    assert all(not frame.feedback.any() for frame in (calibration, development, final))

    selected = validate_selected_method(config, root, require_tracked=True)
    development_responses = load_responses(out / "development/responses.jsonl")
    final_responses = load_responses(out / "final_test/responses.jsonl")
    repeats = config["sampling"]["repeats_per_item"]
    _audit_matrix(
        development_responses,
        development,
        development_methods(config),
        repeats,
        split_ids[0],
    )
    final_methods = ["B0", "B1", selected["selected_method"]]
    _audit_matrix(final_responses, final, final_methods, repeats, split_ids[0])

    load_dotenv(root / ".env")
    live = llm_config(config, run_manifest["provider"])
    secret = os.getenv(live["api_key_env"])
    for path in [
        out / "development/llm_requests.jsonl",
        out / "final_test/llm_requests.jsonl",
    ]:
        text = path.read_text()
        assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", text)
        assert not secret or secret not in text

    result = {
        "verified": True,
        "calibration_items": len(calibration),
        "development_items": len(development),
        "final_test_items": len(final),
        "development_responses": len(development_responses),
        "final_test_responses": len(final_responses),
        "selected_method": selected["selected_method"],
        "primary_population": "no_feedback",
        "cross_split_overlap": 0,
    }
    write_json(out / "evaluation/audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/choices13k_data_driven.json"
    )
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()), ROOT)
