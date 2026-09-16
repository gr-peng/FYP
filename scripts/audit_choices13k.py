#!/usr/bin/env python3
"""Audit a completed Choices13k pilot without making network requests."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.choices13k import verify_manifest  # noqa: E402
from behavioral_market.benchmarks.evaluation import load_responses  # noqa: E402
from behavioral_market.benchmarks.prepare import load_split  # noqa: E402
from behavioral_market.benchmarks.runner import config_hash, output_dir, source_hash  # noqa: E402
from behavioral_market.data.archive import digest, write_json  # noqa: E402


def audit(config, root):
    out = output_dir(config, root)
    manifest = json.loads((root / config["dataset"]["manifest"]).read_text())
    verify_manifest(manifest, root)
    run_manifest = json.loads((out / "manifest.json").read_text())
    assert run_manifest["dataset_manifest_hash"] == digest(
        (root / config["dataset"]["manifest"]).read_bytes()
    )
    assert run_manifest["config_hash"] == config_hash(config)
    assert run_manifest["source_hash"] == source_hash()
    calibration, _ = load_split(config, root, "calibration")
    heldout, _ = load_split(config, root, "heldout")
    selected = json.loads((out / "selected_prompt.json").read_text())
    assert selected["selected_prompt"] in config["prompts"]["candidate_ids"]
    assert selected["calibration_split_hash"] == digest(
        (root / config["dataset"]["processed_dir"] / "calibration_items.parquet").read_bytes()
    )
    results = {}
    for split, items, prompt_ids in [
        ("calibration", calibration, config["prompts"]["candidate_ids"]),
        (
            "heldout",
            heldout,
            list(dict.fromkeys([config["prompts"]["vanilla_id"], selected["selected_prompt"]])),
        ),
    ]:
        responses = load_responses(out / split / "responses.jsonl")
        expected = len(items) * len(prompt_ids) * config["sampling"]["repeats_per_item"]
        assert len(responses) == expected
        assert set(responses.prompt_id) == set(prompt_ids)
        assert set(responses.status) == {"ok"}
        assert set(responses.validated_choice) <= {"A", "B"}
        counts = responses.groupby(["prompt_id", "item_id"]).size()
        assert (counts == config["sampling"]["repeats_per_item"]).all()
        for item_id in items.item_id:
            for prompt_id in prompt_ids:
                group = responses[
                    (responses.item_id == item_id) & (responses.prompt_id == prompt_id)
                ]
                assert set(group.display_order) == {"A_first", "B_first"}
                assert (group.display_order == "A_first").sum() == (
                    config["sampling"]["repeats_per_item"] // 2
                )
        results[split] = {"responses": len(responses), "items": len(items), "prompts": prompt_ids}
    for path in [out / "calibration/llm_requests.jsonl", out / "heldout/llm_requests.jsonl"]:
        text = path.read_text()
        assert not re.search(r"sk-[A-Za-z0-9]{20,}|c96b0738b70c48cd", text)
    audit_result = {
        "verified": True,
        "calibration": results["calibration"],
        "heldout": results["heldout"],
    }
    write_json(out / "evaluation/audit.json", audit_result)
    print(json.dumps(audit_result, indent=2))
    return audit_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/choices13k_pilot.json")
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()), ROOT)
