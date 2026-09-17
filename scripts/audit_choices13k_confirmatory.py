#!/usr/bin/env python3
"""Audit a completed Choices13k confirmatory run without network requests."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.choices13k import verify_manifest  # noqa: E402
from behavioral_market.benchmarks.confirmatory import (  # noqa: E402
    validate_confirmatory_config,
)
from behavioral_market.benchmarks.evaluation import load_responses  # noqa: E402
from behavioral_market.benchmarks.prepare import load_split, split_paths  # noqa: E402
from behavioral_market.benchmarks.runner import (  # noqa: E402
    config_hash,
    llm_config,
    output_dir,
    source_hash,
)
from behavioral_market.data.archive import digest, write_json  # noqa: E402


def audit(config, root):
    validate_confirmatory_config(config, root)
    manifest_path = root / config["dataset"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest, root)
    out = output_dir(config, root)
    run_manifest = json.loads((out / "manifest.json").read_text())
    assert run_manifest["dataset_manifest_hash"] == digest(manifest_path.read_bytes())
    assert run_manifest["config_hash"] == config_hash(config)
    assert run_manifest["source_hash"] == source_hash()
    assert run_manifest["dirty_worktree"] is False

    items, _ = load_split(config, root, "confirmatory")
    pilot = pd.read_parquet(split_paths(config, root)["pilot"])
    assert len(items) == config["dataset"]["confirmatory_items"]
    assert set(items.item_id).isdisjoint(pilot.item_id)
    assert not items.feedback.any()
    assert digest(split_paths(config, root)["confirmatory"].read_bytes()) == config["dataset"][
        "confirmatory_split_sha256"
    ]

    responses = load_responses(out / "confirmatory/responses.jsonl")
    prompt_ids = config["prompts"]["candidate_ids"]
    repeats = config["sampling"]["repeats_per_item"]
    assert len(responses) == len(items) * len(prompt_ids) * repeats
    assert set(responses.prompt_id) == set(prompt_ids)
    assert set(responses.status) == {"ok"}
    assert set(responses.validated_choice) <= {"A", "B"}
    counts = responses.groupby(["prompt_id", "item_id"]).size()
    assert (counts == repeats).all()
    order_counts = responses.groupby(["prompt_id", "item_id", "display_order"]).size()
    assert set(order_counts.index.get_level_values("display_order")) == {
        "A_first",
        "B_first",
    }
    assert (order_counts == repeats // 2).all()

    load_dotenv(root / ".env")
    live = llm_config(config, run_manifest["provider"])
    secret = os.getenv(live["api_key_env"])
    request_log = out / "confirmatory/llm_requests.jsonl"
    log_text = request_log.read_text()
    assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", log_text)
    assert not secret or secret not in log_text

    result = {
        "verified": True,
        "items": len(items),
        "responses": len(responses),
        "prompts": prompt_ids,
        "primary_population": "no_feedback",
        "pilot_overlap": 0,
        "split_sha256": config["dataset"]["confirmatory_split_sha256"],
    }
    write_json(out / "evaluation/audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/choices13k_confirmatory.json",
    )
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()), ROOT)
