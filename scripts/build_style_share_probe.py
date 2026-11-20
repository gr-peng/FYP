#!/usr/bin/env python3
"""Prepare four paired inputs from a fixed opened archive, without executing a model."""

import argparse
import json
import os
import resource
from pathlib import Path

import pandas as pd

from behavioral_market.evaluation.style_diagnostics import bind_archived_answer, normalize_review
from behavioral_market.evaluation.style_probe_materials import (
    build_share_probes,
    validate_share_probes,
)
from scripts.audit_style_switching import (
    checked_path,
    identity,
    load_contexts,
    validate_plan,
    verify_answer,
)

SOURCE_FOLDER = (
    "behavioral-llm-market/outputs/meta_compute_2026/aigc_relay/historical/"
    "vanilla_llm/n30_seed42_shock0_e21"
)
REVIEW_DATE = "2026-06-29"


def prepare_sources(workspace, plan):
    """Select by predecision style and ID; never select by whether the agent switched."""
    validate_plan(plan)
    names = (
        "run_metadata.json",
        "config_snapshot.json",
        "style_switches.parquet",
        "llm_requests.jsonl",
        "llm_responses.jsonl",
    )
    expected = {
        SOURCE_FOLDER + "/" + name: plan["source_snapshot"][SOURCE_FOLDER + "/" + name]
        for name in names
    }
    for relative, fingerprint in expected.items():
        if identity(checked_path(workspace, relative)) != fingerprint:
            raise ValueError("Archived probe source differs from its fixed snapshot")
    folder = checked_path(workspace, SOURCE_FOLDER)
    meta = json.loads((folder / "run_metadata.json").read_text())
    required = {
        "status": "complete",
        "seed": 42,
        "treatment": "vanilla_llm",
        "mode": "historical",
        "n_agents": 30,
        "sessions": 23,
        "shock": 0,
        "include_secondary": True,
    }
    if any(meta[name] != value for name, value in required.items()):
        raise ValueError("Source run does not match the fixed minimal design")
    table = folder / "style_switches.parquet"
    if identity(table)["sha256"] != meta["artifact_hashes"][table.name]:
        raise ValueError("Archived table differs from its original publication hash")
    rows = pd.read_parquet(table, use_threads=False).to_dict("records")
    keys = {(row["agent_id"], row["session_date"]) for row in rows}
    fixed = {(f"A{i:03}", day) for i in range(30) for day in plan["review_dates"]}
    if len(rows) != len(keys) or keys != fixed:
        raise ValueError("Source table is missing or duplicating review identities")
    requests, responses, contexts, _counts = load_contexts(folder, meta, keys)
    sources = []
    for style in ("fundamental", "technical"):
        candidates = sorted(
            (
                row
                for row in rows
                if row["session_date"] == REVIEW_DATE and row["from_style"] == style
            ),
            key=lambda row: row["agent_id"],
        )
        if not candidates:
            raise ValueError("A fixed initial strategy is missing")
        row = candidates[0]
        matched = verify_answer(row, requests, responses)
        record = bind_archived_answer(
            normalize_review(
                row, SOURCE_FOLDER, 42, "vanilla_llm", contexts[(row["agent_id"], REVIEW_DATE)]
            ),
            matched,
        )
        if not record["valid_choice"] or matched is None:
            raise ValueError("The predetermined source lacks a valid verified answer")
        request = requests[matched]
        if request["repair"] != 0 or request["payload"]["model"] != meta["model"]:
            raise ValueError("Use only a verified initial prompt with the archived model setting")
        sources.append(
            {
                "identity": {
                    "run_id": SOURCE_FOLDER,
                    "review_date": REVIEW_DATE,
                    "seed": 42,
                    "treatment": "vanilla_llm",
                    "agent_id": row["agent_id"],
                    "from_style": style,
                    "original_request_hash": matched,
                    "matched_answer_request_hash": matched,
                    "original_prompt_version": meta["prompt_version"],
                    "original_source_hash": meta["source_hash"],
                },
                "payload": request["payload"],
            }
        )
    return sources, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--archive-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 2**30, 8 * 2**30))
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; preserve earlier material sets")
    workspace = args.workspace.resolve()
    plan_identity = identity(args.archive_plan)
    plan = json.loads(args.archive_plan.read_text())
    sources, inputs = prepare_sources(workspace, plan)
    cases = build_share_probes(sources)
    checks = validate_share_probes(cases, sources)
    for relative, expected in inputs.items():
        if identity(checked_path(workspace, relative)) != expected:
            raise ValueError("Source changed while building materials")
    if identity(args.archive_plan) != plan_identity:
        raise ValueError("Archive plan changed while building materials")
    checks.update(
        new_model_calls=0,
        new_market_runs=0,
        material_namespace="style-share-probe-v1",
        new_inputs_are_archived_answers=False,
        human_alignment_established=False,
        causal_effect_established=False,
    )
    receipt = {
        "inputs": inputs,
        "archive_plan": plan_identity,
        "selection": (
            "First agent ID per predecision style on the fixed date; no outcome filtering."
        ),
        "source_identities": [s["identity"] for s in sources],
        "source_payloads": {s["identity"]["agent_id"]: s["payload"] for s in sources},
        "source_hashes_match_before_and_after": True,
        "provenance_limit": (
            "Current fingerprints and request/answer linkage, not an original external signature."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n" for case in cases)
    )
    for name, data in (("checks.json", checks), ("source_receipt.json", receipt)):
        (args.output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"output": str(args.output), **checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
