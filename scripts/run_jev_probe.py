#!/usr/bin/env python3
"""Credential status or a four-request contract probe; no human-fidelity claims."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.benchmarks.choices13k import frame_items  # noqa: E402
from behavioral_market.data.archive import digest, write_json  # noqa: E402
from behavioral_market.simulation.event_runner import git_state  # noqa: E402
from experiments.calibration_v2 import display_item, load_plan  # noqa: E402
from experiments.jev_client import JevClient  # noqa: E402


async def probe(key, trust_env=False):
    commit, dirty = git_state(ROOT)
    if dirty:
        raise RuntimeError("commit the Jev probe and protocol before live execution")
    config = json.loads((ROOT / "config/choices13k_calibration_v2.json").read_text())
    plan, frames = load_plan(config, ROOT)
    items = {x.item_id: x for x in frame_items(frames["development"])}
    out = ROOT / "outputs/jev/contract_probe_20260922"
    code_hash = digest(
        b"".join(
            (ROOT / p).read_bytes()
            for p in (
                "experiments/jev_client.py",
                "scripts/run_jev_probe.py",
                "docs/jev_agent_protocol_draft.md",
            )
        )
    )
    client = JevClient(key, out, ROOT / ".cache/jev/contract_probe", trust_env=trust_env)
    records = []
    try:
        for item_id in plan["target_ids"][:2]:
            for orientation in ("A_first", "B_first"):
                state, mapping = display_item(items[item_id], orientation)
                answer = await client.evaluate_choice(
                    state,
                    "Which displayed option would a randomly selected adult participant "
                    "choose for themselves? Use only the supplied information.",
                    {"A": "Choose displayed Option A", "B": "Choose displayed Option B"},
                    {
                        "study": "contract_probe_20260922",
                        "item_id": item_id,
                        "orientation": orientation,
                        "code_hash": code_hash,
                    },
                )
                p_b = sum(p for key, p in answer["probabilities"].items() if mapping[key] == "B")
                records.append(
                    {
                        "item_id": item_id,
                        "orientation": orientation,
                        "original_b_probability": p_b,
                        **answer,
                    }
                )
                print(json.dumps(records[-1]), flush=True)
        write_json(
            out / "summary.json",
            {
                "git_commit": commit,
                "code_hash": code_hash,
                "records": records,
                "environment_proxy_enabled": trust_env,
                "interpretation": "contract smoke only; no fidelity claim",
            },
        )
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["status", "smoke"])
    parser.add_argument("--trust-env", action="store_true", help="use configured environment proxy")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("TYPESAFE_API_KEY") or os.getenv("TYPESAFE_AI_API_KEY")
    if args.stage == "status":
        result_path = ROOT / "outputs/jev/contract_probe_20260922/summary.json"
        print(
            json.dumps(
                {
                    "model": "jev-1.13.0",
                    "direct_key_configured": bool(key),
                    "gateway_key_configured": bool(os.getenv("AI_GATEWAY_API_KEY")),
                    "real_probe_summary_present": result_path.exists(),
                }
            )
        )
    else:
        asyncio.run(probe(key, args.trust_env))
