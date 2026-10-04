"""Check pinned model, native thinking toggle, and actual order JSON before a matrix."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behavioral_market.data.archive import digest, write_json  # noqa: E402
from behavioral_market.llm.client import OpenAICompatibleClient, RequestPacer  # noqa: E402
from behavioral_market.llm.schemas import OrderDecision  # noqa: E402
from behavioral_market.simulation.event_runner import git_state, llm_config  # noqa: E402
from experiments.two_event_data import verify_manifest  # noqa: E402
from experiments.two_event_market import SYSTEM, code_hash, context, setup  # noqa: E402


async def probe(config, provider):
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("freeze probe before live calls")
    study = json.loads((ROOT / config["data_config"]).read_text())
    directory = ROOT / "data/processed" / study["study_id"]
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_manifest(manifest, ROOT)
    output = ROOT / "outputs" / study["study_id"] / config["experiment_id"] / "contract_probe"
    provider_spec = llm_config(json.loads((ROOT / config["provider_config"]).read_text()), provider)
    provider_spec.update(config["provider_overrides"])
    load_dotenv(ROOT / ".env")
    client = OpenAICompatibleClient(
        provider_spec,
        os.getenv(provider_spec["api_key_env"], ""),
        output,
        ROOT / ".cache" / config["experiment_id"] / "contract_probe",
        pacer=RequestPacer(2),
        cache_namespace="two-event-provider-contract-v1",
    )
    metadata = {
        "status": "running",
        "git_commit": commit,
        "source_hash": code_hash(),
        "config_hash": digest(json.dumps(config, sort_keys=True).encode()),
        "provider_config": provider_spec,
    }
    write_json(output / "metadata.json", metadata)
    try:
        await client.verify_model()
        results = []
        for spec in config["scenarios"]:
            bars = pd.read_parquet(directory / f"{spec['scenario_id']}_daily.parquet")
            agents, targets, env = setup(config, spec, bars, 42, "finite")
            env.start_session(spec["start"])
            env.requote_background()
            for treatment in config["treatments"]:
                for condition in config["event_conditions"]:
                    payload = context(
                        config,
                        spec,
                        agents[0],
                        targets[0],
                        env,
                        spec["events"][-1]["at"],
                        condition,
                        treatment,
                        {},
                    )
                    identity = {
                        "scenario": spec["scenario_id"],
                        "treatment": treatment,
                        "condition": condition,
                        "task": "contract_probe",
                    }
                    body, response_meta = await client.completion(
                        [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                        ],
                        identity,
                    )
                    if body is None:
                        raise RuntimeError("probe transport exhausted")
                    choice = body["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise RuntimeError("probe response was truncated or did not stop normally")
                    decision = OrderDecision.model_validate_json(choice["message"]["content"])
                    results.append({**identity, "decision": decision.model_dump(), **response_meta})
                    write_json(output / "results.json", results)
                    print(
                        json.dumps(
                            {**identity, "status": "valid", "cache_hit": response_meta["cache_hit"]}
                        ),
                        flush=True,
                    )
        metadata.update(status="complete", valid=len(results))
        write_json(output / "metadata.json", metadata)
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "metadata.json", metadata)
        raise
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/two_event_market_pilot_v2.json"
    )
    parser.add_argument("--provider", default="aigc_relay", choices=["aigc_relay", "siliconflow"])
    args = parser.parse_args()
    asyncio.run(probe(json.loads(args.config.read_text()), args.provider))


if __name__ == "__main__":
    main()
