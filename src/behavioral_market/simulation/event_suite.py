"""Resumable first real pilot: historical 12 + endogenous 36 runs, 30 agents each."""

import argparse
import asyncio
import copy
import json
from pathlib import Path

from behavioral_market.data.archive import write_json
from behavioral_market.simulation.event_runner import ROOT, run_experiment


async def run_suite(config, provider, modes, treatments, seeds, shocks, parallel_runs=2):
    requests = asyncio.Semaphore(config["llm"]["max_concurrency"])
    runs = asyncio.Semaphore(parallel_runs)
    jobs = []
    results = []
    directory = ROOT / "outputs/meta_compute_2026" / provider

    async def execute(spec, mode, treatment):
        async with runs:
            metrics = await run_experiment(spec, mode, treatment, provider, semaphore=requests)
            results.append(
                {
                    "mode": mode,
                    "treatment": treatment,
                    "seed": spec["experiment"]["seed"],
                    "shock": spec["fundamental_shock"]["magnitude"],
                    **metrics,
                }
            )
            write_json(
                directory / "suite_progress.json",
                {
                    "completed_in_invocation": len(results),
                    "planned_in_invocation": len(jobs),
                    "results": results,
                },
            )
            if metrics["fallback_rate"] > 0.05:
                raise RuntimeError("pilot fallback rate exceeds 5%; inspect logs before expansion")

    for mode in modes:
        for shock in [0.0] if mode == "historical" else shocks:
            for seed in seeds:
                for treatment in treatments:
                    spec = copy.deepcopy(config)
                    spec["experiment"]["seed"] = seed
                    spec["fundamental_shock"]["magnitude"] = shock
                    jobs.append((spec, mode, treatment))
    await asyncio.gather(*(execute(*job) for job in jobs))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument("--provider", choices=["mock", "siliconflow"], default="siliconflow")
    parser.add_argument("--modes", nargs="+", default=["historical", "endogenous"])
    parser.add_argument("--treatments", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shocks", nargs="+", type=float)
    parser.add_argument("--parallel-runs", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    asyncio.run(
        run_suite(
            config,
            args.provider,
            args.modes,
            args.treatments or config["suite"]["treatments"],
            args.seeds or config["suite"]["seeds"],
            args.shocks if args.shocks is not None else config["suite"]["shocks"],
            args.parallel_runs,
        )
    )


if __name__ == "__main__":
    main()
