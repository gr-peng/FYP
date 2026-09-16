"""Resumable first real pilot: historical 12 + endogenous 36 runs, 30 agents each."""

import argparse
import asyncio
import copy
import json
from pathlib import Path

from behavioral_market.data.archive import write_json
from behavioral_market.llm.client import RequestPacer
from behavioral_market.simulation.event_runner import (
    LIVE_PROVIDERS,
    ROOT,
    output_root,
    run_experiment,
)


async def run_suite(
    config,
    provider,
    modes,
    treatments,
    seeds,
    shocks,
    parallel_runs=2,
    max_rps=4.0,
    cache_only=False,
    event_conditions=None,
    symbols=None,
):
    requests = asyncio.Semaphore(config["llm"]["max_concurrency"])
    pacer = RequestPacer(max_rps)
    runs = asyncio.Semaphore(parallel_runs)
    jobs = []
    results = []
    directory = output_root(config, provider)

    async def execute(spec, mode, treatment):
        async with runs:
            metrics = await run_experiment(
                spec,
                mode,
                treatment,
                provider,
                semaphore=requests,
                pacer=pacer,
                cache_only=cache_only,
            )
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

    event_conditions = event_conditions or [config["events"].get("condition", "e1_e2")]
    symbols = symbols or [config["data"]["primary_symbol"]]
    for symbol in symbols:
        for condition in event_conditions:
            for mode in modes:
                for shock in [0.0] if mode == "historical" else shocks:
                    for seed in seeds:
                        for treatment in treatments:
                            spec = copy.deepcopy(config)
                            spec["experiment"]["seed"] = seed
                            spec["data"]["primary_symbol"] = symbol
                            spec["events"]["condition"] = condition
                            spec["fundamental_shock"]["magnitude"] = shock
                            jobs.append((spec, mode, treatment))
    await asyncio.gather(*(execute(*job) for job in jobs))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument(
        "--provider", choices=["mock", *LIVE_PROVIDERS], default="siliconflow"
    )
    parser.add_argument("--modes", nargs="+", default=["historical", "endogenous"])
    parser.add_argument("--treatments", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shocks", nargs="+", type=float)
    parser.add_argument("--parallel-runs", type=int, default=2)
    parser.add_argument("--max-rps", type=float, default=4.0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--event-conditions", nargs="+", choices=["no_event", "e1_only", "e1_e2"]
    )
    parser.add_argument("--symbols", nargs="+")
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
            args.max_rps,
            args.cache_only,
            args.event_conditions or config["suite"].get("event_conditions"),
            args.symbols or config["suite"].get("symbols"),
        )
    )


if __name__ == "__main__":
    main()
