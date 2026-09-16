import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.client import OpenAICompatibleClient, RequestPacer
from behavioral_market.simulation.event_runner import git_state, source_hash

from .choices13k import frame_items
from .evaluation import evaluate_stage, load_responses
from .prepare import load_split, prepare
from .prompts import PROMPT_VERSION, build_prompt, prompt_registry
from .sampling import orientations
from .schemas import BenchmarkChoice


def output_dir(config, root):
    return root / config["experiment"]["output_dir"]


def llm_config(config, provider):
    common = {
        key: value
        for key, value in config["llm"].items()
        if key not in {"providers", "provider", "max_rps"}
    }
    return {**common, **config["llm"]["providers"][provider]}


def config_hash(config):
    return digest(json.dumps(config, sort_keys=True).encode())


def prerequisite_status(config, root):
    out = output_dir(config, root)
    return {
        "data_prepared": (root / config["dataset"]["manifest"]).exists(),
        "calibration_responses": (out / "calibration/responses.jsonl").exists(),
        "prompt_frozen": (out / "selected_prompt.json").exists(),
        "heldout_responses": (out / "heldout/responses.jsonl").exists(),
        "final_report": (out / "evaluation/report.md").exists(),
    }


def planned_calls(config):
    repeats = config["sampling"]["repeats_per_item"]
    candidates = len(config["prompts"]["candidate_ids"])
    calibration = config["dataset"]["calibration_items"] * repeats * candidates
    maximum_heldout = config["dataset"]["heldout_items"] * repeats * 2
    return {
        "calibration": calibration,
        "heldout_maximum": maximum_heldout,
        "pilot_maximum": calibration + maximum_heldout,
        "note": "heldout is half this maximum if P0 wins calibration",
    }


def write_run_manifest(config, root, provider):
    out = output_dir(config, root)
    out.mkdir(parents=True, exist_ok=True)
    dataset_manifest = root / config["dataset"]["manifest"]
    commit, dirty = git_state(root)
    value = {
        "experiment": config["experiment"]["name"],
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "model": "mock" if provider == "mock" else llm_config(config, provider)["model"],
        "git_commit": commit,
        "dirty_worktree": dirty,
        "source_hash": source_hash(),
        "config_hash": config_hash(config),
        "dataset_manifest_hash": digest(dataset_manifest.read_bytes()),
        "prompt_registry_hash": digest((root / config["prompts"]["file"]).read_bytes()),
        "prompt_version": PROMPT_VERSION,
    }
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        fixed = [
            "provider",
            "model",
            "git_commit",
            "source_hash",
            "config_hash",
            "dataset_manifest_hash",
            "prompt_registry_hash",
            "prompt_version",
        ]
        if any(old.get(key) != value.get(key) for key in fixed):
            raise RuntimeError("existing benchmark output belongs to different frozen inputs")
        return old
    write_json(path, value)
    write_json(out / "config_snapshot.json", config)
    return value


def response_identity(split, prompt_id, item_id, repeat_id):
    return (split, prompt_id, int(item_id), int(repeat_id))


class MockBenchmarkClient:
    async def verify_model(self):
        return "mock"

    async def close(self):
        return None

    async def choose(self, system, user, identity):
        hashed = hashlib.sha256((system + user + json.dumps(identity)).encode()).hexdigest()
        number = int(hashed, 16)
        choice = "A" if number % 2 == 0 else "B"
        return choice, json.dumps({"choice": choice}), {
            "status": "ok",
            "cache_hit": False,
            "retry_count": 0,
            "repair_count": 0,
            "usage": {},
        }


class LiveBenchmarkClient:
    def __init__(self, client):
        self.client = client

    async def verify_model(self):
        return await self.client.verify_model()

    async def close(self):
        await self.client.close()

    async def choose(self, system, user, identity):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_raw = None
        last_meta = {}
        for repair in range(2):
            body, meta = await self.client.completion(messages, {**identity, "repair": repair})
            last_meta = meta
            if body is None:
                return None, last_raw, {
                    **meta,
                    "status": "transport_exhausted",
                    "repair_count": repair,
                }
            try:
                last_raw = body["choices"][0]["message"]["content"]
                choice = BenchmarkChoice.model_validate_json(last_raw).choice
                return choice, last_raw, {**meta, "status": "ok", "repair_count": repair}
            except (ValidationError, KeyError, IndexError, TypeError):
                if repair == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Return only valid JSON with exactly one field: "
                                '{"choice":"A"} or {"choice":"B"}.'
                            ),
                        }
                    )
        return None, last_raw, {
            **last_meta,
            "status": "invalid_json_or_schema",
            "repair_count": 1,
        }


def make_client(config, provider, root, stage_dir, cache_only, max_rps):
    if provider == "mock":
        return MockBenchmarkClient()
    load_dotenv(root / ".env")
    live = llm_config(config, provider)
    client = OpenAICompatibleClient(
        live,
        os.getenv(live["api_key_env"]),
        stage_dir,
        root / live["cache_dir"],
        pacer=RequestPacer(max_rps),
        cache_only=cache_only,
        cache_namespace=PROMPT_VERSION,
    )
    return LiveBenchmarkClient(client)


async def run_responses(
    config,
    root,
    split,
    prompt_ids,
    provider,
    cache_only=False,
    max_rps=None,
    allow_dirty=False,
):
    if provider != "mock":
        _, dirty = git_state(root)
        if dirty and not allow_dirty:
            raise RuntimeError("live benchmark stages require a clean committed worktree")
    items, _ = load_split(config, root, split)
    item_models = {item.item_id: item for item in frame_items(items)}
    _, registry = prompt_registry(root / config["prompts"]["file"])
    unknown = set(prompt_ids) - set(registry)
    if unknown:
        raise ValueError(f"unknown prompt IDs: {sorted(unknown)}")
    destination = output_dir(config, root) / split
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "responses.jsonl"
    completed = set()
    if path.exists():
        old = load_responses(path)
        completed = {
            response_identity(*row)
            for row in old[["split", "prompt_id", "item_id", "repeat_id"]].itertuples(
                index=False, name=None
            )
        }
    repeats = config["sampling"]["repeats_per_item"]
    jobs = []
    for item_id in sorted(item_models):
        item_orientations = orientations(item_id, repeats, config["experiment"]["seed"])
        for prompt_id in prompt_ids:
            for repeat_id, orientation in enumerate(item_orientations):
                identity = response_identity(split, prompt_id, item_id, repeat_id)
                if identity not in completed:
                    jobs.append((*identity, orientation))
    client = make_client(
        config,
        provider,
        root,
        destination,
        cache_only,
        max_rps or config["llm"]["max_rps"],
    )
    await client.verify_model()

    async def execute(split_name, prompt_id, item_id, repeat_id, orientation):
        item = item_models[item_id]
        system, user, display = build_prompt(item, registry[prompt_id], orientation)
        identity = {
            "run_id": config["experiment"]["name"],
            "split": split_name,
            "prompt_id": prompt_id,
            "item_id": item_id,
            "problem_id": item.problem_id,
            "repeat_id": repeat_id,
            "display_order": orientation,
            "model": "mock" if provider == "mock" else llm_config(config, provider)["model"],
            "temperature": config["llm"]["temperature"],
        }
        started = time.perf_counter()
        choice, raw, meta = await client.choose(system, user, identity)
        usage = meta.get("usage", {})
        record = {
            **identity,
            "raw_response": raw,
            "validated_choice": choice,
            "original_b_choice": display.get(choice) == "B" if choice else None,
            "first_display_selected": choice == "A" if choice else None,
            "latency_seconds": time.perf_counter() - started,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "retry_count": meta.get("retry_count", 0),
            "repair_count": meta.get("repair_count", 0),
            "cache_hit": meta.get("cache_hit", False),
            "status": meta["status"],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    try:
        await asyncio.gather(*(execute(*job) for job in jobs))
    finally:
        await client.close()
    responses = load_responses(path)
    expected = len(items) * len(prompt_ids) * repeats
    if len(responses) != expected:
        raise RuntimeError(f"incomplete {split} matrix: {len(responses)}/{expected}")
    return responses, items


def evaluate_calibration(config, root):
    out = output_dir(config, root)
    responses = load_responses(out / "calibration/responses.jsonl")
    items, _ = load_split(config, root, "calibration")
    return evaluate_stage(
        responses,
        items,
        config["sampling"]["repeats_per_item"],
        out / "calibration",
        config,
    )


def freeze_prompt(config, root):
    out = output_dir(config, root)
    rates, metrics = evaluate_calibration(config, root)
    candidates = config["prompts"]["candidate_ids"]
    if set(metrics) != set(candidates):
        raise RuntimeError("calibration response matrix does not contain every candidate prompt")
    gate = config["evaluation"]["invalid_rate_gate"]
    if any(metrics[prompt_id]["invalid_rate"] > gate for prompt_id in candidates):
        raise RuntimeError("calibration invalid rate exceeds the preregistered gate")
    tie_order = {
        prompt_id: index
        for index, prompt_id in enumerate(config["prompts"]["tie_break_order"])
    }
    selected = min(
        candidates,
        key=lambda prompt_id: (metrics[prompt_id]["mae"], tie_order[prompt_id]),
    )
    calibration_path = root / config["dataset"]["processed_dir"] / "calibration_items.parquet"
    prompts_path = root / config["prompts"]["file"]
    value = {
        "selected_prompt": selected,
        "selection_metric": config["calibration"]["selection_metric"],
        "candidate_metrics": metrics,
        "calibration_split_hash": digest(calibration_path.read_bytes()),
        "calibration_item_rates_hash": digest(
            (out / "calibration/item_rates.parquet").read_bytes()
        ),
        "prompt_registry_hash": digest(prompts_path.read_bytes()),
        "config_hash": config_hash(config),
        "source_hash": source_hash(),
        "frozen_at": datetime.now(UTC).isoformat(),
        "items": len(rates),
    }
    path = out / "selected_prompt.json"
    if path.exists():
        old = json.loads(path.read_text())
        comparable = {key: value[key] for key in value if key != "frozen_at"}
        old_comparable = {key: old.get(key) for key in comparable}
        if old_comparable != comparable:
            raise RuntimeError("refusing to overwrite a different frozen prompt selection")
        return old
    write_json(path, value)
    return value


def frozen_prompt(config, root):
    out = output_dir(config, root)
    path = out / "selected_prompt.json"
    if not path.exists():
        raise RuntimeError("prompt is not frozen; run --stage freeze first")
    value = json.loads(path.read_text())
    expected = {
        "calibration_split_hash": digest(
            (root / config["dataset"]["processed_dir"] / "calibration_items.parquet").read_bytes()
        ),
        "prompt_registry_hash": digest((root / config["prompts"]["file"]).read_bytes()),
        "config_hash": config_hash(config),
        "source_hash": source_hash(),
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise RuntimeError("frozen prompt provenance no longer matches current inputs")
    return value


def final_evaluation(config, root):
    frozen = frozen_prompt(config, root)
    out = output_dir(config, root)
    responses = load_responses(out / "heldout/responses.jsonl")
    items, _ = load_split(config, root, "heldout")
    rates, metrics = evaluate_stage(
        responses,
        items,
        config["sampling"]["repeats_per_item"],
        out / "heldout",
        config,
        frozen["selected_prompt"],
    )
    evaluation = out / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ["subgroup_metrics.csv", "calibration_curve.csv", "bootstrap.csv"]:
        shutil.copyfile(out / "heldout" / name, evaluation / name)
    selected = frozen["selected_prompt"]
    vanilla = config["prompts"]["vanilla_id"]
    bootstrap_frame = pd.read_csv(evaluation / "bootstrap.csv")
    delta = bootstrap_frame[bootstrap_frame.kind == "paired_delta"].mae
    ci = [float(delta.quantile(0.025)), float(delta.quantile(0.975))]
    intervals = {}
    for (kind, prompt_id), group in bootstrap_frame.groupby(["kind", "prompt_id"]):
        intervals[f"{kind}:{prompt_id}"] = {
            metric: [float(group[metric].quantile(0.025)), float(group[metric].quantile(0.975))]
            for metric in ["mae", "mean_js"]
        }
    write_json(evaluation / "bootstrap_intervals.json", intervals)
    subgroup = pd.read_csv(evaluation / "subgroup_metrics.csv")
    vanilla_groups = subgroup[subgroup.prompt_id == vanilla]
    selected_groups = subgroup[subgroup.prompt_id == selected]
    subgroup_comparison = vanilla_groups.merge(
        selected_groups,
        on=["dimension", "subgroup"],
        suffixes=("_vanilla", "_selected"),
    )
    subgroup_comparison["delta_mae_selected_minus_vanilla"] = (
        subgroup_comparison.mae_selected - subgroup_comparison.mae_vanilla
    )
    subgroup_comparison.to_csv(evaluation / "subgroup_comparisons.csv", index=False)
    warning = any(
        metric["invalid_rate"] > config["evaluation"]["invalid_rate_gate"]
        for metric in metrics.values()
    )
    degeneracy_warning = any(
        metric["degenerate_item_rate"] is not None
        and metric["degenerate_item_rate"] > config["evaluation"]["degenerate_item_rate_gate"]
        for metric in metrics.values()
    )
    report = [
        "# Choices13k DeepSeek pilot",
        "",
        f"Selected calibration prompt: `{selected}`; held-out prompts: {', '.join(metrics)}.",
        "",
        "| Prompt | MAE | RMSE | mean JS | Pearson | Spearman | invalid | first-position |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def display(value, percent=False):
        if value is None:
            return "NA"
        return f"{value:.2%}" if percent else f"{value:.4f}"

    for prompt_id, metric in metrics.items():
        report.append(
            f"| {prompt_id} | {display(metric['mae'])} | {display(metric['rmse'])} | "
            f"{display(metric['mean_js'])} | {display(metric['pearson'])} | "
            f"{display(metric['spearman'])} | {display(metric['invalid_rate'], True)} | "
            f"{display(metric['first_option_rate'], True)} |"
        )
    if selected == vanilla:
        conclusion = "P0 won calibration; no non-vanilla candidate advanced to held-out."
    else:
        observed = metrics[selected]["mae"] - metrics[vanilla]["mae"]
        conclusion = (
            f"Held-out paired ΔMAE ({selected} − {vanilla}) = {observed:.4f}; "
            f"bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]."
        )
    report += [
        "",
        conclusion,
        "",
        f"Invalid-rate gate exceeded: {warning}. Statistical unit: item "
        f"({len(rates)} item-prompt rows).",
        f"Sampling-degeneracy diagnostic exceeded: {degeneracy_warning}.",
        "",
        "This public-data pilot may be affected by benchmark contamination. Feedback rows "
        "are diagnostic only because participant-level realized feedback trajectories are "
        "unavailable.",
    ]
    (evaluation / "report.md").write_text("\n".join(report) + "\n")
    write_json(
        evaluation / "summary.json",
        {
            "selected_prompt": selected,
            "metrics": metrics,
            "bootstrap_intervals": intervals,
            "delta_mae_ci": ci,
            "invalid_rate_warning": warning,
            "sampling_degeneracy_warning": degeneracy_warning,
        },
    )
    return metrics


async def run_stage(
    config,
    root,
    stage,
    provider=None,
    cache_only=False,
    max_rps=None,
    allow_dirty=False,
):
    provider = provider or config["llm"]["provider"]
    if stage == "plan":
        return {"calls": planned_calls(config), "status": prerequisite_status(config, root)}
    if stage == "prepare":
        return prepare(config, root)
    if stage in {"calibration", "heldout"} and provider != "mock":
        _, dirty = git_state(root)
        if dirty and not allow_dirty:
            raise RuntimeError("live benchmark stages require a clean committed worktree")
    if provider != "mock":
        if provider not in config["llm"]["providers"]:
            raise ValueError(f"unsupported provider {provider}")
        write_run_manifest(config, root, provider)
    elif not (root / config["dataset"]["manifest"]).exists():
        raise RuntimeError("prepare data before running mock responses")
    if stage == "calibration":
        return await run_responses(
            config,
            root,
            "calibration",
            config["prompts"]["candidate_ids"],
            provider,
            cache_only,
            max_rps,
            allow_dirty,
        )
    if stage == "freeze":
        return freeze_prompt(config, root)
    if stage == "heldout":
        selected = frozen_prompt(config, root)["selected_prompt"]
        prompt_ids = list(dict.fromkeys([config["prompts"]["vanilla_id"], selected]))
        return await run_responses(
            config,
            root,
            "heldout",
            prompt_ids,
            provider,
            cache_only,
            max_rps,
            allow_dirty,
        )
    if stage == "evaluate":
        return final_evaluation(config, root)
    raise ValueError(f"unsupported stage {stage}")
