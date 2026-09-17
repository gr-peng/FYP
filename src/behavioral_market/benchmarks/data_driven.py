"""Data-driven Choices13k few-shot calibration experiment."""

import asyncio
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime

import pandas as pd

from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation.event_runner import git_state, source_hash

from .choices13k import frame_items, verify_manifest
from .evaluation import bootstrap, evaluate_stage, load_responses
from .prepare import load_split, split_paths
from .prompts import build_prompt, prompt_registry
from .retrieval import (
    build_few_shot_prompt,
    prompt_template_hash,
    select_demonstrations,
)
from .runner import (
    config_hash,
    llm_config,
    make_client,
    output_dir,
    response_identity,
    write_run_manifest,
)
from .sampling import data_driven_splits, orientations

FROZEN_BASELINES = {
    "B0": "Choose the option you prefer.",
    "B1": "Respond as an ordinary adult participant making the choice for themselves.",
}


def b2_method_id(strategy, k):
    return f"B2_{strategy}_k{k}"


def b2_method_specs(config):
    return {
        b2_method_id(strategy, k): {"retrieval_strategy": strategy, "k": k}
        for strategy in config["few_shot"]["retrieval_strategies"]
        for k in config["few_shot"]["k_values"]
    }


def development_methods(config):
    return ["B0", "B1", *config["development"]["tie_break_order"]]


def validate_data_driven_config(config, root):
    """Reject drift from the preregistered first data-driven experiment."""
    dataset = config["dataset"]
    sampling = config["sampling"]
    llm = config["llm"]
    evaluation = config["evaluation"]
    few_shot = config["few_shot"]
    _, registry = prompt_registry(root / config["prompts"]["file"])
    specs = b2_method_specs(config)
    checks = {
        "primary no-feedback population": dataset["primary_feedback"] is False,
        "three EV-gap strata": dataset["ev_gap_quantiles"] == 3,
        "split sizes": [
            dataset["calibration_pool_items"],
            dataset["development_validation_items"],
            dataset["final_test_items"],
        ]
        == [600, 300, 400],
        "10 repeats": sampling["repeats_per_item"] == 10,
        "counterbalancing": sampling["counterbalance_option_order"] is True,
        "temperature 0.7": llm["temperature"] == 0.7,
        "thinking disabled": llm["enable_thinking"] is False,
        "max tokens 32": llm["max_tokens"] == 32,
        "provider": llm["provider"] == "aigc_relay",
        "pseudo-human count": few_shot["pseudo_participants_per_item"] == 10,
        "retrieval strategies": few_shot["retrieval_strategies"] == ["random", "matched"],
        "K values": few_shot["k_values"] == [4, 8, 16],
        "baseline text": {
            "B0": registry.get(config["prompts"]["B0_registry_id"]),
            "B1": registry.get(config["prompts"]["B1_registry_id"]),
        }
        == FROZEN_BASELINES,
        "tie order": config["development"]["tie_break_order"] == list(specs),
        "primary metric MAE": evaluation["primary_metric"] == "mae",
        "5000 bootstraps": evaluation["bootstrap_samples"] == 5000,
        "consensus threshold": evaluation["strong_consensus_threshold"] == 0.8,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("data-driven protocol drift: " + ", ".join(failures))
    for split in ("calibration_pool", "development_validation", "final_test"):
        value = dataset[f"{split}_sha256"]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{split} hash is not frozen")


def build_data_driven_splits(config, root):
    dataset = config["dataset"]
    base_manifest = json.loads((root / dataset["base_manifest"]).read_text())
    confirmatory_manifest = json.loads((root / dataset["confirmatory_manifest"]).read_text())
    verify_manifest(base_manifest, root)
    verify_manifest(confirmatory_manifest, root)
    paths = split_paths(config, root)
    all_items = pd.read_parquet(paths["all"])
    pilot = pd.read_parquet(paths["pilot"])
    confirmatory = pd.read_parquet(paths["confirmatory"])
    if digest(paths["pilot"].read_bytes()) != dataset["excluded_pilot_sha256"]:
        raise ValueError("pilot exclusion hash mismatch")
    if digest(paths["confirmatory"].read_bytes()) != dataset["excluded_confirmatory_sha256"]:
        raise ValueError("confirmatory exclusion hash mismatch")
    excluded = set(pilot.item_id) | set(confirmatory.item_id)
    return data_driven_splits(
        all_items,
        excluded,
        dataset["calibration_pool_items"],
        dataset["development_validation_items"],
        dataset["final_test_items"],
        config["experiment"]["seed"],
        dataset["ev_gap_quantiles"],
    )


def prepare_data_driven(config, root):
    """Materialize and hash all three frozen no-feedback splits."""
    dataset = config["dataset"]
    base_manifest_path = root / dataset["base_manifest"]
    confirmatory_manifest_path = root / dataset["confirmatory_manifest"]
    base_manifest = json.loads(base_manifest_path.read_text())
    verify_manifest(base_manifest, root)
    splits = build_data_driven_splits(config, root)
    names = ("calibration_pool", "development_validation", "final_test")
    paths = split_paths(config, root)
    records = []
    split_manifest = {
        "seed": config["experiment"]["seed"],
        "primary_feedback": False,
    }
    for name, frame in zip(names, splits, strict=True):
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        actual = digest(path.read_bytes())
        expected = dataset[f"{name}_sha256"]
        if actual != expected:
            raise ValueError(f"{name} split hash mismatch: expected {expected}, got {actual}")
        records.append({"path": str(path.relative_to(root)), "sha256": actual})
        split_manifest[f"{name}_items"] = len(frame)
        split_manifest[f"{name}_sha256"] = actual
    manifest = {
        "dataset": "choices13k_data_driven_calibration",
        "source_repo": dataset["source_repo"],
        "source_commit": dataset["source_commit"],
        "row_identity": "zero_based_csv_row_index_equals_c13k_problems_json_key",
        "base_manifest": str(base_manifest_path.relative_to(root)),
        "base_manifest_sha256": digest(base_manifest_path.read_bytes()),
        "confirmatory_manifest": str(confirmatory_manifest_path.relative_to(root)),
        "confirmatory_manifest_sha256": digest(confirmatory_manifest_path.read_bytes()),
        "excluded_pilot_sha256": dataset["excluded_pilot_sha256"],
        "excluded_confirmatory_sha256": dataset["excluded_confirmatory_sha256"],
        "raw_files": base_manifest["raw_files"],
        "processed_files": records,
        "splits": split_manifest,
    }
    write_json(root / dataset["manifest"], manifest)
    return manifest


def planned_calls(config):
    repeats = config["sampling"]["repeats_per_item"]
    return {
        "development": config["dataset"]["development_validation_items"]
        * len(development_methods(config))
        * repeats,
        "final_test": config["dataset"]["final_test_items"] * 3 * repeats,
    }


def status(config, root):
    out = output_dir(config, root)
    return {
        "data_prepared": (root / config["dataset"]["manifest"]).exists(),
        "development_responses": (out / "development/responses.jsonl").exists(),
        "method_frozen": (out / "selected_method.json").exists(),
        "final_test_responses": (out / "final_test/responses.jsonl").exists(),
        "final_report": (out / "evaluation/report.md").exists(),
    }


def _method_plan(config, root, split, method_ids):
    items, _ = load_split(config, root, split)
    calibration, _ = load_split(config, root, "calibration_pool")
    targets = {item.item_id: item for item in frame_items(items)}
    calibration_items = frame_items(calibration)
    ev_gap_bins = {
        int(row.item_id): str(row.ev_gap_bin)
        for frame in (calibration, items)
        for row in frame.itertuples()
    }
    specs = b2_method_specs(config)
    demonstrations = {}
    for method_id in method_ids:
        if method_id not in specs:
            continue
        spec = specs[method_id]
        for target in targets.values():
            chosen = select_demonstrations(
                calibration_items,
                target,
                ev_gap_bins,
                spec["k"],
                spec["retrieval_strategy"],
                config["experiment"]["seed"],
            )
            demonstrations[(method_id, target.item_id)] = chosen
    return items, targets, demonstrations


async def run_method_responses(
    config,
    root,
    split,
    method_ids,
    provider,
    cache_only=False,
    max_rps=None,
    allow_dirty=False,
):
    if provider != "mock":
        _, dirty = git_state(root)
        if dirty and not allow_dirty:
            raise RuntimeError("live benchmark stages require a clean committed worktree")
    items, targets, demonstrations = _method_plan(config, root, split, method_ids)
    _, registry = prompt_registry(root / config["prompts"]["file"])
    b2_specs = b2_method_specs(config)
    destination_name = "development" if split == "development_validation" else split
    destination = output_dir(config, root) / destination_name
    destination.mkdir(parents=True, exist_ok=True)
    response_path = destination / "responses.jsonl"
    completed = set()
    if response_path.exists():
        old = load_responses(response_path)
        completed = {
            response_identity(*row)
            for row in old[["split", "prompt_id", "item_id", "repeat_id"]].itertuples(
                index=False, name=None
            )
        }
    repeats = config["sampling"]["repeats_per_item"]
    jobs = []
    for item_id in sorted(targets):
        item_orientations = orientations(item_id, repeats, config["experiment"]["seed"])
        for method_id in method_ids:
            for repeat_id, orientation in enumerate(item_orientations):
                identity = response_identity(split, method_id, item_id, repeat_id)
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

    async def execute(split_name, method_id, item_id, repeat_id, orientation):
        target = targets[item_id]
        demonstration_ids = []
        if method_id == "B0":
            system, user, display = build_prompt(
                target, registry[config["prompts"]["B0_registry_id"]], orientation
            )
        elif method_id == "B1":
            system, user, display = build_prompt(
                target, registry[config["prompts"]["B1_registry_id"]], orientation
            )
        else:
            if method_id not in b2_specs:
                raise ValueError(f"unknown data-driven method {method_id}")
            demos = demonstrations[(method_id, item_id)]
            demonstration_ids = [item.item_id for item in demos]
            system, user, display = build_few_shot_prompt(
                target,
                demos,
                orientation,
                config["few_shot"]["pseudo_participants_per_item"],
                config["experiment"]["seed"],
            )
        identity = {
            "run_id": config["experiment"]["name"],
            "split": split_name,
            "prompt_id": method_id,
            "item_id": item_id,
            "problem_id": target.problem_id,
            "repeat_id": repeat_id,
            "display_order": orientation,
            "demonstration_item_ids": demonstration_ids,
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
        with response_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    try:
        await asyncio.gather(*(execute(*job) for job in jobs))
    finally:
        await client.close()
    responses = load_responses(response_path)
    expected = len(items) * len(method_ids) * repeats
    if len(responses) != expected:
        raise RuntimeError(f"incomplete {split} matrix: {len(responses)}/{expected}")
    return responses, items


def evaluate_development(config, root):
    out = output_dir(config, root)
    responses = load_responses(out / "development/responses.jsonl")
    items, _ = load_split(config, root, "development_validation")
    expected = set(development_methods(config))
    if set(responses.prompt_id) != expected:
        raise RuntimeError("development response matrix does not contain every candidate")
    rates, metrics = evaluate_stage(
        responses,
        items,
        config["sampling"]["repeats_per_item"],
        out / "development",
        config,
    )
    rows = []
    for method_id in development_methods(config):
        rows.append(
            {
                "method_id": method_id,
                "eligible_B2": method_id.startswith("B2_"),
                **metrics[method_id],
            }
        )
    pd.DataFrame(rows).to_csv(out / "development/candidate_comparison.csv", index=False)
    return rates, metrics


def select_development_method(metrics, tie_break_order):
    """Select exactly one B2 candidate from development metrics only."""
    if set(tie_break_order) - set(metrics):
        raise ValueError("development metrics are missing a B2 candidate")
    order = {method_id: index for index, method_id in enumerate(tie_break_order)}
    return min(tie_break_order, key=lambda method_id: (metrics[method_id]["mae"], order[method_id]))


def freeze_selected_method(config, root):
    out = output_dir(config, root)
    rates, metrics = evaluate_development(config, root)
    gate = config["evaluation"]["invalid_rate_gate"]
    if any(metric["invalid_rate"] > gate for metric in metrics.values()):
        raise RuntimeError("development invalid rate exceeds the preregistered gate")
    selected = select_development_method(metrics, config["development"]["tie_break_order"])
    spec = b2_method_specs(config)[selected]
    paths = split_paths(config, root)
    commit, dirty = git_state(root)
    if dirty:
        raise RuntimeError("freeze requires the unchanged clean protocol worktree")
    value = {
        "selected_method": selected,
        "method": "few_shot_human_calibrated",
        "retrieval_strategy": spec["retrieval_strategy"],
        "k": spec["k"],
        "selection_metric": "mae",
        "candidate_metrics": {
            key: metrics[key] for key in config["development"]["tie_break_order"]
        },
        "calibration_split_hash": digest(paths["calibration_pool"].read_bytes()),
        "development_split_hash": digest(paths["development_validation"].read_bytes()),
        "prompt_template_hash": prompt_template_hash(
            config["few_shot"]["pseudo_participants_per_item"]
        ),
        "development_item_rates_hash": digest(
            (out / "development/item_rates.parquet").read_bytes()
        ),
        "config_hash": config_hash(config),
        "source_hash": source_hash(),
        "code_commit": commit,
        "model_settings": {
            "provider": config["llm"]["provider"],
            "model": llm_config(config, config["llm"]["provider"])["model"],
            "temperature": config["llm"]["temperature"],
            "enable_thinking": config["llm"]["enable_thinking"],
            "max_tokens": config["llm"]["max_tokens"],
            "repeats_per_item": config["sampling"]["repeats_per_item"],
        },
        "frozen_at": datetime.now(UTC).isoformat(),
        "development_item_prompt_rows": len(rates),
    }
    path = out / "selected_method.json"
    if path.exists():
        old = json.loads(path.read_text())
        comparable = {key: value[key] for key in value if key != "frozen_at"}
        if {key: old.get(key) for key in comparable} != comparable:
            raise RuntimeError("refusing to overwrite a different selected method")
        return old
    write_json(path, value)
    return value


def validate_selected_method(config, root, require_tracked=False):
    out = output_dir(config, root)
    path = out / "selected_method.json"
    if not path.exists():
        raise RuntimeError("selected_method.json does not exist; run and commit freeze first")
    value = json.loads(path.read_text())
    paths = split_paths(config, root)
    expected = {
        "calibration_split_hash": digest(paths["calibration_pool"].read_bytes()),
        "development_split_hash": digest(paths["development_validation"].read_bytes()),
        "prompt_template_hash": prompt_template_hash(
            config["few_shot"]["pseudo_participants_per_item"]
        ),
        "config_hash": config_hash(config),
        "source_hash": source_hash(),
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise RuntimeError("selected method provenance no longer matches frozen inputs")
    if value.get("selected_method") not in b2_method_specs(config):
        raise RuntimeError("selected method is not a preregistered B2 candidate")
    if require_tracked:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if tracked.returncode:
            raise RuntimeError("selected_method.json must be committed before final_test")
    return value


def _validate_run_manifest(config, root, provider):
    out = output_dir(config, root)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("development run manifest is missing")
    value = json.loads(manifest_path.read_text())
    expected = {
        "provider": provider,
        "model": llm_config(config, provider)["model"],
        "source_hash": source_hash(),
        "config_hash": config_hash(config),
        "dataset_manifest_hash": digest((root / config["dataset"]["manifest"]).read_bytes()),
        "prompt_registry_hash": digest((root / config["prompts"]["file"]).read_bytes()),
        "data_driven_prompt_template_hash": prompt_template_hash(
            config["few_shot"]["pseudo_participants_per_item"]
        ),
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise RuntimeError("run manifest no longer matches frozen development inputs")
    return value


def _bootstrap_intervals(frame):
    result = {}
    metrics = ["mae", "mean_js", "position_bias", "b_position_effect"]
    for (contrast, kind, prompt_id), group in frame.groupby(["contrast", "kind", "prompt_id"]):
        result[f"{contrast}:{kind}:{prompt_id}"] = {
            metric: [float(group[metric].quantile(0.025)), float(group[metric].quantile(0.975))]
            for metric in metrics
        }
    return result


def final_evaluation(config, root):
    selected = validate_selected_method(config, root)
    selected_id = selected["selected_method"]
    out = output_dir(config, root)
    responses = load_responses(out / "final_test/responses.jsonl")
    items, _ = load_split(config, root, "final_test")
    expected = {"B0", "B1", selected_id}
    if set(responses.prompt_id) != expected:
        raise RuntimeError("final-test matrix must contain B0, B1, and frozen B2")
    items = items.copy()
    threshold = config["evaluation"]["strong_consensus_threshold"]
    items["human_consensus"] = items.human_b_rate.apply(
        lambda rate: "strong" if rate >= threshold or rate <= 1 - threshold else "mixed"
    )
    rates, metrics = evaluate_stage(
        responses,
        items,
        config["sampling"]["repeats_per_item"],
        out / "final_test",
        config,
    )
    evaluation = out / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ["subgroup_metrics.csv", "calibration_curve.csv"]:
        shutil.copyfile(out / "final_test" / name, evaluation / name)

    bootstrap_frames = []
    for offset, baseline in enumerate(("B0", "B1")):
        frame = bootstrap(
            rates,
            config["evaluation"]["bootstrap_samples"],
            config["experiment"]["seed"] + 100 + offset,
            baseline,
            selected_id,
        )
        frame["contrast"] = f"{selected_id}_vs_{baseline}"
        bootstrap_frames.append(frame)
    bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_frame.to_csv(evaluation / "bootstrap.csv", index=False)
    intervals = _bootstrap_intervals(bootstrap_frame)
    write_json(evaluation / "bootstrap_intervals.json", intervals)

    contrasts = {}
    for baseline in ("B0", "B1"):
        name = f"{selected_id}_vs_{baseline}"
        estimate = metrics[selected_id]["mae"] - metrics[baseline]["mae"]
        ci = intervals[f"{name}:paired_delta:{selected_id}-{baseline}"]["mae"]
        contrasts[name] = {"estimate": estimate, "bootstrap_95_ci": ci}
    contrasts[f"{selected_id}_vs_B0"]["success"] = (
        contrasts[f"{selected_id}_vs_B0"]["bootstrap_95_ci"][1] < 0
    )

    subgroup = pd.read_csv(evaluation / "subgroup_metrics.csv")
    comparisons = []
    selected_groups = subgroup[subgroup.prompt_id == selected_id]
    for baseline in ("B0", "B1"):
        baseline_groups = subgroup[subgroup.prompt_id == baseline]
        merged = baseline_groups.merge(
            selected_groups,
            on=["dimension", "subgroup"],
            suffixes=("_baseline", "_selected"),
        )
        merged.insert(0, "baseline", baseline)
        merged["delta_mae_selected_minus_baseline"] = (
            merged.mae_selected - merged.mae_baseline
        )
        comparisons.append(merged)
    pd.concat(comparisons, ignore_index=True).to_csv(
        evaluation / "subgroup_comparisons.csv", index=False
    )

    warning = any(
        metric["invalid_rate"] > config["evaluation"]["invalid_rate_gate"]
        for metric in metrics.values()
    )
    report = [
        "# Choices13k data-driven calibration final test",
        "",
        f"Frozen B2 method: `{selected_id}` ({selected['retrieval_strategy']}, K={selected['k']}).",
        "",
        "| Method | MAE | RMSE | mean JS | Pearson | Spearman | invalid | position bias | "
        "B-position effect | degenerate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def display(value, percent=False):
        if value is None:
            return "NA"
        return f"{value:.2%}" if percent else f"{value:.4f}"

    for method_id in ["B0", "B1", selected_id]:
        metric = metrics[method_id]
        report.append(
            f"| {method_id} | {display(metric['mae'])} | {display(metric['rmse'])} | "
            f"{display(metric['mean_js'])} | {display(metric['pearson'])} | "
            f"{display(metric['spearman'])} | {display(metric['invalid_rate'], True)} | "
            f"{display(metric['position_bias'])} | {display(metric['b_position_effect'])} | "
            f"{display(metric['degenerate_item_rate'], True)} |"
        )
    for baseline in ("B0", "B1"):
        name = f"{selected_id}_vs_{baseline}"
        value = contrasts[name]
        report.append(
            f"\nPaired ΔMAE ({selected_id} − {baseline}) = {value['estimate']:.4f}; "
            f"item-bootstrap 95% CI [{value['bootstrap_95_ci'][0]:.4f}, "
            f"{value['bootstrap_95_ci'][1]:.4f}]."
        )
    report += [
        "",
        f"Primary success criterion (B2 vs B0 CI upper bound < 0): "
        f"{contrasts[f'{selected_id}_vs_B0']['success']}.",
        f"Invalid-rate gate exceeded: {warning}.",
        f"Statistical unit: item ({len(items)} items; {len(rates)} item-method rows).",
        "",
        "Choices13k is public and may be present in model pretraining. This result concerns "
        "aggregate risky binary choices and does not establish financial-market fidelity.",
    ]
    (evaluation / "report.md").write_text("\n".join(report) + "\n")
    summary = {
        "selected_method": selected,
        "items": len(items),
        "metrics": metrics,
        "primary_contrasts": contrasts,
        "bootstrap_intervals": intervals,
        "invalid_rate_warning": warning,
    }
    write_json(evaluation / "summary.json", summary)
    return summary


async def run_data_driven_stage(
    config,
    root,
    stage,
    provider=None,
    cache_only=False,
    max_rps=None,
    allow_dirty=False,
):
    validate_data_driven_config(config, root)
    provider = provider or config["llm"]["provider"]
    if stage == "plan":
        return {"calls": planned_calls(config), "status": status(config, root)}
    if stage == "prepare":
        return prepare_data_driven(config, root)
    if stage == "development":
        if provider != "mock":
            _, dirty = git_state(root)
            if dirty and not allow_dirty:
                raise RuntimeError("development requires a clean committed worktree")
            manifest = write_run_manifest(config, root, provider)
            template_hash = prompt_template_hash(
                config["few_shot"]["pseudo_participants_per_item"]
            )
            existing_hash = manifest.get("data_driven_prompt_template_hash")
            if existing_hash not in {None, template_hash}:
                raise RuntimeError("run manifest uses a different few-shot prompt template")
            manifest["data_driven_prompt_template_hash"] = template_hash
            write_json(output_dir(config, root) / "manifest.json", manifest)
        elif not (root / config["dataset"]["manifest"]).exists():
            raise RuntimeError("prepare data before mock development")
        return await run_method_responses(
            config,
            root,
            "development_validation",
            development_methods(config),
            provider,
            cache_only,
            max_rps,
            allow_dirty,
        )
    if stage == "freeze":
        return freeze_selected_method(config, root)
    if stage == "final_test":
        selected = validate_selected_method(config, root, require_tracked=provider != "mock")
        if provider != "mock":
            _, dirty = git_state(root)
            if dirty and not allow_dirty:
                raise RuntimeError("final_test requires a clean committed worktree")
            manifest = _validate_run_manifest(config, root, provider)
            commit, _ = git_state(root)
            if "final_test_git_commit" in manifest and manifest["final_test_git_commit"] != commit:
                raise RuntimeError("final test was already bound to a different commit")
            manifest["final_test_git_commit"] = commit
            write_json(output_dir(config, root) / "manifest.json", manifest)
        return await run_method_responses(
            config,
            root,
            "final_test",
            ["B0", "B1", selected["selected_method"]],
            provider,
            cache_only,
            max_rps,
            allow_dirty,
        )
    if stage == "evaluate":
        return final_evaluation(config, root)
    raise ValueError(f"unsupported data-driven stage {stage}")
