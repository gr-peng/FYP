"""Frozen Choices13k confirmatory experiment orchestration and reporting."""

import re
import shutil

import pandas as pd

from behavioral_market.data.archive import write_json
from behavioral_market.simulation.event_runner import git_state

from .evaluation import evaluate_stage, load_responses
from .prepare import load_split, prepare_confirmatory
from .prompts import prompt_registry
from .runner import output_dir, run_responses, write_run_manifest

FROZEN_PROMPTS = {
    "P0": "Choose the option you prefer.",
    "P1": "Respond as an ordinary adult participant making the choice for themselves.",
}


def validate_confirmatory_config(config, root):
    """Reject any drift from the preregistered confirmatory treatment."""
    dataset = config["dataset"]
    sampling = config["sampling"]
    prompts = config["prompts"]
    llm = config["llm"]
    evaluation = config["evaluation"]
    _, registry = prompt_registry(root / prompts["file"])
    checks = {
        "experiment seed": config["experiment"]["seed"] == 20260917,
        "1000 items": dataset["confirmatory_items"] == 1000,
        "no-feedback primary population": dataset["primary_feedback"] is False,
        "three EV-gap strata": dataset["ev_gap_quantiles"] == 3,
        "10 repeats": sampling["repeats_per_item"] == 10,
        "counterbalancing": sampling["counterbalance_option_order"] is True,
        "only P0/P1": prompts["candidate_ids"] == ["P0", "P1"],
        "vanilla P0": prompts["vanilla_id"] == "P0",
        "treatment P1": prompts["treatment_id"] == "P1",
        "frozen prompt text": {key: registry.get(key) for key in FROZEN_PROMPTS}
        == FROZEN_PROMPTS,
        "temperature 0.7": llm["temperature"] == 0.7,
        "thinking disabled": llm["enable_thinking"] is False,
        "primary metric MAE": evaluation["primary_metric"] == "mae",
        "5000 bootstraps": evaluation["bootstrap_samples"] == 5000,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("confirmatory protocol drift: " + ", ".join(failures))
    split_hash = dataset["confirmatory_split_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", split_hash):
        raise ValueError("confirmatory split hash is not frozen")


def planned_confirmatory_calls(config):
    return (
        config["dataset"]["confirmatory_items"]
        * len(config["prompts"]["candidate_ids"])
        * config["sampling"]["repeats_per_item"]
    )


def confirmatory_status(config, root):
    out = output_dir(config, root)
    return {
        "data_prepared": (root / config["dataset"]["manifest"]).exists(),
        "responses": (out / "confirmatory/responses.jsonl").exists(),
        "final_report": (out / "evaluation/report.md").exists(),
    }


def _bootstrap_intervals(frame):
    intervals = {}
    metrics = ["mae", "mean_js", "position_bias", "b_position_effect"]
    for (kind, prompt_id), group in frame.groupby(["kind", "prompt_id"]):
        intervals[f"{kind}:{prompt_id}"] = {
            metric: [float(group[metric].quantile(0.025)), float(group[metric].quantile(0.975))]
            for metric in metrics
        }
    return intervals


def final_confirmatory_evaluation(config, root):
    """Evaluate the frozen response matrix without making API requests."""
    validate_confirmatory_config(config, root)
    out = output_dir(config, root)
    responses = load_responses(out / "confirmatory/responses.jsonl")
    items, _ = load_split(config, root, "confirmatory")
    expected_prompts = set(config["prompts"]["candidate_ids"])
    if set(responses.prompt_id) != expected_prompts:
        raise RuntimeError("confirmatory responses must contain exactly P0 and P1")
    treatment = config["prompts"]["treatment_id"]
    vanilla = config["prompts"]["vanilla_id"]
    rates, metrics = evaluate_stage(
        responses,
        items,
        config["sampling"]["repeats_per_item"],
        out / "confirmatory",
        config,
        treatment,
    )
    evaluation = out / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ["subgroup_metrics.csv", "calibration_curve.csv", "bootstrap.csv"]:
        shutil.copyfile(out / "confirmatory" / name, evaluation / name)
    bootstrap_frame = pd.read_csv(evaluation / "bootstrap.csv")
    intervals = _bootstrap_intervals(bootstrap_frame)
    delta_key = f"paired_delta:{treatment}-{vanilla}"
    delta_mae_ci = intervals[delta_key]["mae"]
    write_json(evaluation / "bootstrap_intervals.json", intervals)

    subgroup = pd.read_csv(evaluation / "subgroup_metrics.csv")
    vanilla_groups = subgroup[subgroup.prompt_id == vanilla]
    treatment_groups = subgroup[subgroup.prompt_id == treatment]
    comparison = vanilla_groups.merge(
        treatment_groups,
        on=["dimension", "subgroup"],
        suffixes=("_vanilla", "_treatment"),
    )
    comparison["delta_mae_treatment_minus_vanilla"] = (
        comparison.mae_treatment - comparison.mae_vanilla
    )
    comparison.to_csv(evaluation / "subgroup_comparisons.csv", index=False)

    observed_delta = metrics[treatment]["mae"] - metrics[vanilla]["mae"]
    strong_evidence = observed_delta < 0 and delta_mae_ci[1] < 0
    warning = any(
        metric["invalid_rate"] > config["evaluation"]["invalid_rate_gate"]
        for metric in metrics.values()
    )
    degeneracy_warning = any(
        metric["degenerate_item_rate"] is not None
        and metric["degenerate_item_rate"] > config["evaluation"]["degenerate_item_rate_gate"]
        for metric in metrics.values()
    )

    def display(value, percent=False):
        if value is None:
            return "NA"
        return f"{value:.2%}" if percent else f"{value:.4f}"

    report = [
        "# Choices13k confirmatory experiment",
        "",
        "Frozen comparison: `P0` versus `P1` on 1,000 untouched no-feedback items.",
        "",
        "| Prompt | MAE | RMSE | mean JS | Pearson | Spearman | invalid | position bias | "
        "B-position effect | degenerate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_id in config["prompts"]["candidate_ids"]:
        metric = metrics[prompt_id]
        report.append(
            f"| {prompt_id} | {display(metric['mae'])} | {display(metric['rmse'])} | "
            f"{display(metric['mean_js'])} | {display(metric['pearson'])} | "
            f"{display(metric['spearman'])} | {display(metric['invalid_rate'], True)} | "
            f"{display(metric['position_bias'])} | {display(metric['b_position_effect'])} | "
            f"{display(metric['degenerate_item_rate'], True)} |"
        )
    report += [
        "",
        f"Primary paired ΔMAE ({treatment} − {vanilla}) = {observed_delta:.4f}; "
        f"item-bootstrap 95% CI [{delta_mae_ci[0]:.4f}, {delta_mae_ci[1]:.4f}].",
        f"Strong confirmatory criterion (CI upper bound < 0): {strong_evidence}.",
        "",
        f"Invalid-rate gate exceeded: {warning}. Sampling-degeneracy diagnostic exceeded: "
        f"{degeneracy_warning}.",
        f"Statistical unit: item ({len(items)} items; {len(rates)} item-prompt rows).",
        "",
        "This public-data experiment may be affected by benchmark contamination and supports "
        "claims only about aggregate risky binary choices.",
    ]
    (evaluation / "report.md").write_text("\n".join(report) + "\n")
    summary = {
        "prompts": config["prompts"]["candidate_ids"],
        "primary_population": "no_feedback",
        "items": len(items),
        "metrics": metrics,
        "primary_contrast": {
            "name": f"mae_{treatment}_minus_{vanilla}",
            "estimate": observed_delta,
            "bootstrap_95_ci": delta_mae_ci,
            "bootstrap_samples": config["evaluation"]["bootstrap_samples"],
            "strong_confirmatory_evidence": strong_evidence,
        },
        "bootstrap_intervals": intervals,
        "invalid_rate_warning": warning,
        "sampling_degeneracy_warning": degeneracy_warning,
    }
    write_json(evaluation / "summary.json", summary)
    return summary


async def run_confirmatory_stage(
    config,
    root,
    stage,
    provider=None,
    cache_only=False,
    max_rps=None,
    allow_dirty=False,
):
    validate_confirmatory_config(config, root)
    provider = provider or config["llm"]["provider"]
    if stage == "plan":
        return {
            "calls": planned_confirmatory_calls(config),
            "status": confirmatory_status(config, root),
        }
    if stage == "prepare":
        return prepare_confirmatory(config, root)
    if stage == "run":
        if provider != "mock":
            _, dirty = git_state(root)
            if dirty and not allow_dirty:
                raise RuntimeError("live benchmark stages require a clean committed worktree")
            if provider not in config["llm"]["providers"]:
                raise ValueError(f"unsupported provider {provider}")
            write_run_manifest(config, root, provider)
        elif not (root / config["dataset"]["manifest"]).exists():
            raise RuntimeError("prepare confirmatory data before running mock responses")
        return await run_responses(
            config,
            root,
            "confirmatory",
            config["prompts"]["candidate_ids"],
            provider,
            cache_only,
            max_rps,
            allow_dirty,
        )
    if stage == "evaluate":
        return final_confirmatory_evaluation(config, root)
    raise ValueError(f"unsupported confirmatory stage {stage}")
