import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from behavioral_market.data.archive import write_json


def bernoulli_js(left, right):
    p = np.array([left, 1 - left], dtype=float)
    q = np.array([right, 1 - right], dtype=float)
    midpoint = (p + q) / 2

    def kl(value, reference):
        mask = value > 0
        return float(np.sum(value[mask] * np.log2(value[mask] / reference[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def mean_or_none(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if len(values) else None


def aggregate_responses(responses, items, repeats):
    expected = len(items) * repeats
    grouped = []
    for (prompt_id, item_id), group in responses.groupby(["prompt_id", "item_id"]):
        valid = group[group.status == "ok"]
        b_first = valid[valid.display_order == "B_first"]
        b_second = valid[valid.display_order == "A_first"]
        grouped.append(
            {
                "prompt_id": prompt_id,
                "item_id": int(item_id),
                "responses": len(group),
                "valid_responses": len(valid),
                "invalid_responses": int((group.status != "ok").sum()),
                "llm_b_rate": float(valid.original_b_choice.mean()) if len(valid) else np.nan,
                "first_option_rate": float(valid.first_display_selected.mean())
                if len(valid)
                else np.nan,
                "b_first_responses": len(b_first),
                "b_second_responses": len(b_second),
                "b_rate_when_b_first": float(b_first.original_b_choice.mean())
                if len(b_first)
                else np.nan,
                "b_rate_when_b_second": float(b_second.original_b_choice.mean())
                if len(b_second)
                else np.nan,
            }
        )
    rates = pd.DataFrame(grouped).merge(items, on="item_id", validate="many_to_one")
    prompt_count = responses.prompt_id.nunique()
    if len(responses) != expected * prompt_count:
        raise ValueError("response matrix is incomplete or contains duplicate identities")
    expected_keys = {
        (prompt_id, int(item_id))
        for prompt_id in responses.prompt_id.unique()
        for item_id in items.item_id
    }
    actual_sizes = responses.groupby(["prompt_id", "item_id"]).size()
    if set(actual_sizes.index) != expected_keys or not (actual_sizes == repeats).all():
        raise ValueError("response matrix is not a complete item/prompt/repeat product")
    return rates


def prompt_metrics(frame):
    usable = frame.dropna(subset=["llm_b_rate"])
    error = usable.llm_b_rate - usable.human_b_rate
    if len(usable) > 1 and usable.llm_b_rate.nunique() > 1 and usable.human_b_rate.nunique() > 1:
        pearson = float(pearsonr(usable.llm_b_rate, usable.human_b_rate).statistic)
        spearman = float(spearmanr(usable.llm_b_rate, usable.human_b_rate).statistic)
    else:
        pearson = spearman = None
    risk_mask = usable.var_a != usable.var_b
    riskier_llm = np.where(
        usable.loc[risk_mask, "var_b"] > usable.loc[risk_mask, "var_a"],
        usable.loc[risk_mask, "llm_b_rate"],
        1 - usable.loc[risk_mask, "llm_b_rate"],
    )
    riskier_human = np.where(
        usable.loc[risk_mask, "var_b"] > usable.loc[risk_mask, "var_a"],
        usable.loc[risk_mask, "human_b_rate"],
        1 - usable.loc[risk_mask, "human_b_rate"],
    )
    ev_mask = usable.ev_a != usable.ev_b
    higher_ev_llm = np.where(
        usable.loc[ev_mask, "ev_b"] > usable.loc[ev_mask, "ev_a"],
        usable.loc[ev_mask, "llm_b_rate"],
        1 - usable.loc[ev_mask, "llm_b_rate"],
    )
    higher_ev_human = np.where(
        usable.loc[ev_mask, "ev_b"] > usable.loc[ev_mask, "ev_a"],
        usable.loc[ev_mask, "human_b_rate"],
        1 - usable.loc[ev_mask, "human_b_rate"],
    )
    position = frame[frame.valid_responses > 0]
    first_option_rate = (
        float(np.average(position.first_option_rate, weights=position.valid_responses))
        if len(position)
        else None
    )
    b_first = frame.dropna(subset=["b_rate_when_b_first"])
    b_second = frame.dropna(subset=["b_rate_when_b_second"])
    b_rate_when_b_first = (
        float(np.average(b_first.b_rate_when_b_first, weights=b_first.b_first_responses))
        if len(b_first)
        else None
    )
    b_rate_when_b_second = (
        float(np.average(b_second.b_rate_when_b_second, weights=b_second.b_second_responses))
        if len(b_second)
        else None
    )
    return {
        "items": len(frame),
        "usable_items": len(usable),
        "mae": mean_or_none(error.abs()),
        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "mean_js": mean_or_none(
            [bernoulli_js(row.llm_b_rate, row.human_b_rate) for row in usable.itertuples()]
        ),
        "pearson": pearson,
        "spearman": spearman,
        "invalid_rate": float(frame.invalid_responses.sum() / frame.responses.sum()),
        "first_option_rate": first_option_rate,
        "position_bias": first_option_rate - 0.5 if first_option_rate is not None else None,
        "b_position_effect": b_rate_when_b_first - b_rate_when_b_second
        if b_rate_when_b_first is not None and b_rate_when_b_second is not None
        else None,
        "degenerate_item_rate": float(usable.llm_b_rate.isin([0.0, 1.0]).mean())
        if len(usable)
        else None,
        "llm_riskier_choice_rate": mean_or_none(riskier_llm),
        "human_riskier_choice_rate": mean_or_none(riskier_human),
        "llm_higher_ev_choice_rate": mean_or_none(higher_ev_llm),
        "human_higher_ev_choice_rate": mean_or_none(higher_ev_human),
    }


def subgroup_metrics(rates):
    rows = []
    for prompt_id, prompt in rates.groupby("prompt_id"):
        dimensions = {
            "has_loss": {True: "loss", False: "no_loss"},
            "ambiguity": {True: "ambiguous", False: "not_ambiguous"},
            "feedback": {True: "feedback", False: "no_feedback"},
        }
        if "ev_gap_bin" in prompt:
            dimensions["ev_gap_bin"] = {
                "small": "small_ev_gap",
                "medium": "medium_ev_gap",
                "large": "large_ev_gap",
            }
        for column, labels in dimensions.items():
            for value, label in labels.items():
                subset = prompt[prompt[column] == value]
                if subset.empty:
                    continue
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "dimension": column,
                        "subgroup": label,
                        **prompt_metrics(subset),
                    }
                )
    return pd.DataFrame(rows)


def calibration_curve(rates, bins):
    value = rates.dropna(subset=["llm_b_rate"]).copy()
    edges = np.linspace(0, 1, bins + 1)
    value["llm_bin"] = pd.cut(value.llm_b_rate, edges, include_lowest=True)
    return (
        value.groupby(["prompt_id", "llm_bin"], observed=True)
        .agg(
            items=("item_id", "size"),
            mean_llm_b_rate=("llm_b_rate", "mean"),
            mean_human_b_rate=("human_b_rate", "mean"),
        )
        .reset_index()
        .assign(llm_bin=lambda frame: frame.llm_bin.astype(str))
    )


def bootstrap(rates, samples, seed, vanilla_id, selected_id):
    by_prompt = {key: value.set_index("item_id") for key, value in rates.groupby("prompt_id")}
    common = sorted(set(by_prompt[vanilla_id].index) & set(by_prompt[selected_id].index))
    common = [
        item_id
        for item_id in common
        if pd.notna(by_prompt[vanilla_id].loc[item_id, "llm_b_rate"])
        and pd.notna(by_prompt[selected_id].loc[item_id, "llm_b_rate"])
    ]
    if not common:
        raise ValueError("no common valid items are available for paired bootstrap")
    arrays = {}
    for prompt_id in dict.fromkeys([vanilla_id, selected_id]):
        frame = by_prompt[prompt_id].loc[common]
        arrays[prompt_id] = {
            "absolute_error": (frame.llm_b_rate - frame.human_b_rate).abs().to_numpy(),
            "js": np.asarray(
                [
                    bernoulli_js(row.llm_b_rate, row.human_b_rate)
                    for row in frame.itertuples()
                ]
            ),
            "first_selections": (
                frame.first_option_rate * frame.valid_responses
            ).to_numpy(),
            "valid_responses": frame.valid_responses.to_numpy(),
            "b_first_choices": (
                frame.b_rate_when_b_first * frame.b_first_responses
            ).to_numpy(),
            "b_first_responses": frame.b_first_responses.to_numpy(),
            "b_second_choices": (
                frame.b_rate_when_b_second * frame.b_second_responses
            ).to_numpy(),
            "b_second_responses": frame.b_second_responses.to_numpy(),
        }

    def sampled_metrics(values, indices):
        position_bias = (
            values["first_selections"][indices].sum()
            / values["valid_responses"][indices].sum()
            - 0.5
        )
        b_position_effect = (
            values["b_first_choices"][indices].sum()
            / values["b_first_responses"][indices].sum()
            - values["b_second_choices"][indices].sum()
            / values["b_second_responses"][indices].sum()
        )
        return {
            "mae": float(values["absolute_error"][indices].mean()),
            "mean_js": float(values["js"][indices].mean()),
            "position_bias": float(position_bias),
            "b_position_effect": float(b_position_effect),
        }

    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(samples):
        indices = rng.choice(len(common), len(common), replace=True)
        metrics = {}
        for prompt_id in dict.fromkeys([vanilla_id, selected_id]):
            current = sampled_metrics(arrays[prompt_id], indices)
            metrics[prompt_id] = current
            rows.append(
                {
                    "replicate": replicate,
                    "kind": "prompt",
                    "prompt_id": prompt_id,
                    "mae": current["mae"],
                    "mean_js": current["mean_js"],
                    "position_bias": current["position_bias"],
                    "b_position_effect": current["b_position_effect"],
                }
            )
        rows.append(
            {
                "replicate": replicate,
                "kind": "paired_delta",
                "prompt_id": f"{selected_id}-{vanilla_id}",
                "mae": metrics[selected_id]["mae"] - metrics[vanilla_id]["mae"],
                "mean_js": metrics[selected_id]["mean_js"] - metrics[vanilla_id]["mean_js"],
                "position_bias": metrics[selected_id]["position_bias"]
                - metrics[vanilla_id]["position_bias"],
                "b_position_effect": metrics[selected_id]["b_position_effect"]
                - metrics[vanilla_id]["b_position_effect"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_stage(responses, items, repeats, destination, config, selected=None):
    destination.mkdir(parents=True, exist_ok=True)
    rates = aggregate_responses(responses, items, repeats)
    rates.to_parquet(destination / "item_rates.parquet", index=False)
    rates.to_csv(destination / "item_rates.csv", index=False)
    metrics = {
        prompt_id: prompt_metrics(frame)
        for prompt_id, frame in rates.groupby("prompt_id", sort=True)
    }
    write_json(destination / "metrics.json", metrics)
    subgroup_metrics(rates).to_csv(destination / "subgroup_metrics.csv", index=False)
    calibration_curve(rates, config["evaluation"]["calibration_bins"]).to_csv(
        destination / "calibration_curve.csv", index=False
    )
    if selected is not None:
        frame = bootstrap(
            rates,
            config["evaluation"]["bootstrap_samples"],
            config["experiment"]["seed"] + 2,
            config["prompts"]["vanilla_id"],
            selected,
        )
        frame.to_csv(destination / "bootstrap.csv", index=False)
    return rates, metrics


def load_responses(path):
    if not path.exists():
        raise RuntimeError(f"missing responses: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    identity = ["split", "prompt_id", "item_id", "repeat_id"]
    if frame.duplicated(identity).any():
        raise ValueError("duplicate benchmark response identities")
    return frame
