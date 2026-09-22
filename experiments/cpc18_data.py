"""CPC18 feature-only parsing and explicitly separated outcome analysis."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.calibration_v2 import visible_fingerprint

FEATURE_COLUMNS = ["GameID", "Amb"] + [
    f"{s}{field}{i}" for s in "ab" for i in range(1, 11) for field in ("p", "v")
]
OUTCOME_COLUMNS = ["SubjID", "Location", "Set", "Condition", "GameID", "Trial", "Feedback", "B"]


def targets_from_frame(frame, eligible_ids):
    if set(frame.columns) != set(FEATURE_COLUMNS):
        raise ValueError("target frame must contain only the explicit feature allowlist")
    selected = frame[frame.GameID.isin(eligible_ids)]
    if selected.GameID.duplicated().any() or set(selected.GameID) != set(eligible_ids):
        raise ValueError("CPC18 target identities mismatch")
    items = []
    for row in selected.sort_values("GameID").to_dict("records"):
        if row["Amb"] not in (0, False):
            raise ValueError("external targets must be non-ambiguous")
        options = {}
        for side in "ab":
            option = [
                (float(row[f"{side}p{i}"]), float(row[f"{side}v{i}"]))
                for i in range(1, 11)
                if pd.notna(row[f"{side}p{i}"]) and pd.notna(row[f"{side}v{i}"])
            ]
            if (
                not option
                or not all(np.isfinite(p) and np.isfinite(x) and 0 <= p <= 1 for p, x in option)
                or abs(sum(p for p, _ in option) - 1) > 1e-6
            ):
                raise ValueError("invalid external gamble")
            options[side] = option
        item = SimpleNamespace(
            item_id=int(row["GameID"]),
            ambiguity=False,
            option_a=options["a"],
            option_b=options["b"],
        )
        for side, option in options.items():
            mean = sum(p * x for p, x in option)
            setattr(item, f"ev_{side}", mean)
            setattr(item, f"var_{side}", sum(p * (x - mean) ** 2 for p, x in option))
        item.has_loss = any(x < 0 and p > 0 for option in options.values() for p, x in option)
        items.append(item)
    if len({visible_fingerprint(x) for x in items}) != len(items):
        raise ValueError("duplicate visible external problems")
    return items


def read_targets(workbook, ids):
    return targets_from_frame(pd.read_excel(workbook, usecols=FEATURE_COLUMNS), ids)


def select_outcomes(frame, ids):
    # Apply scope selection before converting B; irrelevant outcome strings may be unreadable.
    selected = frame.loc[
        frame.GameID.isin(ids) & frame.Trial.between(1, 5) & (frame.Feedback == 0), OUTCOME_COLUMNS
    ].copy()
    selected["B"] = pd.to_numeric(selected["B"], errors="raise")
    if not selected.B.isin([0, 1]).all():
        raise ValueError("CPC18 response is not binary")
    return selected


def read_outcomes(path, ids):
    # This function is called only after a frozen response audit and opening record.
    chunks = [
        select_outcomes(chunk, ids)
        for chunk in pd.read_csv(
            path, usecols=OUTCOME_COLUMNS, dtype={"B": "string"}, chunksize=100000
        )
    ]
    return pd.concat(chunks, ignore_index=True)


def participant_arrays(frame, ids):
    if set(frame.GameID) != set(ids) or frame.duplicated(["SubjID", "GameID", "Trial"]).any():
        raise ValueError("human outcome coverage or trial identities mismatch")
    if not frame.groupby(["SubjID", "GameID"]).size().eq(5).all():
        raise ValueError("expected five repeated trials per participant/game")
    if any(
        frame.groupby("SubjID")[c].nunique().max() != 1 for c in ("Set", "Location", "Condition")
    ):
        raise ValueError("participant changes resampling stratum")
    people = sorted(frame.SubjID.unique())
    grouped = frame.groupby(["SubjID", "GameID"]).B.agg(["sum", "count"])
    numerator = grouped["sum"].unstack().reindex(index=people, columns=ids).fillna(0).to_numpy()
    denominator = grouped["count"].unstack().reindex(index=people, columns=ids).fillna(0).to_numpy()
    meta = frame.drop_duplicates("SubjID").set_index("SubjID").loc[people]
    strata = {}
    for index, values in enumerate(
        meta[["Set", "Location", "Condition"]].itertuples(index=False, name=None)
    ):
        strata.setdefault(tuple(values), []).append(index)
    return numerator, denominator, list(strata.values())


def bootstrap_contrasts(numerator, denominator, strata, predictions, contrasts, samples, seed):
    rng = np.random.default_rng(seed)
    n_people, n_items = numerator.shape
    if denominator.shape != numerator.shape or sorted(i for s in strata for i in s) != list(
        range(n_people)
    ):
        raise ValueError("participant strata must partition the full panel")
    if (denominator.sum(axis=0) <= 0).any():
        raise ValueError("problem without human observations")
    human = numerator.sum(axis=0) / denominator.sum(axis=0)
    weights = np.zeros((samples, n_people), dtype=float)
    for indices in strata:
        n = len(indices)
        weights[:, indices] = rng.multinomial(n, np.full(n, 1 / n), size=samples)
    sampled_denominator = weights @ denominator
    if (sampled_denominator <= 0).any():
        raise ValueError("cluster bootstrap omitted all people for a problem")
    sampled_human = (weights @ numerator) / sampled_denominator
    problem_indices = rng.integers(0, n_items, size=(samples, n_items))
    report = {}
    for left, right in contrasts:
        p, q = np.asarray(predictions[left]), np.asarray(predictions[right])
        delta = np.abs(p - human) - np.abs(q - human)
        sampled_delta = np.abs(p[None, :] - sampled_human) - np.abs(q[None, :] - sampled_human)
        values = {
            "problem_only": delta[problem_indices].mean(axis=1),
            "participant_only": sampled_delta.mean(axis=1),
            "joint": sampled_delta[np.arange(samples)[:, None], problem_indices].mean(axis=1),
        }
        report[f"{left}_minus_{right}"] = {
            "estimate": float(delta.mean()),
            "ci95": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in values.items()},
            "items": n_items,
            "participants": n_people,
            "strata": len(strata),
            "samples": samples,
        }
    return report
