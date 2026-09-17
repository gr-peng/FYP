import hashlib

import numpy as np
import pandas as pd


def _allocate(frame, count):
    sizes = frame.groupby("stratum", observed=True).size().sort_index()
    exact = sizes * count / len(frame)
    allocation = np.floor(exact).astype(int).clip(upper=sizes)
    remaining = count - int(allocation.sum())
    priority = (exact - allocation).sort_values(ascending=False, kind="stable").index.tolist()
    while remaining:
        progressed = False
        for stratum in priority:
            if allocation[stratum] < sizes[stratum]:
                allocation[stratum] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("not enough unique items for requested sample")
    return allocation


def proportional_stratified_sample(frame, count, seed):
    if count > len(frame):
        raise ValueError("requested sample exceeds available items")
    allocation = _allocate(frame, count)
    chosen = []
    for stratum, take in allocation.items():
        group = frame[frame.stratum == stratum]
        random_state = int.from_bytes(
            hashlib.sha256(f"{seed}:{stratum}".encode()).digest()[:4], "big"
        )
        chosen.extend(group.sample(int(take), random_state=random_state).item_id.tolist())
    return frame[frame.item_id.isin(chosen)].sort_values("item_id").copy()


def stratified_split(frame, calibration_count, heldout_count, seed, ev_gap_bins=3):
    value = frame.copy()
    ranks = value["ev_gap"].rank(method="first")
    if ev_gap_bins != 3:
        raise ValueError("the preregistered pilot requires three EV-gap bins")
    value["ev_gap_bin"] = pd.qcut(ranks, q=ev_gap_bins, labels=["small", "medium", "large"])
    value["stratum"] = value.apply(
        lambda row: "|".join(
            [
                f"feedback={int(row.feedback)}",
                f"ambiguity={int(row.ambiguity)}",
                f"loss={int(row.has_loss)}",
                f"ev_gap={row.ev_gap_bin}",
            ]
        ),
        axis=1,
    )
    pilot = proportional_stratified_sample(value, calibration_count + heldout_count, seed)
    calibration = proportional_stratified_sample(pilot, calibration_count, seed + 1)
    heldout = pilot[~pilot.item_id.isin(calibration.item_id)].copy()
    if len(heldout) != heldout_count or set(calibration.item_id) & set(heldout.item_id):
        raise RuntimeError("invalid Choices13k split")
    calibration["split"] = "calibration"
    heldout["split"] = "heldout"
    combined = pd.concat([calibration, heldout]).sort_values("item_id")
    return calibration.sort_values("item_id"), heldout.sort_values("item_id"), combined


def confirmatory_split(frame, excluded_item_ids, count, seed, ev_gap_bins=3):
    """Sample the frozen no-feedback confirmatory population without using labels."""
    if ev_gap_bins != 3:
        raise ValueError("the preregistered confirmatory experiment requires three EV-gap bins")
    excluded = {int(item_id) for item_id in excluded_item_ids}
    value = frame[(~frame.item_id.isin(excluded)) & (~frame.feedback)].copy()
    if value.item_id.isin(excluded).any() or value.feedback.any():
        raise RuntimeError("invalid confirmatory candidate population")
    ranks = value["ev_gap"].rank(method="first")
    value["ev_gap_bin"] = pd.qcut(ranks, q=ev_gap_bins, labels=["small", "medium", "large"])
    value["stratum"] = value.apply(
        lambda row: "|".join(
            [
                f"loss={int(row.has_loss)}",
                f"ambiguity={int(row.ambiguity)}",
                f"ev_gap={row.ev_gap_bin}",
            ]
        ),
        axis=1,
    )
    selected = proportional_stratified_sample(value, count, seed)
    selected["split"] = "confirmatory"
    if len(selected) != count or selected.item_id.isin(excluded).any():
        raise RuntimeError("invalid Choices13k confirmatory split")
    if selected.feedback.any():
        raise RuntimeError("confirmatory primary population must contain only no-feedback rows")
    return selected.sort_values("item_id")


def data_driven_splits(
    frame,
    excluded_item_ids,
    calibration_count,
    development_count,
    final_count,
    seed,
    ev_gap_bins=3,
):
    """Build three disjoint no-feedback splits without using human target labels."""
    if ev_gap_bins != 3:
        raise ValueError("the data-driven protocol requires three EV-gap bins")
    excluded = {int(item_id) for item_id in excluded_item_ids}
    value = frame[(~frame.item_id.isin(excluded)) & (~frame.feedback)].copy()
    requested = calibration_count + development_count + final_count
    if requested > len(value):
        raise ValueError(
            f"requested {requested} no-feedback rows but only {len(value)} untouched rows exist"
        )
    ranks = value["ev_gap"].rank(method="first")
    value["ev_gap_bin"] = pd.qcut(ranks, q=ev_gap_bins, labels=["small", "medium", "large"])
    value["stratum"] = value.apply(
        lambda row: "|".join(
            [
                f"loss={int(row.has_loss)}",
                f"ambiguity={int(row.ambiguity)}",
                f"ev_gap={row.ev_gap_bin}",
            ]
        ),
        axis=1,
    )
    selected = proportional_stratified_sample(value, requested, seed)
    calibration = proportional_stratified_sample(selected, calibration_count, seed + 1)
    remaining = selected[~selected.item_id.isin(calibration.item_id)]
    development = proportional_stratified_sample(remaining, development_count, seed + 2)
    final = remaining[~remaining.item_id.isin(development.item_id)].copy()
    if len(final) != final_count:
        raise RuntimeError("invalid final-test split size")
    for name, split in {
        "calibration_pool": calibration,
        "development_validation": development,
        "final_test": final,
    }.items():
        split["split"] = name
        if split.feedback.any() or split.item_id.isin(excluded).any():
            raise RuntimeError(f"invalid {name} population")
    split_sets = [set(calibration.item_id), set(development.item_id), set(final.item_id)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("data-driven splits overlap")
    return tuple(
        split.sort_values("item_id") for split in (calibration, development, final)
    )


def orientations(item_id, repeats, seed):
    if repeats % 2:
        raise ValueError("exact position counterbalancing requires an even repeat count")
    values = ["A_first"] * (repeats // 2) + ["B_first"] * (repeats // 2)
    random_state = int.from_bytes(
        hashlib.sha256(f"{seed}:{item_id}:orientation".encode()).digest()[:8], "big"
    )
    np.random.default_rng(random_state).shuffle(values)
    return values
