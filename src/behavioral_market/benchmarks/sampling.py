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


def orientations(item_id, repeats, seed):
    if repeats % 2:
        raise ValueError("exact position counterbalancing requires an even repeat count")
    values = ["A_first"] * (repeats // 2) + ["B_first"] * (repeats // 2)
    random_state = int.from_bytes(
        hashlib.sha256(f"{seed}:{item_id}:orientation".encode()).digest()[:8], "big"
    )
    np.random.default_rng(random_state).shuffle(values)
    return values
