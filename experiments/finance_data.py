"""Metadata-only split construction and guarded financial development reads.

Held-out behavior requires a separate, subsequently frozen evaluation entry point.
This module deliberately does not offer a test-data override.
"""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SEED = 20260922
CDA_METADATA = ["SessionID", "Condition (BuyANDSell / BuyORSell)", "Round"]
SCOPIC_METADATA = [
    "participant.code",
    "session.code",
    "group.id_in_subsession",
    "subsession.round_number",
    "participant._current_app_name",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def cda_split(metadata: pd.DataFrame) -> dict:
    pairs = metadata[CDA_METADATA[:2]].drop_duplicates()
    if pairs.groupby("SessionID").size().max() != 1:
        raise ValueError("a CDA session crosses conditions")
    assignments = {name: [] for name in ("train", "development", "test")}
    conditions = {}
    for condition, frame in pairs.groupby(CDA_METADATA[1]):
        ids = sorted(frame.SessionID, key=lambda x: stable_key("cda", x))
        expected = {"BuyANDSell": (9, 5, 2), "BuyORSell": (12, 6, 3)}
        if condition not in expected or len(ids) != expected[condition][0]:
            raise ValueError("unexpected CDA metadata population")
        _, n_train, n_dev = expected[condition]
        assignments["train"].extend(ids[:n_train])
        assignments["development"].extend(ids[n_train : n_train + n_dev])
        assignments["test"].extend(ids[n_train + n_dev :])
        conditions.update(dict.fromkeys(ids, condition))
    if set(conditions.values()) != {"BuyANDSell", "BuyORSell"}:
        raise ValueError("missing CDA condition")
    return {"unit": "SessionID", "assignments": assignments, "conditions": conditions}


def scopic_split(metadata: pd.DataFrame) -> dict:
    ids = sorted(metadata["session.code"].unique())
    if len(ids) != 6:
        raise ValueError("unexpected scopic session population")
    people = metadata.drop_duplicates("participant.code")
    for field in ("session.code", "group.id_in_subsession", "participant._current_app_name"):
        if metadata.groupby("participant.code")[field].nunique().max() != 1:
            raise ValueError(f"participant changes {field}")
    if set(people.groupby(["session.code", "group.id_in_subsession"]).size()) != {3}:
        raise ValueError("unexpected peer group size")
    treatments = {
        sid: sorted(
            set(metadata.loc[metadata["session.code"] == sid, "participant._current_app_name"])
        )
        for sid in ids
    }
    candidates = []
    for train in itertools.combinations(ids, 2):
        remaining = sorted(set(ids) - set(train))
        for dev in itertools.combinations(remaining, 2):
            test = sorted(set(remaining) - set(dev))
            parts = {"train": list(train), "development": list(dev), "test": test}
            if all(
                set().union(*(set(treatments[s]) for s in part)) == {"spt1", "spt2"}
                for part in parts.values()
            ):
                candidates.append(parts)
    if not candidates:
        raise ValueError("cannot preserve both treatments in every session split")
    chosen = min(candidates, key=lambda x: stable_key("scopic", json.dumps(x, sort_keys=True)))
    return {
        "unit": "session.code",
        "assignments": chosen,
        "treatments": treatments,
        "admissible_assignments": len(candidates),
        "rule": "2 sessions per split, both treatments per split; lowest seeded SHA256",
    }


def build_split_plan(root: Path) -> dict:
    sources = {
        "cda": root / "data/raw/finance_cda_2017_20260922/market_data.xlsx",
        "scopic": root / "data/raw/finance_scopic_20260922/data_de_scopic.csv",
    }
    # Explicit column allowlists: no action, price, wealth, risk response or outcome.
    cda = pd.read_excel(sources["cda"], usecols=CDA_METADATA)
    scopic = pd.read_csv(sources["scopic"], usecols=SCOPIC_METADATA)
    datasets = {"cda": cda_split(cda), "scopic": scopic_split(scopic)}
    for name, result in datasets.items():
        result["source_path"] = str(sources[name].relative_to(root))
        result["source_sha256"] = sha256(sources[name])
        result["metadata_columns_read"] = CDA_METADATA if name == "cda" else SCOPIC_METADATA
    return {
        "schema_version": 1,
        "seed": SEED,
        "behavior_used_for_splits": False,
        "scope": "Data partition freeze only; not a frozen model/evaluation protocol",
        "datasets": datasets,
    }


def selected_rows(rows, id_column: str, assignments: dict, split: str):
    """Filter raw records before converting, validating or inspecting behavior."""
    if split not in {"train", "development"}:
        raise ValueError("held-out behavior is closed until a frozen evaluation protocol exists")
    all_ids = [sid for ids in assignments.values() for sid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("overlapping session partitions")
    wanted = set(assignments[split])
    allowed = set(all_ids)
    for row in rows:
        sid = row[id_column]
        if sid not in allowed:
            raise ValueError("unassigned source session")
        if sid in wanted:
            yield row


def load_development_rows(root: Path, plan: dict, dataset: str, split="train") -> list[dict]:
    """Return only the requested train/development rows; verify frozen source bytes."""
    spec = plan["datasets"][dataset]
    path = root / spec["source_path"]
    if sha256(path) != spec["source_sha256"]:
        raise ValueError("financial source differs from frozen partition source")
    if split not in {"train", "development"}:
        raise ValueError("held-out behavior is closed until a frozen evaluation protocol exists")
    if dataset == "cda":
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            cells = workbook.active.iter_rows(values_only=True)
            names = next(cells)
            rows = (dict(zip(names, values, strict=True)) for values in cells)
            return list(selected_rows(rows, spec["unit"], spec["assignments"], split))
        finally:
            workbook.close()
    if dataset == "scopic":
        with path.open(newline="") as handle:
            return list(
                selected_rows(csv.DictReader(handle), spec["unit"], spec["assignments"], split)
            )
    raise ValueError("unknown financial dataset")
