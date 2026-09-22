#!/usr/bin/env python3
"""Inventory external CPC18 features without reading external human responses."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from behavioral_market.data.archive import digest, write_json  # noqa: E402
from experiments.calibration_v2 import visible_fingerprint  # noqa: E402


def audit():
    folder = ROOT / "data/raw/cpc18_external_20260922"
    manifest = json.loads((folder / "manifest.json").read_text())
    workbook = folder / "cpc18_210.xlsx"
    if digest(workbook.read_bytes()) != manifest["sha256"]:
        raise ValueError("CPC18 workbook archive mismatch")
    external = pd.read_excel(workbook, usecols=lambda column: not str(column).startswith("B."))
    if any(str(column).startswith("B.") for column in external):
        raise ValueError("external aggregate responses must not be loaded")
    source_path = ROOT / "data/processed/human_benchmarks/choices13k/all_items.parquet"
    source = pd.read_parquet(
        source_path,
        columns=[
            "item_id",
            "option_a_json",
            "option_b_json",
            "ambiguity",
        ],
    )
    lookup = {}
    for row in source.itertuples():
        item = SimpleNamespace(
            option_a=json.loads(row.option_a_json),
            option_b=json.loads(row.option_b_json),
            ambiguity=bool(row.ambiguity),
        )
        lookup.setdefault(visible_fingerprint(item), []).append(int(row.item_id))
    rows = []
    for row in external.to_dict("records"):
        options = {}
        for side in ("a", "b"):
            options[side] = [
                (float(row[f"{side}p{i}"]), float(row[f"{side}v{i}"]))
                for i in range(1, 11)
                if pd.notna(row[f"{side}p{i}"]) and pd.notna(row[f"{side}v{i}"])
            ]
            if (
                not options[side]
                or abs(sum(p for p, _ in options[side]) - 1) > 1e-6
                or any(not 0 <= p <= 1 for p, _ in options[side])
            ):
                raise ValueError("invalid complete-distribution worksheet")
        if row["Amb"] not in (True, False, 0, 1):
            raise ValueError("unexpected ambiguity encoding")
        item = SimpleNamespace(
            option_a=options["a"], option_b=options["b"], ambiguity=bool(row["Amb"])
        )
        rows.append(
            {
                "cpc_game_id": int(row["GameID"]),
                "ambiguity": item.ambiguity,
                "visible_fingerprint": visible_fingerprint(item),
                "choices13k_matching_rows": lookup.get(visible_fingerprint(item), []),
            }
        )
    eligible = [
        x["cpc_game_id"] for x in rows if not x["ambiguity"] and not x["choices13k_matching_rows"]
    ]
    unique = {}
    duplicate_groups = {}
    for row in sorted(rows, key=lambda x: x["cpc_game_id"]):
        if row["cpc_game_id"] in eligible:
            key = row["visible_fingerprint"]
            unique.setdefault(key, row["cpc_game_id"])
            duplicate_groups.setdefault(key, []).append(row["cpc_game_id"])
    result = {
        "external_labels_loaded": False,
        "comparison": "all Choices13k source rows, visible canonical fingerprint",
        "source_choices13k_sha256": digest(source_path.read_bytes()),
        "source_cpc18_sha256": digest(workbook.read_bytes()),
        "feature_items": len(rows),
        "nonambiguous_items": sum(not x["ambiguity"] for x in rows),
        "risk_only_eligible_ids": eligible,
        "unique_risk_only_eligible_ids": sorted(unique.values()),
        "within_source_duplicate_groups": [x for x in duplicate_groups.values() if len(x) > 1],
        "within_source_deduplication_rule": "retain smallest GameID; no labels read",
        "rows": rows,
        "note": "Risk-only eligibility. Ambiguous supports need a presentation audit. "
        "The workbook covers the calibration 210, not every problem in the later raw archive.",
    }
    write_json(folder / "feature_overlap_screen.json", result)
    print(
        json.dumps(
            {
                "feature_items": len(rows),
                "risk_only_eligible": len(eligible),
                "unique_risk_only_eligible": len(unique),
                "external_labels_loaded": False,
            }
        )
    )

    raw = folder / "all_raw.csv"
    raw_manifest = json.loads((folder / "raw_manifest.json").read_text())
    expected = next(x["sha256"] for x in raw_manifest["files"] if x["path"].endswith("all_raw.csv"))
    if digest(raw.read_bytes()) != expected:
        raise ValueError("CPC18 raw archive mismatch")
    parameters = [
        "Ha",
        "pHa",
        "La",
        "LotShapeA",
        "LotNumA",
        "Hb",
        "pHb",
        "Lb",
        "LotShapeB",
        "LotNumB",
        "Amb",
        "Corr",
    ]
    columns = ["SubjID", "Location", "Set", "Condition", "GameID", "Trial", "Feedback", "block"]
    metadata = pd.read_csv(raw, usecols=columns + parameters)
    definitions = metadata[["GameID"] + parameters].drop_duplicates()
    if definitions.GameID.duplicated().any():
        raise ValueError("raw game definition varies")
    paired = definitions.merge(
        external[["GameID"] + parameters], on="GameID", suffixes=("_r", "_w")
    )
    if len(paired) != len(external) or any(
        not paired[f"{p}_r"].fillna("").equals(paired[f"{p}_w"].fillna(""))
        and not (paired[f"{p}_r"].fillna("") == paired[f"{p}_w"].fillna("")).all()
        for p in parameters
    ):
        raise ValueError("workbook/raw gamble definition mismatch")
    if metadata.duplicated(["SubjID", "GameID", "Trial"]).any():
        raise ValueError("duplicate trial identity")
    if not ((metadata.Trial <= 5) == (metadata.Feedback == 0)).all():
        raise ValueError("no-feedback trial mapping changed")
    usable = metadata[metadata.GameID.isin(unique.values()) & (metadata.Feedback == 0)]
    if not usable.groupby(["SubjID", "GameID"]).size().eq(5).all():
        raise ValueError("incomplete no-feedback subject-game block")
    subject_counts = usable.groupby("GameID").SubjID.nunique()
    inventory = {
        "human_choices_loaded": False,
        "metadata_columns": columns + parameters,
        "raw_sha256": expected,
        "raw_rows": len(metadata),
        "raw_games": int(metadata.GameID.nunique()),
        "workbook_raw_gamble_agreements": len(paired),
        "unique_external_items": len(unique),
        "eligible_no_feedback_rows": len(usable),
        "eligible_subjects": int(usable.SubjID.nunique()),
        "subjects_per_item_min_max": [int(subject_counts.min()), int(subject_counts.max())],
        "eligible_set_ids": sorted(usable.Set.unique().tolist()),
        "interpretation": "Five repeated no-feedback trials per participant-game. Subject IDs "
        "are unique in the full archive; repeated trials are not independent participants.",
    }
    write_json(folder / "raw_metadata_screen.json", inventory)
    print(json.dumps(inventory))


if __name__ == "__main__":
    audit()
