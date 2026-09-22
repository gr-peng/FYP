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
                "choices13k_matching_rows": lookup.get(visible_fingerprint(item), []),
            }
        )
    eligible = [
        x["cpc_game_id"] for x in rows if not x["ambiguity"] and not x["choices13k_matching_rows"]
    ]
    result = {
        "external_labels_loaded": False,
        "comparison": "all Choices13k source rows, visible canonical fingerprint",
        "source_choices13k_sha256": digest(source_path.read_bytes()),
        "source_cpc18_sha256": digest(workbook.read_bytes()),
        "feature_items": len(rows),
        "nonambiguous_items": sum(not x["ambiguity"] for x in rows),
        "risk_only_eligible_ids": eligible,
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
                "external_labels_loaded": False,
            }
        )
    )


if __name__ == "__main__":
    audit()
