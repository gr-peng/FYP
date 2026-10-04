"""Explicit development-only access gated by an immutable executable protocol.

The original train-only loader stays closed to every other partition. Test rows
are consulted for membership only and never retained or analyzed here.
"""

import csv
import io
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime

import pandas as pd

from experiments.replication_metadata import COLUMNS, PLAN, sha
from experiments.replication_training_data import (
    ORDER_COLUMNS,
    ROUND_COLUMNS,
    TRADE_COLUMNS,
    checked_plan,
)

FREEZE = "config/frozen/financial_probability_development_20261004.json"


def selected_development_rows(rows, assignments, columns, session_column):
    ids = [sid for group in assignments.values() for sid in group]
    if len(ids) != len(set(ids)):
        raise ValueError("overlapping partitions")
    wanted = set(assignments["development"])
    for row in rows:
        if row[session_column] in wanted:
            yield {key: row[key] for key in columns}


def checked_freeze(root):
    frozen = json.loads((root / FREEZE).read_text())
    if frozen["scope"] != "development only; test remains closed":
        raise ValueError("invalid development scope")
    committed = subprocess.check_output(["git", "show", f"HEAD:{FREEZE}"], cwd=root)
    if json.loads(committed) != frozen:
        raise ValueError("development protocol must be committed before access")
    for path, expected in frozen["input_hashes"].items():
        if sha((root / path).read_bytes()) != expected:
            raise ValueError(f"frozen input changed: {path}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("freeze clean executable protocol before development access")
    return frozen


def load_development_frames(root):
    frozen = checked_freeze(root)
    plan = checked_plan(root)
    if frozen["partition_sha256"] != sha((root / PLAN).read_bytes()):
        raise ValueError("partition freeze mismatch")
    out = root / "outputs/financial_probability_development_20261004"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "access_log.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "accessed_at": datetime.now(UTC).isoformat(),
                    "scope": frozen["scope"],
                    "freeze_sha256": sha((root / FREEZE).read_bytes()),
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    "source_sha256": plan["source_sha256"],
                }
            )
            + "\n"
        )
    tables = {"orders": [], "trades": [], "rounds": []}
    with zipfile.ZipFile(root / plan["source_path"]) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("data/raw/"):
                continue
            if re.search(r"/custom/Order_[12]\.csv$", name):
                table, columns, unit = "orders", ORDER_COLUMNS, "session_code"
            elif re.search(r"/custom/Trade_[12]\.csv$", name):
                table, columns, unit = "trades", TRADE_COLUMNS, "session_code"
            elif re.search(r"/raw/cda_rep[12]_\d{4}-\d{2}-\d{2}\.csv$", name):
                table, columns, unit = "rounds", ROUND_COLUMNS, "session.code"
            else:
                continue
            with archive.open(name) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                for row in selected_development_rows(reader, plan["assignments"], columns, unit):
                    tables[table].append({**row, "source_member": name})
        metadata = {}
        for name, columns in COLUMNS.items():
            frame = pd.read_stata(
                io.BytesIO(archive.read(f"data/proc/{name}.dta")),
                columns=columns,
                convert_categoricals=False,
            )
            metadata[name] = frame[frame.slab.isin(plan["assignments"]["development"])].copy()
    return {key: pd.DataFrame(rows) for key, rows in tables.items()}, metadata, plan
