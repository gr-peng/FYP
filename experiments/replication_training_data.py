"""Source-checked, train-only reads; no development/test access override."""

import csv
import io
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime

import pandas as pd

from experiments.replication_metadata import COLUMNS, PLAN, build_plan, sha

ORDER_COLUMNS = [
    "uuid",
    "session_code",
    "repetition",
    "group_id",
    "round_number",
    "player_id",
    "kind",
    "side",
    "quantity",
    "price",
    "filled",
    "is_replacement",
    "replaced_by",
    "deleted",
    "created",
]
TRADE_COLUMNS = [
    "uuid",
    "session_code",
    "repetition",
    "group_id",
    "round_number",
    "ask_uuid",
    "bid_uuid",
    "quantity",
    "price",
    "created",
]
ROUND_COLUMNS = [
    "participant.code",
    "player.id_in_group",
    "player.cash",
    "player.assets",
    "player.available_cash",
    "player.available_assets",
    "player.dividend_payment",
    "player.next_cash",
    "group.id_in_subsession",
    "group.dividend",
    "group.closing_price",
    "group.average_price",
    "group.starting_timestamp",
    "subsession.round_number",
    "subsession.num_rounds",
    "subsession.repetition",
    "subsession.practice",
    "session.code",
]


def selected_training_rows(rows, assignments, columns, session_column, split="train"):
    if split != "train":
        raise ValueError("only training behavior is open for the replication audit")
    all_ids = [sid for ids in assignments.values() for sid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("overlapping replication partitions")
    wanted = set(assignments["train"])
    for row in rows:
        # Only membership is consulted for all other rows, including excluded source sessions.
        if row[session_column] in wanted:
            yield {k: row[k] for k in columns}


def checked_plan(root):
    plan = json.loads((root / PLAN).read_text())
    if build_plan(root) != plan:
        raise ValueError("replication partition/source freeze mismatch")
    committed = subprocess.check_output(["git", "show", f"HEAD:{PLAN}"], cwd=root)
    if json.loads(committed) != plan:
        raise ValueError("replication partitions must be committed before training access")
    return plan


def load_training_frames(root):
    plan = checked_plan(root)
    output = root / "outputs/market_replication_training_20260922"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "access_log.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "accessed_at": datetime.now(UTC).isoformat(),
                    "scope": "train only",
                    "plan_sha256": sha((root / PLAN).read_bytes()),
                    "source_sha256": plan["source_sha256"],
                    "loader_sha256": sha(
                        (root / "experiments/replication_training_data.py").read_bytes()
                    ),
                    "git_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    "note": "Training exploration; audit code may change. Held-out stays closed.",
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
                for row in selected_training_rows(reader, plan["assignments"], columns, unit):
                    tables[table].append({**row, "source_member": name})
        metadata = {}
        for name, columns in COLUMNS.items():
            frame = pd.read_stata(
                io.BytesIO(archive.read(f"data/proc/{name}.dta")),
                columns=columns,
                convert_categoricals=False,
            )
            metadata[name] = frame[frame.slab.isin(plan["assignments"]["train"])].copy()
    return {k: pd.DataFrame(v) for k, v in tables.items()}, metadata, plan


def load_training_macro_reference(root):
    """Filter the author's processed benchmark by frozen train membership before analysis."""
    plan = checked_plan(root)
    columns = ["slab", "mlab", "rep", "rd", "rad", "rdmax", "ab"]
    with zipfile.ZipFile(root / plan["source_path"]) as archive:
        content = archive.read("data/proc/outcomes.dta")
    selected = []
    # Binary Stata parsing necessarily decodes chunks. Only selected training rows
    # are retained or analyzed; no held-out summaries or outcome-dependent filtering.
    with pd.read_stata(
        io.BytesIO(content), columns=columns, convert_categoricals=False, chunksize=128
    ) as chunks:
        for chunk in chunks:
            selected.append(chunk.loc[chunk.slab.isin(plan["assignments"]["train"]), columns])
    return pd.concat(selected, ignore_index=True)


def deduplicate(frame, keys):
    """Collapse exact semantic copies only; retain provenance and refuse conflicts."""
    columns = [c for c in frame.columns if c != "source_member"]
    unique = frame.drop_duplicates(columns).copy()
    conflicts = unique.duplicated(keys, keep=False)
    if conflicts.any():
        examples = unique.loc[conflicts, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"conflicting cumulative export records: {examples}")
    provenance = frame.groupby(keys, dropna=False).source_member.agg(lambda x: sorted(set(x)))
    unique = unique.drop(columns="source_member").set_index(keys)
    unique["source_members"] = provenance
    return unique.reset_index(), {
        "raw_rows": len(frame),
        "unique_rows": len(unique),
        "identical_copies_removed": len(frame) - len(unique),
        "conflicting_keys": 0,
    }
