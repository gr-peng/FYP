"""Outcome-free metadata validation and whole-session partitions for human markets."""

import hashlib
import io
import json
import zipfile

import pandas as pd

SEED = 20260926
SOURCE = "data/raw/finance_replication_20260922/reproduction_kit.zip"
SOURCE_SHA256 = "642898545b74210f1d999eac2c2b501ae3f08074ec45ab61bcb19081dcfff6ae"
PLAN = "config/frozen/market_replication_partitions_20260922.json"
COLUMNS = {
    "_sessions": ["slab", "sid", "lab", "study"],
    "_markets": ["slab", "sid", "mlab", "mid", "traders", "study", "cond"],
    "_participants": ["slab", "mlab", "plab", "pid", "mid", "study", "cond"],
}
CODE = [
    "experiments/replication_metadata.py",
    "scripts/freeze_market_replication_partitions.py",
    "docs/market_replication_data_protocol.md",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def partitions(sessions, markets, people):
    if (
        len(sessions) != 83
        or len(markets) != 166
        or len(people) != 1544
        or sessions.slab.duplicated().any()
        or markets.mlab.duplicated().any()
        or people.plab.duplicated().any()
        or set(sessions.study) != {1, 2}
    ):
        raise ValueError("unexpected processed metadata population")
    session_map = sessions.set_index("slab")
    market_map = markets.set_index("mlab")
    if set(markets.slab) != set(sessions.slab) or set(people.mlab) != set(markets.mlab):
        raise ValueError("metadata membership coverage differs")
    if markets.duplicated(["slab", "mid"]).any() or people.duplicated(["slab", "mid", "pid"]).any():
        raise ValueError("duplicate within-session metadata identity")
    for row in markets.itertuples(index=False):
        source = session_map.loc[row.slab]
        if row.study != source.study or row.sid != source.sid:
            raise ValueError("market/session metadata mismatch")
    for row in people.itertuples(index=False):
        source = market_map.loc[row.mlab]
        if any(getattr(row, k) != source[k] for k in ("slab", "mid", "study", "cond")):
            raise ValueError("participant/market metadata mismatch")
    for sid, subset in markets.groupby("slab"):
        study = int(session_map.loc[sid, "study"])
        if len(subset) != 2 or set(subset.cond) != ({1, 2} if study == 1 else {3, 4}):
            raise ValueError("expected both paired conditions within each session")
    assignments = {k: [] for k in ("train", "development", "test")}
    for study, sizes in ((1, (19, 6, 6)), (2, (32, 10, 10))):
        ids = sorted(
            sessions.loc[sessions.study == study, "slab"],
            key=lambda x: sha(f"{SEED}:replication-session:{x}".encode()),
        )
        if len(ids) != sum(sizes):
            raise ValueError("unexpected study population")
        offset = 0
        for name, size in zip(assignments, sizes, strict=True):
            assignments[name].extend(ids[offset : offset + size])
            offset += size
    all_ids = [i for v in assignments.values() for i in v]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(sessions.slab):
        raise ValueError("partitions must cover sessions exactly once")
    counts = {
        name: {
            "sessions": len(ids),
            "markets": int(markets.slab.isin(ids).sum()),
            "participants": int(people.slab.isin(ids).sum()),
        }
        for name, ids in assignments.items()
    }
    return assignments, counts


def build_plan(root):
    data = (root / SOURCE).read_bytes()
    if sha(data) != SOURCE_SHA256:
        raise ValueError("replication archive differs from verified source")
    frames, member_hashes = {}, {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name, columns in COLUMNS.items():
            member = f"data/proc/{name}.dta"
            content = archive.read(member)
            member_hashes[member] = sha(content)
            frames[name] = pd.read_stata(
                io.BytesIO(content), columns=columns, convert_categoricals=False
            )
    assignments, counts = partitions(
        frames["_sessions"], frames["_markets"], frames["_participants"]
    )
    folder = "data/raw/finance_replication_20260922/"
    source_files = [folder + f for f in ("instructions.pdf", "software_20240315.zip")]
    return {
        "schema": "human-market-replication-partitions-1",
        "seed": SEED,
        "unit": "whole session; all paired markets, participants and repetitions",
        "population": "author-processed eligibility metadata, explicit column allowlists",
        "behavior_read_by_this_study": False,
        "source_path": SOURCE,
        "source_sha256": SOURCE_SHA256,
        "metadata_columns": COLUMNS,
        "metadata_member_hashes": member_hashes,
        "input_hashes": {p: sha((root / p).read_bytes()) for p in CODE + source_files},
        "assignments": assignments,
        "counts": counts,
        "session_studies": {
            row.slab: int(row.study)
            for row in frames["_sessions"].sort_values("slab").itertuples(index=False)
        },
    }


def freeze(root):
    plan = build_plan(root)
    path = root / PLAN
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("refusing to replace frozen market partitions")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"behavior_read": False, "counts": plan["counts"], "path": PLAN}
