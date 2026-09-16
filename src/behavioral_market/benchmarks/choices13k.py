import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from behavioral_market.data.archive import digest, write_json

from .schemas import Choices13kItem

SOURCE_FILES = ("c13k_problems.json", "c13k_selections.csv")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def moments(outcomes):
    expected = sum(float(probability) * float(payoff) for probability, payoff in outcomes)
    variance = sum(
        float(probability) * (float(payoff) - expected) ** 2
        for probability, payoff in outcomes
    )
    return expected, variance


def load_official(csv_path, problems_path):
    selections = pd.read_csv(csv_path)
    problems = json.loads(Path(problems_path).read_text())
    if len(selections) != len(problems) or set(problems) != {str(i) for i in selections.index}:
        raise ValueError("Choices13k JSON keys must exactly match CSV row indices")
    rows = []
    for item_id, source in selections.iterrows():
        raw = problems[str(item_id)]
        option_a = [(float(p), float(x)) for p, x in raw["A"]]
        option_b = [(float(p), float(x)) for p, x in raw["B"]]
        ev_a, var_a = moments(option_a)
        ev_b, var_b = moments(option_b)
        row = Choices13kItem(
            item_id=int(item_id),
            problem_id=int(source["Problem"]),
            n_humans=int(source["n"]),
            feedback=parse_bool(source["Feedback"]),
            human_b_rate=float(source["bRate"]),
            option_a=option_a,
            option_b=option_b,
            ambiguity=parse_bool(source["Amb"]),
            correlation=int(source["Corr"]),
            block=int(source["Block"]),
            has_loss=any(payoff < 0 for _, payoff in [*option_a, *option_b]),
            ev_a=ev_a,
            ev_b=ev_b,
            var_a=var_a,
            var_b=var_b,
        )
        rows.append(row)
    return rows


def items_frame(items):
    rows = []
    for item in items:
        row = item.model_dump(exclude={"option_a", "option_b"})
        row["option_a_json"] = json.dumps(item.option_a, separators=(",", ":"))
        row["option_b_json"] = json.dumps(item.option_b, separators=(",", ":"))
        rows.append(row)
    return pd.DataFrame(rows)


def frame_items(frame):
    items = []
    for row in frame.to_dict("records"):
        row["option_a"] = json.loads(row.pop("option_a_json"))
        row["option_b"] = json.loads(row.pop("option_b_json"))
        row.pop("split", None)
        row.pop("ev_gap_bin", None)
        row.pop("stratum", None)
        row.pop("ev_gap", None)
        items.append(Choices13kItem.model_validate(row))
    return items


def clone_snapshot(config, root):
    spec = config["dataset"]
    checkout = root / spec["third_party_dir"]
    commit = spec["source_commit"]
    if not checkout.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["source_repo"], str(checkout)],
            cwd=root,
            check=True,
        )
    remote = subprocess.run(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if remote.rstrip("/") != spec["source_repo"].rstrip("/"):
        raise RuntimeError("existing Choices13k checkout has an unexpected origin")
    subprocess.run(["git", "-C", str(checkout), "fetch", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != commit:
        raise RuntimeError("Choices13k checkout did not resolve to the pinned commit")
    raw = root / spec["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copyfile(checkout / name, raw / name)
    return raw


def file_record(path, root):
    return {"path": str(path.relative_to(root)), "sha256": digest(path.read_bytes())}


def verify_manifest(manifest, root):
    for record in [*manifest["raw_files"], *manifest["processed_files"]]:
        path = root / record["path"]
        if not path.exists() or digest(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Choices13k manifest mismatch: {record['path']}")


def write_manifest(config, root, raw, processed_paths):
    manifest = {
        "dataset": "choices13k",
        "source_repo": config["dataset"]["source_repo"],
        "source_commit": config["dataset"]["source_commit"],
        "row_identity": "zero_based_csv_row_index_equals_c13k_problems_json_key",
        "raw_files": [file_record(raw / name, root) for name in SOURCE_FILES],
        "processed_files": [file_record(path, root) for path in processed_paths],
    }
    write_json(root / config["dataset"]["manifest"], manifest)
    return manifest
