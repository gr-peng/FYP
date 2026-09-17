import json
import tempfile
from pathlib import Path

import pandas as pd

from behavioral_market.data.archive import digest

from .choices13k import (
    clone_snapshot,
    items_frame,
    load_official,
    verify_manifest,
    write_manifest,
)
from .sampling import confirmatory_split, stratified_split


def split_paths(config, root):
    directory = root / config["dataset"]["processed_dir"]
    paths = {
        "all": directory / "all_items.parquet",
        "pilot": directory / "pilot_items.parquet",
        "calibration": directory / "calibration_items.parquet",
        "heldout": directory / "heldout_items.parquet",
    }
    if "confirmatory_file" in config["dataset"]:
        paths["confirmatory"] = directory / config["dataset"]["confirmatory_file"]
    return paths


def prepare(config, root):
    manifest_path = root / config["dataset"]["manifest"]
    paths = split_paths(config, root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_commit"] != config["dataset"]["source_commit"]:
            raise ValueError("existing Choices13k manifest uses a different source commit")
        verify_manifest(manifest, root)
        for split in ("calibration", "heldout"):
            if manifest["splits"][f"{split}_items"] != config["dataset"][f"{split}_items"]:
                raise ValueError("existing Choices13k split size differs from configuration")
        return manifest
    raw = clone_snapshot(config, root)
    items = load_official(raw / "c13k_selections.csv", raw / "c13k_problems.json")
    frame = items_frame(items)
    frame["ev_gap"] = (frame.ev_a - frame.ev_b).abs()
    calibration, heldout, pilot = stratified_split(
        frame,
        config["dataset"]["calibration_items"],
        config["dataset"]["heldout_items"],
        config["experiment"]["seed"],
        config["dataset"]["ev_gap_quantiles"],
    )
    paths["all"].parent.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "all": frame,
        "pilot": pilot,
        "calibration": calibration,
        "heldout": heldout,
    }.items():
        value.to_parquet(paths[name], index=False)
    manifest = write_manifest(config, root, raw, list(paths.values()))
    manifest["splits"] = {
        "seed": config["experiment"]["seed"],
        "calibration_items": len(calibration),
        "heldout_items": len(heldout),
        "calibration_sha256": digest(paths["calibration"].read_bytes()),
        "heldout_sha256": digest(paths["heldout"].read_bytes()),
    }
    from behavioral_market.data.archive import write_json

    write_json(manifest_path, manifest)
    return manifest


def load_split(config, root, split):
    manifest_path = root / config["dataset"]["manifest"]
    if not manifest_path.exists():
        raise RuntimeError("Choices13k data are not prepared; run --stage prepare first")
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest, root)
    path = split_paths(config, root)[split]
    expected = manifest["splits"][f"{split}_sha256"]
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"frozen {split} split hash mismatch")
    return pd.read_parquet(path), manifest


def build_confirmatory_split(config, root):
    """Build the deterministic untouched split in memory from the frozen pilot data."""
    dataset = config["dataset"]
    base_manifest_path = root / dataset["base_manifest"]
    base_manifest = json.loads(base_manifest_path.read_text())
    verify_manifest(base_manifest, root)
    paths = split_paths(config, root)
    all_items = pd.read_parquet(paths["all"])
    pilot_items = pd.read_parquet(paths["pilot"])
    return confirmatory_split(
        all_items,
        pilot_items.item_id,
        dataset["confirmatory_items"],
        config["experiment"]["seed"],
        dataset["ev_gap_quantiles"],
    )


def confirmatory_parquet_hash(config, root):
    """Return the deterministic parquet hash without changing repository data."""
    frame = build_confirmatory_split(config, root)
    with tempfile.TemporaryDirectory(prefix="choices13k-confirmatory-") as directory:
        path = Path(directory) / config["dataset"]["confirmatory_file"]
        frame.to_parquet(path, index=False)
        return digest(path.read_bytes())


def prepare_confirmatory(config, root):
    """Materialize and verify the preregistered untouched confirmatory split."""
    dataset = config["dataset"]
    base_manifest_path = root / dataset["base_manifest"]
    base_manifest = json.loads(base_manifest_path.read_text())
    verify_manifest(base_manifest, root)
    paths = split_paths(config, root)
    frame = build_confirmatory_split(config, root)
    path = paths["confirmatory"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    actual_hash = digest(path.read_bytes())
    expected_hash = dataset["confirmatory_split_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"confirmatory split hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    pilot_hash = digest(paths["pilot"].read_bytes())
    if pilot_hash != dataset["excluded_pilot_sha256"]:
        raise ValueError("pilot exclusion set no longer matches the preregistered hash")
    manifest = {
        "dataset": "choices13k_confirmatory",
        "source_repo": dataset["source_repo"],
        "source_commit": dataset["source_commit"],
        "row_identity": "zero_based_csv_row_index_equals_c13k_problems_json_key",
        "base_manifest": str(base_manifest_path.relative_to(root)),
        "base_manifest_sha256": digest(base_manifest_path.read_bytes()),
        "raw_files": base_manifest["raw_files"],
        "processed_files": [
            {"path": str(path.relative_to(root)), "sha256": actual_hash},
        ],
        "splits": {
            "seed": config["experiment"]["seed"],
            "confirmatory_items": len(frame),
            "confirmatory_sha256": actual_hash,
            "excluded_pilot_items": int(len(pd.read_parquet(paths["pilot"]))),
            "excluded_pilot_sha256": pilot_hash,
            "primary_feedback": False,
        },
    }
    from behavioral_market.data.archive import write_json

    write_json(root / dataset["manifest"], manifest)
    return manifest
