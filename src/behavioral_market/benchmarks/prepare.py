import json

import pandas as pd

from behavioral_market.data.archive import digest

from .choices13k import (
    clone_snapshot,
    items_frame,
    load_official,
    verify_manifest,
    write_manifest,
)
from .sampling import stratified_split


def split_paths(config, root):
    directory = root / config["dataset"]["processed_dir"]
    return {
        "all": directory / "all_items.parquet",
        "pilot": directory / "pilot_items.parquet",
        "calibration": directory / "calibration_items.parquet",
        "heldout": directory / "heldout_items.parquet",
    }


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
