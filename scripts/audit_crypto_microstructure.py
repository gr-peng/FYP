# ruff: noqa: E402
"""Validate frozen public archives; export training/development features explicitly."""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behavioral_market.data.archive import digest, write_json
from experiments.bybit_reconstruction import audit_book_file


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def audit_task(config, config_hash, split, day, kind, export_features):
    url = config["urls"][kind].format(date=day)
    path = ROOT / "data/raw" / config["study_id"] / kind / url.rsplit("/", 1)[-1]
    meta = json.loads(path.with_name(path.name + ".metadata.json").read_text())
    expected = {
        "url": url,
        "split": split,
        "day": day,
        "kind": kind,
        "venue": config["venue"],
        "market": config["market"],
        "symbol": config["symbol"],
        "config_sha256": config_hash,
    }
    if any(meta[k] != v for k, v in expected.items()):
        raise ValueError("Archive provenance or frozen split differs")
    if meta["sha256"] != file_hash(path) or meta["bytes"] != path.stat().st_size:
        raise ValueError("Archive bytes differ from download manifest")
    output = ROOT / "outputs" / config["study_id"] / "integrity"
    output.mkdir(parents=True, exist_ok=True)
    features = None
    if export_features:
        if split == "event_test":
            raise ValueError("Final event features require a future frozen evaluation gate")
        features = ROOT / "data/processed" / config["study_id"] / split / f"{day}_{kind}_1s.parquet"
    if kind == "orderbook":
        result = audit_book_file(path, day, features, output / f"{day}_{kind}.progress.json")
    else:
        from experiments.bybit_trades import audit_trades_file

        result = audit_trades_file(path, day, features)
    result.update(
        {
            "input_sha256": meta["sha256"],
            "split": split,
            "config_sha256": config_hash,
            "kind": kind,
            "day": day,
        }
    )
    write_json(output / f"{day}_{kind}.json", result)
    print(
        json.dumps(
            {"day": day, "kind": kind, "output": str(output), "rows": result.get("feature_rows")}
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/crypto_microstructure_20261011.json"
    )
    parser.add_argument(
        "--split", choices=["train", "development", "event_test", "all"], required=True
    )
    parser.add_argument("--kind", choices=["orderbook", "trades", "all"], default="all")
    parser.add_argument("--export-features", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Choose 1–4 workers")
    if args.export_features and args.split in {"all", "event_test"}:
        parser.error("Event-test feature extraction is closed; choose train or development")
    config = json.loads(args.config.read_text())
    config_hash = digest(args.config.read_bytes())
    tasks = [
        (split, day, kind)
        for split, days in config["splits"].items()
        if args.split in {"all", split}
        for day in days
        for kind in ("orderbook", "trades")
        if args.kind in {kind, "all"}
    ]
    out = ROOT / "outputs" / config["study_id"] / f"audit_{args.split}_{args.kind}.json"
    state = {
        "status": "running",
        "config_sha256": config_hash,
        "results": [],
        "source_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in [
                Path(__file__),
                ROOT / "experiments/bybit_reconstruction.py",
                ROOT / "experiments/bybit_trades.py",
                ROOT / "src/behavioral_market/data/bybit_book.py",
            ]
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write_json(out, state)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(audit_task, config, config_hash, *task, args.export_features)
                for task in tasks
            ]
            for future in as_completed(futures):
                state["results"].append(future.result())
                write_json(out, state)
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "detail": str(exc)[:300]}
        raise
    finally:
        write_json(out, state)


if __name__ == "__main__":
    main()
