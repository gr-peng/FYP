"""Archive public Bybit research files without credentials or trading requests."""

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

from behavioral_market.data.archive import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def download_one(config, directory, split, day, kind, frozen_hash):
    url = config["urls"][kind].format(date=day)
    target = directory / kind / url.rsplit("/", 1)[-1]
    metadata = target.with_name(target.name + ".metadata.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        prior = json.loads(metadata.read_text())
        if prior["sha256"] != file_hash(target) or prior["url"] != url:
            raise ValueError(f"Existing archive differs: {target.name}")
        return prior
    part = target.with_name(target.name + ".part")
    started = datetime.now(UTC).isoformat()
    last_error = None
    for attempt in range(1, 5):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code not in {200, 206}:
                    raise ValueError(f"HTTP {response.status_code} for {target.name}")
                if response.status_code == 206:
                    bounds, total = response.headers["Content-Range"].split(" ", 1)[1].split("/")
                    first, last = map(int, bounds.split("-"))
                    expected = int(total)
                    if first != offset or last != expected - 1:
                        raise ValueError("Unexpected partial HTTP response; full suffix required")
                else:
                    expected = int(response.headers["Content-Length"])
                    offset = 0  # A server ignoring Range must replace, not append the whole file.
                if expected > config["limits"]["maximum_file_bytes"]:
                    raise ValueError("Archive exceeds the declared file-size budget")
                written = offset
                with part.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(2**20):
                        handle.write(chunk)
                        written += len(chunk)
                        if written > expected:
                            raise ValueError("Response exceeds declared archive length")
                if written != expected:
                    raise ValueError(f"Truncated archive: {written} != {expected}")
                record = {
                    "url": url,
                    "venue": config["venue"],
                    "market": config["market"],
                    "symbol": config["symbol"],
                    "kind": kind,
                    "split": split,
                    "day": day,
                    "bytes": written,
                    "sha256": file_hash(part),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "started_at": started,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "attempts_this_invocation": attempt,
                    "config_sha256": frozen_hash,
                    "path": str(target.relative_to(ROOT)),
                    "validation": (
                        "HTTP size and local SHA256; decompressor CRC and data semantics pending"
                    ),
                }
                # Persist metadata before the atomic publish; interrupted partials are resumable.
                write_json(metadata, record)
                part.replace(target)
                print(json.dumps({"complete": target.name, "bytes": written}), flush=True)
                return record
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = {"type": type(exc).__name__, "detail": str(exc)[:300]}
            write_json(target.with_name(target.name + ".error.json"), last_error)
            if attempt < 4:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Archive download failed: {target.name}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/crypto_microstructure_20261011.json"
    )
    parser.add_argument(
        "--split", choices=["train", "development", "event_test", "all"], required=True
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frozen_hash = digest(args.config.read_bytes())
    directory = ROOT / "data/raw" / config["study_id"]
    tasks = [
        (split, day, kind)
        for split, days in config["splits"].items()
        if args.split in {split, "all"}
        for day in days
        for kind in ("orderbook", "trades")
    ]
    out = ROOT / "outputs" / config["study_id"] / f"download_{args.split}.json"
    state = {
        "status": "running",
        "config_sha256": frozen_hash,
        "source_sha256": digest(Path(__file__).read_bytes()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split": args.split,
        "files": [],
    }
    write_json(out, state)
    try:
        with ThreadPoolExecutor(max_workers=config["limits"]["download_workers"]) as pool:
            futures = [
                pool.submit(download_one, config, directory, *task, frozen_hash) for task in tasks
            ]
            for future in futures:
                state["files"].append(future.result())
                if (
                    sum(row["bytes"] for row in state["files"])
                    > config["limits"]["maximum_total_bytes"]
                ):
                    raise ValueError("Archive exceeds declared study byte budget")
                write_json(out, state)
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "detail": str(exc)[:400]}
        raise
    finally:
        write_json(out, state)


if __name__ == "__main__":
    main()
