"""Archive public Bybit research files without credentials or trading requests."""

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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


def archive_paths(config, directory, day, kind):
    url = config["urls"][kind].format(date=day)
    target = directory / kind / url.rsplit("/", 1)[-1]
    metadata = target.with_name(target.name + ".metadata.json")
    return url, target, metadata


def archive_identity(config, split, day, kind, frozen_hash, url):
    return {
        "url": url,
        "venue": config["venue"],
        "market": config["market"],
        "symbol": config["symbol"],
        "kind": kind,
        "split": split,
        "day": day,
        "config_sha256": frozen_hash,
    }


def checked_cache(target, metadata, identity, maximum_file_bytes):
    """Completed evidence is immutable; reject drift instead of relabelling it."""
    if target.exists():
        if not metadata.exists():
            raise ValueError(f"Existing archive lacks metadata: {target.name}")
        prior = json.loads(metadata.read_text())
        if not isinstance(prior, dict):
            raise ValueError(f"Existing archive metadata is not an object: {target.name}")
        for key, value in identity.items():
            if prior.get(key) != value:
                raise ValueError(f"Existing archive identity differs ({key}): {target.name}")
        size = target.stat().st_size
        if (
            type(prior.get("bytes")) is not int
            or prior["bytes"] != size
            or not 0 < size <= maximum_file_bytes
            or prior.get("sha256") != file_hash(target)
        ):
            raise ValueError(f"Existing archive bytes/hash differ: {target.name}")
        return prior
    return None


def source_validator(headers):
    etag = headers.get("ETag", "").strip()
    if etag.startswith('"') and etag.endswith('"') and len(etag) >= 2:
        return {"header": "ETag", "value": etag}
    modified = headers.get("Last-Modified")
    if modified:
        try:
            if parsedate_to_datetime(modified).tzinfo is not None:
                return {"header": "Last-Modified", "value": modified}
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def content_length(headers):
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Encoded archive response cannot be safely budgeted")
    raw = headers.get("Content-Length", "")
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError("A positive, reliable Content-Length is required")
    return int(raw)


def inspect_source(url, maximum_bytes):
    """Reserve a known size before requesting any archive body."""
    with requests.head(
        url, headers={"Accept-Encoding": "identity"}, allow_redirects=True, timeout=(20, 90)
    ) as response:
        if response.status_code != 200:
            raise ValueError(f"HEAD returned HTTP {response.status_code}: {url}")
        total = content_length(response.headers)
        if total > maximum_bytes:
            raise ValueError("Archive exceeds its reserved file-size budget")
        return {
            "url": url,
            "total": total,
            "validator": source_validator(response.headers),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def plan_downloads(config, directory, tasks, frozen_hash):
    """Preflight every study file and freeze aggregate reservations before any GET."""
    plans, targets, reserved = {}, set(), 0
    limits = config["limits"]
    for task in tasks:
        split, day, kind = task
        url, target, metadata = archive_paths(config, directory, day, kind)
        if target in targets:
            raise ValueError(f"Study tasks collide at archive path: {target}")
        targets.add(target)
        identity = archive_identity(config, split, day, kind, frozen_hash, url)
        prior = checked_cache(target, metadata, identity, limits["maximum_file_bytes"])
        source = None if prior else inspect_source(url, limits["maximum_file_bytes"])
        size = prior["bytes"] if prior else source["total"]
        reserved += size
        if reserved > limits["maximum_total_bytes"]:
            raise ValueError("Archive study exceeds declared total byte budget before download")
        plans[task] = {"reserved_bytes": size, "identity": identity, "source": source}
    return plans


def discard_partial(part, partial_metadata):
    part.unlink(missing_ok=True)
    partial_metadata.unlink(missing_ok=True)


def resume_offset(part, partial_metadata, identity, source):
    """An unvalidated prefix is disposable, never a source of bytes to append to."""
    if part.exists() and partial_metadata.exists() and source["validator"]:
        try:
            saved = json.loads(partial_metadata.read_text())
            size = part.stat().st_size
            if (
                saved.get("identity") == identity
                and saved.get("source", {}).get("url") == source["url"]
                and saved["source"].get("total") == source["total"]
                and saved["source"].get("validator") == source["validator"]
                and 0 < size < source["total"]
            ):
                return size
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    discard_partial(part, partial_metadata)
    return 0


def download_one(config, directory, split, day, kind, frozen_hash, plan):
    url, target, metadata = archive_paths(config, directory, day, kind)
    identity = archive_identity(config, split, day, kind, frozen_hash, url)
    if plan["identity"] != identity:
        raise ValueError("Download reservation identity differs")
    reserved = plan["reserved_bytes"]
    prior = checked_cache(target, metadata, identity, config["limits"]["maximum_file_bytes"])
    if prior:
        if prior["bytes"] > reserved:
            raise ValueError("Cached archive exceeds its reservation")
        return prior
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    partial_metadata = target.with_name(target.name + ".part.metadata.json")
    started = datetime.now(UTC).isoformat()
    last_error = None
    for attempt in range(1, 5):
        try:
            source = (
                plan["source"] if attempt == 1 and plan["source"] else inspect_source(url, reserved)
            )
            offset = resume_offset(part, partial_metadata, identity, source)
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers.update(
                    Range=f"bytes={offset}-", **{"If-Range": source["validator"]["value"]}
                )
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code not in {200, 206}:
                    raise ValueError(f"HTTP {response.status_code} for {target.name}")
                validator = source["validator"]
                if validator and response.headers.get(validator["header"]) != validator["value"]:
                    discard_partial(part, partial_metadata)
                    raise ValueError("Archive version changed; discard prefix and recheck HEAD")
                expected = source["total"]
                length = content_length(response.headers)
                if response.status_code == 206:
                    unit, content_range = response.headers["Content-Range"].split(" ", 1)
                    bounds, total = content_range.split("/")
                    first, last = map(int, bounds.split("-"))
                    if (
                        unit != "bytes"
                        or not offset
                        or first != offset
                        or last != expected - 1
                        or int(total) != expected
                        or length != expected - offset
                    ):
                        raise ValueError("Unexpected partial HTTP response; full suffix required")
                else:
                    offset = 0  # A server ignoring Range must replace, not append the whole file.
                    if length != expected:
                        discard_partial(part, partial_metadata)
                        raise ValueError("Archive length changed; recheck HEAD within reservation")
                if expected > reserved:
                    raise ValueError("Archive exceeds its reserved file-size budget")
                # Persist the version identity before accepting even the first body chunk.
                write_json(partial_metadata, {"identity": identity, "source": source})
                written = offset
                with part.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(2**20):
                        if written + len(chunk) > expected:
                            discard_partial(part, partial_metadata)
                            raise ValueError("Response exceeds declared archive length")
                        handle.write(chunk)
                        written += len(chunk)
                if written != expected:
                    raise ValueError(f"Truncated archive: {written} != {expected}")
                record = {
                    **identity,
                    "bytes": written,
                    "sha256": file_hash(part),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "started_at": started,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "attempts_this_invocation": attempt,
                    "download_source_sha256": digest(Path(__file__).read_bytes()),
                    "path": str(target.relative_to(ROOT)),
                    "validation": (
                        "HTTP size and local SHA256; decompressor CRC and data semantics pending"
                    ),
                }
                # Persist metadata before the atomic publish; interrupted partials are resumable.
                write_json(metadata, record)
                part.replace(target)
                partial_metadata.unlink(missing_ok=True)
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
    study_tasks = [
        (split, day, kind)
        for split, days in config["splits"].items()
        for day in days
        for kind in ("orderbook", "trades")
    ]
    tasks = [task for task in study_tasks if args.split in {task[0], "all"}]
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
        plans = plan_downloads(config, directory, study_tasks, frozen_hash)
        state["reserved_study_bytes"] = sum(plan["reserved_bytes"] for plan in plans.values())
        write_json(out, state)
        with ThreadPoolExecutor(max_workers=config["limits"]["download_workers"]) as pool:
            futures = [
                pool.submit(download_one, config, directory, *task, frozen_hash, plans[task])
                for task in tasks
            ]
            for future in futures:
                state["files"].append(future.result())
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
