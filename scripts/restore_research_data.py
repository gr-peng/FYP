"""Restore the versioned public source files listed in research_data/sources.csv."""

import argparse
import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify(path, row):
    if path.stat().st_size != int(row["bytes"]):
        raise ValueError("file size differs from the archive")
    with path.open("rb") as reader:
        if hashlib.file_digest(reader, "sha256").hexdigest() != row["sha256"]:
            raise ValueError("SHA-256 differs from the archive")


def restore(root, row, verify_only):
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root / "data"):
        raise ValueError("source path outside data/")
    if path.exists() or verify_only:
        verify(path, row)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(row["url"], headers={"User-Agent": "FYP research archive reader/1.0"})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        with urlopen(request, timeout=90) as source, name.open("wb") as writer:
            count = 0
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > int(row["bytes"]):
                    raise ValueError("download exceeds recorded size")
                writer.write(chunk)
        verify(name, row)
        try:
            os.link(name, path)
        except FileExistsError:
            verify(path, row)
    finally:
        name.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="dataset column, repeatable")
    parser.add_argument("--path", help="one exact catalog path")
    parser.add_argument("--root", type=Path, default=ROOT, help="directory containing data/")
    parser.add_argument("--verify-only", action="store_true", help="only check local files")
    args = parser.parse_args()
    with (ROOT / "research_data/sources.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.dataset and set(args.dataset) - {r["dataset"] for r in rows}:
        parser.error("unknown dataset name")
    selected = [
        r for r in rows
        if (not args.dataset or r["dataset"] in args.dataset)
        and (not args.path or r["path"] == args.path)
    ]
    if not selected:
        parser.error("no matching source")
    for row in selected:
        try:
            restore(args.root.resolve(), row, args.verify_only)
        except (OSError, ValueError) as error:
            sys.exit(f'{row["path"]}: {error}')
    print(f"Verified {len(selected)} files ({sum(int(r['bytes']) for r in selected)} bytes)")


if __name__ == "__main__":
    main()
