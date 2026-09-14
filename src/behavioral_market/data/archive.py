import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import requests


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Archive:
    """Immutable content-addressed raw bodies and credential-free request provenance."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: list[dict] = []

    def get(
        self, source: str, url: str, params=None, headers=None, force=False
    ) -> tuple[bytes, dict]:
        params = params or {}
        safe = {k: v for k, v in params.items() if k.lower() not in {"apikey", "token", "key"}}
        identity = digest(json.dumps([url, safe], sort_keys=True).encode())
        index = self.root / source / f"{identity}.metadata.json"
        if index.exists() and not force:
            entry = json.loads(index.read_text())
            body = Path(entry["path"]).read_bytes()
            if digest(body) != entry["sha256"]:
                raise ValueError("raw archive hash mismatch")
        else:
            try:
                response = requests.get(url, params=params, headers=headers, timeout=45)
            except requests.RequestException as exc:
                raise RuntimeError(f"{source}: transport failure ({type(exc).__name__})") from None
            if response.status_code != 200:
                raise RuntimeError(f"{source}: HTTP {response.status_code}")
            body = response.content
            path = self.root / source / f"{digest(body)}.raw"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(body)
            entry = {
                "source": source,
                "url": url,
                "request": safe,
                "path": str(path),
                "sha256": digest(body),
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            write_json(index, entry)
        self.entries.append(entry)
        return body, entry
