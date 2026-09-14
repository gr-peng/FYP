from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query)
        if not k.lower().startswith("utm_") and k.lower() not in {"ref", "fbclid"}
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(sorted(query)),
            "",
        )
    )


def deduplicate(items):
    kept = []
    urls = set()
    for item in sorted(items, key=lambda x: x.published_at_utc):
        url = normalize_url(item.url)
        if url in urls or any(
            SequenceMatcher(None, item.title.lower(), k.title.lower()).ratio() >= 0.92 for k in kept
        ):
            continue
        urls.add(url)
        kept.append(item)
    return kept
