import copy
import json
import sys

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from scripts import download_crypto_microstructure as downloader

TASK = ("train", "2024-11-01", "orderbook")


class Response:
    def __init__(self, status=200, headers=None, chunks=(), before_body=None):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})
        self.chunks = chunks
        self.before_body = before_body
        self.body_started = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        self.body_started = True
        if self.before_body:
            self.before_body()
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not make real HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", forbid_network)
    monkeypatch.setattr(downloader, "ROOT", tmp_path)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    config = {
        "study_id": "test_study",
        "venue": "Bybit",
        "market": "linear_perpetual",
        "symbol": "BTCUSDT",
        "urls": {
            "orderbook": "https://example.invalid/book/{date}.gz",
            "trades": "https://example.invalid/trades/{date}.gz",
        },
        "splits": {"train": ["2024-11-01"]},
        "limits": {"maximum_file_bytes": 20, "maximum_total_bytes": 40, "download_workers": 2},
    }
    return config, tmp_path / "data", monkeypatch


def mock_http(monkeypatch, heads, gets):
    calls = {"head": [], "get": []}

    def head(url, **kwargs):
        calls["head"].append((url, kwargs))
        return heads.pop(0)

    def get(url, **kwargs):
        calls["get"].append((url, kwargs))
        return gets.pop(0)

    monkeypatch.setattr(downloader.requests, "head", head)
    monkeypatch.setattr(downloader.requests, "get", get)
    return calls


def headers(version='"A"', length=6):
    result = {"Content-Length": str(length)}
    if version:
        result["ETag"] = version
    return result


def plan(config, directory):
    return downloader.plan_downloads(config, directory, [TASK], "frozen")[TASK]


def run(config, directory, reservation):
    return downloader.download_one(config, directory, *TASK, "frozen", reservation)


def paths(config, directory):
    return downloader.archive_paths(config, directory, TASK[1], TASK[2])


def partial(config, directory, reservation, body=b"old"):
    _, target, _ = paths(config, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(target.name + ".part").write_bytes(body)
    downloader.write_json(
        target.with_name(target.name + ".part.metadata.json"),
        {"identity": reservation["identity"], "source": reservation["source"]},
    )


@pytest.mark.parametrize(
    "version_headers,if_range",
    [
        ({"ETag": '"A"'}, '"A"'),
        ({"Last-Modified": "Fri, 01 Nov 2024 00:00:00 GMT"}, "Fri, 01 Nov 2024 00:00:00 GMT"),
    ],
)
def test_interrupted_download_persists_version_and_resumes_with_if_range(
    setup, version_headers, if_range
):
    config, directory, monkeypatch = setup
    full_headers = {"Content-Length": "6", **version_headers}
    _, target, _ = paths(config, directory)

    def check_before_body():
        evidence = json.loads(target.with_name(target.name + ".part.metadata.json").read_text())
        assert evidence["source"]["total"] == 6
        assert evidence["source"]["url"] == config["urls"]["orderbook"].format(date=TASK[1])
        assert evidence["source"]["validator"]["value"] == if_range

    calls = mock_http(
        monkeypatch,
        [Response(headers=full_headers), Response(headers=full_headers)],
        [
            Response(
                headers=full_headers,
                chunks=[b"abc", requests.ConnectionError("disconnected")],
                before_body=check_before_body,
            ),
            Response(
                status=206,
                headers={
                    "Content-Length": "3",
                    "Content-Range": "bytes 3-5/6",
                    **version_headers,
                },
                chunks=[b"def"],
            ),
        ],
    )
    record = run(config, directory, plan(config, directory))
    assert target.read_bytes() == b"abcdef"
    assert calls["get"][1][1]["headers"]["If-Range"] == if_range
    assert calls["get"][1][1]["headers"]["Range"] == "bytes=3-"
    assert record["attempts_this_invocation"] == 2
    assert "download_source_sha256" in record


def test_changed_version_in_partial_response_cannot_mix_prefix_and_suffix(setup):
    config, directory, monkeypatch = setup
    changed = Response(
        status=206,
        headers={**headers('"B"', 3), "Content-Range": "bytes 3-5/6"},
        chunks=[b"XYZ"],
    )
    calls = mock_http(
        monkeypatch,
        [Response(headers=headers()), Response(headers=headers('"B"'))],
        [changed, Response(headers=headers('"B"'), chunks=[b"NEWXYZ"])],
    )
    reservation = plan(config, directory)
    partial(config, directory, reservation)
    run(config, directory, reservation)
    assert not changed.body_started
    assert paths(config, directory)[1].read_bytes() == b"NEWXYZ"
    assert calls["get"][0][1]["headers"]["If-Range"] == '"A"'
    assert "Range" not in calls["get"][1][1]["headers"]


def test_new_head_version_discards_previous_invocation_partial(setup):
    config, directory, monkeypatch = setup
    calls = mock_http(
        monkeypatch,
        [Response(headers=headers('"B"'))],
        [Response(headers=headers('"B"'), chunks=[b"NEWXYZ"])],
    )
    reservation = plan(config, directory)
    old = copy.deepcopy(reservation)
    old["source"]["validator"]["value"] = '"A"'
    partial(config, directory, old)
    run(config, directory, reservation)
    assert "Range" not in calls["get"][0][1]["headers"]
    assert paths(config, directory)[1].read_bytes() == b"NEWXYZ"


@pytest.mark.parametrize("version", [None, 'W/"weak"'])
def test_no_strong_validator_restarts_instead_of_resuming(setup, version):
    config, directory, monkeypatch = setup
    current = headers(version)
    calls = mock_http(
        monkeypatch,
        [Response(headers=current), Response(headers=current)],
        [
            Response(headers=current, chunks=[b"old", requests.ConnectionError("disconnected")]),
            Response(headers=current, chunks=[b"NEWXYZ"]),
        ],
    )
    run(config, directory, plan(config, directory))
    assert paths(config, directory)[1].read_bytes() == b"NEWXYZ"
    assert all("Range" not in kwargs["headers"] for _, kwargs in calls["get"])


def test_ignored_range_200_replaces_partial_instead_of_appending(setup):
    config, directory, monkeypatch = setup
    calls = mock_http(
        monkeypatch,
        [Response(headers=headers())],
        [Response(headers=headers(), chunks=[b"abcdef"])],
    )
    reservation = plan(config, directory)
    partial(config, directory, reservation)
    run(config, directory, reservation)
    assert calls["get"][0][1]["headers"]["Range"] == "bytes=3-"
    assert paths(config, directory)[1].read_bytes() == b"abcdef"


def write_cache(config, directory):
    url, target, metadata = paths(config, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"abcdef")
    record = {
        **downloader.archive_identity(config, *TASK, "frozen", url),
        "bytes": 6,
        "sha256": downloader.file_hash(target),
        "original_evidence": "must stay unchanged",
    }
    downloader.write_json(metadata, record)
    return target, metadata, record


@pytest.mark.parametrize(
    "field",
    [
        "split",
        "day",
        "venue",
        "market",
        "symbol",
        "kind",
        "config_sha256",
        "url",
        "bytes",
        "sha256",
    ],
)
def test_completed_cache_rejects_every_identity_or_content_mismatch(setup, field):
    config, directory, _ = setup
    _, metadata, record = write_cache(config, directory)
    record[field] = 5 if field == "bytes" else "different"
    downloader.write_json(metadata, record)
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="Existing archive"):
        plan(config, directory)
    assert metadata.read_bytes() == before


def test_split_and_config_drift_are_rejected_without_relabelling_cache(setup):
    config, directory, _ = setup
    _, metadata, _ = write_cache(config, directory)
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="split"):
        downloader.plan_downloads(config, directory, [("event_test", TASK[1], TASK[2])], "frozen")
    with pytest.raises(ValueError, match="config_sha256"):
        downloader.plan_downloads(config, directory, [TASK], "new-config-hash")
    assert metadata.read_bytes() == before


def test_valid_cache_is_verified_offline_and_metadata_stays_immutable(setup):
    config, directory, _ = setup
    _, metadata, original = write_cache(config, directory)
    before = metadata.read_bytes()
    assert run(config, directory, plan(config, directory)) == original
    assert metadata.read_bytes() == before


def test_aggregate_budget_fails_in_main_before_any_get_or_worker_is_started(setup):
    config, _, monkeypatch = setup
    config["limits"]["maximum_total_bytes"] = 10
    # Even a train-only run reserves the complete study, including the later split.
    config["splits"]["event_test"] = ["2024-11-05"]
    config_path = downloader.ROOT / "config.json"
    downloader.write_json(config_path, config)
    monkeypatch.setattr(sys, "argv", ["download", "--config", str(config_path), "--split", "train"])
    monkeypatch.setattr(downloader.subprocess, "check_output", lambda *a, **k: "commit\n")
    calls = mock_http(monkeypatch, [Response(headers=headers(length=3)) for _ in range(4)], [])

    def no_workers(*args, **kwargs):
        raise AssertionError("No worker may start before aggregate budget validation")

    monkeypatch.setattr(downloader, "ThreadPoolExecutor", no_workers)
    with pytest.raises(ValueError, match="total byte budget before download"):
        downloader.main()
    assert len(calls["head"]) == 4
    assert calls["get"] == []
    state = json.loads((downloader.ROOT / "outputs/test_study/download_train.json").read_text())
    assert state["status"] == "failed"


@pytest.mark.parametrize(
    "head",
    [
        Response(status=405),
        Response(headers={"ETag": '"A"'}),
        Response(headers={"Content-Length": "unknown"}),
        Response(headers={"Content-Length": "6", "Content-Encoding": "gzip"}),
    ],
)
def test_unreliable_head_is_rejected_before_body_request(setup, head):
    config, directory, monkeypatch = setup
    calls = mock_http(monkeypatch, [head], [])
    with pytest.raises(ValueError):
        plan(config, directory)
    assert not calls["get"]


def test_file_growth_after_reservation_is_rejected_before_body_consumption(setup):
    config, directory, monkeypatch = setup
    growing = Response(headers=headers(length=7), chunks=[b"too big"])
    calls = mock_http(
        monkeypatch,
        [Response(headers=headers())] + [Response(headers=headers(length=7)) for _ in range(3)],
        [growing],
    )
    with pytest.raises(RuntimeError, match="reserved file-size budget"):
        run(config, directory, plan(config, directory))
    assert len(calls["get"]) == 1
    assert not growing.body_started
    assert not paths(config, directory)[1].exists()


def test_body_exceeding_reserved_bytes_is_never_published(setup):
    config, directory, monkeypatch = setup
    mock_http(
        monkeypatch,
        [Response(headers=headers()) for _ in range(4)],
        [Response(headers=headers(), chunks=[b"abc", b"def-too-many"]) for _ in range(4)],
    )
    with pytest.raises(RuntimeError, match="exceeds declared archive length"):
        run(config, directory, plan(config, directory))
    _, target, _ = paths(config, directory)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()
