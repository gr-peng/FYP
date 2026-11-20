"""Synthetic packets only. Never open the approved real training cache in tests."""

import copy
import json
import stat
from pathlib import Path

import numpy as np
import pytest
from test_conditional_order_memory import no_external_access as no_external_access
from test_conditional_order_memory import records

from scripts import validate_memory_information as tool


def synthetic(n=30):
    rows = records("fictional", n)
    for i, row in enumerate(rows):
        row["provenance"].update(
            session=f"fictional-s{i % 5}",
            participant=f"fictional-p{i % 10}",
            repeat=2,
            period=2,
            second=5,
        )
        row["state"]["repetition"] = 2
        last = row["state"][tool.model.COMPLETED]["last_completed_period_with_own_orders"]
        if last is not None:
            last.update(repetition=2, period=1)
    return rows


def cache(tmp_path, rows):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, tool.sha(path.read_bytes())


def test_cache_hash_checked_before_json_and_decoding_uses_same_bytes(tmp_path, monkeypatch):
    rows = synthetic()
    path, digest = cache(tmp_path, rows)
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.load_cache(path, "wrong", [], 30)
    original = Path.read_bytes

    def read_and_change(p):
        data = original(p)
        if p == path:
            p.write_text("not JSON after original bytes were read")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_change)
    actual = tool.load_cache(path, digest, {r["provenance"]["session"] for r in rows}, len(rows))
    assert actual == rows


@pytest.mark.parametrize(
    "bad", ["duplicate_target", "duplicate_nested", "NaN", "Infinity", "1e999"]
)
def test_verified_cache_still_rejects_ambiguous_or_nonfinite_json(tmp_path, bad):
    rows = synthetic()
    line = json.dumps(rows[0])
    if bad == "duplicate_target":
        line = '{"target": {"buy": 1},' + line[1:]
    elif bad == "duplicate_nested":
        line = line.replace('"buy": 0', '"buy": 1, "buy": 0', 1)
    else:
        line = '{"unused_field": ' + bad + "," + line[1:]
    path = tmp_path / "ambiguous.jsonl"
    path.write_text(line + "\n" + "".join(json.dumps(r) + "\n" for r in rows[1:]))
    with pytest.raises(ValueError, match="duplicate JSON|nonfinite JSON"):
        tool.load_cache(
            path, tool.sha(path.read_bytes()), {r["provenance"]["session"] for r in rows}, len(rows)
        )


@pytest.mark.parametrize(
    "bad",
    [
        "wrong_partition",
        "duplicate",
        "future_public",
        "future_trade",
        "future_own",
        "label",
        "clock",
    ],
)
def test_invalid_cache_has_no_admission_or_fitting(tmp_path, bad):
    rows = synthetic()
    allowed = {r["provenance"]["session"] for r in rows}
    if bad == "wrong_partition":
        rows[0]["provenance"]["session"] = "unapproved"
    elif bad == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif bad == "future_public":
        rows[0]["state"]["earlier_periods"][0]["period"] = 2
    elif bad == "future_trade":
        rows[0]["state"]["earlier_current_period_trades"]["last_completed_trade_second"] = 5
    elif bad == "future_own":
        rows[1]["state"][tool.model.COMPLETED]["last_completed_period_with_own_orders"][
            "period"
        ] = 2
    elif bad == "label":
        rows[0]["target"]["buy"] = 2
    elif bad == "clock":
        rows[0]["provenance"]["second"] = 4
    path, digest = cache(tmp_path, rows)
    with pytest.raises(ValueError):
        tool.load_cache(path, digest, allowed, len(rows))


def test_recorded_person_links_are_transitive_across_sessions():
    rows = [
        {"provenance": {"session": s, "participant": p}}
        for s, p in [
            ("a", "x"),
            ("b", "x"),
            ("b", "y"),
            ("c", "y"),
            ("d", "d"),
            ("e", "e"),
            ("f", "f"),
            ("g", "g"),
        ]
    ]
    result = tool.grouped_folds(rows)
    assert ["a", "b", "c"] in result["components"]
    assert len({result["session_to_fold"][s] for s in ("a", "b", "c")}) == 1
    assert result["cross_session_recorded_participant_codes"] == 2
    with pytest.raises(ValueError, match="fewer than five"):
        tool.grouped_folds(rows[:-1])


def test_singletons_reproduce_the_predeclared_hash_mapping():
    rows = synthetic()
    mapping = tool.grouped_folds(rows)["session_to_fold"]
    ordered = sorted(mapping, key=lambda s: tool.sha((tool.SALT + s).encode()))
    assert mapping == {s: i % 5 for i, s in enumerate(ordered)}


def test_bootstrap_keeps_repeated_components_and_session_weighting(monkeypatch):
    class FixedDraws:
        def integers(self, _high, size):
            assert size == (1, 2)
            return np.array([[0, 0]])

    monkeypatch.setattr(tool.np.random, "default_rng", lambda _seed: FixedDraws())
    result = tool.interval({"a": 0.2, "b": -0.2, "c": 0.4}, [["a"], ["b", "c"]], draws=1)
    assert result["difference"] == pytest.approx(0.4 / 3)
    assert result["percentile_95"] == pytest.approx([0.2, 0.2])


def test_bootstrap_does_not_deduplicate_a_draw(monkeypatch):
    class FixedDraws:
        def integers(self, _high, size):
            assert size == (1, 3)
            return np.array([[0, 0, 1]])

    monkeypatch.setattr(tool.np.random, "default_rng", lambda _seed: FixedDraws())
    # Draw includes a twice and b/c/d once: (.2+.2+.2)/5=.12.
    result = tool.interval(
        {"a": 0.2, "b": 0.2, "c": 0, "d": 0, "e": -0.1}, [["a"], ["b", "c", "d"], ["e"]], draws=1
    )
    assert result["percentile_95"] == pytest.approx([0.12, 0.12])


def test_prepare_never_opens_human_cache(tmp_path, monkeypatch):
    forbidden = tool.ROOT / tool.CACHE
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def check_text(path, *args, **kwargs):
        assert path != forbidden
        return original_text(path, *args, **kwargs)

    def check_bytes(path, *args, **kwargs):
        assert path != forbidden
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", check_text)
    monkeypatch.setattr(Path, "read_bytes", check_bytes)
    design = tool.ROOT.parent / "iterations/memory_training_validation_20261107/design_v1.json"
    plan = tool.prepare(tmp_path / "packet", design)
    assert plan["preparation_opened_human_states"] is False
    assert tool.checked_packet(tmp_path / "packet") == plan


def execution_fixture(tmp_path, monkeypatch):
    rows = synthetic()
    path, digest = cache(tmp_path, rows)
    packet = tmp_path / "packet"
    packet.mkdir()
    tool.save(packet / "plan.json", {"evidence_kind": "synthetic_fixture_not_real_approval"})
    tool.save(
        packet / "independent_preflight.json",
        {
            "accepted_for_execution": True,
            "mustfix": [],
            "plan_sha256": tool.sha((packet / "plan.json").read_bytes()),
        },
    )
    folds = tool.grouped_folds(rows)
    tool.save(
        packet / "partitions.json",
        {"assignments": {"train": list(folds["session_to_fold"]), "development": [], "test": []}},
    )
    tool.save(packet / "engineering_design.json", {"candidate_folds": folds["session_to_fold"]})
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "CACHE", path.name)
    monkeypatch.setattr(tool, "CACHE_SHA", digest)
    monkeypatch.setattr(tool, "checked_packet", lambda _p: {})
    original = tool.load_cache
    monkeypatch.setattr(tool, "load_cache", lambda p, h, t, _n: original(p, h, t, len(rows)))
    return packet, rows


def test_complete_synthetic_cross_validation_has_one_common_row_and_balanced_scores(
    tmp_path, monkeypatch
):
    packet, rows = execution_fixture(tmp_path, monkeypatch)
    out = tmp_path / "result"
    result = tool.execute(packet, out)
    assert result["common_records"] == 30 and result["methods"] == 7
    assert len(list((out / "models").glob("*.json"))) == 15
    saved = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len({(r["session"], r["order_id"]) for r in saved}) == len(rows)
    for name, value in result["overall"]["methods"].items():
        average = np.mean(
            [
                r["methods"][name]["brier_session_person_balanced"]
                for r in result["per_session"].values()
            ]
        )
        assert value["brier_session_person_balanced"] == pytest.approx(average)
    assert tool.read(out / "status.json")["status"] == "complete_pending_independent_audit"


@pytest.mark.parametrize("where", ["approval", "folds", "admission", "fit"])
def test_any_failure_stops_without_success_report(tmp_path, monkeypatch, where):
    packet, _ = execution_fixture(tmp_path, monkeypatch)
    fits = []

    def fail_fit(*_a, **_k):
        fits.append(True)
        raise ValueError("synthetic fit stopped")

    monkeypatch.setattr(tool.model, "split_predictions", fail_fit)
    if where == "approval":
        tool.save(packet / "independent_preflight.json", {"accepted_for_execution": False})
        monkeypatch.setattr(tool, "load_cache", lambda *_a: pytest.fail("unaccepted data access"))
    elif where in ("folds", "admission"):
        original = tool.save

        def fail_write(path, value):
            if path.name == where + ".json":
                raise OSError("synthetic write failure")
            return original(path, value)

        monkeypatch.setattr(tool, "save", fail_write)
    with pytest.raises((ValueError, OSError)):
        tool.execute(packet, tmp_path / "result")
    assert bool(fits) == (where == "fit")
    assert not (tmp_path / "result/summary.json").exists()
    assert not (tmp_path / "result/REPORT.txt").exists()
    if where != "approval":
        assert tool.read(tmp_path / "result/status.json")["status"] == "incomplete"


@pytest.mark.parametrize("directory_failure", [False, True])
def test_failed_fold_durability_blocks_fitting(tmp_path, monkeypatch, directory_failure):
    packet, _ = execution_fixture(tmp_path, monkeypatch)
    original_save, original_fsync = tool.save, tool.os.fsync
    attempted_failure = []

    def durable_save(path, value):
        if path.name != "folds.json":
            return original_save(path, value)

        def failing_sync(fd):
            is_directory = stat.S_ISDIR(tool.os.fstat(fd).st_mode)
            if is_directory == directory_failure:
                attempted_failure.append(True)
                raise OSError("synthetic durability failure")
            return original_fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(tool.os, "fsync", failing_sync)
            return original_save(path, value)

    monkeypatch.setattr(tool, "save", durable_save)
    monkeypatch.setattr(tool.model, "split_predictions", lambda *_a: pytest.fail("fit too early"))
    out = tmp_path / "result"
    with pytest.raises(OSError, match="durability failure"):
        tool.execute(packet, out)
    assert attempted_failure
    assert tool.read(out / "status.json")["status"] == "incomplete"
    assert not (out / "summary.json").exists()
    assert not (out / "REPORT.txt").exists()


def test_file_sync_failure_preserves_previous_file_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    tool.save(path, {"old": True})

    def fail(_fd):
        raise OSError("synthetic durability failure")

    monkeypatch.setattr(tool.os, "fsync", fail)
    with pytest.raises(OSError, match="durability failure"):
        tool.save(path, {"new": True})
    assert tool.read(path) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]
