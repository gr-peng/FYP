import copy
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments import crypto_effective_flow_reader as reader
from experiments.crypto_effective_flow import GRID_US, ObservedPrint
from experiments.crypto_execution_diagnostic import day_start_us

ROOT = Path(__file__).resolve().parents[1]
DAY = "2024-11-01"
START = day_start_us(DAY)


def message(i=0, *, kind=None, **changes):
    timestamp_ms = START // 1000 + 10 + i * 100
    q = 1000 + i
    result = {
        "topic": "orderbook.500.BTCUSDT",
        "type": kind or ("snapshot" if i == 0 else "delta"),
        "cts": timestamp_ms, "ts": timestamp_ms + 10,
        "data": {"s": "BTCUSDT", "u": i + 1, "seq": i + 10,
                 "b": [["99.00", f"{q // 1000}.{q % 1000:03d}"], ["98.00", "2.000"]],
                 "a": [["100.00", "1.000"], ["101.00", "2.000"]]},
    }
    result.update(changes)
    return result


def prints():
    return tuple(ObservedPrint(
        START + (i + 1) * 1_000_000 - 1000, "Buy" if i % 2 == 0 else "Sell",
        i + 1, 10000 if i % 2 == 0 else 9900, i + 1, f"print-{i}",
    ) for i in range(9))


def candidates(messages=None, tape=None, start=6_000_000, end=6_200_000):
    return list(reader.flow_candidates(
        (message(i) for i in range(100)) if messages is None else messages,
        prints() if tape is None else tape, DAY,
        emit_start_us=START + start, emit_end_us=START + end,
    ))


def test_protocol_exact_machine_snapshot_and_no_changed_smoke_window():
    plan = json.loads((ROOT / reader.CONFIG_PATH).read_bytes())
    assert plan == reader.expected_protocol()
    assert plan["smoke"] == {"day": DAY, "start_second_utc": 600, "end_second_utc": 900}


def test_pending_raw_message_never_changes_observed_prefix_or_levels():
    future = message(1, ts=START // 1000 + 6000)
    later = message(2)
    cursor = reader.FlowBookCursor(iter((message(), future, later)), DAY, 0)
    point = cursor.observation(START + 1_000_000)
    assert point.message_index == 1 and point.bids[0] == (9900, 1000)
    assert cursor.pending[0] == 2
    # A lower-ts following message cannot bypass the pending raw-order barrier.
    still = cursor.observation(START + 5_000_000)
    assert still.message_index == 1 and still.bids == point.bids


def test_actual_signal_prefixes_have_exact_zero_one_five_second_lags():
    row = candidates()[0]
    assert row.eligible
    assert [o.observed_at_us for o in row.signal_observations] == [
        START + 6_000_000, START + 5_000_000, START + 1_000_000,
    ]
    assert [o.message_index for o in row.signal_observations] == [60, 50, 10]
    assert [o.bids[0][1] for o in row.signal_observations] == [1059, 1049, 1009]
    assert [f.trade_pressure for f in row.features_by_delay] == [-1, 1, 1]
    assert all(o.available_us < o.observed_at_us for o in row.signal_observations)
    # Five-second-old input is evaluated at its own signal clock, not today's A.
    assert row.signal_observations[-1].engine_us < row.evidence.start.engine_us - 4_000_000


def test_current_interval_future_prints_do_not_change_any_feature():
    first_tape = (*prints()[:6], ObservedPrint(START + 6_020_000, "Buy", 7, 10000, 7, "new"))
    second_tape = (*prints()[:6], ObservedPrint(START + 6_020_000, "Sell", 999, 9900, 7, "new"))
    first, second = candidates(tape=first_tape)[0], candidates(tape=second_tape)[0]
    assert first.features_by_delay == second.features_by_delay
    assert first.signal_observations == second.signal_observations
    assert first.evidence.template.intents != second.evidence.template.intents


def test_each_print_half_open_allocation_exactly_once_in_tape_order():
    tape = tuple(ObservedPrint(START + 6_000_000 + i * 100_000, "Buy", 1, 10000,
                               i + 1, f"tie-{i}") for i in range(10))
    rows = candidates(tape=tape, end=7_000_000)
    assert [t.trade_id for row in rows for t in row.evidence.prints] == [t.trade_id for t in tape]
    assert all(len(row.evidence.prints) == 1 for row in rows)


def test_candidate_context_exclusions_retained_before_six_seconds():
    rows = candidates(tape=(), start=0, end=6_100_000)
    assert len(rows) == 61
    assert not any(row.eligible for row in rows[:60])
    assert rows[60].eligible
    assert "delay_5000_incomplete_past_context" in rows[59].exclusion_reasons


def test_duplicate_raw_snapshot_is_an_explicit_reset_not_an_effective_add():
    original = message()
    cursor = reader.FlowBookCursor(iter((original, copy.deepcopy(original))), DAY, 0)
    point = cursor.observation(START + 100_000)
    assert point.segment_id == 1 and point.message_index == 2
    assert point.bids[0][1] == 1000


def test_reset_inside_lag_context_excludes_common_cohort():
    messages = [message(i) for i in range(100)]
    messages[40]["type"] = "snapshot"
    row = candidates(messages=messages)[0]
    assert row.evidence.eligible and not row.eligible
    assert "delay_5000_segment_or_error_crossing" in row.exclusion_reasons
    assert row.features_by_delay[-1] is None


@pytest.mark.parametrize("day", ["2024-11-04", "2024-11-05", "2024-10-14"])
def test_unapproved_day_rejected_before_advancing_raw_iterators(day):
    def never_read():
        raise AssertionError("Raw reader must remain unopened")
        yield None
    with pytest.raises(ValueError, match="approved"):
        list(reader.flow_candidates(never_read(), never_read(), day,
                                    emit_start_us=day_start_us(day),
                                    emit_end_us=day_start_us(day) + GRID_US))


def test_serialized_evidence_keeps_lag_clocks_and_separates_narrow_template():
    row = candidates()[0]
    record = reader.candidate_record(row)
    assert len(record["signal_prefixes"]) == len(record["features_by_delay"]) == 3
    assert record["start_prefix"]["bid_l1"] == [9900, 1059]
    assert set(record["template"]) == {"row_id", "adds", "removes", "intents"}
    assert all(level["residual_units"] or level["consumed_proxy_units"]
               for level in record["effective_levels"])
    assert "bids" not in record["start_prefix"]
    assert json.loads(json.dumps(record, allow_nan=False))["grid_us"] == record["grid_us"]


def temporary_root(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    for name in (reader.CONFIG_PATH, reader.SOURCE_PATH):
        (root / name).write_bytes((ROOT / name).read_bytes())
    return root


def fake_identity(root, paths):
    names = {name: "a" * 64 for name in reader.READER_SOURCES}
    names.update({reader.CONFIG_PATH: reader.sha256(root / reader.CONFIG_PATH),
                  reader.SOURCE_PATH: reader.SOURCE_SHA})
    return {"git_commit": "synthetic", "source_sha256": names}


def test_prepare_only_validates_inputs_without_market_parse(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    calls = []
    monkeypatch.setattr(reader, "frozen_identity", fake_identity)
    monkeypatch.setattr(reader, "audited_inputs", lambda *args: calls.append(args[3]) or {})
    monkeypatch.setattr(reader, "_write_day", lambda *args: pytest.fail("No parsing in prepare"))
    output = tmp_path / "prepared"
    state = reader.run_extraction(root, output, prepare_only=True)
    assert calls == [DAY, DAY]
    assert state["status"] == "prepared_no_market_rows_read"
    assert not state["outputs_published"] and not state["outputs"]
    assert (output / "prepared_receipt.json").exists()


def test_failed_freeze_leaves_failed_receipt_and_never_opens_market(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    def reject(*args):
        raise ValueError("Not committed")
    monkeypatch.setattr(reader, "frozen_identity", reject)
    monkeypatch.setattr(reader, "audited_inputs", lambda *args: pytest.fail("Not verified"))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="committed"):
        reader.run_extraction(root, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"]
    assert not (output / "prepared_receipt.json").exists()


def test_split_and_existing_output_fail_closed_without_overwrite(tmp_path):
    output = tmp_path / "closed"
    with pytest.raises(ValueError, match="closed"):
        reader.run_extraction(tmp_path, output, split="development")
    assert not output.exists()
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("old identity")
    with pytest.raises(FileExistsError):
        reader.run_extraction(tmp_path, output)
    assert sentinel.read_text() == "old identity"


def test_midstream_failure_never_publishes_complete_batch(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    monkeypatch.setattr(reader, "frozen_identity", fake_identity)
    monkeypatch.setattr(reader, "audited_inputs", lambda *args: {})
    def fail(*args):
        raise ValueError("Synthetic stream failure")
    monkeypatch.setattr(reader, "_write_day", fail)
    output = tmp_path / "stream-failed"
    with pytest.raises(ValueError, match="stream failure"):
        reader.run_extraction(root, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"]
    assert state["outputs"] == []


def test_source_config_change_fails_before_input_open(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    source = json.loads((root / reader.SOURCE_PATH).read_bytes())
    source["urls"]["trades"] = "https://invalid.example/{date}"
    (root / reader.SOURCE_PATH).write_text(json.dumps(source))
    monkeypatch.setattr(reader, "audited_inputs", lambda *args: pytest.fail("Source is altered"))
    with pytest.raises(ValueError, match="frozen train source"):
        reader.run_extraction(root, tmp_path / "wrong-source")


@pytest.mark.parametrize("prepare_only", [False, True])
def test_config_race_during_audits_fails_before_prepared_or_market(tmp_path, monkeypatch,
                                                                prepare_only):
    root = temporary_root(tmp_path)
    monkeypatch.setattr(reader, "frozen_identity", fake_identity)
    def change_config(*args):
        p = root / reader.CONFIG_PATH
        p.write_bytes(p.read_bytes() + b"\n")
        return {}
    monkeypatch.setattr(reader, "audited_inputs", change_config)
    monkeypatch.setattr(reader, "_write_day", lambda *args: pytest.fail("Preflight changed"))
    output = tmp_path / "config-race"
    with pytest.raises(ValueError, match="configuration changed"):
        reader.run_extraction(root, output, prepare_only=prepare_only)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"]
    assert not (output / "prepared_receipt.json").exists()


def test_input_receipt_revocation_during_preflight_never_reaches_market(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    monkeypatch.setattr(reader, "frozen_identity", fake_identity)
    calls = []
    monkeypatch.setattr(reader, "audited_inputs", lambda *args:
                        calls.append(1) or {"identity": len(calls)})
    monkeypatch.setattr(reader, "_write_day", lambda *args: pytest.fail("Input was revoked"))
    output = tmp_path / "input-race"
    with pytest.raises(ValueError, match="receipt changed"):
        reader.run_extraction(root, output)
    assert not (output / "prepared_receipt.json").exists()


def test_changed_finished_day_bytes_cannot_publish_complete_master(tmp_path, monkeypatch):
    root = temporary_root(tmp_path)
    monkeypatch.setattr(reader, "frozen_identity", fake_identity)
    monkeypatch.setattr(reader, "audited_inputs", lambda *args: {})
    def changed_output(inputs, output, day, window):
        path = output / "synthetic.gz"
        path.write_bytes(b"finished output")
        digest = reader.sha256(path)
        path.write_bytes(b"changed after hash")
        return {"path": str(path), "sha256": digest, "rows": 3000}
    monkeypatch.setattr(reader, "_write_day", changed_output)
    output = tmp_path / "changed-output"
    with pytest.raises(ValueError, match="output bytes changed"):
        reader.run_extraction(root, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"]


def test_record_file_roundtrip_is_bounded_sparse_evidence(tmp_path):
    destination = tmp_path / "synthetic.jsonl.gz"
    records = [reader.candidate_record(row) for row in candidates()]
    with gzip.open(destination, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with gzip.open(destination, "rt") as handle:
        restored = [json.loads(line) for line in handle]
    assert len(restored) == 2
    assert restored[0]["features_by_delay"] == records[0]["features_by_delay"]
    assert asdict(candidates()[0].evidence.start)["bids"][0] == (9900, 1059)
