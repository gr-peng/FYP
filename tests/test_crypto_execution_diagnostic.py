import copy
import gzip
import io
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from experiments import crypto_execution_diagnostic as execution
from experiments.crypto_execution_diagnostic import (
    DELAYS,
    PublishedBookCursor,
    Trade,
    day_start_us,
    diagnostic_rows,
    run_diagnostic,
    sha256,
    write_day,
)

DAY = "2024-11-01"
START = day_start_us(DAY)
MS = START // 1000
PROJECT = Path(__file__).resolve().parents[1]


def book(offset=0, *, ts=None, kind="snapshot", u=1, seq=100, bids=None, asks=None):
    return {
        "topic": "orderbook.500.BTCUSDT",
        "type": kind,
        "cts": MS + offset,
        "ts": MS + (offset if ts is None else ts),
        "data": {
            "s": "BTCUSDT",
            "u": u,
            "seq": seq,
            "b": bids if bids is not None else [["99.00", "1.000"], ["98.00", "1.000"]],
            "a": asks if asks is not None else [["100.00", "1.000"], ["101.00", "1.000"]],
        },
    }


def trade(offset_us=400_000, *, q=2000, price=10050, side="Buy", identifier="first", row=1):
    return Trade(START + offset_us, side, q, price, identifier, row)


def rows(messages, trades):
    return list(diagnostic_rows(trades, {d: iter(copy.deepcopy(messages)) for d in DELAYS}, DAY))


def test_delayed_snapshot_and_strict_millisecond_boundary():
    cursor = PublishedBookCursor([book(ts=104)], DAY, 0)
    assert cursor.predict(START + 50_000, "Buy", 1000)["reason"] == "no_state"
    assert cursor.predict(START + 105_000, "Buy", 1000)["reason"] == "no_state"
    prediction = cursor.predict(START + 105_001, "Buy", 1000)
    assert prediction["reason"] == "eligible"
    assert prediction["available_us"] == START + 105_000


def test_engine_timestamp_also_blocks_early_publication_timestamp():
    cursor = PublishedBookCursor([book(100, ts=0)], DAY, 0)
    assert cursor.predict(START + 100_000, "Buy", 1000)["reason"] == "no_state"
    assert cursor.predict(START + 101_000, "Buy", 1000)["reason"] == "no_state"
    assert cursor.predict(START + 101_001, "Buy", 1000)["reason"] == "eligible"


def test_pending_future_ts_does_not_contaminate_old_state_and_prefix_barrier_persists():
    messages = [
        book(),
        book(100, ts=1000, kind="delta", u=2, seq=105, bids=[], asks=[["100.00", "2.000"]]),
        book(200, ts=300, kind="delta", u=3, seq=110, bids=[], asks=[["100.00", "3.000"]]),
    ]
    cursor = PublishedBookCursor(messages, DAY, 0)
    early = cursor.predict(START + 500_000, "Buy", 2000)
    assert early["update_id"] == 1
    assert early["publication_prefix_ms"] == MS
    assert early["available_us"] == START + 1000
    assert early["l1_quantity_units"] == 1000
    late = cursor.predict(START + 1_001_001, "Buy", 2000)
    assert late["update_id"] == 3
    assert late["publication_prefix_ms"] == MS + 1000
    assert late["available_us"] == START + 1_001_000


def test_snapshot_reset_does_not_jump_pending_raw_order_barrier():
    messages = [
        book(),
        book(100, ts=1000, kind="delta", u=2, seq=110, bids=[], asks=[]),
        book(200, ts=200, u=1, seq=120, asks=[["102.00", "2.000"]]),
    ]
    cursor = PublishedBookCursor(messages, DAY, 0)
    assert cursor.predict(START + 500_000, "Buy", 1000)["l1_price_units"] == 10000
    after = cursor.predict(START + 1_001_001, "Buy", 1000)
    assert after["l1_price_units"] == 10200
    assert after["publication_prefix_ms"] == MS + 200
    assert after["available_us"] == START + 1_001_000


def test_gap_invalidates_until_a_new_snapshot():
    messages = [
        book(),
        book(100, kind="delta", u=3, seq=110, bids=[], asks=[]),
        book(200, kind="delta", u=4, seq=120, bids=[], asks=[]),
        book(500, seq=130),
    ]
    cursor = PublishedBookCursor(messages, DAY, 0)
    assert cursor.predict(START + 300_000, "Buy", 1000)["reason"] == "invalid_state"
    assert cursor.invalid_messages == 2
    assert cursor.predict(START + 501_001, "Buy", 1000)["reason"] == "eligible"


def test_exact_integer_cost_partial_coverage_and_readonly_sweeps():
    cursor = PublishedBookCursor([book()], DAY, 0)
    partial = cursor.predict(START + 400_000, "Buy", 3000)
    assert partial["reason"] == "partial_depth"
    assert partial["covered_quantity_units"] == 2000
    assert partial["unresolved_quantity_units"] == 1000
    assert partial["sweep_cost_product_units"] == "20100000"
    assert partial["sweep_price_denominator"] == 2000
    small = cursor.predict(START + 400_000, "Buy", 1)
    assert small["sweep_cost_product_units"] == "10000"
    assert small["l1_quantity_units"] == 1000
    sell = cursor.predict(START + 400_000, "Sell", 2000)
    assert sell["sweep_cost_product_units"] == "19700000"
    assert sell["covered_quantity_units"] == 2000
    repeated = cursor.predict(START + 400_000, "Buy", 2000)
    assert repeated["sweep_cost_product_units"] == "20100000"


def test_same_timestamp_prints_are_not_grouped_or_consumed_as_parent_order():
    result = rows(
        [book()],
        [
            trade(q=1000, price=10000),
            trade(q=1000, price=10100, identifier="second", row=2),
        ],
    )
    assert len(result) == 10
    assert [r["trade_id"] for r in result[::5]] == ["first", "second"]
    assert all(r["sweep_cost_product_units"] == "10000000" for r in result)
    assert result[0]["sweep_abs_bps"] == 0
    assert result[5]["sweep_abs_bps"] == 100
    assert not any(r["cross_l1"] for r in result)


def test_partial_sweeps_never_get_full_print_price_errors():
    result = rows([book()], [trade(q=3000)])
    assert all(r["reason"] == "partial_depth" for r in result)
    assert all("sweep_abs_bps" not in r for r in result)
    assert not any(r["shared_eligible"] for r in result)


def test_all_delays_and_shared_intersection_aggregation(tmp_path):
    result = rows([book()], [trade(100_000), trade(identifier="late", row=2)])
    assert [r["delay_ms"] for r in result[:5]] == list(DELAYS)
    assert not any(r["shared_eligible"] for r in result[:5])
    assert all(r["shared_cross_l1"] for r in result[5:])
    outputs = write_day(result, tmp_path, DAY, START, START + 600_000_000, row_group_size=3)
    stored = pq.read_table(tmp_path / f"{DAY}_prints.parquet").to_pylist()
    assert len(stored) == 10
    assert stored[5]["sweep_cost_product_units"] == "20100000"
    assert stored[5]["sweep_price_denominator"] == 2000
    assert stored[5]["sweep_abs_bps"] == 0
    report = json.loads((tmp_path / f"{DAY}_summary.json").read_text())
    zero, delayed = report["delays"]["0"], report["delays"]["250"]
    assert zero["candidate_count"] == delayed["candidate_count"] == 2
    assert zero["groups"]["eligible"]["n"] == 2
    assert delayed["groups"]["eligible"]["n"] == 1
    assert zero["groups"]["shared_cross_l1"]["n"] == 1
    assert zero["groups"]["cross_l1"]["sweep_minus_l1_mae_bps"] == -50
    assert report["quantiles_computed"] is False
    block_rows = pq.read_table(tmp_path / f"{DAY}_blocks.parquet").to_pylist()
    assert len(block_rows) == 10
    empty = [r for r in block_rows if r["bucket_ms"] == (START + 300_000_000) // 1000]
    assert all(r["candidate_count"] == 0 for r in empty)
    assert all(
        json.loads(r["report_json"])["groups"]["eligible"]["sweep_abs_bps_mean"] is None
        for r in empty
    )
    assert all(sha256(r["path"]) == r["sha256"] for r in outputs)


def test_future_prices_do_not_affect_current_prediction():
    original = [book(), book(500, kind="delta", u=2, seq=101, bids=[], asks=[["100.00", "5"]])]
    changed = copy.deepcopy(original)
    changed[1]["data"]["a"] = [["100.00", "500"]]
    assert rows(original, [trade()]) == rows(changed, [trade()])


def test_empty_window_has_zero_coverage_and_no_fabricated_price_errors(tmp_path):
    write_day([], tmp_path, DAY, START, START + 300_000_000)
    assert pq.read_table(tmp_path / f"{DAY}_prints.parquet").num_rows == 0
    summary = json.loads((tmp_path / f"{DAY}_summary.json").read_text())
    assert summary["print_delay_rows"] == 0
    for report in summary["delays"].values():
        assert report["candidate_count"] == 0
        assert report["groups"]["eligible"]["sweep_abs_bps_mean"] is None


def test_partial_block_window_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="whole UTC five-minute blocks"):
        write_day([], tmp_path, DAY, START + 1, START + 300_000_000)


def test_previous_day_snapshot_cannot_seed_today_even_after_today_delta():
    cursor = PublishedBookCursor(
        [
            book(-100),
            book(100, kind="delta", u=2, seq=105, bids=[], asks=[]),
        ],
        DAY,
        0,
    )
    assert cursor.predict(START + 400_000, "Buy", 1000)["reason"] == "outside_day_state"
    with pytest.raises(ValueError, match="selected UTC day"):
        cursor.predict(START + 86_400_000_000, "Buy", 1000)


def test_stale_book_and_reverse_time_are_not_repaired():
    cursor = PublishedBookCursor([book()], DAY, 0)
    assert cursor.predict(START + 1_000_000, "Buy", 1000)["reason"] == "stale"
    with pytest.raises(ValueError, match="nondecreasing"):
        cursor.predict(START + 500_000, "Buy", 1000)
    with pytest.raises(ValueError, match="reversed"):
        rows([book()], [trade(), trade(100_000, identifier="reverse")])


def test_trade_parser_preserves_microseconds_and_source_order():
    body = (
        "timestamp,symbol,side,size,price,trdMatchID\n"
        "1730419200.123456,BTCUSDT,Buy,0.001,100.01,a\n"
        "1730419200.123456,BTCUSDT,Buy,0.002,100.02,b\n"
    )
    parsed = list(execution._trade_rows(io.StringIO(body), DAY, START, START + 1_000_000))
    assert [r.timestamp_us for r in parsed] == [START + 123456, START + 123456]
    assert [r.trade_id for r in parsed] == ["a", "b"]
    assert parsed[0].quantity_units == 1 and parsed[0].price_units == 10001


def source_fixture(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "crypto_execution_diagnostic_20261011.json"
    config_path.write_bytes((PROJECT / "config" / config_path.name).read_bytes())
    source_path = config_dir / "crypto_microstructure_20261011.json"
    source_path.write_bytes((PROJECT / "config" / source_path.name).read_bytes())
    source = json.loads(source_path.read_text())
    source_hash = sha256(source_path)
    for name in execution.EXECUTION_SOURCES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic source identity " + name)

    def identity(root, paths):
        return {
            "git_commit": "synthetic-frozen-commit",
            "source_sha256": {str(p.relative_to(root)): sha256(p) for p in paths},
        }

    monkeypatch.setattr(execution, "frozen_identity", identity)
    current_sources = {name: sha256(tmp_path / name) for name in execution.AUDIT_SOURCES}
    for kind in ["orderbook", "trades"]:
        url = source["urls"][kind].format(date=DAY)
        raw = tmp_path / "data/raw" / source["study_id"] / kind / url.rsplit("/", 1)[-1]
        raw.parent.mkdir(parents=True, exist_ok=True)
        if kind == "orderbook":
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"{DAY}_BTCUSDT_ob500.data", json.dumps(book()) + "\n")
        else:
            with gzip.open(raw, "wt", newline="") as handle:
                handle.write(
                    "timestamp,symbol,side,size,price,trdMatchID\n"
                    "1730419200.400000,BTCUSDT,Buy,2.000,100.50,synthetic\n"
                )
        identity_fields = {
            "url": url,
            "split": "train",
            "day": DAY,
            "kind": kind,
            "config_sha256": source_hash,
            **{key: source[key] for key in ["venue", "market", "symbol"]},
        }
        raw.with_name(raw.name + ".metadata.json").write_text(
            json.dumps(
                {
                    **identity_fields,
                    "sha256": sha256(raw),
                    "bytes": raw.stat().st_size,
                }
            )
        )
        audit = {
            **identity_fields,
            "status": "complete",
            "audit_run_id": "synthetic-" + kind,
            "input_sha256": sha256(raw),
            "processing_source_sha256": current_sources,
            "structural_pass": True,
            "zip_crc_verified": True,
            "errors": {},
            "quality": "valid",
            "records_scan_complete": True,
            "gzip_integrity": {"status": "passed", "read_to_eof": True, "crc_verified": True},
        }
        audit_path = tmp_path / "outputs" / source["study_id"] / "integrity" / f"{DAY}_{kind}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit))
    return config_path, source


def test_synthetic_runner_receipt_precedes_raw_parsing(tmp_path, monkeypatch):
    config, _ = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "execution-output"
    actual_score = execution.score_raw_day

    def checking_score(*args, **kwargs):
        prepared = json.loads((output / "prepared_receipt.json").read_text())
        assert prepared["stage"] == "inputs_verified_before_market_parsing"
        assert prepared["outputs_published"] is False
        assert prepared["windows_us"] == {DAY: [START, START + 3_600_000_000]}
        assert len(prepared["inputs"][DAY]) == 2
        return actual_score(*args, **kwargs)

    monkeypatch.setattr(execution, "score_raw_day", checking_score)
    result = run_diagnostic(tmp_path, config, output)
    assert result["status"] == "complete" and result["outputs_published"]
    assert result["days"] == [DAY]
    assert result["quantiles_computed"] is False
    assert datetime.fromisoformat(result["started_at"]).tzinfo is not None
    assert datetime.fromisoformat(result["completed_at"]) >= datetime.fromisoformat(
        result["started_at"]
    )
    assert result["runtime"]["python"] and result["runtime"]["pyarrow"]
    assert len(result["outputs"]) == 3
    assert pq.read_table(output / f"{DAY}_prints.parquet").num_rows == 5
    assert pq.read_table(output / f"{DAY}_blocks.parquet").num_rows == 12 * 5


def test_prepare_only_never_parses_market_bodies(tmp_path, monkeypatch):
    config, _ = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(execution, "score_raw_day", lambda *args: pytest.fail("Raw market parsing"))
    result = run_diagnostic(tmp_path, config, tmp_path / "prepared", prepare_only=True)
    assert result["status"] == "prepared" and not result["outputs_published"]
    assert result["outputs"] == []


@pytest.mark.parametrize("split", ["development", "event_test", "all"])
def test_runner_hardgate_precedes_any_input_access(tmp_path, split):
    with pytest.raises(ValueError, match="closed"):
        run_diagnostic(tmp_path, tmp_path / "missing", tmp_path / "output", split=split)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", ["raw", "audit_failed", "gzip_crc", "duplicate_ids", "config"])
def test_preflight_failures_never_publish(tmp_path, monkeypatch, failure):
    config, source = source_fixture(tmp_path, monkeypatch)
    audit_path = tmp_path / "outputs" / source["study_id"] / "integrity" / f"{DAY}_trades.json"
    if failure == "raw":
        raw = tmp_path / "data/raw" / source["study_id"] / "trades" / f"BTCUSDT{DAY}.csv.gz"
        raw.write_bytes(raw.read_bytes() + b"changed")
    elif failure == "config":
        plan = json.loads(config.read_text())
        plan["splits"]["train"] = ["2024-11-05"]
        config.write_text(json.dumps(plan))
    else:
        audit = json.loads(audit_path.read_text())
        if failure == "audit_failed":
            audit["status"] = "failed"
        elif failure == "gzip_crc":
            audit["gzip_integrity"]["crc_verified"] = False
        else:
            audit["quality"] = "invalid"
            audit["anomaly_counts"] = {"duplicate_trade_id": 1}
        audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(execution, "score_raw_day", lambda *args: pytest.fail("Raw market parsing"))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run_diagnostic(tmp_path, config, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and not manifest["outputs_published"]


def test_midrun_receipt_mutation_revokes_output_publication(tmp_path, monkeypatch):
    config, source = source_fixture(tmp_path, monkeypatch)
    actual_score = execution.score_raw_day

    def changing_score(*args, **kwargs):
        result = actual_score(*args, **kwargs)
        path = tmp_path / "outputs" / source["study_id"] / "integrity" / f"{DAY}_trades.json"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(execution, "score_raw_day", changing_score)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="evidence changed"):
        run_diagnostic(tmp_path, config, output)
    state = json.loads((output / "manifest.json").read_text())
    assert state["status"] == "failed" and not state["outputs_published"]


@pytest.mark.parametrize("changed_receipt", ["audit", "metadata"])
def test_preflight_receipt_race_cannot_bind_old_success_to_new_bytes(
    tmp_path, monkeypatch, changed_receipt
):
    config, source = source_fixture(tmp_path, monkeypatch)
    raw = tmp_path / "data/raw" / source["study_id"] / "trades" / f"BTCUSDT{DAY}.csv.gz"
    audit = tmp_path / "outputs" / source["study_id"] / "integrity" / f"{DAY}_trades.json"
    receipt = audit if changed_receipt == "audit" else raw.with_name(raw.name + ".metadata.json")
    original_hash = execution.sha256

    def race(path):
        result = original_hash(path)
        if Path(path) == raw:
            content = json.loads(receipt.read_text())
            content["status"] = "failed"
            receipt.write_text(json.dumps(content))
        return result

    monkeypatch.setattr(execution, "sha256", race)
    monkeypatch.setattr(execution, "score_raw_day", lambda *args: pytest.fail("Raw market parsing"))
    output = tmp_path / "race"
    with pytest.raises(ValueError, match="receipt changed during preflight"):
        run_diagnostic(tmp_path, config, output)
    state = json.loads((output / "manifest.json").read_text())
    assert state["status"] == "failed" and not state["outputs_published"]
    assert not (output / "prepared_receipt.json").exists()


def test_uncommitted_source_is_rejected_before_real_run(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "frozen.py"
    source.write_text("original\n")
    subprocess.run(["git", "add", source.name], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture freeze",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert execution.frozen_identity(tmp_path, [source])["source_sha256"][source.name] == sha256(
        source
    )
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="not frozen"):
        execution.frozen_identity(tmp_path, [source])
