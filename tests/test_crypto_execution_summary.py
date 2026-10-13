import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import crypto_execution_summary as summary
from experiments.crypto_execution_diagnostic import (
    AUDIT_SOURCES,
    DELAYS,
    EXECUTION_SOURCES,
    ROW_SCHEMA,
    Trade,
    day_start_us,
    diagnostic_rows,
    write_day,
)

PROJECT = Path(__file__).resolve().parents[1]
DAY = "2024-11-01"


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def synthetic_rows(day=DAY, *, price=10050, copies=None):
    start = day_start_us(day)
    book = {
        "topic": "orderbook.500.BTCUSDT",
        "type": "snapshot",
        "cts": start // 1000,
        "ts": start // 1000,
        "data": {
            "s": "BTCUSDT",
            "u": 1,
            "seq": 100,
            "b": [["99.00", "1.000"], ["98.00", "1.000"]],
            "a": [["100.00", "1.000"], ["101.00", "1.000"]],
        },
    }
    if copies is not None:
        trades = [
            Trade(start + 400_000 + i, "Buy", 2000, price, f"id-{i}", i + 1) for i in range(copies)
        ]
    else:
        trades = [
            Trade(start + 100_000, "Buy", 1000, 10000, "early-buy", 1),
            Trade(start + 200_000, "Sell", 1000, 9900, "early-sell", 2),
            Trade(start + 400_000, "Buy", 2000, 10050, "cross", 3),
            Trade(start + 500_000, "Buy", 1000, 10100, "small", 4),
            Trade(start + 600_000, "Buy", 3000, 10050, "partial", 5),
            Trade(start + 1_000_000, "Buy", 1000, 10000, "stale", 6),
        ]
    return list(
        diagnostic_rows(trades, {delay: iter([copy.deepcopy(book)]) for delay in DELAYS}, day)
    )


def fixture_run(tmp_path, *, mode="smoke", rows_by_day=None, relative_outputs=False):
    root = tmp_path / "project"
    root.mkdir()
    for name in dict.fromkeys(
        (
            summary.CONFIG_NAME,
            summary.SOURCE_CONFIG_NAME,
            *EXECUTION_SOURCES,
            *summary.SUMMARY_SOURCES,
        )
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT / name).read_bytes())
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "frozen synthetic source",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    plan = json.loads((root / summary.CONFIG_NAME).read_text())
    source = json.loads((root / summary.SOURCE_CONFIG_NAME).read_text())
    days = [DAY] if mode == "smoke" else plan["splits"]["train"]
    hashes = {
        name: summary.sha256(root / name)
        for name in (summary.CONFIG_NAME, summary.SOURCE_CONFIG_NAME, *EXECUTION_SOURCES)
    }
    directory = root / "outputs/synthetic_execution"
    directory.mkdir(parents=True)
    state = {
        "status": "complete",
        "outputs_published": True,
        "stage": "outputs_verified",
        "run_id": "a" * 32,
        "split": "train",
        "mode": mode,
        "started_at": "2026-10-01T00:00:00+00:00",
        "completed_at": "2026-10-01T00:00:01+00:00",
        "runtime": {
            "python": "3.11",
            "python_implementation": "CPython",
            "pyarrow": pa.__version__,
            "zlib_runtime": "synthetic",
            "platform": "synthetic",
        },
        "git_commit": commit,
        "source_sha256": hashes,
        "config_sha256": hashes[summary.CONFIG_NAME],
        "source_config_sha256": hashes[summary.SOURCE_CONFIG_NAME],
        "config": plan,
        "days": days,
        "inputs": {},
        "windows_us": {},
        "interpretation": plan["interpretation"],
        "quantiles_computed": False,
        "outputs": [],
    }
    for day in days:
        start = day_start_us(day)
        end = start + (3_600_000_000 if mode == "smoke" else 86_400_000_000)
        state["windows_us"][day] = [start, end]
        state["inputs"][day] = {}
        for kind in ("orderbook", "trades"):
            url = source["urls"][kind].format(date=day)
            raw = root / "data/raw" / source["study_id"] / kind / url.rsplit("/", 1)[-1]
            meta = raw.with_name(raw.name + ".metadata.json")
            audit = root / "outputs" / source["study_id"] / "integrity" / f"{day}_{kind}.json"
            identity = {
                "url": url,
                "split": "train",
                "day": day,
                "kind": kind,
                "config_sha256": state["source_config_sha256"],
                **{key: source[key] for key in ("venue", "market", "symbol")},
            }
            _json(meta, {**identity, "sha256": "0" * 64, "bytes": 77})
            _json(
                audit,
                {
                    **identity,
                    "status": "complete",
                    "audit_run_id": f"audit-{day}-{kind}",
                    "input_sha256": "0" * 64,
                    "processing_source_sha256": {name: hashes[name] for name in AUDIT_SOURCES},
                    "zip_crc_verified": True,
                    "structural_pass": True,
                    "errors": {},
                    "quality": "valid",
                    "records_scan_complete": True,
                    "gzip_integrity": {
                        "status": "passed",
                        "read_to_eof": True,
                        "crc_verified": True,
                    },
                },
            )
            state["inputs"][day][kind] = {
                "path": str(raw),
                "sha256": "0" * 64,
                "bytes": 77,
                "metadata_path": str(meta),
                "metadata_sha256": summary.sha256(meta),
                "audit_path": str(audit),
                "audit_sha256": summary.sha256(audit),
                "audit_run_id": f"audit-{day}-{kind}",
            }
            assert not raw.exists()  # The summary cannot require or open raw market bodies.
        rows = synthetic_rows(day) if rows_by_day is None else rows_by_day.get(day, [])
        state["outputs"].extend(write_day(rows, directory, day, start, end, row_group_size=3))
    if relative_outputs:
        for output in state["outputs"]:
            output["path"] = str(Path(output["path"]).relative_to(root))
    prepared = {
        **state,
        "status": "running",
        "outputs_published": False,
        "stage": "inputs_verified_before_market_parsing",
        "outputs": [],
        "completed_at": None,
    }
    _json(directory / "prepared_receipt.json", prepared)
    manifest = directory / "manifest.json"
    _json(manifest, state)
    return root, root / summary.CONFIG_NAME, manifest


def update_output(root, manifest, path):
    state = json.loads(manifest.read_text())
    for receipt in state["outputs"]:
        recorded = Path(receipt["path"])
        recorded = recorded if recorded.is_absolute() else root / recorded
        if recorded == path:
            receipt["sha256"] = summary.sha256(path)
            if path.suffix == ".parquet":
                receipt["rows"] = pq.ParquetFile(path).metadata.num_rows
    _json(manifest, state)


def forbid_values(monkeypatch):
    monkeypatch.setattr(
        summary,
        "summarize_day",
        lambda *args, **kwargs: pytest.fail("Unauthorized Parquet value consumption"),
    )


def test_exact_quantiles_are_inverse_empirical_cdf_not_interpolated():
    stats = summary.exact_statistics([0, 1, 2, 3, 10])
    assert stats == {"mean": 3.2, "median": 2, "p90": 10, "p99": 10}
    assert summary.exact_statistics(np.arange(100)) == {
        "mean": 49.5,
        "median": 49,
        "p90": 89,
        "p99": 98,
    }
    assert summary.exact_statistics([]) == dict.fromkeys(("mean", "median", "p90", "p99"))
    with pytest.raises(ValueError, match="finite"):
        summary.exact_statistics([1, float("nan")])


def test_exact_paired_counts_strata_shared_groups_and_smoke_label(tmp_path):
    root, config, manifest = fixture_run(tmp_path, relative_outputs=True)
    state = summary.summarize_execution(root, config, manifest, root / "summary-output")
    assert state["status"] == "complete" and state["outputs_published"] is True
    path = Path(state["outputs"][0]["path"])
    assert summary.sha256(path) == state["outputs"][0]["sha256"]
    report = json.loads(path.read_text())
    assert report["quantile_policy"]["method"] == "inverted_cdf"
    assert report["delays_ms"] == list(DELAYS)
    assert report["equal_day"]["available"] is False
    assert report["days"][DAY]["whole_utc_day"] is False
    zero = report["days"][DAY]["delays"]["0"]
    delayed = report["days"][DAY]["delays"]["250"]
    assert zero["candidate_count"] == delayed["candidate_count"] == 6
    assert zero["eligible_count"] == 4 and zero["exclusion_count"] == 2
    assert delayed["eligible_count"] == 2 and delayed["reasons"]["no_state"] == 2
    assert zero["reasons"]["partial_depth"] == zero["reasons"]["stale"] == 1
    assert zero["groups"]["eligible"]["errors_bps"]["sweep_abs_bps"] == {
        "mean": 25,
        "median": 0,
        "p90": 100,
        "p99": 100,
    }
    assert zero["groups"]["cross_l1"]["n"] == 1
    assert zero["groups"]["cross_l1"]["sweep_minus_l1_mae_bps"] == -50
    for result in report["days"][DAY]["delays"].values():
        assert result["groups"]["shared_eligible"]["n"] == 2
        assert result["groups"]["shared_cross_l1"]["n"] == 1
    assert zero["by_side"]["Sell"]["candidate_count"] == 1
    assert zero["by_side"]["Sell"]["eligible_count"] == 1
    assert zero["by_size_bin"][">5"]["candidate_count"] == 0
    assert (
        zero["by_size_bin"][">5"]["groups"]["eligible"]["errors_bps"]["sweep_abs_bps"]["mean"]
        is None
    )
    assert zero["by_age_bin"]["[1000,+inf)ms"]["exclusion_count"] == 1
    assert report["producer_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert "does not reopen or rehash raw archives" in report["source_evidence"]


def test_whole_day_equal_weights_ignore_print_multiplicity_and_list_empty_days(tmp_path):
    rows = {
        DAY: synthetic_rows(DAY, copies=1),
        "2024-11-02": synthetic_rows("2024-11-02", price=10000, copies=10),
        "2024-11-03": [],
    }
    root, config, manifest = fixture_run(tmp_path, mode="train", rows_by_day=rows)
    state = summary.summarize_execution(root, config, manifest, root / "summary-output")
    report = json.loads(Path(state["outputs"][0]["path"]).read_text())
    result = report["equal_day"]["delays"]["0"]["groups"]["cross_l1"]
    assert result["days_with_samples"] == [DAY, "2024-11-02"]
    assert result["days_without_samples"] == ["2024-11-03"]
    assert result["day_count"] == 2
    expected = (-50 + 10000 * 0.5 / 100.5) / 2
    assert result["equal_day_sweep_minus_l1_mae_bps"] == pytest.approx(expected)
    assert result["equal_day_sweep_minus_l1_mae_bps"] != pytest.approx(
        (-50 + 10 * 10000 * 0.5 / 100.5) / 11
    )
    assert result["days_sweep_lower_l1_mae"] == result["days_sweep_higher_l1_mae"] == 1
    assert all(day["whole_utc_day"] for day in report["days"].values())


@pytest.mark.parametrize("split", ["development", "event_test", "all"])
def test_split_hardgate_precedes_any_filesystem_consumption(tmp_path, split):
    with pytest.raises(ValueError, match="closed"):
        summary.summarize_execution(
            tmp_path, tmp_path / "missing", tmp_path / "missing", tmp_path / "output", split=split
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change",
    [
        "event_day",
        "extra_day",
        "window",
        "failed",
        "unpublished",
        "prepared",
        "provenance",
        "nonutc_time",
        "future_time",
    ],
)
def test_manifest_identity_failures_precede_output_value_consumption(tmp_path, monkeypatch, change):
    root, config, manifest = fixture_run(tmp_path)
    state = json.loads(manifest.read_text())
    if change == "event_day":
        state["days"] = ["2024-11-05"]
    elif change == "extra_day":
        state["days"].append("2024-11-04")
    elif change == "window":
        state["windows_us"][DAY][1] += 300_000_000
    elif change == "failed":
        state["status"] = "failed"
    elif change == "unpublished":
        state["outputs_published"] = False
    elif change == "prepared":
        prepared_path = manifest.parent / "prepared_receipt.json"
        prepared = json.loads(prepared_path.read_text())
        prepared["run_id"] = "mismatched"
        _json(prepared_path, prepared)
    elif change == "provenance":
        state["source_sha256"]["experiments/crypto_execution_diagnostic.py"] = "0" * 64
    elif change == "nonutc_time":
        state["completed_at"] = "2026-10-01T08:00:01+08:00"
    else:
        state["completed_at"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    _json(manifest, state)
    forbid_values(monkeypatch)
    output = root / "rejected"
    with pytest.raises(ValueError):
        summary.summarize_execution(root, config, manifest, output)
    failed = json.loads((output / "manifest.json").read_text())
    assert failed["status"] == "failed" and failed["outputs_published"] is False
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("target", ["prints", "blocks", "summary", "metadata", "audit"])
def test_every_output_and_current_receipt_is_hashed_before_values(tmp_path, monkeypatch, target):
    root, config, manifest = fixture_run(tmp_path)
    state = json.loads(manifest.read_text())
    if target in ("metadata", "audit"):
        path = Path(state["inputs"][DAY]["trades"][target + "_path"])
    else:
        suffix = {
            "prints": "prints.parquet",
            "blocks": "blocks.parquet",
            "summary": "summary.json",
        }[target]
        path = manifest.parent / f"{DAY}_{suffix}"
    path.write_bytes(path.read_bytes() + b"tampered")
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="differs"):
        summary.summarize_execution(root, config, manifest, root / "rejected")


def test_missing_print_schema_field_rejected_before_values(tmp_path, monkeypatch):
    root, config, manifest = fixture_run(tmp_path)
    path = manifest.parent / f"{DAY}_prints.parquet"
    table = pq.read_table(path).drop(["sweep_cost_product_units"])
    pq.write_table(table, path)
    update_output(root, manifest, path)
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="schema"):
        summary.summarize_execution(root, config, manifest, root / "rejected")


@pytest.mark.parametrize(
    "change", ["delay", "shared", "duplicate_id", "out_of_window", "excluded_residual", "stratum"]
)
def test_corrupted_paired_row_semantics_fail_closed(tmp_path, change):
    root, config, manifest = fixture_run(tmp_path)
    path = manifest.parent / f"{DAY}_prints.parquet"
    rows = pq.read_table(path).to_pylist()
    if change == "delay":
        rows[1]["delay_ms"] = 0
    elif change == "shared":
        rows[0]["shared_eligible"] = True
    elif change == "duplicate_id":
        for row in rows[5:10]:
            row["trade_id"] = rows[0]["trade_id"]
    elif change == "out_of_window":
        for row in rows[-5:]:
            row["timestamp_us"] = day_start_us(DAY) + 3_600_000_000
    elif change == "excluded_residual":
        rows[-1]["sweep_abs_bps"] = 0
    else:
        rows[0]["age_bin"] = "[0,25)ms"
    pq.write_table(pa.Table.from_pylist(rows, schema=ROW_SCHEMA), path)
    update_output(root, manifest, path)
    with pytest.raises(ValueError):
        summary.summarize_execution(root, config, manifest, root / "rejected")


def test_unaligned_read_batches_keep_five_print_rows_together(tmp_path, monkeypatch):
    root, _, manifest = fixture_run(tmp_path)
    original = pq.ParquetFile

    class TinyBatches:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def iter_batches(self, **kwargs):
            kwargs["batch_size"] = 3
            return self.file.iter_batches(**kwargs)

    monkeypatch.setattr(summary.pq, "ParquetFile", TinyBatches)
    report = summary.summarize_day(
        manifest.parent / f"{DAY}_prints.parquet",
        DAY,
        [day_start_us(DAY), day_start_us(DAY) + 3_600_000_000],
    )
    assert report["print_delay_rows"] == 30
    assert report["delays"]["0"]["groups"]["shared_eligible"]["n"] == 2


def test_large_integer_quantities_keep_exact_stratum_and_coverage_sums(tmp_path):
    quantity = 2**53 + 1
    l1 = 2**53
    rows = synthetic_rows(copies=1)
    for row in rows:
        row.update(
            quantity_units=quantity,
            covered_quantity_units=quantity,
            unresolved_quantity_units=0,
            l1_quantity_units=l1,
            sweep_price_denominator=quantity,
            size_bin="(1,5]",
            cross_l1=True,
            shared_cross_l1=True,
        )
    path = tmp_path / "big.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=ROW_SCHEMA), path)
    result = summary.summarize_day(
        path, DAY, [day_start_us(DAY), day_start_us(DAY) + 3_600_000_000]
    )
    assert result["delays"]["0"]["covered_quantity_units_sum"] == quantity
    assert result["delays"]["0"]["by_size_bin"]["(1,5]"]["eligible_count"] == 1


def test_midread_receipt_change_revokes_publication(tmp_path, monkeypatch):
    root, config, manifest = fixture_run(tmp_path)
    original = summary.summarize_day

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        prepared = manifest.parent / "prepared_receipt.json"
        prepared.write_text(prepared.read_text() + "\n")
        return result

    monkeypatch.setattr(summary, "summarize_day", changing)
    with pytest.raises(ValueError, match="changed"):
        summary.summarize_execution(root, config, manifest, root / "rejected")
    assert not (root / "rejected/summary.json").exists()


def test_summary_directory_and_final_artifact_cannot_be_replaced(tmp_path, monkeypatch):
    root, config, manifest = fixture_run(tmp_path)
    output = root / "summary-output"
    summary.summarize_execution(root, config, manifest, output)
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        summary.summarize_execution(root, config, manifest, output)
    assert (output / "summary.json").read_bytes() == before
    original = summary._publish_exclusive

    def competing(path, content):
        path.write_text("existing artifact")
        return original(path, content)

    monkeypatch.setattr(summary, "_publish_exclusive", competing)
    with pytest.raises(FileExistsError):
        summary.summarize_execution(root, config, manifest, root / "race")
    assert (root / "race/summary.json").read_text() == "existing artifact"
    assert json.loads((root / "race/manifest.json").read_text())["outputs_published"] is False
