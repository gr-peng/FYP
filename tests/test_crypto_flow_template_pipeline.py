import copy
import gzip
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from experiments import crypto_effective_flow_reader as reader
from experiments import crypto_flow_template_pipeline as pipeline
from experiments.crypto_effective_flow import (
    GRID_US,
    TRAIN_DAYS,
    FlowTemplate,
    Intent,
    ObservedPrint,
    StateFeatures,
    TrainingRow,
    fit_template_pool,
)
from experiments.crypto_execution_diagnostic import audited_inputs, day_start_us, sha256

ROOT = Path(__file__).resolve().parents[1]
DAY = TRAIN_DAYS[0]
START = day_start_us(DAY)


def message(i):
    ts = START // 1000 + 10 + i * 100
    q = 1000 + i
    return {
        "topic": "orderbook.500.BTCUSDT",
        "type": "snapshot" if i == 0 else "delta",
        "cts": ts,
        "ts": ts + 10,
        "data": {
            "s": "BTCUSDT",
            "u": i + 1,
            "seq": i + 10,
            "b": [["99.00", f"{q // 1000}.{q % 1000:03d}"], ["98.00", "2.000"]],
            "a": [["100.00", "1.000"], ["101.00", "2.000"]],
        },
    }


def synthetic_tape():
    return tuple(
        ObservedPrint(
            START + (i + 1) * 1_000_000 - 1000,
            "Buy" if i % 2 == 0 else "Sell",
            i + 1,
            10000 if i % 2 == 0 else 9900,
            i + 1,
            f"synthetic-{i}",
        )
        for i in range(6)
    )


def records(*, tape=None, messages=None, end=6_200_000):
    candidates = reader.flow_candidates(
        [message(i) for i in range(100)] if messages is None else messages,
        synthetic_tape() if tape is None else tape,
        DAY,
        emit_start_us=START,
        emit_end_us=START + end,
    )
    return [json.loads(json.dumps(reader.candidate_record(candidate))) for candidate in candidates]


def parse(values, end=6_200_000):
    return pipeline._parse_records(iter(values), DAY, end_us=START + end)


def test_machine_protocol_is_fixed_and_includes_runtime_freeze_closure():
    plan = json.loads((ROOT / pipeline.CONFIG_PATH).read_bytes())
    assert plan == pipeline.expected_protocol()
    assert plan["candidate_rows_per_day"] == 864000
    assert plan["train_windows"]["total_candidates"] == 72
    assert plan["generator"]["initial_view_max_depth_per_side"] == 500
    assert not plan["generator"]["topups_allowed"]
    assert plan["generator"]["decimal_precision"] == 80
    assert plan["metrics"]["train_minute_candidates"] == 4320
    assert plan["metrics"]["return_bps"] == "10000*log(mid_end/mid_start); also for P"
    assert plan["metrics"]["quantile_method"] == "inverted_cdf"
    assert plan["metrics"]["market_pass_threshold"] is None
    assert len(pipeline.FROZEN_PATHS) == len(set(pipeline.FROZEN_PATHS))
    assert "src/behavioral_market/market/order_book.py" in pipeline.FROZEN_PATHS
    assert "experiments/crypto_autonomous_metrics.py" in pipeline.FROZEN_PATHS


def test_complete_sequential_prefix_preserves_all_candidates_and_three_features():
    values = records()
    rows, counts = parse(values)
    assert counts["candidate_rows"] == 62 and counts["eligible_rows"] == 2
    assert counts["observed_prints"] == 6 and counts["observed_quantity_units"] == 21
    assert [row.template.row_id for row in rows] == [
        f"{DAY}:{START + x}"
        for x in (
            6_000_000,
            6_100_000,
        )
    ]
    assert rows[0].features_by_delay[0].trade_pressure == -1
    assert rows[0].features_by_delay[1].trade_pressure == 1
    assert rows[0].features_by_delay[2].trade_pressure == 1
    assert all(
        type(feature.spread_bps) is Fraction for row in rows for feature in row.features_by_delay
    )
    # Earlier candidates can have valid templates but cannot enter the common cohort.
    assert values[10]["template"] is not None and not values[10]["eligible"]
    assert counts["exclusions"]["delay_5000_incomplete_past_context"] == 60


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda values: values.pop(10), "grid order"),
        (lambda values: values.append(copy.deepcopy(values[-1])), "Extra candidate"),
        (lambda values: values[60].update(eligible=False), "Eligibility"),
        (lambda values: values[60]["template"].update(row_id=f"{DAY}:{START}"), "row_id"),
        (lambda values: values[60]["features_by_delay"][0].update(spread_bps=[1.0, 1]), "rational"),
        (lambda values: values[60]["features_by_delay"][0].update(spread_bps=[1, 0]), "rational"),
        (lambda values: values[60]["features_by_delay"][0].update(spread_bps=[2, 2]), "reduced"),
        (lambda values: values[60]["features_by_delay"].__setitem__(2, None), "Feature rational"),
        (
            lambda values: values[60]["features_by_delay"][0].update(future_mid=999),
            "feature fields",
        ),
        (lambda values: values[60]["template"].update(next_book=[]), "narrow template fields"),
        (
            lambda values: values[60]["signal_prefixes"][1].update(message_index=None),
            "Unknown message prefix",
        ),
        (
            lambda values: values[60]["signal_prefixes"][2].update(observed_at_us=START),
            "precise observation clock",
        ),
        (
            lambda values: values[20]["end_prefix"].update(engine_us=START + 1_900_001),
            "millisecond",
        ),
        (lambda values: values[20]["end_prefix"].update(available_us=START + 2_100_000), "clocks"),
        (lambda values: values[20]["end_prefix"].update(segment_start_us=START + 20_000), "origin"),
        (lambda values: values[20]["end_prefix"].update(invalid_count=-1), "integer"),
        (lambda values: values[30]["observed_prints"][0].update(source_row=99), "source-row"),
    ],
)
def test_parser_rejects_corrupted_clocks_rationals_and_leaky_fields(edit, match):
    values = records()
    # The fourth print is in candidate 39, rather than candidate 30.
    if match == "source-row":
        values[30]["observed_prints"] = copy.deepcopy(values[39]["observed_prints"])
        values[30]["observed_prints"][0]["timestamp_us"] = START + 3_010_000
    edit(values)
    with pytest.raises(ValueError, match=match):
        parse(values)


def test_pressure_is_checked_before_current_label_prints_enter_the_buffer():
    current = ObservedPrint(START + 6_010_000, "Buy", 500, 10000, 7, "current-label")
    values = records(tape=(*synthetic_tape(), current))
    rows, _ = parse(values)
    assert rows[0].features_by_delay[0].trade_pressure == -1
    assert rows[0].template.intents == (Intent("Buy", 500),)
    values[60]["features_by_delay"][0]["trade_pressure"] = [247, 253]
    with pytest.raises(ValueError, match="past-print pressure"):
        parse(values)


def test_source_reset_inside_lag_context_is_retained_and_not_fitted():
    messages = [message(i) for i in range(100)]
    messages[40]["type"] = "snapshot"
    values = records(messages=messages, end=9_200_000)
    rows, counts = parse(values, end=9_200_000)
    assert counts["candidate_rows"] == 92 and len(rows) == 1
    assert rows[0].template.row_id == f"{DAY}:{START + 9_100_000}"
    assert counts["exclusions"]["snapshot_or_reset_crossing"] == 1
    assert counts["exclusions"]["delay_5000_segment_or_error_crossing"] > 0


def test_prior_day_origin_exclusion_keeps_computable_features_without_fitting_labels():
    messages = [message(i) for i in range(100)]
    messages[0]["cts"] = START // 1000 - 90
    messages[0]["ts"] = START // 1000 - 80
    values = records(messages=messages)
    assert values[60]["features_by_delay"][0] is not None
    rows, counts = parse(values)
    assert not rows and counts["candidate_rows"] == 62
    assert counts["exclusions"]["start_outside_day_state"] == 62
    assert "delay_0_invalid_or_stale_signal" not in counts["exclusions"]


def test_effective_residual_and_intent_tape_are_independently_checked():
    values = records()
    values[60]["template"]["adds"][0]["quantity_units"] += 1
    with pytest.raises(ValueError, match="minimal net residuals"):
        parse(values)
    values = records()
    values[9]["template"]["intents"][0]["price_units"] = 10000
    with pytest.raises(ValueError, match="narrow intent fields"):
        parse(values)


def test_unknown_source_clock_and_missing_grid_do_not_consume_later_records():
    values = records()
    values[5]["end_prefix"]["message_index"] = None

    def guarded():
        yield from values[:6]
        pytest.fail("Parser read beyond the bad source prefix")

    with pytest.raises(ValueError, match="Unknown message prefix"):
        pipeline._parse_records(guarded(), DAY, end_us=START + 6_200_000)
    with pytest.raises(ValueError, match="Missing candidate"):
        parse(values[:4])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def run_root(tmp_path):
    """Small real Git/source/hash fixtures, never the workspace's real archives."""
    root = tmp_path / "synthetic-repository"
    root.mkdir()
    for name in pipeline.FROZEN_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            (ROOT / name).read_bytes()
            if name.endswith(".json")
            else f"# synthetic committed {name}\n".encode()
        )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-qm",
        "synthetic fixture sources",
    )
    commit = git(root, "rev-parse", "HEAD")
    source_names = (reader.CONFIG_PATH, reader.SOURCE_PATH, *reader.READER_SOURCES)
    source_hashes = {name: sha256(root / name) for name in source_names}
    source = json.loads((root / reader.SOURCE_PATH).read_bytes())
    audit_sources = {name: source_hashes[name] for name in pipeline.AUDIT_SOURCES}
    inputs = {}
    for day in TRAIN_DAYS:
        for kind in ("orderbook", "trades"):
            url = source["urls"][kind].format(date=day)
            path = root / "data/raw" / source["study_id"] / kind / url.rsplit("/", 1)[-1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic raw hash-only {day} {kind}".encode())
            metadata = {
                "url": url,
                "split": "train",
                "day": day,
                "kind": kind,
                "config_sha256": reader.SOURCE_SHA,
                **{name: source[name] for name in ("venue", "market", "symbol")},
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            write_json(path.with_name(path.name + ".metadata.json"), metadata)
            audit = {
                "status": "complete",
                "audit_run_id": f"synthetic-{day}-{kind}",
                "split": "train",
                "day": day,
                "kind": kind,
                "config_sha256": reader.SOURCE_SHA,
                "input_sha256": sha256(path),
                "processing_source_sha256": audit_sources,
            }
            if kind == "orderbook":
                audit.update(zip_crc_verified=True, structural_pass=True, errors={})
            else:
                audit.update(
                    quality="valid",
                    records_scan_complete=True,
                    gzip_integrity={"status": "passed", "read_to_eof": True, "crc_verified": True},
                )
            write_json(
                root / "outputs" / source["study_id"] / "integrity" / f"{day}_{kind}.json", audit
            )
        inputs[day] = audited_inputs(root, source, reader.SOURCE_SHA, day, audit_sources)
    extraction = root / "outputs/synthetic-extraction"
    extraction.mkdir()
    outputs = []
    for day in TRAIN_DAYS:
        path = extraction / f"{day}_effective_flow.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            handle.write("{}\n")
        outputs.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": pipeline.CANDIDATES_PER_DAY,
                "eligible_rows": 1,
                "exclusions": {"start_invalid_or_absent": pipeline.CANDIDATES_PER_DAY - 1},
                "observed_prints": 0,
                "observed_quantity_units": 0,
                "print_alignment": {},
            }
        )
    state = {
        "status": "complete",
        "outputs_published": True,
        "outputs": outputs,
        "run_id": "synthetic-master",
        "split": "train",
        "mode": "train",
        "started_at": "2026-10-01T00:00:00+00:00",
        "completed_at": "2026-10-01T01:00:00+00:00",
        "runtime": {"python": "synthetic", "python_implementation": "synthetic"},
        "git_commit": commit,
        "source_sha256": source_hashes,
        "config": reader.expected_protocol(),
        "config_sha256": sha256(root / reader.CONFIG_PATH),
        "source_config_sha256": reader.SOURCE_SHA,
        "inputs": inputs,
        "days": list(TRAIN_DAYS),
        "windows_us": {
            day: [day_start_us(day), day_start_us(day) + pipeline.DAY_US] for day in TRAIN_DAYS
        },
    }
    write_json(extraction / "manifest.json", state)
    write_json(
        extraction / "prepared_receipt.json",
        dict(
            state,
            status="running",
            outputs_published=False,
            outputs=[],
            completed_at=None,
        ),
    )
    return root, extraction / "manifest.json"


def fake_complete_rows(monkeypatch):
    original = pipeline._gzip_rows
    monkeypatch.setattr(
        pipeline,
        "_gzip_rows",
        lambda path: pipeline.CANDIDATES_PER_DAY
        if path.name.endswith("_effective_flow.jsonl.gz")
        else original(path),
    )


def fake_parse_day(path, day):
    feature = StateFeatures(Fraction(0), Fraction(TRAIN_DAYS.index(day) + 1), Fraction(0))
    row = TrainingRow(
        day, FlowTemplate(f"{day}:{day_start_us(day) + 6_000_000}", (), (), ()), (feature,) * 3
    )
    counts = {
        "candidate_rows": pipeline.CANDIDATES_PER_DAY,
        "eligible_rows": 1,
        "exclusions": {"start_invalid_or_absent": pipeline.CANDIDATES_PER_DAY - 1},
        "observed_prints": 0,
        "observed_quantity_units": 0,
        "print_alignment": {},
    }
    return (row,), counts


def fit_fake(run_root, monkeypatch):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    monkeypatch.setattr(pipeline, "_parse_day", fake_parse_day)
    destination = root / "outputs/synthetic-model"
    state = pipeline.fit_flow_templates(root, manifest, destination)
    return destination, state


def change_master(manifest, update, *, sync_prepared=True):
    state = json.loads(manifest.read_bytes())
    update(state)
    write_json(manifest, state)
    if sync_prepared:
        write_json(
            manifest.parent / "prepared_receipt.json",
            dict(
                state,
                status="running",
                outputs_published=False,
                outputs=[],
                completed_at=None,
            ),
        )


@pytest.mark.parametrize(
    "update",
    [
        lambda state: state.update(mode="smoke"),
        lambda state: state.update(split="development"),
        lambda state: state.update(split="event_test"),
        lambda state: state.update(status="failed", outputs_published=False),
        lambda state: state.update(outputs_published=False),
        lambda state: state.update(days=list(TRAIN_DAYS[:2])),
        lambda state: state["windows_us"][DAY].__setitem__(1, START + 300_000_000),
        lambda state: state["outputs"][0].update(rows=3000),
        lambda state: state["outputs"].pop(),
        lambda state: state["outputs"][0].update(sha256="0" * 64),
        lambda state: state.update(config_sha256="0" * 64),
        lambda state: state.update(completed_at="2099-01-01T00:00:00+00:00"),
        lambda state: state.update(completed_at="2026-10-01T01:00:00"),
    ],
)
def test_master_rejections_precede_candidate_values_and_leave_immutable_failed_receipt(
    run_root,
    monkeypatch,
    update,
):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    change_master(manifest, update)
    monkeypatch.setattr(
        pipeline, "_parse_day", lambda *args: pytest.fail("Candidate values consumed")
    )
    destination = root / "outputs/rejected-model"
    with pytest.raises(ValueError):
        pipeline.fit_flow_templates(root, manifest, destination)
    state = json.loads((destination / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"] and state["outputs"] == []
    assert not (destination / "rows.jsonl.gz").exists()
    previous = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        pipeline.fit_flow_templates(root, manifest, destination)
    assert (destination / "manifest.json").read_bytes() == previous


def test_smoke_refusal_never_hashes_or_opens_raw_or_gzip(run_root, monkeypatch):
    root, manifest = run_root
    change_master(manifest, lambda state: state.update(mode="smoke"))
    monkeypatch.setattr(pipeline, "audited_inputs", lambda *args: pytest.fail("Raw read"))
    monkeypatch.setattr(pipeline, "_gzip_rows", lambda *args: pytest.fail("Gzip read"))
    with pytest.raises(ValueError, match="three-full-day"):
        pipeline.verify_train_extraction(root, manifest)


def test_current_gzip_crc_rows_and_all_output_hashes_are_verified_before_parser(
    run_root, monkeypatch
):
    root, manifest = run_root
    with pytest.raises(ValueError, match="gzip row count"):
        pipeline.verify_train_extraction(root, manifest)
    fake_complete_rows(monkeypatch)
    state = json.loads(manifest.read_bytes())
    Path(state["outputs"][2]["path"]).write_bytes(b"late-file-tamper")
    monkeypatch.setattr(pipeline, "_parse_day", lambda *args: pytest.fail("Earlier day parsed"))
    with pytest.raises(ValueError, match="daily gzip hash"):
        pipeline.fit_flow_templates(root, manifest, root / "outputs/late-file-reject")


@pytest.mark.parametrize("which", ["raw", "metadata", "audit", "source", "consumer", "prepared"])
def test_stale_current_inputs_or_dirty_sources_fail_before_values(run_root, monkeypatch, which):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    state = json.loads(manifest.read_bytes())
    receipt = state["inputs"][DAY]["trades"]
    paths = {
        "raw": Path(receipt["path"]),
        "metadata": Path(receipt["metadata_path"]),
        "audit": Path(receipt["audit_path"]),
        "source": root / reader.READER_SOURCES[0],
        "consumer": root / "experiments/crypto_autonomous_flow.py",
        "prepared": manifest.parent / "prepared_receipt.json",
    }
    path = paths[which]
    if which == "prepared":
        prepared = json.loads(path.read_bytes())
        prepared["run_id"] = "different-preflight-run"
        write_json(path, prepared)
    else:
        path.write_bytes(path.read_bytes() + b"\n ")
    monkeypatch.setattr(
        pipeline, "_parse_day", lambda *args: pytest.fail("Candidate values consumed")
    )
    with pytest.raises(ValueError):
        pipeline.fit_flow_templates(root, manifest, root / "outputs/stale-reject")


def test_prepare_only_fences_every_day_and_never_constructs_a_pool(run_root, monkeypatch):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    monkeypatch.setattr(pipeline, "_parse_day", lambda *args: pytest.fail("Parse in prepare"))
    monkeypatch.setattr(pipeline, "fit_template_pool", lambda *args: pytest.fail("Fit in prepare"))
    output = root / "outputs/prepared-model"
    state = pipeline.fit_flow_templates(root, manifest, output, prepare_only=True)
    assert state["status"] == "prepared_no_candidate_values_read" and not state["outputs_published"]
    assert (output / "prepared_receipt.json").exists()
    assert not (output / "rows.jsonl.gz").exists()
    assert sha256(output / "prepared_receipt.json") == state["prepared_receipt_sha256"]


@pytest.mark.parametrize("prepare_only", [False, True])
def test_corrupted_own_prepared_write_rejected_before_model_values(
    run_root, monkeypatch, prepare_only,
):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    output = root / "outputs/corrupt-prepared-write"
    original = pipeline.write_json
    def corrupt(path, state):
        original(path, state)
        if path == output / "prepared_receipt.json":
            original(path, dict(state, status="failed"))
    monkeypatch.setattr(pipeline, "write_json", corrupt)
    monkeypatch.setattr(pipeline, "_parse_day", lambda *args: pytest.fail("Candidate values read"))
    with pytest.raises(ValueError, match="prepared receipt differs"):
        pipeline.fit_flow_templates(root, manifest, output, prepare_only=prepare_only)
    assert not json.loads((output / "manifest.json").read_bytes())["outputs_published"]


@pytest.mark.parametrize("mutation_day", [0, 2])
def test_changed_own_prepared_stops_next_parser_or_publication(
    run_root, monkeypatch, mutation_day,
):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)
    output = root / "outputs/late-prepared-change"
    days = []
    def parse_and_change(path, day):
        days.append(day)
        if day == TRAIN_DAYS[mutation_day]:
            target = output / "prepared_receipt.json"
            target.write_bytes(target.read_bytes() + b"\n")
        return fake_parse_day(path, day)
    monkeypatch.setattr(pipeline, "_parse_day", parse_and_change)
    with pytest.raises(ValueError, match="prepared receipt changed"):
        pipeline.fit_flow_templates(root, manifest, output)
    assert days == list(TRAIN_DAYS[:mutation_day + 1])
    assert not json.loads((output / "manifest.json").read_bytes())["outputs_published"]


def test_model_loader_requires_recorded_own_prepared_byte_identity(run_root, monkeypatch):
    root, _ = run_root
    destination, _ = fit_fake(run_root, monkeypatch)
    target = destination / "prepared_receipt.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="prepared receipt hash"):
        pipeline.load_flow_templates(root, destination / "manifest.json")


def test_whole_pool_content_roundtrip_uses_saved_parameters_and_no_raw_reader(
    run_root, monkeypatch
):
    root, _ = run_root
    destination, state = fit_fake(run_root, monkeypatch)
    assert state["candidate_rows"] == 3 * 864000 and state["eligible_rows"] == 3
    monkeypatch.setattr(pipeline, "fit_template_pool", lambda *args: pytest.fail("Refit in loader"))
    monkeypatch.setattr(pipeline, "audited_inputs", lambda *args: pytest.fail("Raw in loader"))
    for inputs in state["input_extraction"]["inputs"].values():
        for receipt in inputs.values():
            Path(receipt["path"]).unlink()
            Path(receipt["metadata_path"]).unlink()
            Path(receipt["audit_path"]).unlink()
    model = pipeline.load_flow_templates(root, destination / "manifest.json")
    assert model.pool.spread_median == 2
    assert model.identity.manifest_sha256 == sha256(destination / "manifest.json")
    assert model.identity.row_content_sha256 == state["row_content_sha256"]
    with gzip.open(destination / "rows.jsonl.gz", "rt") as handle:
        narrow = [json.loads(line) for line in handle]
    assert all(set(row) == {"template", "features_by_delay"} for row in narrow)
    assert all(set(row["template"]) == {"row_id", "adds", "removes", "intents"} for row in narrow)
    assert len(model.pool.rows) == 3 and len(model.pool.indexes) == 3


def rewrite_parameters(destination, update):
    path = destination / "parameters.json"
    parameters = json.loads(path.read_bytes())
    update(parameters)
    write_json(path, parameters)
    manifest = destination / "manifest.json"
    state = json.loads(manifest.read_bytes())
    state["outputs"][1]["sha256"] = sha256(path)
    write_json(manifest, state)


@pytest.mark.parametrize(
    "update,match",
    [
        (lambda params: params.update(spread_median=[999, 1]), "threshold"),
        (lambda params: params["indexes"][0][0].append(0), "indexes"),
        (lambda params: params["indexes"][0][0].append(True), "integer arrays"),
        (lambda params: params.update(fallback_min_rows=1), "fixed fit"),
    ],
)
def test_loader_validates_frozen_cells_threshold_and_fixed_policy_without_replacing_them(
    run_root,
    monkeypatch,
    update,
    match,
):
    root, _ = run_root
    destination, _ = fit_fake(run_root, monkeypatch)
    rewrite_parameters(destination, update)
    with pytest.raises(ValueError, match=match):
        pipeline.load_flow_templates(root, destination / "manifest.json")


def test_row_value_tamper_is_detected_even_when_row_identities_and_compressed_hash_are_retained(
    run_root,
    monkeypatch,
):
    root, _ = run_root
    destination, state = fit_fake(run_root, monkeypatch)
    path = destination / "rows.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        values = [json.loads(line) for line in handle]
    identities = [value["template"]["row_id"] for value in values]
    values[0]["features_by_delay"][0]["trade_pressure"] = [1, 1]
    with gzip.open(path, "wb") as handle:
        for value in values:
            handle.write(pipeline._canonical(value) + b"\n")
    manifest = destination / "manifest.json"
    state["outputs"][0]["sha256"] = sha256(path)
    write_json(manifest, state)
    assert identities == [value["template"]["row_id"] for value in values]
    with pytest.raises(ValueError, match="row content"):
        pipeline.load_flow_templates(root, manifest)


def test_late_source_revocation_during_parse_never_publishes_model(run_root, monkeypatch):
    root, manifest = run_root
    fake_complete_rows(monkeypatch)

    def revoke(path, day):
        if day == TRAIN_DAYS[-1]:
            source = root / "experiments/crypto_autonomous_metrics.py"
            source.write_bytes(source.read_bytes() + b"# late mutation\n")
        return fake_parse_day(path, day)

    monkeypatch.setattr(pipeline, "_parse_day", revoke)
    destination = root / "outputs/late-revocation"
    with pytest.raises(ValueError, match="source differs"):
        pipeline.fit_flow_templates(root, manifest, destination)
    state = json.loads((destination / "manifest.json").read_bytes())
    assert state["status"] == "failed" and not state["outputs_published"] and state["outputs"] == []


def test_loader_rejects_failed_receipt_before_any_model_values(run_root, monkeypatch):
    root, _ = run_root
    destination, state = fit_fake(run_root, monkeypatch)
    state["status"] = "failed"
    state["outputs_published"] = False
    write_json(destination / "manifest.json", state)
    monkeypatch.setattr(pipeline, "_gzip_rows", lambda *args: pytest.fail("Model values consumed"))
    with pytest.raises(ValueError, match="published train models"):
        pipeline.load_flow_templates(root, destination / "manifest.json")


def test_canonical_content_hash_distinguishes_identical_row_identities():
    first, _ = fake_parse_day(None, DAY)
    f = StateFeatures(Fraction(0), Fraction(1), Fraction(1))
    second = TrainingRow(DAY, first[0].template, (f,) * 3)
    a = hashlib.sha256(pipeline._canonical(pipeline._row_json(first[0]))).hexdigest()
    b = hashlib.sha256(pipeline._canonical(pipeline._row_json(second))).hexdigest()
    assert first[0].template.row_id == second.template.row_id and a != b


def test_no_unapproved_day_parser_or_consumer_split_is_enabled(run_root):
    def never():
        pytest.fail("Unapproved day parsed")
        yield None

    with pytest.raises(ValueError, match="approved"):
        pipeline._parse_records(never(), "2024-11-04", end_us=day_start_us("2024-11-04") + GRID_US)
    root, manifest = run_root
    with pytest.raises(ValueError, match="closed"):
        pipeline.fit_flow_templates(root, manifest, root / "outputs/closed", split="development")


def test_day_filtered_print_prefix_may_start_after_row_one_but_cannot_skip_inside_day():
    tape = tuple(
        ObservedPrint(
            value.timestamp_us,
            value.side,
            value.quantity_units,
            value.price_units,
            value.source_row + 10,
            value.trade_id,
        )
        for value in synthetic_tape()
    )
    values = records(tape=tape)
    rows, counts = parse(values)
    assert len(rows) == 2 and counts["observed_prints"] == len(tape)
    values[29]["observed_prints"][0]["source_row"] += 1
    with pytest.raises(ValueError, match="source-row"):
        parse(values)


def test_raw_revoked_during_gzip_preflight_is_rejected_before_candidate_parsing(
    run_root, monkeypatch
):
    root, manifest = run_root
    state = json.loads(manifest.read_bytes())
    raw = Path(state["inputs"][DAY]["trades"]["path"])

    def count_and_revoke(path):
        if path.name.startswith(TRAIN_DAYS[-1]):
            raw.write_bytes(raw.read_bytes() + b"late raw mutation")
        return pipeline.CANDIDATES_PER_DAY

    monkeypatch.setattr(pipeline, "_gzip_rows", count_and_revoke)
    monkeypatch.setattr(pipeline, "_parse_day", lambda *args: pytest.fail("Revoked raw parsed"))
    destination = root / "outputs/revoked-before-parser"
    with pytest.raises(ValueError):
        pipeline.fit_flow_templates(root, manifest, destination)
    assert json.loads((destination / "manifest.json").read_bytes())["status"] == "failed"
    assert not (destination / "prepared_receipt.json").exists()


@pytest.mark.parametrize("which", ["source", "output", "audit"])
def test_final_raw_preflight_is_followed_by_receipt_and_output_fences(
    run_root, monkeypatch, which
):
    root, manifest = run_root
    state = json.loads(manifest.read_bytes())
    fake_complete_rows(monkeypatch)
    original, calls = pipeline.audited_inputs, 0
    paths = {
        "source": root / reader.READER_SOURCES[0],
        "output": Path(state["outputs"][0]["path"]),
        "audit": Path(state["inputs"][DAY]["trades"]["audit_path"]),
    }

    def mutate_after_last_raw_check(*args):
        nonlocal calls
        result = original(*args)
        calls += 1
        if calls == 2 * len(TRAIN_DAYS):
            path = paths[which]
            path.write_bytes(path.read_bytes() + b"\n ")
        return result

    monkeypatch.setattr(pipeline, "audited_inputs", mutate_after_last_raw_check)
    monkeypatch.setattr(pipeline, "_parse_day", lambda *args: pytest.fail("Revoked source parsed"))
    with pytest.raises(ValueError):
        pipeline.fit_flow_templates(root, manifest, root / f"outputs/late-{which}")


def test_loader_does_not_replace_missing_daily_rows_with_extra_rows_from_another_day(
    run_root, monkeypatch
):
    root, _ = run_root
    destination, state = fit_fake(run_root, monkeypatch)
    first = fake_parse_day(None, DAY)[0][0]
    duplicate_day = TrainingRow(
        DAY,
        FlowTemplate(f"{DAY}:{START + 6_100_000}", (), (), ()),
        first.features_by_delay,
    )
    last = fake_parse_day(None, TRAIN_DAYS[-1])[0][0]
    pool = fit_template_pool((first, duplicate_day, last))
    (destination / "rows.jsonl.gz").unlink()
    outputs, digest = pipeline._write_pool(pool, destination)
    state.update(
        outputs=outputs,
        row_content_sha256=digest,
        row_identity_sha256=pool.row_identity_sha256,
    )
    write_json(destination / "manifest.json", state)
    with pytest.raises(ValueError, match="daily common cohorts"):
        pipeline.load_flow_templates(root, destination / "manifest.json")


def test_loader_rejects_rows_before_full_lag_context_even_if_training_day_is_valid():
    row = fake_parse_day(None, DAY)[0][0]
    value = pipeline._row_json(row)
    value["template"]["row_id"] = f"{DAY}:{START + 5_900_000}"
    with pytest.raises(ValueError, match="complete lag/print context"):
        pipeline._narrow_row(value)


@pytest.mark.parametrize("which", ["manifest", "prepared", "rows", "source"])
def test_loader_rechecks_gzip_and_receipts_before_any_model_values(
    run_root, monkeypatch, which
):
    root, _ = run_root
    destination, _ = fit_fake(run_root, monkeypatch)
    original = pipeline._gzip_rows
    paths = {
        "manifest": destination / "manifest.json",
        "prepared": destination / "prepared_receipt.json",
        "rows": destination / "rows.jsonl.gz",
        "source": root / "experiments/crypto_autonomous_flow.py",
    }

    def count_and_revoke(path):
        result = original(path)
        changed = paths[which]
        changed.write_bytes(changed.read_bytes() + b"\n ")
        return result

    monkeypatch.setattr(pipeline, "_gzip_rows", count_and_revoke)
    monkeypatch.setattr(pipeline, "_narrow_row", lambda *args: pytest.fail("Model values parsed"))
    with pytest.raises(ValueError):
        pipeline.load_flow_templates(root, destination / "manifest.json")
