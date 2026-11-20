import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from experiments import crypto_autonomous_target_pipeline as pipe
from experiments.crypto_autonomous_flow import FeatureContext, InitialBook
from experiments.crypto_autonomous_metrics import BlockSummary
from experiments.crypto_autonomous_targets import (
    DayTargets,
    ObservedPriceBlock,
    WindowInitialization,
)
from experiments.crypto_effective_flow import StateFeatures

ROOT = Path(__file__).resolve().parents[1]


def synthetic_day(day):
    start = pipe.day_start_us(day)
    labels = tuple(BlockSummary(start // 1000 + i * 60_000,
                                (i / 10, i / 1000, i, i / 100, i / 9, i / 7,
                                 (i % 3 - 1) / 2),
                                ("missing_prefix",) if i == 0 else (),
                                0 if i == 0 else 600, 1.0, 1.0)
                   for i in range(1440))
    book = InitialBook(((9999, 1000),), ((10001, 2000),))
    features = StateFeatures(Fraction(-1, 3), Fraction(2), Fraction(0))
    inits = []
    for hour in range(24):
        at = start + hour * 3_600_000_000 + 600_000_000
        context = tuple(FeatureContext(t, features)
                        for t in range(at - 5_000_000, at + 1, 100_000))
        inits.append(WindowInitialization(at, book, (), context, (), ()))
    blocks = tuple(ObservedPriceBlock(start + i * 300_000_000,
                                      tuple(b.values[0] for b in labels[i * 5:i * 5 + 5]),
                                      ("minute_0:missing_prefix",) if i == 0 else ())
                   for i in range(288))
    return DayTargets(day, (start, start + pipe.DAY_US), labels, tuple(inits), blocks, 0, 0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for name in pipe.TARGET_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    def identity(root, paths):
        return {"git_commit": "synthetic-test-only", "source_sha256": {
            str(Path(p).relative_to(root)): pipe.sha256(p) for p in paths}}
    monkeypatch.setattr(pipe, "frozen_identity", identity)
    monkeypatch.setattr(pipe, "audited_inputs",
                        lambda root, source, digest, day, audit: {"day": day})
    def preflight(root, output, **kwargs):
        assert kwargs == {"split": "train", "mode": "train", "prepare_only": True}
        state = {"status": "prepared_no_market_rows_read", "days": list(pipe.TRAIN_DAYS),
                 "completed_at": "synthetic-time",
                 "outputs_published": False, "outputs": [],
                 "source_sha256": identity(root, [root / p for p in pipe.reader.READER_SOURCES])[
                     "source_sha256"],
                 "inputs": {day: {"day": day} for day in pipe.TRAIN_DAYS}}
        pipe.write_json(output / "manifest.json", state)
        pipe.write_json(output / "prepared_receipt.json",
                        dict(state, status="running", completed_at=None))
        return state
    monkeypatch.setattr(pipe.reader, "run_extraction", preflight)
    monkeypatch.setattr(pipe, "_read_day", lambda inputs, day: synthetic_day(day))
    return root, tmp_path / "output"


def test_all_candidates_retained_and_separate_generator_interface(setup):
    root, output = setup
    state = pipe.build_train_targets(root, output)
    assert state["status"] == "complete" and state["outputs_published"]
    assert len(state["outputs"]) == 10
    for day in pipe.TRAIN_DAYS:
        counts = state["day_counts"][day]
        assert counts["minute_candidates"] == 1440 and counts["complete_minutes"] == 1439
        assert counts["initialization_candidates"] == 24
        assert counts["price_block_candidates"] == 288 and counts["complete_price_blocks"] == 287
        narrow = json.loads((output / f"{day}_initializations.json").read_text())
        assert set(narrow) == {"day", "span_us", "split", "initializations"}
        init = narrow["initializations"][0]
        assert not {"labels", "returns_bps", "price_units", "observed_prints"} & set(init)
        assert init["feature_context"][0]["features"]["imbalance_l1"] == [-1, 3]
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["training_candidate_labels"] == 4320
    assert metrics["training_complete_labels"] == 4317
    for record in state["outputs"]:
        assert pipe.sha256(record["path"]) == record["sha256"]
    prepared = json.loads((output / "prepared_receipt.json").read_text())
    assert not prepared["outputs_published"] and prepared["outputs"] == []
    assert pipe.sha256(output / "prepared_receipt.json") == state["prepared_receipt_sha256"]


@pytest.mark.parametrize("split", ["development", "event_test", "unknown"])
def test_split_rejected_before_filesystem_access(tmp_path, split):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="closed"):
        pipe.build_train_targets(tmp_path / "missing", output, split=split)
    assert not output.exists()


def test_prepare_never_reads_market_or_fits_scales(setup, monkeypatch):
    root, output = setup
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    monkeypatch.setattr(pipe, "fit_metrics", lambda *args, **kwargs: pytest.fail("fit"))
    state = pipe.build_train_targets(root, output, prepare_only=True)
    assert state["status"] == "prepared_no_market_rows_read"
    assert not state["outputs_published"] and not state["outputs"]


def test_changed_config_during_preflight_fails_before_parse(setup, monkeypatch):
    root, output = setup
    previous = pipe.reader.run_extraction
    def corrupt(*args, **kwargs):
        result = previous(*args, **kwargs)
        path = root / pipe.CONFIG_PATH
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(pipe.reader, "run_extraction", corrupt)
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    with pytest.raises(ValueError, match="protocol changed"):
        pipe.build_train_targets(root, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_revoked_audit_never_enters_market_reader(setup, monkeypatch):
    root, output = setup
    monkeypatch.setattr(pipe, "audited_inputs", lambda *args: {"revoked": True})
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    with pytest.raises(ValueError, match="input/audit"):
        pipe.build_train_targets(root, output)


@pytest.mark.parametrize("filename", ["manifest.json", "prepared_receipt.json"])
@pytest.mark.parametrize("prepare_only", [False, True])
def test_changed_preflight_receipt_cannot_bind_new_unverified_bytes(
    setup, monkeypatch, filename, prepare_only,
):
    root, output = setup
    previous = pipe.reader.run_extraction
    def corrupt(*args, **kwargs):
        result = previous(*args, **kwargs)
        path = output / "source_preflight" / filename
        changed = json.loads(path.read_bytes())
        changed.update(status="failed", inputs={"revoked": True})
        pipe.write_json(path, changed)
        return result
    monkeypatch.setattr(pipe.reader, "run_extraction", corrupt)
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    with pytest.raises(ValueError, match="preflight receipt differs"):
        pipe.build_train_targets(root, output, prepare_only=prepare_only)
    assert not json.loads((output / "manifest.json").read_text())["outputs_published"]


def test_preflight_revocation_during_long_input_hashing_fails_before_parse(setup, monkeypatch):
    root, output = setup
    previous = pipe.audited_inputs
    def corrupt(*args):
        result = previous(*args)
        if args[3] == pipe.TRAIN_DAYS[-1]:
            path = output / "source_preflight/manifest.json"
            path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(pipe, "audited_inputs", corrupt)
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    with pytest.raises(ValueError, match="artifact changed"):
        pipe.build_train_targets(root, output)


def test_second_day_failure_keeps_existing_day_unpublished(setup, monkeypatch):
    root, output = setup
    def read(inputs, day):
        if day == pipe.TRAIN_DAYS[1]:
            raise ValueError("synthetic malformed raw day")
        return synthetic_day(day)
    monkeypatch.setattr(pipe, "_read_day", read)
    with pytest.raises(ValueError, match="malformed raw"):
        pipe.build_train_targets(root, output)
    state = json.loads((output / "manifest.json").read_text())
    assert state["status"] == "failed" and not state["outputs_published"]
    assert len(state["outputs"]) == 3


@pytest.mark.parametrize("prepare_only", [False, True])
def test_corrupted_own_prepared_write_rejected_before_parser(setup, monkeypatch, prepare_only):
    root, output = setup
    previous = pipe.write_json
    def corrupt(path, state):
        previous(path, state)
        if path == output / "prepared_receipt.json":
            previous(path, dict(state, status="failed"))
    monkeypatch.setattr(pipe, "write_json", corrupt)
    monkeypatch.setattr(pipe, "_read_day", lambda *args: pytest.fail("market parse"))
    with pytest.raises(ValueError, match="preflight receipt differs"):
        pipe.build_train_targets(root, output, prepare_only=prepare_only)
    assert not json.loads((output / "manifest.json").read_text())["outputs_published"]


@pytest.mark.parametrize("mutation_day", [0, 2])
def test_own_prepared_mutation_stops_next_parser_or_final_publication(
    setup, monkeypatch, mutation_day,
):
    root, output = setup
    days_read = []
    def read(inputs, day):
        days_read.append(day)
        if day == pipe.TRAIN_DAYS[mutation_day]:
            path = output / "prepared_receipt.json"
            path.write_bytes(path.read_bytes() + b"\n")
        return synthetic_day(day)
    monkeypatch.setattr(pipe, "_read_day", read)
    with pytest.raises(ValueError, match="artifact changed"):
        pipe.build_train_targets(root, output)
    assert days_read == list(pipe.TRAIN_DAYS[:mutation_day + 1])
    assert not json.loads((output / "manifest.json").read_text())["outputs_published"]


@pytest.mark.parametrize("field", ["minute_labels", "initializations", "price_blocks"])
def test_missing_calendar_candidate_fails_without_fitting(setup, monkeypatch, field):
    root, output = setup
    def read(inputs, day):
        result = synthetic_day(day)
        return replace(result, **{field: getattr(result, field)[1:]})
    monkeypatch.setattr(pipe, "_read_day", read)
    monkeypatch.setattr(pipe, "fit_metrics", lambda *args, **kwargs: pytest.fail("fit"))
    with pytest.raises(ValueError, match="calendar"):
        pipe.build_train_targets(root, output)


def test_post_write_output_mutation_prevents_complete_publication(setup, monkeypatch):
    root, output = setup
    previous = pipe._publish
    def corrupt(*args):
        result = previous(*args)
        if result["kind"] == "frozen_train_metrics":
            p = output / f"{pipe.TRAIN_DAYS[0]}_minute_labels.json"
            p.write_bytes(p.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(pipe, "_publish", corrupt)
    with pytest.raises(ValueError, match="artifact changed"):
        pipe.build_train_targets(root, output)
    assert not json.loads((output / "manifest.json").read_text())["outputs_published"]


def test_existing_output_not_overwritten(setup):
    root, output = setup
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        pipe.build_train_targets(root, output)
    assert sentinel.read_text() == "prior evidence"
