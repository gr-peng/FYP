"""Train-only effective-flow extraction from verified publication-order prefixes.

No model fit or autonomous generation happens here. Sparse output retains raw
prefix references and all nonzero residual/consumption prices; unchanged levels
are reconstructible from the hash-bound raw archive, rather than copied per grid.
"""

import gzip
import hashlib
import json
import platform
import zipfile
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from behavioral_market.data.archive import write_json
from behavioral_market.data.bybit_book import BookIntegrityError

from .crypto_effective_flow import (
    DAY_US,
    DELAYS_MS,
    GRID_US,
    TRAIN_DAYS,
    BookObservation,
    FlowEvidence,
    ObservedPrint,
    StateFeatures,
    extract_interval,
    state_features,
)
from .crypto_execution_diagnostic import (
    AUDIT_SOURCES,
    EXECUTION_SOURCES,
    SPLITS,
    PublishedBookCursor,
    _trade_rows,
    audited_inputs,
    day_start_us,
    frozen_identity,
    sha256,
)

CONFIG_PATH = "config/crypto_effective_flow_extraction_20261013.json"
SOURCE_PATH = "config/crypto_microstructure_20261011.json"
SOURCE_SHA = "e056498b7296591841d64ea4700ed1bb4fdf821ead965c943e0ee3520a361184"
READER_SOURCES = (
    "experiments/crypto_effective_flow_reader.py",
    "experiments/crypto_effective_flow.py",
    "scripts/extract_crypto_effective_flow.py",
    *EXECUTION_SOURCES,
)


def expected_protocol():
    return {
        "study_id": "crypto_effective_flow_extraction_20261013",
        "source_config": SOURCE_PATH,
        "source_config_sha256": SOURCE_SHA,
        "allowed_splits": ["train"],
        "days": list(TRAIN_DAYS),
        "grid_us": GRID_US,
        "delays_ms": list(DELAYS_MS),
        "max_book_age_us": 1_000_000,
        "past_print_pressure_us": 1_000_000,
        "common_delay_cohort": True,
        "require_same_signal_label_segment_and_invalid_count": True,
        "snapshot_policy": ("Every valid raw snapshot, including an exact duplicate, "
                            "is a reset boundary"),
        "net_event_order": "bid descending then ask ascending; raw print order preserved",
        "smoke": {"day": "2024-11-01", "start_second_utc": 600, "end_second_utc": 900},
        "priming": "Consume from UTC day start; no prior-day or future state initialization",
        "evidence": "Sparse changed/consumed levels plus hash-bound raw prefix references",
        "interpretation": ("Effective net residuals, not gross orders "
                           "or identified parent intentions"),
    }


class FlowBookCursor(PublishedBookCursor):
    """Reuse the reviewed pending-message clock; raw snapshots are explicit resets."""

    def advance(self, timestamp_us):
        if timestamp_us < self.last_time:
            raise ValueError("Observation grids must be nondecreasing")
        self.last_time = timestamp_us
        while True:
            self._peek()
            if self.pending is None or self.pending[3] >= timestamp_us:
                break
            index, message, prefix, available = self.pending
            self.pending = None
            self.prefix_ts, self.available_us = prefix, available
            self.message_index = index
            try:
                update = self.book.apply(message)
                # BybitBook calls an identical repeat 'duplicate'. Its raw kind
                # remains a snapshot, so do not silently cross its clock reset.
                if message["type"] == "snapshot":
                    self.segment += 1
                    self.segment_start_us = update.engine_ms * 1000
            except BookIntegrityError:
                self.invalid_messages += 1

    def observation(self, timestamp_us):
        self.advance(timestamp_us)
        valid = self.book.valid
        update = self.book.last_update
        return BookObservation(
            timestamp_us,
            tuple(self.book.levels("bid")) if valid else (),
            tuple(self.book.levels("ask")) if valid else (),
            valid,
            self.segment if self.segment >= 0 else None,
            self.segment_start_us,
            self.message_index,
            update.engine_ms * 1000 if update is not None else None,
            self.available_us if self.message_index is not None else None,
            self.prefix_ts,
            self.invalid_messages,
        )


@dataclass(frozen=True, slots=True)
class FlowCandidate:
    evidence: FlowEvidence
    signal_observations: tuple[BookObservation | None, ...]
    features_by_delay: tuple[StateFeatures | None, ...]
    exclusion_reasons: tuple[str, ...]

    @property
    def eligible(self):
        return not self.exclusion_reasons


def flow_candidates(messages, prints, day, *, emit_start_us, emit_end_us):
    """One prefix cursor, a 5s book ring and a 6s tape ring; no hidden future feature."""
    start = day_start_us(day)
    if (day not in TRAIN_DAYS or emit_start_us % GRID_US or emit_end_us % GRID_US
            or not start <= emit_start_us < emit_end_us <= start + DAY_US):
        raise ValueError("Only aligned approved train-day extraction is allowed")
    cursor = FlowBookCursor(messages, day, 0)
    observations = deque(maxlen=51)
    tape = deque()
    prints = iter(prints)
    def next_print():
        value = next(prints, None)
        if value is not None and type(value) is not ObservedPrint:
            raise ValueError("Raw tape reader must produce exact immutable ObservedPrint")
        return value

    pending = next_print()
    previous_row, previous_time = 0, -1
    current = cursor.observation(start)
    observations.append(current)
    for a in range(start, emit_end_us, GRID_US):
        b = a + GRID_US
        signals, features, reasons = [], [], []
        # Capture all inputs before consuming this interval's label prints/book.
        for delay in DELAYS_MS:
            signal_time = a - delay * 1000
            offset = delay * 1000 // GRID_US
            signal = observations[-1 - offset] if len(observations) > offset else None
            signals.append(signal)
            if signal is None or signal_time - 1_000_000 < start:
                features.append(None)
                reasons.append(f"delay_{delay}_incomplete_past_context")
                continue
            if signal.observed_at_us != signal_time:
                raise RuntimeError("Observer ring failed exact lag-grid identity")
            if (signal.segment_id != current.segment_id
                    or signal.invalid_count != current.invalid_count):
                features.append(None)
                reasons.append(f"delay_{delay}_segment_or_error_crossing")
                continue
            past = tuple(t for t in tape if signal_time - 1_000_000 <= t.timestamp_us < signal_time)
            try:
                features.append(state_features(signal, past))
            except ValueError:
                features.append(None)
                reasons.append(f"delay_{delay}_invalid_or_stale_signal")
        interval = []
        while pending is not None and pending.timestamp_us < b:
            if (pending.timestamp_us < previous_time or pending.source_row <= previous_row
                    or not start <= pending.timestamp_us < start + DAY_US):
                raise ValueError("Tape identity/order/day differs from audited source")
            previous_time, previous_row = pending.timestamp_us, pending.source_row
            if pending.timestamp_us < a:
                raise ValueError("Tape reader reassigned a print across a grid boundary")
            interval.append(pending)
            tape.append(pending)
            pending = next_print()
        next_observation = cursor.observation(b)
        evidence = extract_interval(day, current, next_observation, tuple(interval))
        if a >= emit_start_us:
            yield FlowCandidate(evidence, tuple(signals), tuple(features),
                                (*evidence.exclusion_reasons, *reasons))
        current = next_observation
        observations.append(current)
        while tape and tape[0].timestamp_us < b - 6_000_000:
            tape.popleft()


def _observation_reference(observation):
    if observation is None:
        return None
    result = {name: getattr(observation, name) for name in (
        "observed_at_us", "valid", "segment_id", "segment_start_us", "message_index",
        "engine_us", "available_us", "publication_prefix_ms", "invalid_count",
    )}
    for name, levels in (("bid", observation.bids), ("ask", observation.asks)):
        result[f"{name}_l1"] = list(levels[0]) if levels else None
        result[f"{name}_outer_price_units"] = levels[-1][0] if levels else None
        result[f"{name}_visible_levels"] = len(levels)
    return result


def candidate_record(candidate):
    e = candidate.evidence
    return {
        "day": e.day,
        "grid_us": [e.start.observed_at_us, e.end.observed_at_us],
        "eligible": candidate.eligible,
        "exclusion_reasons": list(candidate.exclusion_reasons),
        "start_prefix": _observation_reference(e.start),
        "end_prefix": _observation_reference(e.end),
        "signal_prefixes": [_observation_reference(s) for s in candidate.signal_observations],
        "features_by_delay": [None if f is None else {
            name: [getattr(f, name).numerator, getattr(f, name).denominator]
            for name in ("imbalance_l1", "spread_bps", "trade_pressure")
        } for f in candidate.features_by_delay],
        "observed_prints": [asdict(t) for t in e.prints],
        "print_alignment": [asdict(t) for t in e.alignment],
        "censored_endpoints": list(e.censored_endpoints),
        "effective_levels": [asdict(level) for level in e.levels
                             if level.residual_units or level.consumed_proxy_units],
        # Even an excluded lag candidate retains the label decomposition. A
        # later fit must require eligible=true, never mere template existence.
        "template": asdict(e.template) if e.template is not None else None,
    }


def _read_prints(handle, day, end):
    for trade in _trade_rows(handle, day, day_start_us(day), end):
        yield ObservedPrint(trade.timestamp_us, trade.side, trade.quantity_units,
                            trade.price_units, trade.source_row, trade.trade_id)


def _write_day(inputs, output, day, window):
    destination = output / f"{day}_effective_flow.jsonl.gz"
    temporary = destination.with_suffix(".gz.tmp")
    rows, eligible, print_count, quantity = 0, 0, 0, 0
    exclusions, alignment = Counter(), Counter()
    try:
        with zipfile.ZipFile(inputs["orderbook"]["path"]) as archive:
            names = archive.namelist()
            if names != [f"{day}_BTCUSDT_ob500.data"]:
                raise ValueError("Unexpected audited LOB archive member")
            with archive.open(names[0]) as book_handle, gzip.open(
                inputs["trades"]["path"], "rt", encoding="utf-8", newline="",
            ) as trade_handle, gzip.open(temporary, "wt", encoding="utf-8") as writer:
                candidates = flow_candidates(
                    map(json.loads, book_handle), _read_prints(trade_handle, day, window[1]), day,
                    emit_start_us=window[0], emit_end_us=window[1],
                )
                for candidate in candidates:
                    writer.write(json.dumps(candidate_record(candidate), separators=(",", ":"),
                                            allow_nan=False) + "\n")
                    rows += 1
                    eligible += candidate.eligible
                    exclusions.update(candidate.exclusion_reasons)
                    print_count += len(candidate.evidence.prints)
                    quantity += candidate.evidence.observed_quantity_units
                    alignment.update(a.domain for a in candidate.evidence.alignment)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if rows != (window[1] - window[0]) // GRID_US:
        raise ValueError("Extraction lost a candidate grid; no complete publication allowed")
    return {"path": str(destination), "sha256": sha256(destination), "rows": rows,
            "eligible_rows": eligible, "exclusions": dict(exclusions),
            "observed_prints": print_count, "observed_quantity_units": quantity,
            "print_alignment": dict(alignment)}


def run_extraction(root, output, *, split="train", mode="smoke", prepare_only=False):
    if split != "train" or mode not in {"smoke", "train"}:
        raise ValueError("Development/event access is closed; choose fixed smoke or train")
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    state = {"status": "running", "run_id": uuid4().hex, "outputs_published": False,
             "outputs": [], "split": split, "mode": mode,
             "runtime": {"python": platform.python_version(),
                         "python_implementation": platform.python_implementation()},
             "started_at": datetime.now(UTC).isoformat(), "completed_at": None}
    write_json(manifest, state)
    try:
        config = root / CONFIG_PATH
        plan_bytes = config.read_bytes()
        plan = json.loads(plan_bytes)
        if plan != expected_protocol():
            raise ValueError("Extraction plan differs from the fixed first-window protocol")
        source_path = root / SOURCE_PATH
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        if (hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA or source["splits"] != SPLITS
                or source["venue"] != "bybit" or source["market"] != "linear_usdt_perpetual"
                or source["symbol"] != "BTCUSDT" or source["book_depth"] != 500
                or source["price_storage_scale"] != 100
                or source["quantity_storage_scale"] != 1000):
            raise ValueError("Only the original frozen train source is allowed")
        identity = frozen_identity(root, [config, source_path,
                                         *[root / name for name in READER_SOURCES]])
        plan_hash = hashlib.sha256(plan_bytes).hexdigest()
        if (config.read_bytes() != plan_bytes or source_path.read_bytes() != source_bytes
                or identity["source_sha256"][CONFIG_PATH] != plan_hash
                or identity["source_sha256"][SOURCE_PATH] != SOURCE_SHA):
            raise ValueError("Parsed configuration changed during preflight")
        days = list(TRAIN_DAYS) if mode == "train" else [plan["smoke"]["day"]]
        audit_sources = {name: identity["source_sha256"][name] for name in AUDIT_SOURCES}
        inputs, windows = {}, {}
        for day in days:
            inputs[day] = audited_inputs(root, source, SOURCE_SHA, day, audit_sources)
            start = day_start_us(day)
            windows[day] = ([start, start + DAY_US] if mode == "train" else [
                start + plan["smoke"]["start_second_utc"] * 1_000_000,
                start + plan["smoke"]["end_second_utc"] * 1_000_000,
            ])
        # A later day's audit/raw hashing can outlive a configuration change or
        # revoke an earlier source receipt. Fence all identities again before
        # publishing prepared status or entering any market parser.
        if config.read_bytes() != plan_bytes or source_path.read_bytes() != source_bytes:
            raise ValueError("Parsed configuration changed during input verification")
        if frozen_identity(root, [root / p for p in identity["source_sha256"]])[
            "source_sha256"
        ] != identity["source_sha256"]:
            raise ValueError("Frozen source changed during input verification")
        for day in days:
            if audited_inputs(root, source, SOURCE_SHA, day, audit_sources) != inputs[day]:
                raise ValueError("Input receipt changed before market parsing")
        if config.read_bytes() != plan_bytes or source_path.read_bytes() != source_bytes:
            raise ValueError("Parsed configuration changed during final input verification")
        if frozen_identity(root, [root / p for p in identity["source_sha256"]])[
            "source_sha256"
        ] != identity["source_sha256"]:
            raise ValueError("Frozen source changed before market parsing")
        state.update(**identity, config_sha256=plan_hash, config=plan,
                     source_config_sha256=SOURCE_SHA, inputs=inputs, days=days, windows_us=windows)
        write_json(output / "prepared_receipt.json", state)
        if prepare_only:
            state["status"] = "prepared_no_market_rows_read"
        else:
            for day in days:
                state["outputs"].append(_write_day(inputs[day], output, day, windows[day]))
                write_json(manifest, state)
            # Revalidate bytes, current audit and metadata, not just earlier stat.
            for day in days:
                if audited_inputs(root, source, SOURCE_SHA, day, audit_sources) != inputs[day]:
                    raise ValueError("Source evidence changed during extraction")
            if frozen_identity(root, [root / p for p in identity["source_sha256"]])[
                "source_sha256"
            ] != identity["source_sha256"]:
                raise ValueError("Frozen extraction source changed during processing")
            for record in state["outputs"]:
                if sha256(record["path"]) != record["sha256"]:
                    raise ValueError("Published output bytes changed before completion")
            state.update(status="complete", outputs_published=True)
        state["completed_at"] = datetime.now(UTC).isoformat()
        write_json(manifest, state)
        return state
    except BaseException as error:
        state.update(status="failed", outputs_published=False, error=repr(error),
                     completed_at=datetime.now(UTC).isoformat())
        write_json(manifest, state)
        raise
