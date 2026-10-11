"""Archive integrity and one-second observed-state extraction, with no gap filling."""

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

from behavioral_market.data.archive import write_json
from behavioral_market.data.bybit_book import BookIntegrityError, BybitBook


def audit_book_file(path, expected_day, features_path=None, progress_path=None):
    path = Path(path)
    start = datetime.fromisoformat(expected_day).replace(tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((start + timedelta(days=1)).timestamp() * 1000)
    book = BybitBook()
    counts, errors = Counter(), Counter()
    examples, features, segments = [], [], []
    bucket, segment, segment_start = None, -1, None
    bucket_counts = Counter()
    crc_verified = False
    first_cts, last_cts = None, None
    same_version_snapshots = 0

    def flush():
        if features_path is not None and book.valid and bucket is not None:
            row = book.features()
            row.update(
                {
                    "bucket_ms": bucket,
                    "segment": segment,
                    "available_at_ms": max(bucket + 1000, row["ts_ms"]),
                    "state_age_at_bucket_end_ms": bucket + 1000 - row["cts_ms"],
                    **bucket_counts,
                }
            )
            features.append(row)

    def close_segment():
        nonlocal segment_start
        if segment_start is not None:
            segments.append(
                {
                    "segment": segment,
                    "first_cts_ms": segment_start,
                    "last_cts_ms": book.last_update.engine_ms,
                }
            )
            segment_start = None

    try:
        with ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) != 1 or members[0].filename != path.name.removesuffix(".zip"):
                raise BookIntegrityError("Expected exactly the named daily data member")
            with archive.open(members[0]) as source:
                for line_number, raw in enumerate(source, 1):
                    counts["messages"] += 1
                    try:
                        if len(raw) > 2**20:
                            raise BookIntegrityError("Oversized archive message")
                        message = json.loads(raw)
                        cts = message["cts"]
                        if type(cts) is not int or not start_ms <= cts < end_ms:
                            raise BookIntegrityError(
                                "Matching-engine timestamp outside expected day"
                            )
                        first_cts = cts if first_cts is None else min(first_cts, cts)
                        last_cts = cts if last_cts is None else max(last_cts, cts)
                        next_bucket = cts // 1000 * 1000
                        if next_bucket != bucket:
                            flush()
                            bucket, bucket_counts = (
                                next_bucket,
                                Counter(
                                    {
                                        "delta_messages": 0,
                                        "positive_visible_depth_change_units": 0,
                                        "negative_visible_depth_change_units": 0,
                                    }
                                ),
                            )
                        prior = book.last_update
                        comparable = (
                            message["type"] == "snapshot"
                            and book.valid
                            and prior is not None
                            and message["data"]["u"] == prior.update_id
                            and message["data"]["seq"] == prior.sequence
                        )
                        # A genuinely new snapshot begins a distinct known-state segment.
                        result = book.apply(message)
                        counts[result.kind] += 1
                        same_version_snapshots += int(comparable)
                        if result.kind == "snapshot":
                            # The previous segment ends before this snapshot, not at its new time.
                            if segment_start is not None:
                                segments.append(
                                    {
                                        "segment": segment,
                                        "first_cts_ms": segment_start,
                                        "last_cts_ms": prior.engine_ms,
                                    }
                                )
                            segment += 1
                            segment_start = result.engine_ms
                            bucket_counts.clear()
                        elif result.kind == "delta":
                            bucket_counts["delta_messages"] += 1
                            bucket_counts["positive_visible_depth_change_units"] += (
                                result.positive_depth_change_units
                            )
                            bucket_counts["negative_visible_depth_change_units"] += (
                                result.negative_depth_change_units
                            )
                    except (BookIntegrityError, ValueError, KeyError, TypeError) as exc:
                        close_segment()
                        book.valid = False
                        counts["invalid_messages"] += 1
                        errors[type(exc).__name__] += 1
                        if len(examples) < 20:
                            examples.append({"line": line_number, "error": str(exc)[:200]})
                    if progress_path is not None and line_number % 50000 == 0:
                        write_json(
                            progress_path,
                            {
                                "status": "running",
                                "file": path.name,
                                "counts": dict(counts),
                                "last_cts_ms": last_cts,
                            },
                        )
            # ZipExtFile verifies its central-directory CRC when read to EOF.
            crc_verified = True
        flush()
        close_segment()
    except (BadZipFile, EOFError, OSError, BookIntegrityError) as exc:
        errors[type(exc).__name__] += 1
        examples.append({"archive_error": str(exc)[:200]})
    result = {
        "file": str(path),
        "expected_day": expected_day,
        "counts": dict(counts),
        "errors": dict(errors),
        "error_examples": examples,
        "zip_crc_verified": crc_verified,
        "first_cts_ms": first_cts,
        "last_cts_ms": last_cts,
        "initial_uncovered_ms": None if first_cts is None else first_cts - start_ms,
        "ending_uncovered_ms": None if last_cts is None else end_ms - last_cts,
        "segments": segments,
        "same_version_snapshot_comparisons": same_version_snapshots,
        "independent_exchange_checksum": "not supplied by this archive",
        "no_hidden_exchange_gaps_proven": False,
        "structural_pass": crc_verified and not errors,
        "feature_rows": 0,
        "interpretation": "Only known-state segments; u-step check is conservative. "
        "Visible L2 changes are not identified order additions/cancellations. "
        "No counterfactual fills or endogenous-price validity established.",
    }
    if features_path is not None and crc_verified and features:
        features_path = Path(features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(features)
        for name in (
            "delta_messages",
            "positive_visible_depth_change_units",
            "negative_visible_depth_change_units",
        ):
            frame[name] = frame[name].fillna(0).astype("int64")
        # Exclude seconds straddling initial snapshots, gaps, resets or file endings.
        # A known state early in a second does not prove the whole second was observed.
        starts = {s["segment"]: s["first_cts_ms"] for s in segments}
        ends = {s["segment"]: s["last_cts_ms"] for s in segments}
        frame = frame[
            (frame.bucket_ms >= frame.segment.map(starts))
            & (frame.bucket_ms + 1000 <= frame.segment.map(ends))
        ]
        temporary = features_path.with_name(features_path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(features_path)
        result["feature_rows"] = len(frame)
        result["features"] = str(features_path)
    if progress_path is not None:
        write_json(progress_path, {"status": "complete", **result})
    return result
