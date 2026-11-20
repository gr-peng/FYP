"""Post-result format sensitivity from already accepted, complete derived choices.

This re-computes choices and arithmetic. It does not authenticate HTTP or remote
generation anew and has no raw-log, dataset, credential or network interface.
"""

import math
from collections import defaultdict
from fractions import Fraction

from behavioral_market.benchmarks import tradeoff_comparison as study
from behavioral_market.evaluation.matched_choice_collection import identity

TOLERANCE = 1e-12
EXTRA_ROW_FIELDS = {
    "compatible",
    "status",
    "choice",
    "returned_mode",
    "cache_key_header",
    "strong_signature",
    "body_sha256",
}
require = study.require


def _number(value, expected):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and math.isclose(value, float(expected), rel_tol=0, abs_tol=TOLERANCE),
        "accepted primary arithmetic differs",
    )


def _pair(left, right, a, b):
    difference = a - b
    return {
        "left_method": left,
        "right_method": right,
        "direction": "left_minus_right",
        "left_absolute_error": float(a),
        "right_absolute_error": float(b),
        "absolute_error_difference": float(difference),
        "absolute_error_difference_exact": str(difference),
    }


def analyze(packet, target, evaluation):
    packet = study.validate_packet(packet)
    require(
        isinstance(target, dict)
        and set(target) == {"item_id", "human_b_rate"}
        and type(target["item_id"]) is int
        and target["item_id"] == study.TARGET,
        "local target schema",
    )
    value = target["human_b_rate"]
    require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
        "overall target rate",
    )
    q = Fraction(str(value))
    require(
        evaluation["evidence_kind"] == packet["evidence_kind"]
        and evaluation["protocol_complete"] is True
        and evaluation["local_logs_match_frozen_packet"] is True,
        "accepted complete evidence required",
    )
    expected = {row["slot_id"]: row for row in packet["requests"]}
    require(
        type(evaluation["logged_dispatch_attempts"]) is int
        and evaluation["logged_dispatch_attempts"] == len(expected) == 24,
        "complete dispatch count",
    )
    require(
        isinstance(evaluation["rows"], list) and len(evaluation["rows"]) == len(expected),
        "complete24 choice rows required",
    )
    common = {
        study.panel_tool.key(row): row["presentation"] for row in packet["panel"]["opportunities"]
    }
    observed = {}
    for row in evaluation["rows"]:
        require(isinstance(row, dict), "derived row schema")
        sid = row.get("slot_id")
        require(sid in expected and sid not in observed, "foreign or duplicate choice slot")
        base = identity(expected[sid])
        require(set(row) == set(base) | EXTRA_ROW_FIELDS, "derived row fields")
        require(
            all(type(row.get(k)) is type(v) and row[k] == v for k, v in base.items()),
            "choice identity differs",
        )
        require(
            row["compatible"] is True
            and row["status"] == "compatible"
            and type(row["choice"]) is str
            and row["choice"] in ("A", "B")
            and row["returned_mode"] == "no_returned_reasoning_observed",
            "valid accepted choice required",
        )
        observed[sid] = row
    require(set(observed) == set(expected), "missing choice slots")
    reconstructed, grouped = [], defaultdict(list)
    for sid, request in expected.items():
        original = common[study.panel_tool.key(request)]["mapping"][observed[sid]["choice"]]
        row = {
            **identity(request),
            "choice": observed[sid]["choice"],
            "original_choice": original["original_option"],
            "original_position": original["position"],
        }
        reconstructed.append(row)
        grouped[request["method_id"], request["content_order"], request["letter_order"]].append(row)
    pooled, lookup, cells = [], {}, []
    for method in study.METHODS:
        all_rows = [row for row in reconstructed if row["method_id"] == method]
        require(len(all_rows) == 8, "eight planned responses per method")
        count = sum(r["original_choice"] == "B" for r in all_rows)
        rate = Fraction(count, 8)
        error = abs(rate - q)
        pooled.append(
            {
                "method_id": method,
                "planned_records": 8,
                "received_records": 8,
                "original_B_count": count,
                "pooled_original_B_rate": float(rate),
                "pooled_original_B_rate_exact": str(rate),
                "absolute_error": float(error),
                "absolute_error_exact": str(error),
            }
        )
        lookup[method] = error
        for content, letters in study.panel_tool.FORMATS:
            group = sorted(grouped[method, content, letters], key=lambda r: r["repeat_id"])
            require(
                len(group) == 2 and [r["repeat_id"] for r in group] == [0, 1],
                "complete repeat pair",
            )
            n = sum(r["original_choice"] == "B" for r in group)
            rate = Fraction(n, 2)
            cells.append(
                {
                    "method_id": method,
                    "content_order": content,
                    "letter_order": letters,
                    "planned_records": 2,
                    "received_records": 2,
                    "slot_ids": [r["slot_id"] for r in group],
                    "display_choices": [r["choice"] for r in group],
                    "original_choices": [r["original_choice"] for r in group],
                    "original_B_count": n,
                    "original_B_rate": float(rate),
                    "original_B_rate_exact": str(rate),
                    "overall_human_reference": float(q),
                    "format_specific_human_rate": None,
                    "signed_deviation_from_overall": float(rate - q),
                    "absolute_deviation_from_overall": float(abs(rate - q)),
                    "same_input_disagreement": group[0]["original_choice"]
                    != group[1]["original_choice"],
                }
            )
    wanted_names = {f"{left}_vs_{right}" for left, right in study.COMPARISONS}
    require(set(evaluation["comparisons"]) == wanted_names, "two fixed comparisons required")
    primary = []
    for left, right in study.COMPARISONS:
        accepted = evaluation["comparisons"][f"{left}_vs_{right}"]
        require(
            accepted["left_method"] == left
            and accepted["right_method"] == right
            and accepted["difference_direction"] == "left_minus_right",
            "primary direction differs",
        )
        require(
            all(
                type(accepted[k]) is int and accepted[k] == v
                for k, v in {
                    "planned_records": 24,
                    "received_records": 24,
                    "valid_records": 24,
                    "invalid_records": 0,
                    "missing_records": 0,
                }.items()
            ),
            "accepted comparison counts",
        )
        p = accepted["primary"]
        require(
            type(p["planned_items"]) is int
            and p["planned_items"] == 1
            and type(p["complete_pair_items"]) is int
            and p["complete_pair_items"] == 1
            and p["comparison_complete"] is True,
            "accepted primary completeness",
        )
        for field, number in {
            "left_MAE": lookup[left],
            "right_MAE": lookup[right],
            "left_minus_right_MAE": lookup[left] - lookup[right],
        }.items():
            _number(p[field], number)
        primary.append(
            {
                **_pair(left, right, lookup[left], lookup[right]),
                "prespecified_pooled_metric": True,
                "matches_accepted_primary": True,
            }
        )
    cell_lookup = {(c["method_id"], c["content_order"], c["letter_order"]): c for c in cells}
    paired = []
    for content, letters in study.panel_tool.FORMATS:
        for left, right in study.COMPARISONS:
            a, b = (
                Fraction(cell_lookup[m, content, letters]["original_B_count"], 2)
                for m in (left, right)
            )
            diff = abs(a - q) - abs(b - q)
            paired.append(
                {
                    "content_order": content,
                    "letter_order": letters,
                    **_pair(left, right, abs(a - q), abs(b - q)),
                    "relative_to_same_overall_reference": True,
                    "format_specific_human_rate": None,
                    "description": "closer" if diff < 0 else "farther" if diff > 0 else "same",
                }
            )
    summaries = [
        {
            "left_method": left,
            "right_method": right,
            **{
                name: sum(
                    r["left_method"] == left
                    and r["right_method"] == right
                    and r["description"] == name
                    for r in paired
                )
                for name in ("closer", "same", "farther")
            },
        }
        for left, right in study.COMPARISONS
    ]
    return {
        "analysis_kind": "post_result_format_sensitivity",
        "new_model_POSTs": 0,
        "source_evidence_kind": packet["evidence_kind"],
        "target_item_id": study.TARGET,
        "overall_human_B_rate": float(q),
        "overall_human_B_rate_exact": str(q),
        "format_specific_human_rates_available": False,
        "planned_records": 24,
        "received_records": 24,
        "reconstructed_rows": reconstructed,
        "pooled_methods": pooled,
        "primary_comparisons": primary,
        "format_method_cells": cells,
        "posthoc_format_pairs": paired,
        "posthoc_format_counts": summaries,
        "same_input_disagreement_cells": sum(c["same_input_disagreement"] for c in cells),
        "same_input_cells": 12,
        "independent_generation_count": None,
        "provenance_boundary": "Derived-choice arithmetic only;no new remote authentication.",
        "limit": "Same overall human proportion is used only as a sensitivity reference. "
        "No format-specific human measurements,significance or general human-alignment conclusion; "
        "the prespecified pooled primary metric is retained.",
    }
