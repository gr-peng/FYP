"""Pure four-format rendering and complete-cohort comparison of supplied choices.

This module neither sends model requests nor authenticates their provenance. Method
specific references belong outside the common displayed question panel.
"""

import json
import math
from collections import defaultdict
from copy import deepcopy
from itertools import product
from statistics import fmean

from .prompts import format_option

VERSION = "matched-choice-panel-v1"
FORMATS = tuple(product(("AB", "BA"), repeat=2))
KEY_FIELDS = ("item_id", "method_id", "content_order", "letter_order", "repeat_id")
SYSTEM = (
    "You are participating in a behavioral decision experiment. "
    "Choose one displayed option. Return exactly one JSON object and no explanation."
)
SUFFIX = "\n\nReturn JSON only with exactly one field named choice whose value is A or B."


def require(value, message):
    if not value:
        raise ValueError(message)


def same(left, right):
    """Compare JSON structures without bool/int or float/int equality aliases."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def render_presentations(
    option_a, option_b, prompt_text="Choose the option you prefer.", system_text=SYSTEM
):
    require(
        all(isinstance(x, str) for x in (option_a, option_b, prompt_text, system_text)),
        "question text must be strings",
    )
    require(
        bool(option_a.strip()) and bool(option_b.strip()), "two nonempty original blocks required"
    )
    originals, result = {"A": option_a, "B": option_b}, []
    for content, letters in FORMATS:
        mapping = {
            label: {"original_option": original, "position": i + 1}
            for i, (original, label) in enumerate(zip(content, letters, strict=True))
        }
        blocks = [
            f"Option {label}:\n{originals[original]}"
            for original, label in zip(content, letters, strict=True)
        ]
        result.append(
            {
                "content_order": content,
                "letter_order": letters,
                "mapping": mapping,
                "messages": [
                    {"role": "system", "content": system_text},
                    {
                        "role": "user",
                        "content": prompt_text + "\n\n" + "\n\n".join(blocks) + SUFFIX,
                    },
                ],
            }
        )
    return result


def render_item(item, prompt_text="Choose the option you prefer."):
    """Legacy item adapter; accesses only option text inputs, never a dataset loader."""
    a = "\n".join(format_option(item.option_a, True))
    b = "\n".join(format_option(item.option_b, not item.ambiguity))
    return render_presentations(a, b, prompt_text)


def key(row):
    item, method, content, letters, repeat = (row[k] for k in KEY_FIELDS)
    require(type(item) is int and item >= 0, "strict nonnegative integer item ID required")
    require(
        isinstance(method, str) and bool(method.strip()) and method == method.strip(),
        "nonempty method ID required",
    )
    require(
        type(content) is str and type(letters) is str and (content, letters) in FORMATS,
        "unknown presentation format",
    )
    require(type(repeat) is int and repeat >= 0, "strict nonnegative repeat ID required")
    return item, method, content, letters, repeat


def make_panel(
    items, methods, repeats, *, prompt_text="Choose the option you prefer.", system_text=SYSTEM
):
    """Bind shared item text to a complete item×method×format×repeat plan.

    items maps integer item IDs to a pair of original option text blocks. Method IDs
    identify conditions; actual reference prompts and transport need a later adapter.
    """
    require(isinstance(items, dict) and bool(items), "nonempty shared items required")
    require(all(type(i) is int and i >= 0 for i in items), "strict integer item IDs required")
    require(
        isinstance(methods, (list, tuple))
        and len(methods) >= 2
        and all(isinstance(m, str) and m.strip() and m == m.strip() for m in methods)
        and len(set(methods)) == len(methods),
        "at least two unique method IDs required",
    )
    require(type(repeats) is int and repeats > 0, "positive integer repeat count required")
    sources, opportunities = [], []
    for item in sorted(items):
        blocks = items[item]
        require(
            isinstance(blocks, (list, tuple)) and len(blocks) == 2, "two shared blocks required"
        )
        panels = render_presentations(*blocks, prompt_text=prompt_text, system_text=system_text)
        sources.append({"item_id": item, "option_a_text": blocks[0], "option_b_text": blocks[1]})
        for presentation in panels:
            for repeat in range(repeats):
                for method in sorted(methods):
                    opportunities.append(
                        {
                            "item_id": item,
                            "method_id": method,
                            "content_order": presentation["content_order"],
                            "letter_order": presentation["letter_order"],
                            "repeat_id": repeat,
                            "presentation": deepcopy(presentation),
                        }
                    )
    return {
        "version": VERSION,
        "items": sources,
        "methods": sorted(methods),
        "repeats": repeats,
        "prompt_text": prompt_text,
        "system_text": system_text,
        "opportunities": opportunities,
    }


def validate_panel(panel):
    require(
        isinstance(panel, dict)
        and set(panel)
        == {"version", "items", "methods", "repeats", "prompt_text", "system_text", "opportunities"}
        and panel["version"] == VERSION,
        "panel schema/version differs",
    )
    items = {}
    for row in panel["items"]:
        require(set(row) == {"item_id", "option_a_text", "option_b_text"}, "shared item schema")
        item = row["item_id"]
        require(
            type(item) is int and item >= 0 and item not in items, "duplicate or invalid shared ID"
        )
        items[item] = (row["option_a_text"], row["option_b_text"])
    expected = make_panel(
        items,
        panel["methods"],
        panel["repeats"],
        prompt_text=panel["prompt_text"],
        system_text=panel["system_text"],
    )
    actual, wanted = {}, {key(row): row for row in expected["opportunities"]}
    for row in panel["opportunities"]:
        require(set(row) == set(KEY_FIELDS) | {"presentation"}, "opportunity schema differs")
        k = key(row)
        require(k not in actual, "duplicate planned key")
        actual[k] = row
    require(set(actual) == set(wanted), "unmatched planned items/methods/formats/repeats")
    require(all(same(actual[k], wanted[k]) for k in wanted), "planned content or mapping changed")
    return expected


def _received(panel, records):
    expected = {key(row): row for row in panel["opportunities"]}
    observed = {}
    for row in records:
        require(
            isinstance(row, dict)
            and set(row) == set(KEY_FIELDS) | {"presentation", "status", "choice"},
            "received schema differs",
        )
        k = key(row)
        require(k in expected and k not in observed, "foreign or duplicate received key")
        require(
            same(row["presentation"], expected[k]["presentation"]),
            "received question text or mapping differs",
        )
        require(
            row["status"] in ("valid", "invalid")
            and (row["choice"] is None or isinstance(row["choice"], str)),
            "received status/choice type",
        )
        observed[k] = row
    result = []
    for k, planned in expected.items():
        row = observed.get(k)
        valid = row is not None and row["status"] == "valid" and row["choice"] in ("A", "B")
        selected = planned["presentation"]["mapping"][row["choice"]] if valid else None
        binfo = next(
            x for x in planned["presentation"]["mapping"].items() if x[1]["original_option"] == "B"
        )
        result.append(
            {
                **deepcopy(planned),
                "received": row is not None,
                "status": "valid" if valid else "invalid" if row else "missing",
                "reported_status": row["status"] if row else None,
                "choice": row["choice"] if row else None,
                "selected_original": selected["original_option"] if selected else None,
                "selected_position": selected["position"] if selected else None,
                "original_B_position": binfo[1]["position"],
                "original_B_letter": binfo[0],
            }
        )
    return result


def validate_received(panel, records):
    return _received(validate_panel(panel), records)


def _contrasts(rates):
    result = []
    for letter in ("A", "B"):
        a, b = rates[1, letter], rates[2, letter]
        result.append(
            {
                "factor": "position_first_minus_second",
                "fixed_original_B_letter": letter,
                "difference": a - b if a is not None and b is not None else None,
            }
        )
    for position in (1, 2):
        a, b = rates[position, "B"], rates[position, "A"]
        result.append(
            {
                "factor": "letter_B_minus_A",
                "fixed_original_B_position": position,
                "difference": a - b if a is not None and b is not None else None,
            }
        )
    return result


def _mean_known(values):
    return fmean(values) if values and all(x is not None for x in values) else None


def compare_panel(
    panel, records, targets, *, left_method, right_method, evidence_kind="unverified_inputs"
):
    """Compare explicit left minus right over the whole preplanned item cohort.

    targets is one complete shared item→human-B-rate mapping. Missing targets or
    malformed plans are errors; missing/invalid received choices remain unknown.
    """
    plan = validate_panel(panel)
    require(
        left_method != right_method
        and left_method in plan["methods"]
        and right_method in plan["methods"],
        "two explicit planned methods required",
    )
    require(
        evidence_kind in {"unverified_inputs", "synthetic_tool_test"}, "no provenance attestation"
    )
    ids = [row["item_id"] for row in plan["items"]]
    require(
        isinstance(targets, dict)
        and all(type(i) is int for i in targets)
        and set(targets) == set(ids),
        "one complete shared target mapping required",
    )
    require(
        all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in targets.values()),
        "targets must be finite nonbool numeric[0,1]",
    )
    rows = _received(plan, records)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["item_id"], row["method_id"], row["content_order"], row["letter_order"]].append(
            row
        )
    item_methods, lookup = [], {}
    for item in ids:
        for method in plan["methods"]:
            cells, rates = [], {}
            for content, letters in FORMATS:
                group = grouped[item, method, content, letters]
                complete = (
                    len(group) == plan["repeats"]
                    and sum(r["status"] == "valid" for r in group) == plan["repeats"]
                )
                rate = (
                    sum(r["selected_original"] == "B" for r in group) / plan["repeats"]
                    if complete
                    else None
                )
                pos, letter = group[0]["original_B_position"], group[0]["original_B_letter"]
                rates[pos, letter] = rate
                cells.append(
                    {
                        "content_order": content,
                        "letter_order": letters,
                        "original_B_position": pos,
                        "original_B_letter": letter,
                        "planned_records": plan["repeats"],
                        "received_records": sum(r["received"] for r in group),
                        "valid_records": sum(r["status"] == "valid" for r in group),
                        "display_choices": [r["choice"] for r in group],
                        "original_choices": [r["selected_original"] for r in group],
                        "received_original_B_rate": rate,
                    }
                )
            pooled = _mean_known(list(rates.values()))
            contrasts = _contrasts(rates)
            entry = {
                "item_id": item,
                "method_id": method,
                "human_b_rate": targets[item],
                "planned_records": 4 * plan["repeats"],
                "cells": cells,
                "contrasts": contrasts,
                "pooled_original_B_rate": pooled,
                "absolute_error": abs(pooled - targets[item]) if pooled is not None else None,
            }
            for factor, edges in (("position", contrasts[:2]), ("letter", contrasts[2:])):
                entry[f"mean_signed_{factor}_difference"] = _mean_known(
                    [e["difference"] for e in edges]
                )
                entry[f"mean_absolute_{factor}_difference"] = _mean_known(
                    [abs(e["difference"]) if e["difference"] is not None else None for e in edges]
                )
            item_methods.append(entry)
            lookup[item, method] = entry
    pairs = []
    for item in ids:
        left, right = lookup[item, left_method], lookup[item, right_method]
        complete = left["absolute_error"] is not None and right["absolute_error"] is not None
        pairs.append(
            {
                "item_id": item,
                "human_b_rate": targets[item],
                "left_original_B_rate": left["pooled_original_B_rate"],
                "right_original_B_rate": right["pooled_original_B_rate"],
                "left_absolute_error": left["absolute_error"],
                "right_absolute_error": right["absolute_error"],
                "complete_pair": complete,
                "left_minus_right_absolute_error": left["absolute_error"] - right["absolute_error"]
                if complete
                else None,
            }
        )
    methods = []
    for method in plan["methods"]:
        subset = [lookup[item, method] for item in ids]
        summary = {
            "method_id": method,
            "planned_items": len(ids),
            "complete_items": sum(r["pooled_original_B_rate"] is not None for r in subset),
            "MAE": _mean_known([r["absolute_error"] for r in subset]),
        }
        for factor, kind in product(("position", "letter"), ("signed", "absolute")):
            field = f"mean_{kind}_{factor}_difference"
            summary[field] = _mean_known([r[field] for r in subset])
        methods.append(summary)
    complete = sum(p["complete_pair"] for p in pairs)
    return {
        "evidence_kind": evidence_kind,
        "methods": plan["methods"],
        "left_method": left_method,
        "right_method": right_method,
        "difference_direction": "left_minus_right",
        "planned_records": len(rows),
        "received_records": sum(r["received"] for r in rows),
        "valid_records": sum(r["status"] == "valid" for r in rows),
        "invalid_records": sum(r["status"] == "invalid" for r in rows),
        "missing_records": sum(r["status"] == "missing" for r in rows),
        "panel_complete": all(r["status"] == "valid" for r in rows),
        "rows": rows,
        "item_methods": item_methods,
        "item_pairs": pairs,
        "method_summaries": methods,
        "primary": {
            "planned_items": len(ids),
            "complete_pair_items": complete,
            "comparison_complete": complete == len(ids),
            "left_MAE": _mean_known([p["left_absolute_error"] for p in pairs])
            if complete == len(ids)
            else None,
            "right_MAE": _mean_known([p["right_absolute_error"] for p in pairs])
            if complete == len(ids)
            else None,
            "left_minus_right_MAE": _mean_known(
                [p["left_minus_right_absolute_error"] for p in pairs]
            ),
        },
        "model_calls_made_by_tool": 0,
        "limit": "Supplied-record comparison only;does not attest actual sending,independence "
        "or causal effects or human-reference effectiveness. References need a separate adapter.",
    }
