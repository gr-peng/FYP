"""Exact payoff diagnostics and a fixed three-condition, four-format comparison."""

import math
import re
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

from behavioral_market.evaluation.style_mechanism_design import canonical, digest

from . import presentation_comparison as panel_tool
from .reference_comparison import CONTROLS, ENDPOINT, INTRO, MODEL

PROTOCOL = "tradeoff-reference-comparison-v1"
TARGET = 166
REFERENCES = (211, 384, 542)
METHODS = ("plain", "true_reference", "shuffled_reference")
ORDERS = (METHODS, METHODS[1:] + METHODS[:1], METHODS[2:] + METHODS[:2], METHODS)
PAIRINGS = {"true_reference": REFERENCES, "shuffled_reference": (384, 542, 211)}
TEXT_FIELDS = {"item_id", "option_a_text", "option_b_text"}
COMPARISONS = (("true_reference", "plain"), ("true_reference", "shuffled_reference"))
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
LINE = re.compile(rf"- ({NUMBER})% chance of receiving ({NUMBER})")
require = panel_tool.require


def parse_option(text):
    """Parse every complete line, preserving exact decimal probabilities/payoffs."""
    require(isinstance(text, str) and bool(text), "nonempty option text")
    outcomes, total = defaultdict(Fraction), Fraction(0)
    for line in text.splitlines():
        matched = LINE.fullmatch(line)
        require(matched is not None, "unsupported or incomplete probability/payoff line")
        probability, payoff = Fraction(matched[1]) / 100, Fraction(matched[2])
        require(0 <= probability <= 1, "probability outside[0,100]")
        total += probability
        outcomes[payoff] += probability
    require(total == 1, "probabilities must sum exactly100%")
    return tuple(sorted((payoff, p) for payoff, p in outcomes.items() if p))


def text_row(row, expected_id=None, extra=()):
    require(isinstance(row, dict) and set(row) == TEXT_FIELDS | set(extra), "text-only schema")
    require(type(row["item_id"]) is int and row["item_id"] >= 0, "strict integer item ID")
    require(expected_id is None or row["item_id"] == expected_id, "fixed item differs")
    require(
        all(isinstance(row[k], str) and row[k] for k in TEXT_FIELDS - {"item_id"}), "option text"
    )


def select_tradeoff(rows):
    """Select using only IDs and complete original text, never human/model outcomes."""
    require(isinstance(rows, list) and bool(rows), "nonempty candidate rows")
    ids, diagnostics = set(), []
    for row in rows:
        text_row(row)
        item = row["item_id"]
        require(item not in ids, "duplicate candidate")
        ids.add(item)
        a, b = (parse_option(row[k]) for k in ("option_a_text", "option_b_text"))
        grid = sorted({x for x, _ in a + b})
        differences, points = [], []
        for x in grid:
            fa = sum((p for payoff, p in a if payoff <= x), Fraction(0))
            fb = sum((p for payoff, p in b if payoff <= x), Fraction(0))
            differences.append(fa - fb)
            points.append(
                {"payoff": str(x), "CDF_A": str(fa), "CDF_B": str(fb), "A_minus_B": str(fa - fb)}
            )
        positive, negative = any(x > 0 for x in differences), any(x < 0 for x in differences)
        classification = (
            "crossing"
            if positive and negative
            else "B_distribution_dominates"
            if positive
            else "A_distribution_dominates"
            if negative
            else "equal_distributions"
        )
        diagnostics.append(
            {
                "item_id": item,
                "classification": classification,
                "eligible": positive and negative,
                "strict_outcome_A_better": min(x for x, _ in a) > max(x for x, _ in b),
                "strict_outcome_B_better": min(x for x, _ in b) > max(x for x, _ in a),
                "support_A": [[str(x), str(p)] for x, p in a],
                "support_B": [[str(x), str(p)] for x, p in b],
                "CDF_points": points,
            }
        )
    eligible = sorted(d["item_id"] for d in diagnostics if d["eligible"])
    return {
        "selected_item_id": eligible[0] if eligible else None,
        "eligible_item_ids": eligible,
        "items": sorted(diagnostics, key=lambda d: d["item_id"]),
        "limit": "Crossing distributions show tradeoffs; subjective difficulty is unproven.",
    }


def checked_materials(materials):
    require(
        isinstance(materials, dict) and set(materials) == {"target", "references"},
        "materials schema",
    )
    text_row(materials["target"], TARGET)
    refs = materials["references"]
    require(isinstance(refs, list) and len(refs) == 3, "three fixed references")
    target_pair = tuple(materials["target"][k] for k in ("option_a_text", "option_b_text"))
    for row, item in zip(refs, REFERENCES, strict=True):
        text_row(row, item, ("human_b_rate",))
        rate = row["human_b_rate"]
        require(
            type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1, "reference rate"
        )
        pair = tuple(row[k] for k in ("option_a_text", "option_b_text"))
        require(pair != target_pair and pair[::-1] != target_pair, "target/reference duplicate")
    require(len({r["human_b_rate"] for r in refs}) > 1, "identical reference conditions")
    candidates = [{k: row[k] for k in TEXT_FIELDS} for row in [materials["target"], *refs]]
    diagnostic = select_tradeoff(candidates)
    require(diagnostic["selected_item_id"] == TARGET, "fixed diagnostic target differs")
    return deepcopy(materials), diagnostic


def reference_cards(materials, method):
    materials, _ = checked_materials(materials)
    require(method in METHODS, "unknown method")
    if method == "plain":
        return []
    rates = {r["item_id"]: r["human_b_rate"] for r in materials["references"]}
    return [
        {
            "option_a_text": row["option_a_text"],
            "option_b_text": row["option_b_text"],
            "assigned_rate": rates[source],
        }
        for row, source in zip(materials["references"], PAIRINGS[method], strict=True)
    ]


def _context(cards):
    return INTRO + "".join(
        f"\n\nReference {i}:\nOriginal Option A:\n{row['option_a_text']}"
        f"\n\nOriginal Option B:\n{row['option_b_text']}"
        f"\n\nFraction choosing original Option B: {canonical(row['assigned_rate'])}"
        for i, row in enumerate(cards, 1)
    )


def make_packet(materials, run_id, evidence_kind="synthetic_tool_test"):
    materials, diagnostic = checked_materials(materials)
    require(isinstance(run_id, str) and bool(run_id.strip()), "run ID")
    require(evidence_kind in {"synthetic_tool_test", "recorded_model_logs"}, "evidence kind")
    target = materials["target"]
    panel = panel_tool.make_panel(
        {TARGET: (target["option_a_text"], target["option_b_text"])}, list(METHODS), 2
    )
    presentations = {
        (r["content_order"], r["letter_order"]): r["presentation"] for r in panel["opportunities"]
    }
    requests = []
    for (content, letters), order in zip(panel_tool.FORMATS, ORDERS, strict=True):
        for repeat in (0, 1):
            for method in order if repeat == 0 else order[::-1]:
                common = presentations[content, letters]
                payload = {**deepcopy(CONTROLS), "messages": deepcopy(common["messages"])}
                if method != "plain":
                    payload["messages"][1]["content"] = (
                        _context(reference_cards(materials, method))
                        + "\n\nNew question:\n"
                        + payload["messages"][1]["content"]
                    )
                identity = {
                    "run_id": run_id,
                    "evidence_kind": evidence_kind,
                    "slot_id": f"slot_{len(requests) + 1:02d}",
                    "position": len(requests) + 1,
                    "item_id": TARGET,
                    "method_id": method,
                    "content_order": content,
                    "letter_order": letters,
                    "repeat_id": repeat,
                    "wire_sha256": digest(payload),
                    "presentation_sha256": digest(common),
                }
                requests.append(
                    {
                        **identity,
                        "request_hash": digest([payload, identity, PROTOCOL]),
                        "payload": payload,
                    }
                )
    core = {
        "protocol": PROTOCOL,
        "run_id": run_id,
        "evidence_kind": evidence_kind,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "materials": materials,
        "diagnostic": diagnostic,
        "panel": panel,
        "requests": requests,
        "comparisons": [
            {"left": a, "right": b, "difference": "left_minus_right_absolute_error"}
            for a, b in COMPARISONS
        ],
    }
    return {**core, "packet_sha256": digest(core)}


def validate_packet(packet):
    rebuilt = make_packet(packet["materials"], packet["run_id"], packet["evidence_kind"])
    require(canonical(packet) == canonical(rebuilt), "packet differs from rebuilt materials")
    return rebuilt


def evaluate(packet, requests, attempts, responses, targets):
    from behavioral_market.evaluation.matched_choice_collection import authenticate_logs

    packet = validate_packet(packet)
    authenticated = authenticate_logs(packet, requests, attempts, responses)
    comparisons = {}
    for left, right in COMPARISONS:
        comparisons[f"{left}_vs_{right}"] = panel_tool.compare_panel(
            packet["panel"],
            authenticated["adapted"],
            targets,
            left_method=left,
            right_method=right,
        )
    return {
        "evidence_kind": packet["evidence_kind"],
        **{k: v for k, v in authenticated.items() if k != "adapted"},
        "comparisons": comparisons,
        "used_completion_POSTs": 91 + len(attempts),
        "overall_human_alignment_established": False,
        "limit": "One post-result exploratory calibration target. Neither independent generation, "
        "personal memory learning nor general human alignment or market realism is established.",
    }
