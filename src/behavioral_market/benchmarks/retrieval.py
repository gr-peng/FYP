"""Leakage-safe demonstration generation and retrieval for Choices13k."""

import hashlib
import json
import math

import numpy as np

from .prompts import format_option

DATA_DRIVEN_TEMPLATE_VERSION = "choices13k-data-driven-v1"
SYSTEM_TEMPLATE = (
    "You are participating in a behavioral decision experiment. "
    "Choose one displayed option. Return exactly one JSON object and no explanation."
)
INTRO_TEMPLATE = (
    "Review the following choices made by individual human participants in earlier "
    "decision problems. Use them as behavioral examples, then make your own choice for "
    "the final problem."
)
TARGET_TEMPLATE = "Now make your own choice for this new problem."
OUTPUT_TEMPLATE = (
    "Return JSON only with exactly one field named choice whose value is A or B."
)


def _stable_number(*parts):
    value = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def pseudo_human_choices(item, count, seed):
    """Convert an empirical rate into a deterministic shuffled pseudo-participant sample."""
    b_choices = int(math.floor(item.human_b_rate * count + 0.5))
    values = ["B"] * b_choices + ["A"] * (count - b_choices)
    np.random.default_rng(_stable_number(seed, item.item_id, "pseudo-human")).shuffle(values)
    return values


def _payoff_scale(item):
    return max(abs(payoff) for _, payoff in [*item.option_a, *item.option_b])


def _probability_signature(item):
    visible = [probability for probability, _ in item.option_a]
    if not item.ambiguity:
        visible.extend(probability for probability, _ in item.option_b)
    return tuple(sorted(round(value, 8) for value in visible))


def observable_retrieval_features(item, ev_gap_bin):
    """Return features allowed by protocol; excludes all human response fields."""
    return {
        "has_loss": item.has_loss,
        "ambiguity": item.ambiguity,
        "ev_gap_bin": str(ev_gap_bin),
        "payoff_scale": _payoff_scale(item),
        "probability_signature": _probability_signature(item),
        "outcome_count": len(item.option_a) + len(item.option_b),
    }


def select_demonstrations(calibration_items, target, ev_gap_bins, k, strategy, seed):
    """Select demonstration IDs without consulting the target or pool human rates."""
    if k <= 0 or k > len(calibration_items):
        raise ValueError("invalid demonstration count")
    if strategy == "random":
        ranked = sorted(
            calibration_items,
            key=lambda item: _stable_number(seed, target.item_id, item.item_id, "random"),
        )
    elif strategy == "matched":
        target_features = observable_retrieval_features(target, ev_gap_bins[target.item_id])

        def score(item):
            features = observable_retrieval_features(item, ev_gap_bins[item.item_id])
            categorical = sum(
                features[key] != target_features[key]
                for key in ("has_loss", "ambiguity", "ev_gap_bin")
            )
            scale_distance = abs(
                math.log1p(features["payoff_scale"])
                - math.log1p(target_features["payoff_scale"])
            )
            return (
                categorical,
                features["ambiguity"] != target_features["ambiguity"],
                features["has_loss"] != target_features["has_loss"],
                features["ev_gap_bin"] != target_features["ev_gap_bin"],
                abs(features["outcome_count"] - target_features["outcome_count"]),
                features["probability_signature"] != target_features["probability_signature"],
                scale_distance,
                _stable_number(seed, target.item_id, item.item_id, "matched-tie"),
            )

        ranked = sorted(calibration_items, key=score)
    else:
        raise ValueError(f"unsupported retrieval strategy {strategy!r}")
    return ranked[:k]


def _display_item(item, orientation):
    if orientation not in {"A_first", "B_first"}:
        raise ValueError("invalid option orientation")
    originals = [
        ("A", item.option_a, True),
        ("B", item.option_b, not item.ambiguity),
    ]
    if orientation == "B_first":
        originals.reverse()
    display = {}
    blocks = []
    for display_label, (original_label, outcomes, probabilities_visible) in zip(
        ("A", "B"), originals, strict=True
    ):
        display[display_label] = original_label
        blocks.append(
            f"Option {display_label}:\n"
            + "\n".join(format_option(outcomes, probabilities_visible))
        )
    return "\n\n".join(blocks), display


def build_few_shot_prompt(target, demonstrations, orientation, pseudo_count, seed):
    """Build a prompt with calibration labels and no target response metadata."""
    examples = []
    for index, item in enumerate(demonstrations, start=1):
        demo_orientation = (
            "A_first"
            if _stable_number(seed, item.item_id, "demo-orientation") % 2 == 0
            else "B_first"
        )
        gamble, display = _display_item(item, demo_orientation)
        original_to_display = {original: shown for shown, original in display.items()}
        choices = pseudo_human_choices(item, pseudo_count, seed)
        choice_lines = [
            f"Human participant {participant}: {original_to_display[choice]}"
            for participant, choice in enumerate(choices, start=1)
        ]
        examples.append(
            f"Earlier problem {index}:\n{gamble}\n\n" + "\n".join(choice_lines)
        )
    target_gamble, target_display = _display_item(target, orientation)
    user = (
        INTRO_TEMPLATE
        + "\n\n"
        + "\n\n".join(examples)
        + f"\n\n{TARGET_TEMPLATE}\n{target_gamble}\n\n{OUTPUT_TEMPLATE}"
    )
    return SYSTEM_TEMPLATE, user, target_display


def prompt_template_hash(pseudo_count):
    value = {
        "version": DATA_DRIVEN_TEMPLATE_VERSION,
        "system": SYSTEM_TEMPLATE,
        "intro": INTRO_TEMPLATE,
        "target": TARGET_TEMPLATE,
        "output": OUTPUT_TEMPLATE,
        "pseudo_participants_per_item": pseudo_count,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
