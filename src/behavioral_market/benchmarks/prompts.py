import json

from .schemas import Choices13kItem

PROMPT_VERSION = "choices13k-v1"


def format_number(value):
    return f"{value:g}"


def format_option(outcomes, probabilities_visible=True):
    if probabilities_visible:
        return [
            f"- {format_number(probability * 100)}% chance of receiving {format_number(payoff)}"
            for probability, payoff in outcomes
        ]
    return [
        "- receiving one of these payoffs, with fixed but undisclosed probabilities: "
        + ", ".join(format_number(payoff) for _, payoff in outcomes)
    ]


def build_prompt(item: Choices13kItem, prompt_text, orientation):
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
            f"Option {display_label}:\n" + "\n".join(format_option(outcomes, probabilities_visible))
        )
    system = (
        "You are participating in a behavioral decision experiment. "
        "Choose one displayed option. Return exactly one JSON object and no explanation."
    )
    user = (
        f"{prompt_text}\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON only with exactly one field named choice whose value is A or B."
    )
    return system, user, display


def prompt_registry(path):
    value = json.loads(path.read_text())
    version = value.pop("version")
    return version, value
