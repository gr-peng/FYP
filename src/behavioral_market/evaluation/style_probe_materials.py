"""Prepare paired synthetic inputs; this module cannot call a model or simulate a market."""

import copy
import hashlib
import json
import re
from datetime import date

SHARES = ("fundamental_share", "technical_share")
STYLES = ("fundamental", "technical")
CONDITIONS = ("own_style_majority", "other_style_majority")
CASE_KEYS = {
    "case_id",
    "pair_id",
    "condition",
    "material_type",
    "source",
    "payload",
    "payload_sha256",
}


def payload_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def share_tokens(text, replacements=None):
    """Change exactly the two numeric tokens, preserving all other prompt bytes."""
    for name in SHARES:
        pattern = re.compile(
            r'("' + name + r'"\s*:\s*)'
            r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?(?=\s*[,}])"
        )
        value = "<share>" if replacements is None else str(replacements[name])
        text, count = pattern.subn(lambda match, value=value: match[1] + value, text)
        if count != 1:
            raise ValueError("Expected exactly one numeric token for each population share")
    return text


def user_content(payload):
    messages = payload["messages"]
    if [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Use only an initial system/user request, without repair or answers")
    if not all(isinstance(m["content"], str) for m in messages):
        raise ValueError("Prompt contents must be strings")
    return messages[1]["content"]


def validate_sources(sources):
    if len(sources) != 2 or {s["identity"]["from_style"] for s in sources} != set(STYLES):
        raise ValueError("Require one verified source for each initial strategy")
    if len({(s["identity"]["run_id"], s["identity"]["review_date"]) for s in sources}) != 1:
        raise ValueError("Source pairs must share their run and review date")
    if len({s["identity"]["agent_id"] for s in sources}) != 2:
        raise ValueError("Sources must be two distinct agents")
    for source in sources:
        origin = source["identity"]
        if origin["treatment"] != "vanilla_llm":
            raise ValueError("This minimal material set uses ordinary AI prompts only")
        for key in ("original_request_hash", "matched_answer_request_hash"):
            h = origin[key]
            if (
                not isinstance(h, str)
                or len(h) != 64
                or any(c not in "0123456789abcdef" for c in h)
            ):
                raise ValueError("Source must identify its verified request and answer")
        if origin["original_request_hash"] != origin["matched_answer_request_hash"]:
            raise ValueError("Use an initial request with a directly matched original answer")
        context = json.loads(user_content(source["payload"]))
        state, observation = context["agent_state"], context["observation"]
        if (state["persona"]["agent_id"], state["current_style"]) != (
            origin["agent_id"],
            origin["from_style"],
        ):
            raise ValueError("Source agent/strategy identity differs from its context")
        if observation["session_date"] != origin["review_date"]:
            raise ValueError("Source review date differs from its context")
        if observation["phase"] != "post_close_style_review":
            raise ValueError("Source is not a completed-session strategy review")
        if date.fromisoformat(observation["mean_field"]["as_of_session"]) >= date.fromisoformat(
            origin["review_date"]
        ):
            raise ValueError("Population summary must strictly precede the review")
        share_tokens(user_content(source["payload"]))


def build_share_probes(sources):
    """Construct two complete pairs without changing even other prompt whitespace."""
    validate_sources(sources)
    cases = []
    for source in sorted(sources, key=lambda s: s["identity"]["from_style"]):
        origin = source["identity"]
        pair_id = origin["agent_id"] + "_" + origin["review_date"]
        for condition in CONDITIONS:
            own = 0.8 if condition == "own_style_majority" else 0.2
            shares = {origin["from_style"] + "_share": own}
            other = (set(STYLES) - {origin["from_style"]}).pop()
            shares[other + "_share"] = 0.2 if own == 0.8 else 0.8
            payload = copy.deepcopy(source["payload"])
            payload["messages"][1]["content"] = share_tokens(user_content(payload), shares)
            cases.append(
                {
                    "case_id": pair_id + "_" + condition,
                    "pair_id": pair_id,
                    "condition": condition,
                    "material_type": "synthetic_population_share_input_only",
                    "source": copy.deepcopy(origin),
                    "payload": payload,
                    "payload_sha256": payload_digest(payload),
                }
            )
    validate_share_probes(cases, sources)
    return cases


def validate_share_probes(cases, sources):
    """Reject incomplete pairs, hidden changes, wrong clocks and invented result fields."""
    validate_sources(sources)
    if len(cases) != 4 or len({c["case_id"] for c in cases}) != 4:
        raise ValueError("Require exactly four unique inputs in two complete pairs")
    by_pair = {}
    for source in sources:
        origin = source["identity"]
        by_pair[origin["agent_id"] + "_" + origin["review_date"]] = source
    coverage = set()
    for case in cases:
        if (
            set(case) != CASE_KEYS
            or case["material_type"] != "synthetic_population_share_input_only"
        ):
            raise ValueError("Only input material fields are allowed; no model results")
        if case["pair_id"] not in by_pair or case["condition"] not in CONDITIONS:
            raise ValueError("Unknown pair or condition")
        key = case["pair_id"], case["condition"]
        if key in coverage or case["case_id"] != "_".join(key):
            raise ValueError("Duplicate or inconsistent pair condition")
        coverage.add(key)
        source = by_pair[case["pair_id"]]
        if case["source"] != source["identity"]:
            raise ValueError("Changed source identity")
        original, candidate = source["payload"], case["payload"]
        old_text, new_text = user_content(original), user_content(candidate)
        if share_tokens(old_text) != share_tokens(new_text):
            raise ValueError("Changed input beyond the two population share tokens")
        masked = copy.deepcopy(candidate)
        masked["messages"][1]["content"] = old_text
        if masked != original:
            raise ValueError("Changed system prompt, model settings or request envelope")
        context = json.loads(new_text)
        population = context["observation"]["mean_field"]
        shares = [population[name] for name in SHARES]
        if any(type(x) not in (int, float) or not 0 <= x <= 1 for x in shares):
            raise ValueError("Population shares must be finite proportions")
        if abs(sum(shares) - 1) > 1e-12:
            raise ValueError("Population shares must sum to one")
        own = population[case["source"]["from_style"] + "_share"]
        expected = 0.8 if case["condition"] == "own_style_majority" else 0.2
        if own != expected:
            raise ValueError("Pair must reverse the majority at the fixed 80/20 proportions")
        if case["payload_sha256"] != payload_digest(candidate):
            raise ValueError("Input payload fingerprint differs")
    if coverage != {(pair, condition) for pair in by_pair for condition in CONDITIONS}:
        raise ValueError("Missing one side of a complete pair")
    return {
        "pairs": 2,
        "input_cases": 4,
        "both_initial_strategies_present": True,
        "only_population_share_tokens_changed": True,
        "all_other_prompt_bytes_and_model_settings_unchanged": True,
        "population_clock_is_prior": True,
        "model_answers_present": 0,
        "execution_status": "materials_ready_not_run",
    }
