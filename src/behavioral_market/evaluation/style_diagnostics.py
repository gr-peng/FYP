"""Describe archived strategy choices without treating them as human/causal evidence."""

import math
from collections import defaultdict
from datetime import date, datetime

STYLES = {"fundamental", "technical"}
FEATURES = ("loss", "proxy_better", "other_style_majority")
LLM_TREATMENTS = {"vanilla_llm", "low_herding", "high_herding"}
LLM_REVIEW_STATUSES = {
    "ok",
    "inconsistent_style",
    "invalid_json",
    "invalid_json_or_schema",
    "transport_exhausted",
}


def validate_review_status(status, treatment):
    """A programmed choice and a model response have different authorities."""
    if treatment == "rule_abm":
        allowed = {"rule"}
    elif treatment in LLM_TREATMENTS:
        allowed = LLM_REVIEW_STATUSES
    else:
        raise ValueError("Unknown review treatment")
    if not isinstance(status, str) or status not in allowed:
        raise ValueError("Review status does not belong to this treatment")


def bind_archived_answer(record, request_hash):
    """Require the script's unique raw-answer match before publishing a valid AI choice."""
    if request_hash is not None and (
        not isinstance(request_hash, str)
        or len(request_hash) != 64
        or any(c not in "0123456789abcdef" for c in request_hash)
    ):
        raise ValueError("Invalid raw answer request identity")
    if record["authority"] == "programmed_rule":
        if request_hash is not None:
            raise ValueError("A programmed choice cannot carry an AI answer identity")
    elif record["authority"] == "archived_llm":
        if record["valid_choice"] and request_hash is None:
            raise ValueError("A valid AI choice requires a matched archived answer")
    else:
        raise ValueError("Unknown review authority")
    return {**record, "raw_answer_request_hash": request_hash}


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or invalid {name}")
    return float(value)


def context_features(row, context):
    """Use only the archived context for this exact post-close review."""
    state, observation = context["agent_state"], context["observation"]
    current = date.fromisoformat(row["session_date"])
    if observation["session_date"] != row["session_date"]:
        raise ValueError("Context belongs to another review date")
    if state["persona"]["agent_id"] != row["agent_id"]:
        raise ValueError("Context belongs to another agent")
    if state["current_style"] != row["from_style"]:
        raise ValueError("Predecision style differs from recorded origin")
    if observation["phase"] != "post_close_style_review":
        raise ValueError("Expected a post-close review context")
    timestamp = datetime.fromisoformat(observation["decision_at_utc"])
    if timestamp.tzinfo is None or timestamp.date() != current:
        raise ValueError("Review clock must be aware and match the session")
    if date.fromisoformat(observation["history_last_session"]) != current:
        raise ValueError("Review must use the completed current session, not a future one")
    if any(date.fromisoformat(item["session"]) > current for item in state["memory"]):
        raise ValueError("Future memory in review context")
    population = observation["mean_field"]
    if date.fromisoformat(population["as_of_session"]) >= current:
        raise ValueError("Population information must precede the review session")
    actual = finite(state["counterfactual"]["actual_block_return"], "actual return")
    proxy = finite(state["counterfactual"]["alternative_block_return"], "proxy return")
    for name, value in [("actual_block_return", actual), ("counterfactual_block_return", proxy)]:
        if not math.isclose(value, finite(row[name], name), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Published return differs from decision-time context")
    gap = finite(state["counterfactual"]["gap"], "proxy gap")
    if not math.isclose(gap, proxy - actual, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Inconsistent proxy gap")
    shares = {s: finite(population[s + "_share"], "style share") for s in STYLES}
    if any(not 0 <= value <= 1 for value in shares.values()) or not math.isclose(
        sum(shares.values()),
        1,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Invalid population style shares")
    own = shares[row["from_style"]]
    other = shares[(STYLES - {row["from_style"]}).pop()]
    majority = None if math.isclose(own, other, abs_tol=1e-12) else other > own
    price = finite(observation["last_close"], "price")
    anchor = finite(observation["fundamental_value"], "value anchor")
    if price <= 0 or anchor <= 0:
        raise ValueError("Expected positive price and value anchor")
    return {
        "other_style_majority": majority,
        "population_tied": majority is None,
        "population_as_of": population["as_of_session"],
        "valuation_gap": anchor / price - 1,
        "valuation_basis": observation.get("fundamental_method", "not_recorded"),
        "proxy_method": state["counterfactual"]["method"],
        "context_verified": True,
    }


def normalize_review(row, run_id, seed, treatment, context=None):
    """Retain invalid/fallback outcomes; never count a corrected stay as a valid choice."""
    validate_review_status(row["validation_status"], treatment)
    if row["from_style"] not in STYLES or row["target_style"] not in STYLES:
        raise ValueError("Unknown strategy style")
    if row["decision"] not in {"stay", "switch"}:
        raise ValueError("Unknown strategy decision")
    actual = finite(row["actual_block_return"], "actual return")
    proxy = finite(row["counterfactual_block_return"], "proxy return")
    coherent = (row["decision"] == "stay") == (row["from_style"] == row["target_style"])
    valid = row["validation_status"] in {"ok", "rule"} and coherent
    result = {
        "run_id": run_id,
        "seed": seed,
        "treatment": treatment,
        "agent_id": row["agent_id"],
        "review_date": row["session_date"],
        "from_style": row["from_style"],
        "decision": row["decision"],
        "target_style": row["target_style"],
        "validation_status": row["validation_status"],
        "valid_choice": valid,
        "switched": row["decision"] == "switch",
        "actual_block_return": actual,
        "proxy_block_return": proxy,
        "loss": actual < 0,
        "proxy_better": proxy > actual,
        "other_style_majority": None,
        "valuation_gap": None,
        "context_verified": False,
        "authority": "programmed_rule" if treatment == "rule_abm" else "archived_llm",
    }
    if context is not None:
        result.update(context_features(row, context))
    elif treatment != "rule_abm":
        raise ValueError("LLM review requires an authenticated archived context")
    return result


def condition_counts(records, feature):
    """Report opportunities and ties/missingness explicitly; undefined rates are null."""
    if feature not in FEATURES:
        raise ValueError("Unknown diagnostic feature")
    cells = {True: [], False: []}
    invalid = unavailable = 0
    for row in records:
        if not row["valid_choice"]:
            invalid += 1
            continue
        value = row[feature]
        if value is None:
            unavailable += 1
        elif type(value) is bool:
            cells[value].append(row["switched"])
        else:
            raise ValueError("Diagnostic conditions must be boolean or unavailable")
    answer = {
        "total_opportunities": len(records),
        "invalid_choices": invalid,
        "condition_unavailable_or_tied": unavailable,
    }
    for value, label in [(True, "condition_true"), (False, "condition_false")]:
        n = len(cells[value])
        answer[label] = {
            "n": n,
            "switches": sum(cells[value]),
            "switch_rate": sum(cells[value]) / n if n else None,
        }
    a, b = answer["condition_true"]["switch_rate"], answer["condition_false"]["switch_rate"]
    answer["rate_difference"] = a - b if a is not None and b is not None else None
    return answer


def summarize_reviews(records, expected_runs):
    """Keep the run/seed unit; never pool repeated agent reviews into independent people."""
    if not records:
        raise ValueError("No reviews")
    by_run = defaultdict(list)
    seen = set()
    for row in records:
        key = row["run_id"], row["agent_id"], row["review_date"]
        if key in seen:
            raise ValueError("Duplicate strategy-review identity")
        seen.add(key)
        by_run[row["run_id"]].append(row)
    if set(by_run) != set(expected_runs):
        raise ValueError("Missing or extra fixed primary runs")
    per_run = []
    for run_id, rows in sorted(by_run.items()):
        if len(rows) != expected_runs[run_id]:
            raise ValueError("Missing review opportunities; do not average a survivor subset")
        valid = [r for r in rows if r["valid_choice"]]
        per_run.append(
            {
                "run_id": run_id,
                "treatment": rows[0]["treatment"],
                "seed": rows[0]["seed"],
                "opportunities": len(rows),
                "valid_choices": len(valid),
                "invalid_choices": len(rows) - len(valid),
                "switches": sum(r["switched"] for r in valid),
                "switch_rate": sum(r["switched"] for r in valid) / len(valid) if valid else None,
                "conditions": {name: condition_counts(rows, name) for name in FEATURES},
            }
        )
    groups = []
    for treatment in sorted({r["treatment"] for r in records}):
        rows = [r for r in records if r["treatment"] == treatment]
        runs = [r for r in per_run if r["treatment"] == treatment]
        rates = [r["switch_rate"] for r in runs]
        conditions = {}
        for name in FEATURES:
            counts = condition_counts(rows, name)
            differences = [r["conditions"][name]["rate_difference"] for r in runs]
            usable = [value for value in differences if value is not None]
            blocks = defaultdict(list)
            for row in rows:
                blocks[row["run_id"], row["review_date"], row["from_style"]].append(row)
            comparable = sum(
                condition_counts(block, name)["rate_difference"] is not None
                for block in blocks.values()
            )
            counts.update(
                runs_with_both_conditions=len(usable),
                expected_runs=len(runs),
                equal_run_mean_difference=(sum(usable) / len(usable))
                if len(usable) == len(runs)
                else None,
                same_run_date_style_blocks=len(blocks),
                same_run_date_style_blocks_with_both_conditions=comparable,
            )
            conditions[name] = counts
        groups.append(
            {
                "treatment": treatment,
                "runs": len(runs),
                "opportunities": len(rows),
                "valid_choices": sum(r["valid_choices"] for r in runs),
                "invalid_choices": sum(r["invalid_choices"] for r in runs),
                "switches": sum(r["switches"] for r in runs),
                "equal_run_switch_rate": sum(rates) / len(rates)
                if all(value is not None for value in rates)
                else None,
                "conditions": conditions,
            }
        )
    llm = [r for r in records if r["authority"] == "archived_llm"]
    valuation = []
    for day in sorted({r["review_date"] for r in llm}):
        values = {r["valuation_gap"] for r in llm if r["review_date"] == day}
        values.discard(None)
        valuation.append(
            {
                "review_date": day,
                "distinct_values": len(values),
                "values": sorted(values),
                "varies_within_review_date": len(values) > 1,
            }
        )
    dated = []
    for treatment, day in sorted({(r["treatment"], r["review_date"]) for r in records}):
        rows = [r for r in records if r["treatment"] == treatment and r["review_date"] == day]
        valid = [r for r in rows if r["valid_choice"]]
        dated.append(
            {
                "treatment": treatment,
                "review_date": day,
                "opportunities": len(rows),
                "valid_choices": len(valid),
                "loss_count": sum(r["loss"] for r in valid),
                "switches": sum(r["switched"] for r in valid),
                "conditions": {name: condition_counts(rows, name) for name in FEATURES},
            }
        )
    return {
        "total_opportunities": len(records),
        "runs": len(by_run),
        "per_run": per_run,
        "per_treatment": groups,
        "valuation_by_review_date": valuation,
        "by_review_date": dated,
        "limits": [
            "Retrospective descriptive associations, not randomized factor effects.",
            "Repeated agent reviews are not independent participants or human evidence.",
            "The alternative return is a frictionless signal proxy, not an executable account.",
            "The value anchor is a synthetic reference, not an estimated true firm value.",
            "No new confidence intervals, significance tests or scientific success threshold.",
        ],
        "human_alignment_established": False,
        "causal_effect_established": False,
    }
