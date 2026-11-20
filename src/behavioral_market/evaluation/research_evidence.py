"""Conservative interpretation of recorded research evidence, without running studies.

Execution, publication and scientific support are deliberately separate questions.
These helpers interpret supplied records; they do not independently authenticate
human labels, identify causality, or authorize an experiment.
"""

import math
from collections.abc import Mapping


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def execution_status(*, planned, entered, returned, verified, exceptions):
    """Classify actual counts, never a top-level success flag or absence of errors."""
    values = {name: _count(value, name) for name, value in locals().copy().items()}
    if not values["planned"]:
        raise ValueError("planned must be positive")
    if not 0 <= verified <= returned <= entered <= planned:
        raise ValueError("inconsistent execution denominators")
    if exceptions > entered - returned:
        raise ValueError("exceptions exceed entries without a return")
    if not entered:
        state = "not_started"
    elif exceptions or returned != entered or verified != returned:
        state = "incomplete_or_failed"
    elif verified < planned:
        state = "partial"
    else:
        state = "complete"
    return {
        "state": state,
        **values,
        "not_entered": planned - entered,
        "scientific_success_inferred": False,
    }


def error_comparison(candidate, reference, interval=None):
    """Lower error is better; an interval touching zero establishes no direction.

    Intervals must already have been computed by the original study. This does
    not resample observations or reinterpret failure to reject as equivalence.
    """
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (candidate, reference)
    ):
        raise ValueError("errors must be finite nonnegative numbers")
    delta = candidate - reference
    direction = "lower" if delta < 0 else "higher" if delta > 0 else "equal"
    support = "not_assessed"
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("interval requires two bounds")
        low, high = interval
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in interval):
            raise ValueError("interval must be finite")
        if low > high:
            raise ValueError("reversed interval")
        if (high < 0 and delta >= 0) or (low > 0 and delta <= 0):
            raise ValueError("point estimate contradicts interval direction")
        support = "lower_error" if high < 0 else "higher_error" if low > 0 else "not_established"
    return {
        "candidate": candidate,
        "reference": reference,
        "difference": delta,
        "point_direction": direction,
        "interval": interval,
        "directional_support": support,
        "equivalence_inferred": False,
    }


def publication_status(manifest, actual_outputs):
    """Accept only explicitly published outputs with matching supplied identities.

    actual_outputs is an allowlisted path -> SHA256 map populated by the caller.
    Never follow arbitrary manifest paths to find/read private or sealed data.
    A historical 'running' flag is not evidence that its process is still alive.
    """
    if not isinstance(manifest, Mapping) or not isinstance(actual_outputs, Mapping):
        raise ValueError("manifest and actual_outputs must be mappings")
    if manifest.get("status") != "complete" or manifest.get("outputs_published") is not True:
        return {"state": "not_published", "live_process_status": "not_checked"}
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return {"state": "not_published", "reason": "no published output identities"}
    seen = set()
    for item in outputs:
        if not isinstance(item, Mapping) or not {"path", "sha256"} <= item.keys():
            return {"state": "identity_missing"}
        name, expected = item["path"], item["sha256"]
        if not isinstance(name, str) or name in seen:
            return {"state": "identity_missing"}
        seen.add(name)
        if not isinstance(expected, str) or len(expected) != 64:
            return {"state": "identity_missing"}
        if actual_outputs.get(name) != expected:
            return {"state": "identity_mismatch", "path": name}
    return {"state": "published", "verified_output_count": len(seen)}


def market_comparison_readiness(**requirements):
    """Check necessary declared links; 'ready' is not proof of a scientific claim."""
    names = {
        "human_financial_improvement",
        "independent_market_comparison",
        "same_policy",
        "same_market_rules",
        "autonomous_prices",
    }
    if set(requirements) != names or any(type(v) is not bool for v in requirements.values()):
        raise ValueError("supply all five boolean market-comparison requirements")
    missing = sorted(name for name, value in requirements.items() if not value)
    return {
        "ready_for_review": not missing,
        "missing": missing,
        "scientific_claim_proven": False,
    }
