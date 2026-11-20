"""Validate already-verified derived bytes against saved aggregate receipts.

No file loading, model fitting or original-response authentication occurs here.
"""

from collections import Counter

import numpy as np

from experiments import probability_recalibration as calibration
from experiments.market_quote_baselines import metrics
from scripts.validate_memory_information import sha, strict_json

METHODS = {
    "constant_half",
    "demonstration_mean",
    "lifetime_history_prior4",
    "memory_ridge_logistic",
    "neutral_labels",
    "no_examples",
    "real_labels",
    "shuffled_labels",
    "training_constant",
}
TOLERANCE = 1e-10
require = calibration.require


def compare_metrics(actual, expected, name="metrics"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(), name + " keys differ"
        )
        return sum(
            compare_metrics(actual[k], value, name + "." + k) for k, value in expected.items()
        )
    require(type(expected) in (int, float), "unexpected archived metric type at " + name)
    require(type(actual) in (int, float), "unexpected computed metric type at " + name)
    require(np.isfinite(expected) and np.isfinite(actual), "nonfinite metric at " + name)
    if type(expected) is int:
        require(type(actual) is int and actual == expected, "count differs at " + name)
    else:
        require(abs(actual - expected) <= TOLERANCE, "archived metric differs at " + name)
    return 1


def admit(raw, expected_sha, preparation, analysis, development, *, count=400, quota=25):
    require(type(raw) is bytes, "decode only the immutable source bytes")
    require(sha(raw) == expected_sha, "derived table hash mismatch before decoding")
    table = strict_json(raw)
    require(isinstance(table, list) and len(table) == count, "unexpected table coverage")
    require(
        len(preparation["target_ids"]) == count
        and len(set(preparation["target_ids"])) == count
        and preparation["sessions"] == len(development),
        "unexpected archived cohort",
    )
    require(set(analysis["overall"]["methods"]) == METHODS, "unexpected archived method set")
    records = []
    predictions = {name: [] for name in sorted(METHODS)}
    for row in table:
        require(
            isinstance(row, dict)
            and set(row) == {"target_id", "provenance", "observed_buy", "predictions"},
            "unexpected derived row schema",
        )
        provenance = row["provenance"]
        require(isinstance(provenance, dict), "invalid provenance")
        record = {
            "target_id": row["target_id"],
            "session": provenance["session"],
            "participant": provenance["participant"],
            "order_id": provenance["order_id"],
            "observed_buy": row["observed_buy"],
            "probability_buy": row["predictions"]["no_examples"],
        }
        require(set(row["predictions"]) == METHODS, "unexpected saved method set")
        for name in predictions:
            value = row["predictions"][name]
            require(
                type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
                "invalid saved probability",
            )
            predictions[name].append(float(value))
        records.append(record)
    calibration.validate(records)
    require(
        [r["target_id"] for r in records] == preparation["target_ids"],
        "target identity or ordering differs from preparation",
    )
    counts = Counter(r["session"] for r in records)
    require(
        set(counts) == set(development) and all(n == quota for n in counts.values()),
        "development membership or per-session quota differs",
    )
    require(set(analysis["per_session"]) == set(development), "archived session set differs")
    scoring = calibration.scoring_records(records)
    predictions = {k: np.array(v) for k, v in predictions.items()}
    checked = compare_metrics(metrics(scoring, predictions), analysis["overall"], "overall")
    for session in sorted(development):
        ids = [i for i, r in enumerate(records) if r["session"] == session]
        value = metrics([scoring[i] for i in ids], {k: p[ids] for k, p in predictions.items()})
        checked += compare_metrics(value, analysis["per_session"][session], "session." + session)
    return records, {
        "rows": len(records),
        "sessions": len(counts),
        "checked_archived_methods": sorted(METHODS),
        "checked_metric_numbers": checked,
        "absolute_tolerance": TOLERANCE,
        "source_sha256": expected_sha,
        "retained_method": "no_examples",
        "consistency_only": True,
        "original_responses_labels_and_menu_averaging_reauthenticated": False,
    }
