"""Descriptive accounting of already saved forecasts; no fitting or data loading.

Observed personal means are post-outcome arithmetic terms, never predictors.
"""

from collections import defaultdict

import numpy as np

METHODS = (
    "account_public_task_15",
    "plus_current_period_17",
    "plus_completed_history_23",
    "fold_training_constant",
    "constant_half",
    "current_history_prior4",
    "combined_history_prior4",
)
TERMS = (
    "total_error",
    "personal_mean_bias",
    "within_person_error",
    "prediction_variance",
    "label_variance",
    "prediction_label_covariance",
)
PAIRS = ((METHODS[1], METHODS[0]), (METHODS[2], METHODS[1]))
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_identity(values):
    require(
        abs(values["total_error"] - values["personal_mean_bias"] - values["within_person_error"])
        <= TOLERANCE,
        "total-error reconstruction failed",
    )
    require(
        abs(
            values["within_person_error"]
            - values["prediction_variance"]
            - values["label_variance"]
            + 2 * values["prediction_label_covariance"]
        )
        <= TOLERANCE,
        "centered-error reconstruction failed",
    )


def person_terms(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    require(
        y.ndim == 1
        and y.shape == p.shape
        and len(y) > 0
        and np.isfinite(y).all()
        and np.isfinite(p).all()
        and np.isin(y, (0, 1)).all()
        and ((0 <= p) & (p <= 1)).all(),
        "invalid person arrays",
    )
    centered_y, centered_p = y - y.mean(), p - p.mean()
    result = {
        "total_error": float(np.mean((p - y) ** 2)),
        "personal_mean_bias": float((p.mean() - y.mean()) ** 2),
        "within_person_error": float(np.mean((centered_p - centered_y) ** 2)),
        "prediction_variance": float(np.mean(centered_p**2)),
        "label_variance": float(np.mean(centered_y**2)),
        "prediction_label_covariance": float(np.mean(centered_p * centered_y)),
    }
    check_identity(result)
    return result


def validate(records):
    require(bool(records), "empty forecast table")
    keys = []
    for r in records:
        require(
            set(r)
            == {"session", "participant", "order_id", "fold", "observed_buy", "probabilities"},
            "unexpected forecast schema",
        )
        require(
            all(
                isinstance(r[k], str) and r[k].strip()
                for k in ("session", "participant", "order_id")
            ),
            "missing forecast identity",
        )
        require(type(r["fold"]) is int and 0 <= r["fold"] < 5, "invalid original fold")
        require(type(r["observed_buy"]) is int and r["observed_buy"] in (0, 1), "invalid label")
        require(set(r["probabilities"]) == set(METHODS), "incomplete common forecast series")
        require(
            all(
                type(p) in (int, float) and np.isfinite(p) and 0 <= p <= 1
                for p in r["probabilities"].values()
            ),
            "invalid probability",
        )
        keys.append((r["session"], r["order_id"]))
    require(len(keys) == len(set(keys)), "duplicate target")


def average(units):
    result = {
        method: {term: float(np.mean([r[method][term] for r in units])) for term in TERMS}
        for method in METHODS
    }
    for value in result.values():
        check_identity(value)
    return result


def decompose(records):
    validate(records)
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["session"], r["participant"]].append(i)
    sessions = defaultdict(list)
    people = []
    for (session, participant), ids in sorted(groups.items()):
        y = np.array([records[i]["observed_buy"] for i in ids], dtype=float)
        values = {
            m: person_terms(y, [records[i]["probabilities"][m] for i in ids]) for m in METHODS
        }
        person = {
            "session": session,
            "participant": participant,
            "events": len(ids),
            "observed_personal_mean_for_accounting_only": float(y.mean()),
            "constant_observed_labels": bool(np.all(y == y[0])),
            "methods": values,
        }
        sessions[session].append(values)
        people.append(person)
    per_session = {s: average(units) for s, units in sessions.items()}
    overall = average(list(per_session.values()))
    for p in people:
        p["balanced_weight"] = 1 / (len(sessions) * len(sessions[p["session"]]))
    population = {
        "records": len(records),
        "sessions": len(sessions),
        "session_person_units": len(people),
        "unique_recorded_codes": len({p["participant"] for p in people}),
    }
    for name, selected in {
        "single_target": [p for p in people if p["events"] == 1],
        "constant_labels_including_single_target": [
            p for p in people if p["constant_observed_labels"]
        ],
        "constant_labels_multiple_targets": [
            p for p in people if p["constant_observed_labels"] and p["events"] > 1
        ],
    }.items():
        population[name] = {
            "people": len(selected),
            "balanced_weight": sum(p["balanced_weight"] for p in selected),
        }
    return {
        "overall": overall,
        "per_session": per_session,
        "people": people,
        "population": population,
    }


def contrasts(result, components, *, draws=10000):
    sessions = result["per_session"]
    flat = [s for group in components for s in group]
    require(
        bool(components)
        and all(components)
        and len(flat) == len(set(flat))
        and set(flat) == set(sessions),
        "incomplete or overlapping components",
    )
    require(type(draws) is int and draws > 0, "invalid bootstrap draws")
    selected = np.random.default_rng(20261107).integers(
        len(components), size=(draws, len(components))
    )
    counts = np.array([len(group) for group in components])
    denominator = counts[selected].sum(axis=1)
    output = {}
    for first, second in PAIRS:
        differences = {
            s: {t: value[first][t] - value[second][t] for t in TERMS}
            for s, value in sessions.items()
        }
        point, samples = {}, {}
        for term in TERMS:
            sums = np.array([sum(differences[s][term] for s in group) for group in components])
            point[term] = float(sums.sum() / counts.sum())
            samples[term] = sums[selected].sum(axis=1) / denominator
            require(
                abs(
                    point[term] - (result["overall"][first][term] - result["overall"][second][term])
                )
                <= TOLERANCE,
                "global/session contrast mismatch",
            )
        check_identity(point)
        require(
            np.max(np.abs(samples["label_variance"])) <= TOLERANCE, "labels changed between methods"
        )
        error_total = (
            samples["total_error"] - samples["personal_mean_bias"] - samples["within_person_error"]
        )
        error_within = (
            samples["within_person_error"]
            - samples["prediction_variance"]
            + 2 * samples["prediction_label_covariance"]
        )
        require(
            max(np.max(np.abs(error_total)), np.max(np.abs(error_within))) <= TOLERANCE,
            "bootstrap algebra failed",
        )
        output[first + "-minus-" + second] = {
            "differences": point,
            "per_session": differences,
            "percentile_95": {
                t: np.quantile(samples[t], [0.025, 0.975]).tolist() for t in TERMS[:3]
            },
            "draws": draws,
            "component_clusters": len(components),
            "max_bootstrap_reconstruction_error": float(
                max(np.max(np.abs(error_total)), np.max(np.abs(error_within)))
            ),
            "scope": "Six pointwise descriptive intervals conditional on fixed forecasts; "
            "no refitting, no simultaneous inference; percentile endpoints need not sum.",
        }
    return output
