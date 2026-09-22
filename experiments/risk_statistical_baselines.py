"""Risk-only statistical controls with explicit A/B symmetry and train-only fitting."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.calibration_v2 import visible_fingerprint, visible_option

FEATURES = (
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "loss_probability",
    "gain_probability",
    "certainty",
    "entropy",
    "concentration",
    "log_support_size",
)


def features(item):
    if item.ambiguity:
        raise ValueError("risk-only model cannot consume ambiguous latent probabilities")
    options = [visible_option(item.option_a)["outcomes"], visible_option(item.option_b)["outcomes"]]
    scale = max(1.0, *(abs(x) for option in options for _, x in option))
    vectors = []
    for option in options:
        probabilities = np.array([p for p, _ in option])
        payoffs = np.array([x / scale for _, x in option])
        mean = float(probabilities @ payoffs)
        vectors.append(
            np.array(
                [
                    mean,
                    np.sqrt(probabilities @ (payoffs - mean) ** 2),
                    payoffs.min(),
                    payoffs.max(),
                    probabilities[payoffs < 0].sum(),
                    probabilities[payoffs > 0].sum(),
                    float(len(option) == 1),
                    -np.sum(probabilities * np.log(probabilities)),
                    np.sum(probabilities**2),
                    math.log1p(len(option)),
                ]
            )
        )
    return vectors[1] - vectors[0]


def fit(pool, ridge=0.01, neighbors=16):
    pool = [item for item in pool if not item.ambiguity]
    x = np.array([features(item) for item in pool])
    y = np.array([item.human_b_rate for item in pool])
    # RMS scaling and no intercept preserve exact option-swap complementarity.
    scale = np.sqrt(np.mean(x**2, axis=0))
    scale[scale < 1e-12] = 1
    z = x / scale

    def objective(beta):
        logits = z @ beta
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + ridge / 2 * (beta @ beta)
        gradient = z.T @ (expit(logits) - y) / len(y) + ridge * beta
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise ValueError(f"fractional logistic fit failed: {result.message}")
    return {
        "features": list(FEATURES),
        "ridge": ridge,
        "neighbors": neighbors,
        "scale": scale.tolist(),
        "beta": result.x.tolist(),
        "train_z": z.tolist(),
        "train_y": y.tolist(),
        "train_ids": [item.item_id for item in pool],
        "train_fingerprints": [visible_fingerprint(item) for item in pool],
    }


def predict(model, targets):
    z_train = np.array(model["train_z"])
    y_train = np.array(model["train_y"])
    predictions = {"symmetric_fractional_logistic": [], "aligned_knn_16": []}
    for target in targets:
        z = features(target) / model["scale"]
        predictions["symmetric_fractional_logistic"].append(float(expit(z @ model["beta"])))
        direct = np.sum((z_train - z) ** 2, axis=1)
        flipped = np.sum((z_train + z) ** 2, axis=1)
        same_structure = np.array(model["train_fingerprints"]) == visible_fingerprint(target)
        distance = np.minimum(direct, flipped)
        distance[same_structure] = np.inf
        if np.isfinite(distance).sum() < model["neighbors"]:
            raise ValueError("not enough structurally distinct neighbors")
        # Stable secondary key; pool identity does not encode target labels.
        selected = np.lexsort((np.array(model["train_ids"]), distance))[: model["neighbors"]]
        oriented = np.where(direct <= flipped, y_train, 1 - y_train)
        oriented[np.isclose(direct, flipped, atol=1e-12, rtol=1e-12)] = 0.5
        predictions["aligned_knn_16"].append(float(oriented[selected].mean()))
    return {name: np.array(values) for name, values in predictions.items()}
