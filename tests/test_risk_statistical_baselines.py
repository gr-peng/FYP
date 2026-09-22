import numpy as np
import pytest
from test_choices13k_data_driven import make_item

from experiments.risk_statistical_baselines import features, fit, predict


def test_option_swap_complements_probabilities_and_target_labels_are_unused():
    pool = [make_item(i, (i % 8) / 8) for i in range(1, 41)]
    model = fit(pool)
    target = make_item(101, 0.123)
    swapped = target.model_copy(update={"option_a": target.option_b, "option_b": target.option_a})
    changed = target.model_copy(update={"human_b_rate": 0.987, "ev_b": 1e9, "var_a": 1e8})
    np.testing.assert_allclose(features(swapped), -features(target))
    outputs = predict(model, [target, swapped, changed])
    for p in outputs.values():
        assert np.isclose(p[0] + p[1], 1)
        assert p[0] == p[2]
    with pytest.raises(ValueError, match="ambiguous"):
        features(target.model_copy(update={"ambiguity": True}))
