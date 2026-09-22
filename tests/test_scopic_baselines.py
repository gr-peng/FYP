import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.scopic_baselines import fit, person_weights, predict  # noqa: E402


def test_person_weighting_and_target_blind_prediction():
    items = []
    for person, count in (("many", 20), ("few", 2)):
        for i in range(count):
            items.append(
                {
                    "provenance": {"participant": person, "session": "one"},
                    "asset": "a",
                    "sell_any": int(person == "few"),
                    "state": {
                        "current_prices": {"a": 100 + i},
                        "holdings": {"a": 2},
                        "average_acquisition_prices": {"a": 100},
                        "cash": 800,
                        "main_period": 2,
                        "total_main_periods": 14,
                        "price_history": [{"a": 100}, {"a": 100}, {"a": 100}, {"a": 100 + i}],
                    },
                }
            )
    weights = person_weights(items)
    assert np.isclose(weights[:20].sum(), 0.5)
    assert np.isclose(weights[20:].sum(), 0.5)
    model = fit(items)
    assert np.isclose(model["train_constant"], 0.5)
    before = predict(model, items)
    poisoned = copy.deepcopy(items)
    for item in poisoned:
        item["sell_any"] = "forbidden_label"
        item["provenance"] = "forbidden_identity"
    after = predict(model, poisoned)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
        assert np.isfinite(before[name]).all()
        assert ((before[name] >= 0) & (before[name] <= 1)).all()
