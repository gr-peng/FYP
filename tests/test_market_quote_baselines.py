import numpy as np

from experiments.market_quote_baselines import metrics, weights


def test_scoring_weights_sessions_then_people_not_number_of_submissions():
    records = [
        {"provenance": {"session": s, "participant": p}, "target": {"buy": y}}
        for s, p, y in [("s1", "p1", 1), ("s1", "p1", 1), ("s1", "p2", 0), ("s2", "p3", 0)]
    ]
    np.testing.assert_allclose(weights(records), [0.125, 0.125, 0.25, 0.5])
    result = metrics(records, {"always_buy": np.ones(4), "always_sell": np.zeros(4)})
    assert result["human_buy_rate_session_person_balanced"] == 0.25
    assert result["methods"]["always_buy"]["brier_session_person_balanced"] == 0.75
    assert result["methods"]["always_sell"]["brier_session_person_balanced"] == 0.25
