import numpy as np

from experiments.probability_analysis import mean_bernoulli_js


def test_js_is_a_mean_in_bits_and_independent_of_dataset_repetition():
    assert mean_bernoulli_js([0.0, 1.0], [1.0, 0.0]) == 1.0
    assert mean_bernoulli_js([0.0, 0.2, 1.0], [0.0, 0.2, 1.0]) == 0.0
    p, y = [0.1, 0.7, 0.9], [0.8, 0.5, 0.2]
    once = mean_bernoulli_js(p, y)
    assert 0 < once < 1
    assert np.isclose(once, mean_bernoulli_js(p * 7, y * 7))
