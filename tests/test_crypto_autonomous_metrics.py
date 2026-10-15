import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from experiments import crypto_autonomous_metrics as m


def point(timestamp, *, bid=9999, ask=10001, quantity=1000):
    return m.BookPoint(timestamp, ((bid, quantity),), ((ask, quantity),))


def points(*, start=0, blocks=5):
    return tuple(point(start + offset) for offset in range(0, blocks * m.BLOCK_MS + 1, m.GRID_MS))


def observed(*, start=0):
    return m.WindowSummary(
        start,
        tuple(
            m.BlockSummary(start + i * m.BLOCK_MS, (0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0))
            for i in range(5)
        ),
    )


def paths(*, prints=()):
    shared = points()
    return tuple(m.SimulationPath(seed, shared, prints) for seed in range(20))


def labels():
    low = (-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -0.5)
    high = (1.0, 3.0, 3.0, 3.0, 3.0, 8.0, 0.5)
    return tuple(
        m.BlockSummary(start, low if i % 2 == 0 else high) for i, start in enumerate(m.TRAIN_STARTS)
    )


def test_inputs_are_immutable_copies_and_live_objects_are_rejected():
    p = point(0)
    with pytest.raises(FrozenInstanceError):
        p.timestamp_ms = 100
    with pytest.raises(TypeError, match="immutable"):
        m.BookPoint(0, [(9999, 1000)], ((10001, 1000),))
    with pytest.raises(TypeError, match="immutable"):
        m.BookPoint(0, ([9999, 1000],), ((10001, 1000),))

    class Reader:
        @property
        def timestamp_ms(self):
            raise AssertionError("A live reader must never be called")

    with pytest.raises(TypeError, match="Bridge must copy"):
        m.summarize_block((Reader(),), (), start_ms=0)


def test_mid_keeps_half_storage_unit_and_price_quantity_scales():
    initial = point(0)
    terminal = point(60_000, bid=10000, ask=10001)
    assert initial.mid_twice_units == 20000
    assert terminal.mid_twice_units == 20001
    assert terminal.mid_price_usdt == pytest.approx(100.005)
    block = m.summarize_block(points(blocks=1)[:-1] + (terminal,), (), start_ms=0)
    assert block.values[0] == pytest.approx(10_000 * math.log1p(1 / 20000))
    assert block.values[4:6] == (1.0, 1.0)
    assert block.values[0] > 0  # Integer-rounded midpoints would falsely give zero.


def test_numpy_integer_storage_does_not_overflow_fixed_width_price_quantity_arithmetic():
    price = np.int64(10**15)
    quantity = np.int64(8 * 10**18)
    p = m.BookPoint(0, ((price, quantity),), ((price + 1, quantity),))
    assert p.mid_twice_units == 2 * int(price) + 1
    print_rows = (
        m.PublishedPrint(1, "Buy", quantity, price),
        m.PublishedPrint(1, "Sell", quantity, price),
    )
    block = m.summarize_block(
        tuple(replace(p, timestamp_ms=t) for t in range(0, 60_001, 100)),
        print_rows,
        start_ms=0,
    )
    assert block.complete and block.values[1] == 2 * int(quantity) / 1000
    assert block.values[6] == 0.0


def test_depth_exact_5bps_boundary_and_visible_truncation_diagnostic():
    p = m.BookPoint(
        0, ((9999, 1000), (9995, 4000), (9994, 9000)), ((10001, 1000), (10005, 5000), (10006, 9000))
    )
    view = tuple(replace(p, timestamp_ms=t) for t in range(0, 60_001, 100))
    block = m.summarize_block(view, (), start_ms=0)
    assert block.complete
    assert block.values[4:6] == pytest.approx((5.0, 6.0))
    assert block.bid_5bps_coverage_fraction == 1
    assert block.ask_5bps_coverage_fraction == 1
    # Half-unit mid changes the exact band edge; do not round the threshold.
    half = m.BookPoint(
        0,
        ((10000, 1000), (9996, 2000), (9995, 9000)),
        ((10001, 1000), (10005, 3000), (10006, 9000)),
    )
    b = m.summarize_block(
        tuple(replace(half, timestamp_ms=t) for t in range(0, 60_001, 100)), (), start_ms=0
    )
    assert b.values[4:6] == pytest.approx((3.0, 4.0))
    thin = m.summarize_block(points(blocks=1), (), start_ms=0)
    assert thin.complete and thin.values[4:6] == (1.0, 1.0)
    assert thin.bid_5bps_coverage_fraction == 0
    assert (
        thin.ask_5bps_coverage_fraction == 0
    )  # Visible quantity is defined, full band is censored.


def test_600_left_clock_points_have_equal_time_weight_and_exclude_terminal():
    view = tuple(
        point(
            t,
            bid=9999 if t < 30_000 else 9998,
            ask=10001 if t < 30_000 else 10002,
            quantity=1000 if t < 30_000 else 3000,
        )
        for t in range(0, 60_000, 100)
    )
    view += (point(60_000, bid=9900, ask=10100, quantity=900_000),)
    block = m.summarize_block(view, (), start_ms=0)
    assert block.complete and block.valid_left_points == 600
    assert block.values[0] == 0
    assert block.values[3:6] == pytest.approx((3.0, 2.0, 2.0))


def test_published_prints_are_half_open_and_pressure_uses_quantity_not_count():
    prints = (
        m.PublishedPrint(59_999, "Buy", 900_000, 10000),
        m.PublishedPrint(60_000, "Buy", 1000, 10000),
        m.PublishedPrint(60_000, "Buy", 1000, 10000),
        m.PublishedPrint(119_999, "Sell", 2000, 10000),
        m.PublishedPrint(120_000, "Sell", 3000, 10000),
    )
    window = m.summarize_window(points(start=60_000), prints, start_ms=60_000)
    assert window.blocks[0].values[1:3] == (4.0, 3.0)
    assert window.blocks[0].values[6] == 0.0
    assert window.blocks[1].values[1:3] == (3.0, 1.0)
    assert window.blocks[1].values[6] == -1.0
    assert sum(block.values[2] for block in window.blocks) == 4.0


def test_zero_return_and_no_prints_are_true_zero_missing_stream_is_not():
    block = m.summarize_block(points(blocks=1), (), start_ms=0)
    assert block.complete
    assert (block.values[0], block.values[1], block.values[2], block.values[6]) == (0, 0, 0, 0)
    missing = m.summarize_block(points(blocks=1), None, start_ms=0)
    assert not missing.complete
    assert missing.values[1:3] == (None, None) and missing.values[6] is None
    assert "missing_print_stream" in missing.issues


@pytest.mark.parametrize(
    "bad", [None, (), ((10001, 1000),), ((9999, 0),), ((9999, 1.5),), ((9998, 1000), (9999, 1000))]
)
def test_missing_crossed_zero_quantity_noninteger_and_unsorted_books_are_not_filled(bad):
    view = list(points(blocks=1))
    view[1] = m.BookPoint(100, bad, ((10001, 1000),))
    block = m.summarize_block(tuple(view), (), start_ms=0)
    assert not block.complete and block.valid_left_points == 599
    assert block.values[3:6] == (None, None, None)
    assert block.values[0] == 0  # Independently known return endpoints remain known.


def test_missing_clock_point_is_not_carried_forward_or_averaged_out():
    view = points(blocks=1)
    block = m.summarize_block(view[:5] + view[6:], (), start_ms=0)
    assert not block.complete and block.valid_left_points == 599
    assert block.values[3] is None
    end_missing = m.summarize_block(view[:-1], (), start_ms=0)
    assert end_missing.values[0] is None
    assert end_missing.values[3] == pytest.approx(2.0)


def test_duplicate_and_off_grid_points_are_not_selected_by_favourable_price():
    view = points(blocks=1)
    duplicate = m.summarize_block(view + (point(100, bid=9990, ask=10100),), (), start_ms=0)
    assert not duplicate.complete and duplicate.valid_left_points == 599
    extra = m.summarize_block(view + (point(101),), (), start_ms=0)
    assert not extra.complete and "off_grid_point" in extra.issues


def test_invalid_and_nonmonotone_prints_do_not_get_silently_removed():
    wrong = m.summarize_block(
        points(blocks=1), (m.PublishedPrint(1, "unknown", 1000, 10000),), start_ms=0
    )
    assert not wrong.complete and wrong.values[1:3] == (None, None)
    reversed_time = m.summarize_block(
        points(blocks=1),
        (m.PublishedPrint(2, "Buy", 1000, 10000), m.PublishedPrint(1, "Buy", 1000, 10000)),
        start_ms=0,
    )
    assert not reversed_time.complete and "nonmonotone_print_stream" in reversed_time.issues


def test_transform_is_fixed_seven_dimensional_and_not_centered():
    raw = (-3, 2, 3, 4, 5, 6, -0.5)
    transformed = m.transform_vector(raw)
    assert transformed.dtype == np.float64
    np.testing.assert_allclose(transformed, [-3, *np.log1p([2, 3, 4, 5, 6]), -0.5])


def test_transform_does_not_mutate_a_caller_owned_float64_array():
    raw = np.array([-3, 2, 3, 4, 5, 6, -0.5], dtype=np.float64)
    original = raw.copy()
    transformed = m.transform_vector(raw)
    np.testing.assert_array_equal(raw, original)
    assert not np.shares_memory(transformed, raw)


def test_fit_uses_all_utc_training_candidates_population_ddof_zero():
    frozen = m.fit_metrics(labels(), split="train")
    expected = (1, math.log(2), math.log(2), math.log(2) / 2, math.log(2) / 2, math.log(3) / 2, 0.5)
    np.testing.assert_allclose(frozen.scales, expected, atol=2e-14, rtol=2e-14)
    assert frozen.training_candidate_labels == 4320 and frozen.training_complete_labels == 4320
    sample_std = np.asarray([m.transform_vector(label.values) for label in labels()]).std(
        axis=0, ddof=1
    )
    assert not np.allclose(frozen.scales, sample_std, rtol=1e-6, atol=0)


def test_incomplete_training_labels_keep_calendar_denominator_and_are_not_zero():
    candidate = labels()
    broken = replace(candidate[0], values=(None,) * 7, issues=("missing_source_state",))
    frozen = m.fit_metrics((broken,) + candidate[1:], split="train")
    assert frozen.training_candidate_labels == 4320 and frozen.training_complete_labels == 4319
    expected = np.asarray([m.transform_vector(label.values) for label in candidate[1:]]).std(
        axis=0, ddof=0
    )
    np.testing.assert_allclose(frozen.scales, expected)
    with pytest.raises(ValueError, match="complete candidate calendar"):
        m.fit_metrics(candidate[1:], split="train")


@pytest.mark.parametrize("split", ["development", "event_test", "unknown"])
def test_fit_date_split_gate_before_input_consumption(split):
    class Unreadable:
        def __iter__(self):
            raise AssertionError("Forbidden labels must not be touched")

    with pytest.raises(ValueError, match="cannot fit"):
        m.fit_metrics(Unreadable(), split=split)


def test_future_date_and_duplicate_minutes_rejected_without_selection():
    candidate = labels()
    nov4 = int(m.TRAIN_STARTS[-1] + 60_000)
    with pytest.raises(ValueError, match="November 1-3"):
        m.fit_metrics((replace(candidate[0], start_ms=nov4),) + candidate[1:], split="train")
    with pytest.raises(ValueError, match="complete candidate calendar"):
        m.fit_metrics((candidate[1],) + candidate[1:], split="train")


def test_zero_std_and_nonfinite_claimed_training_values_fail_instead_of_dropping_dimensions():
    candidate = labels()
    same = tuple(replace(label, values=candidate[0].values) for label in candidate)
    with pytest.raises(ValueError, match="zero population"):
        m.fit_metrics(same, split="train")
    for value in (float("inf"), float("nan")):
        poisoned = replace(candidate[0], values=(value,) + candidate[0].values[1:])
        with pytest.raises(ValueError, match="finite raw"):
            m.fit_metrics((poisoned,) + candidate[1:], split="train")
    with pytest.raises(ValueError, match="Invalid fixed training scale"):
        m.FrozenMetrics((0.0,) * 7)


def test_fair_energy_hand_example_differs_from_biased_k_squared_estimator():
    draws = np.array([[-1.0]] * 10 + [[1.0]] * 10)
    fair = m.fair_energy_score(draws, [0.0])
    assert fair == pytest.approx(9 / 19)
    biased = 1 - 200 / 400
    assert biased == 0.5 and fair != biased
    with pytest.raises(ValueError, match="exactly 20"):
        m.fair_energy_score(draws[:-1], [0.0])


def test_identical_complete_paths_and_inclusive_inverted_cdf_intervals():
    result = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), paths())
    assert result["complete"] and result["energy_score"] == 0.0
    assert result["price_complete"] and result["price_energy_score"] == 0.0
    assert result["expected_path_blocks"] == 100
    for block in result["blocks"]:
        interval = block["intervals_90"]["return_bps"]
        assert interval == {
            "expected_predictions": 20,
            "valid_predictions": 20,
            "lower": 0,
            "upper": 0,
            "covered": True,
        }
    prices = tuple(m.PricePath(i, (float(i),) * 5) for i in range(20))
    actual = replace(
        observed(),
        blocks=tuple(replace(b, values=(0.0,) + b.values[1:]) for b in observed().blocks),
    )
    interval = m.score_price_window(m.FrozenMetrics((1.0,) * 7), actual, prices)["blocks"][0][
        "interval_90"
    ]
    assert interval["lower"] == 0 and interval["upper"] == 18 and interval["covered"] is True


def test_all_five_blocks_equal_weight_and_actual_published_counts():
    prints = tuple(m.PublishedPrint(i * 60_000 + 1, "Buy", (i + 1) * 1000, 10000) for i in range(5))
    result = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), paths(prints=prints))
    expected = [math.hypot(math.log1p(i + 1), math.log(2), 1) for i in range(5)]
    assert result["energy_score"] == pytest.approx(math.fsum(expected) / 5)
    assert [b["joint"]["energy_score"] for b in result["blocks"]] == pytest.approx(expected)


def test_twenty_failed_paths_keep_all_seed_and_block_denominators_no_penalty():
    failed = tuple(m.SimulationPath(i, None, None) for i in range(20))
    result = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), failed)
    assert not result["complete"] and result["energy_score"] is None
    assert result["expected_paths"] == 20 and result["expected_path_blocks"] == 100
    assert result["incomplete_expected_paths"] == 20
    assert len(result["blocks"]) == 5
    assert all(
        len(b["failed_seeds"]) == 20 and b["complete_predictions"] == 0 for b in result["blocks"]
    )


def test_one_missing_seed_or_one_bad_block_prevents_complete_main_score():
    good = paths()
    missing = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), good[:-1])
    assert not missing["complete"] and missing["energy_score"] is None
    assert missing["blocks"][0]["intervals_90"]["return_bps"]["covered"] is None
    broken_points = good[-1].points[:601] + good[-1].points[602:]
    broken = good[:-1] + (replace(good[-1], points=broken_points),)
    result = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), broken)
    assert result["blocks"][0]["joint"]["complete"]
    assert not result["blocks"][1]["joint"]["complete"]
    assert not result["complete"] and result["energy_score"] is None
    assert result["incomplete_expected_paths"] == 1


def test_duplicate_and_extra_seeds_cannot_be_used_to_choose_twenty():
    good = paths()
    for altered in (good[:-1] + (good[0],), good + (replace(good[0], seed=999),)):
        result = m.score_window(m.FrozenMetrics((1.0,) * 7), observed(), altered)
        assert not result["complete"] and result["energy_score"] is None


def test_price_only_p_cannot_fake_other_dimensions_and_uses_return_scale():
    p = tuple(m.PricePath(i, ((-1.0 if i < 10 else 1.0),) * 5) for i in range(20))
    metrics = m.FrozenMetrics((2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0))
    result = m.score_price_window(metrics, observed(), p)
    assert result["scope"] == "price_only" and result["price_energy_score"] == pytest.approx(9 / 38)
    assert "energy_score" not in result and "joint" not in result
    with pytest.raises(TypeError, match="only SimulationPath"):
        m.score_window(metrics, observed(), p)
    bad = p[:-1] + (replace(p[-1], returns_bps=(0, 0, None, 0, 0)),)
    assert m.score_price_window(metrics, observed(), bad)["price_energy_score"] is None


def test_evaluation_never_refits_from_future_prices(monkeypatch):
    metrics = m.fit_metrics(labels(), split="train")
    original = metrics.scales
    monkeypatch.setattr(
        m, "fit_metrics", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No future fit"))
    )
    p = tuple(m.PricePath(i, (1, 2, 3, 4, 5)) for i in range(20))
    a = m.score_price_window(metrics, observed(), p)
    shifted = replace(
        observed(),
        blocks=tuple(replace(b, values=(1e6,) + b.values[1:]) for b in observed().blocks),
    )
    b = m.score_price_window(metrics, shifted, p)
    assert a["price_energy_score"] == pytest.approx(3 / original[0])
    assert b["price_energy_score"] != a["price_energy_score"]
    assert metrics.scales == original
