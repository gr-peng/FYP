from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from random import Random
from types import SimpleNamespace

import pytest

from experiments.crypto_effective_flow import (
    DAY_US,
    GRID_US,
    BookObservation,
    FlowTemplate,
    Intent,
    ObservedPrint,
    RelativeLiquidity,
    StateFeatures,
    TrainingRow,
    extract_interval,
    fit_template_pool,
    select_template,
    state_cell,
    state_features,
)

DAY = "2024-11-01"
START = 1730419200000000
A = START + 10_000_000


def observation(at=A, bids=((9900, 1000), (9800, 2000)),
                asks=((10000, 1000), (10100, 2000)), **kwargs):
    fields = dict(observed_at_us=at, bids=bids, asks=asks, valid=True,
                  segment_id=0, segment_start_us=START + 10_000,
                  message_index=100 + (at - A) // GRID_US,
                  engine_us=at // 1000 * 1000 - 30_000,
                  available_us=at // 1000 * 1000 - 20_000,
                  publication_prefix_ms=(at - 21_000) // 1000)
    fields.update(kwargs)
    return BookObservation(**fields)


def print_(side="Buy", quantity=1000, price=10000, timestamp=A + 20_000, row=1):
    return ObservedPrint(timestamp, side, quantity, price, row, f"trade-{row}")


def features(i=Fraction(0), spread=Fraction(1), pressure=Fraction(0)):
    return StateFeatures(i, spread, pressure)


def training_rows(n=64, spread=Fraction(1)):
    return tuple(TrainingRow(
        DAY, FlowTemplate(f"{DAY}:{A + j * GRID_US}", (), (), (Intent("Buy", j + 1),)),
        (features(spread=spread), features(spread=spread), features(spread=spread)),
    ) for j in range(n))


def test_exact_consumption_is_not_removed_twice():
    begin = observation(asks=((10000, 1500), (10100, 2000)))
    end = observation(A + GRID_US, asks=((10000, 500), (10100, 2000)))
    result = extract_interval(DAY, begin, end, (print_(),))
    assert result.eligible
    assert not result.template.adds and not result.template.removes
    assert result.template.intents == (Intent("Buy", 1000),)
    assert result.observed_quantity_units == 1000
    assert next(x for x in result.levels if x.price_units == 10000).residual_units == 0


def test_gross_add_cancel_cannot_be_identified_from_equal_endpoints():
    begin, end = observation(), observation(A + GRID_US)
    result = extract_interval(DAY, begin, end)
    # No activity and add 7 BTC followed by cancel 7 BTC have identical evidence.
    assert result.template.adds == result.template.removes == ()
    assert all(level.residual_units == 0 for level in result.levels)


def test_consumption_plus_replenishment_is_net_residual_not_gross_add():
    begin, end = observation(), observation(A + GRID_US)
    result = extract_interval(DAY, begin, end, (print_(),))
    assert result.template.adds == (RelativeLiquidity("ask", 0, 1000),)
    assert result.template.removes == ()
    assert result.alignment[0].within_engine_span


def test_absolute_prices_align_when_ranks_shift_and_edges_remain_censored():
    begin = observation(bids=((9900, 1000), (9800, 2000), (9700, 3000)),
                        asks=((10000, 1000), (10100, 2000), (10200, 3000)))
    end = observation(A + GRID_US, bids=((9950, 4000), (9900, 900), (9800, 1500)),
                      asks=((10000, 1500), (10100, 2500)))
    result = extract_interval(DAY, begin, end)
    assert result.template.adds == (
        RelativeLiquidity("bid", 50, 4000), RelativeLiquidity("ask", 0, 500),
        RelativeLiquidity("ask", 100, 500),
    )
    assert result.template.removes == (
        RelativeLiquidity("bid", 0, 100), RelativeLiquidity("bid", -100, 500),
    )
    assert ("bid", "start", 1, 3000) in result.censored_endpoints
    assert ("ask", "start", 1, 3000) in result.censored_endpoints
    assert {x.price_units for x in result.levels} == {9950, 9900, 9800, 10000, 10100}


def test_outside_domain_and_absent_display_prints_are_retained_and_labeled():
    prints = (print_(price=10200), print_(price=10050, quantity=7, row=2))
    result = extract_interval(DAY, observation(), observation(A + GRID_US), prints)
    assert result.observed_quantity_units == 1007
    assert result.template.intents == (Intent("Buy", 1000), Intent("Buy", 7))
    assert result.alignment[0].domain == "outside_common_visible_domain"
    assert result.alignment[0].absent_at_both_endpoints is None
    assert result.alignment[1].absent_at_both_endpoints
    assert result.template.adds == (RelativeLiquidity("ask", 50, 7),)


def test_interval_inside_spread_add_exposes_ordering_uncertainty():
    end = observation(A + GRID_US, bids=((10050, 1000), (9900, 1000), (9800, 2000)),
                      asks=((10100, 2000),))
    result = extract_interval(DAY, observation(), end, (print_(),))
    assert result.template.adds == (RelativeLiquidity("bid", 150, 1000),)
    assert result.template.removes == ()
    # Before Buy consumption this bid crosses ask 100; after it ask is 101.
    assert observation().bids[0][0] + result.template.adds[0].price_offset_units > 10000
    assert result.levels


@pytest.mark.parametrize("change,reason", [
    ({"segment_id": 1}, "snapshot_or_reset_crossing"),
    ({"invalid_count": 1}, "invalid_message_crossing"),
    ({"engine_us": A + GRID_US - 1_000_000}, "end_stale"),
    ({"segment_start_us": START - 1000}, "end_outside_day_state"),
])
def test_ineligible_candidates_keep_prints_without_manufactured_flow(change, reason):
    end = observation(A + GRID_US, **change)
    begin = observation(engine_us=end.engine_us) if reason == "end_stale" else observation()
    if reason == "end_outside_day_state":
        begin = observation(segment_start_us=end.segment_start_us)
    prints = (print_(),)
    result = extract_interval(DAY, begin, end, prints)
    assert not result.eligible and reason in result.exclusion_reasons
    assert result.prints == prints and result.observed_quantity_units == 1000
    assert result.template is None and not result.levels
    assert result.alignment[0].domain == "unusable_endpoints"


def test_missing_initial_state_is_not_zero_liquidity():
    absent = BookObservation(A, (), (), False, None, None, None, None, None, None)
    result = extract_interval(DAY, absent, observation(A + GRID_US), (print_(),))
    assert result.exclusion_reasons == ("start_invalid_or_absent",)
    assert result.template is None


@pytest.mark.parametrize("change", [
    {"bids": ((9900, 2000), (9800, 2000))},
    {"engine_us": A - 29_000},
    {"available_us": A - 19_000},
    {"publication_prefix_ms": (A - 22_000) // 1000},
    {"segment_id": 1},
    {"invalid_count": 1},
])
def test_same_consumed_prefix_cannot_manufacture_change(change):
    begin = observation()
    end = replace(begin, observed_at_us=A + GRID_US, **change)
    with pytest.raises(ValueError, match="Same consumed prefix"):
        extract_interval(DAY, begin, end)


def test_unchanged_prefix_at_later_grid_is_legitimate_and_has_no_flow():
    begin = observation()
    end = replace(begin, observed_at_us=A + GRID_US)
    result = extract_interval(DAY, begin, end)
    assert result.eligible
    assert result.template.adds == result.template.removes == ()


def test_segment_origin_and_publication_prefix_require_identity_and_order():
    with pytest.raises(ValueError, match="snapshot origin"):
        extract_interval(DAY, observation(),
                         observation(A + GRID_US, segment_start_us=START + 11_000))
    with pytest.raises(ValueError, match="publication prefix"):
        extract_interval(DAY, observation(), observation(
            A + GRID_US, publication_prefix_ms=(A - 22_000) // 1000))
    with pytest.raises(ValueError, match="cumulative"):
        extract_interval(DAY, observation(invalid_count=1), observation(A + GRID_US))


def test_valid_source_index_and_cts_precision_are_not_caller_guesses():
    with pytest.raises(ValueError, match="positive"):
        observation(message_index=0)
    with pytest.raises(ValueError, match="millisecond"):
        observation(engine_us=A - 30_001)
    with pytest.raises(ValueError, match="millisecond"):
        observation(segment_start_us=START + 10_001)


@pytest.mark.parametrize("time", [A - 1, A + GRID_US])
def test_half_open_print_allocation_rejects_wrong_interval(time):
    with pytest.raises(ValueError, match="half-open"):
        extract_interval(DAY, observation(), observation(A + GRID_US),
                         (print_(timestamp=time),))


def test_timestamp_ties_keep_tape_row_order_without_price_sort():
    prints = (print_(side="Sell", price=9900, row=10), print_(side="Buy", row=11))
    result = extract_interval(DAY, observation(), observation(A + GRID_US), prints)
    assert result.template.intents == (Intent("Sell", 1000), Intent("Buy", 1000))
    with pytest.raises(ValueError, match="source-row"):
        extract_interval(DAY, observation(), observation(A + GRID_US), tuple(reversed(prints)))


def test_invalid_prefix_future_and_boolean_storage_never_enter_evidence():
    with pytest.raises(ValueError, match="available-prefix"):
        observation(available_us=A)
    with pytest.raises(ValueError, match="available-prefix"):
        observation(publication_prefix_ms=A // 1000)
    with pytest.raises(ValueError, match="integer"):
        observation(bids=((9900, True),))
    with pytest.raises(ValueError, match="immutable"):
        observation(bids=[(9900, 1000)])
    with pytest.raises(ValueError, match="ordered"):
        observation(bids=((9800, 1000), (9900, 1000)))


def test_utc_day_and_grid_reject_reassignment():
    with pytest.raises(ValueError, match="aligned"):
        extract_interval("2024-11-02", observation(), observation(A + GRID_US))
    with pytest.raises(ValueError, match="aligned"):
        extract_interval(DAY, observation(A + 1), observation(A + GRID_US + 1))
    with pytest.raises(ValueError, match="aligned"):
        extract_interval(DAY, observation(START + DAY_US), observation(START + DAY_US + GRID_US))


def test_template_is_separate_immutable_narrow_interface_without_future_prices():
    before, end = observation(), observation(A + GRID_US)
    snapshot = asdict(before)
    first = extract_interval(DAY, before, end, (print_(price=10000),))
    second = extract_interval(DAY, before, end, (print_(price=10010),))
    assert first.prints[0].price_units != second.prints[0].price_units
    assert first.template.intents == second.template.intents
    assert set(asdict(first.template)) == {"row_id", "adds", "removes", "intents"}
    assert set(asdict(first.template.intents[0])) == {"side", "quantity_units"}
    assert asdict(before) == snapshot
    with pytest.raises(FrozenInstanceError):
        first.template.row_id = "changed"


def test_past_pressure_is_causal_exact_and_quantity_weighted():
    past = (print_(quantity=3, timestamp=A - 800_000, row=1),
            print_(side="Sell", quantity=1, price=9900, timestamp=A - 1, row=2))
    state = state_features(observation(bids=((9900, 3000), (9800, 2000))), past)
    assert state.imbalance_l1 == Fraction(1, 2)
    assert state.trade_pressure == Fraction(1, 2)
    assert state.spread_bps == Fraction(20000, 199)
    assert state_features(observation()).trade_pressure == 0
    with pytest.raises(ValueError, match="half-open"):
        state_features(observation(), (print_(timestamp=A),))


def test_fixed_eighteen_cells_have_inclusive_center_boundaries():
    median = Fraction(1)
    cells = {
        state_cell(features(i, s, p), median)
        for i in (Fraction(-1), Fraction(0), Fraction(1))
        for s in (Fraction(1), Fraction(2))
        for p in (Fraction(-1), Fraction(0), Fraction(1))
    }
    assert cells == set(range(18))
    assert state_cell(features(Fraction(-1, 3), pressure=Fraction(1, 3)), median) == 7
    assert state_cell(features(Fraction(1, 3), pressure=Fraction(-1, 3)), median) == 7


def test_empirical_pool_uses_same_rows_all_delays_and_exact_zero_delay_median():
    rows = training_rows(4)
    rows = tuple(replace(row, features_by_delay=(
        features(spread=Fraction(j + 1)), features(Fraction(1), Fraction(100)),
        features(Fraction(-1), Fraction(200)),
    )) for j, row in enumerate(rows))
    pool = fit_template_pool(rows)
    assert pool.spread_median == Fraction(5, 2)
    assert all(sorted(i for cell in delay for i in cell) == list(range(4))
               for delay in pool.indexes)
    assert pool.indexes[0] != pool.indexes[1] != pool.indexes[2]


@pytest.mark.parametrize("day", ["2024-11-04", "2024-11-05", "2024-10-14"])
def test_training_pool_refuses_nonapproved_dates_before_fit(day):
    with pytest.raises(ValueError, match="closed"):
        TrainingRow(day, training_rows(1)[0].template, (features(),) * 3)


def test_pool_duplicate_reordered_and_alias_row_ids_are_rejected():
    rows = training_rows(2)
    for bad in ((rows[0], rows[0]), tuple(reversed(rows))):
        with pytest.raises(ValueError, match="unique"):
            fit_template_pool(bad)
    with pytest.raises(ValueError, match="identity"):
        replace(rows[0], template=replace(rows[0].template, row_id=f"{DAY}:0{A}"))


def test_m1_fallback_is_fixed_and_joint_template_remains_whole():
    pool = fit_template_pool(training_rows(31))
    first = select_template(pool, features(), method="M0", delay_ms=0, rng=Random(17))
    second = select_template(pool, features(), method="M1", delay_ms=0, rng=Random(17))
    assert second.fallback and first.template == second.template
    assert not first.fallback
    pool32 = fit_template_pool(training_rows(32))
    selected = select_template(pool32, features(), method="M1", delay_ms=0, rng=Random(17))
    assert not selected.fallback and selected.template in [r.template for r in pool32.rows]
    unpopulated = select_template(pool32, features(Fraction(1)), method="M1",
                                 delay_ms=0, rng=Random(17))
    assert unpopulated.fallback
    with pytest.raises(ValueError, match="method"):
        select_template(pool32, features(), method="M1", delay_ms=10, rng=Random(17))


def test_same_seed_consumes_one_draw_for_all_pool_strategies():
    pool = fit_template_pool(training_rows(64))
    r0, r1 = Random(77), Random(77)
    for _ in range(12):
        m0 = select_template(pool, features(), method="M0", delay_ms=0, rng=r0)
        m1 = select_template(pool, features(), method="M1", delay_ms=0, rng=r1)
        assert m0.template == m1.template
    assert r0.getstate() == r1.getstate()


def test_different_nonfallback_pool_sizes_keep_shared_parent_random_stream():
    rows = training_rows(96)
    rows = tuple(replace(row, features_by_delay=(
        features(Fraction(0) if i < 32 else Fraction(1)),
    ) * 3) for i, row in enumerate(rows))
    pool = fit_template_pool(rows)
    r0, r1 = Random(5), Random(5)
    for _ in range(100):
        select_template(pool, features(), method="M0", delay_ms=0, rng=r0)
        m1 = select_template(pool, features(), method="M1", delay_ms=0, rng=r1)
        assert not m1.fallback and 0 <= m1.row_index < 32
        assert r0.getstate() == r1.getstate()


def test_wide_duck_objects_cannot_bypass_narrow_runtime_input_contract():
    row = training_rows(1)[0]
    wide_template = SimpleNamespace(**asdict(row.template), future_price=99999,
                                    historical_path="sealed.csv")
    with pytest.raises(ValueError, match="narrow"):
        TrainingRow(DAY, wide_template, row.features_by_delay)
    wide_book = SimpleNamespace(**asdict(observation()))
    with pytest.raises(ValueError, match="BookObservation"):
        extract_interval(DAY, wide_book, observation(A + GRID_US))
    with pytest.raises(ValueError, match="BookObservation"):
        state_features(wide_book)
    with pytest.raises(ValueError, match="StateFeatures"):
        state_cell(SimpleNamespace(**asdict(features())), Fraction(1))
    pool = fit_template_pool(training_rows(32))
    with pytest.raises(ValueError, match="validated"):
        select_template(SimpleNamespace(**asdict(pool)), features(), method="M0",
                        delay_ms=0, rng=Random(1))


def test_pool_constructor_cannot_replace_threshold_cells_or_row_identity():
    pool = fit_template_pool(training_rows(32))
    with pytest.raises(ValueError, match="threshold"):
        replace(pool, spread_median=Fraction(2))
    with pytest.raises(ValueError, match="indexes"):
        replace(pool, indexes=pool.indexes[:1])
    with pytest.raises(ValueError, match="digest"):
        replace(pool, row_identity_sha256="0" * 64)
