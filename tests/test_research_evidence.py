"""Regression cases for inflated completion and unsupported market-realism claims."""

import pytest

from behavioral_market.evaluation.research_evidence import (
    error_comparison,
    execution_status,
    market_comparison_readiness,
    publication_status,
)


def test_zero_errors_does_not_complete_a_run_that_never_started():
    result = execution_status(planned=4320, entered=0, returned=0, verified=0, exceptions=0)
    assert result["state"] == "not_started"
    assert result["not_entered"] == 4320
    assert result["scientific_success_inferred"] is False


def test_all_observed_calls_verified_is_still_partial():
    result = execution_status(
        planned=4320,
        entered=1526,
        returned=1526,
        verified=1526,
        exceptions=0,
    )
    assert result["state"] == "partial"
    assert result["not_entered"] == 2794


def test_a_complete_engineering_run_is_not_scientific_success():
    result = execution_status(planned=240, entered=240, returned=240, verified=240, exceptions=0)
    assert result["state"] == "complete"
    assert result["scientific_success_inferred"] is False


@pytest.mark.parametrize(
    "changes",
    [{"entered": True}, {"returned": 11}, {"verified": -1}, {"exceptions": 1}],
)
def test_inconsistent_denominators_are_rejected(changes):
    values = dict(planned=10, entered=10, returned=10, verified=10, exceptions=0)
    values.update(changes)
    with pytest.raises(ValueError):
        execution_status(**values)


def test_failure_and_unverified_returns_cannot_be_complete():
    assert (
        execution_status(
            planned=10,
            entered=10,
            returned=9,
            verified=9,
            exceptions=1,
        )["state"]
        == "incomplete_or_failed"
    )
    assert (
        execution_status(
            planned=10,
            entered=10,
            returned=10,
            verified=9,
            exceptions=0,
        )["state"]
        == "incomplete_or_failed"
    )


def test_financial_example_crossing_zero_is_not_improvement_or_equivalence():
    result = error_comparison(0.22621, 0.22516, [-0.00661, 0.00809])
    assert result["point_direction"] == "higher"
    assert result["directional_support"] == "not_established"
    assert result["equivalence_inferred"] is False


def test_positive_synthetic_energy_difference_means_higher_error():
    result = error_comparison(9.296506070630434, 9.288567277550785)
    assert result["point_direction"] == "higher"
    assert result["directional_support"] == "not_assessed"


def test_supported_risk_choice_improvement_does_not_fill_financial_market_links():
    assert (
        error_comparison(0.173899, 0.192307, [-0.03361, -0.00309])["directional_support"]
        == "lower_error"
    )
    result = market_comparison_readiness(
        human_financial_improvement=False,
        independent_market_comparison=False,
        same_policy=False,
        same_market_rules=False,
        autonomous_prices=True,
    )
    assert result["ready_for_review"] is False
    assert "human_financial_improvement" in result["missing"]


@pytest.mark.parametrize("interval", [[0, 0.1], [-0.1, 0]])
def test_touching_zero_does_not_establish_a_direction(interval):
    assert error_comparison(0.2, 0.2, interval)["directional_support"] == "not_established"


@pytest.mark.parametrize("candidate", [float("nan"), float("inf"), -1, True])
def test_invalid_scores_cannot_enter_a_progress_report(candidate):
    with pytest.raises(ValueError):
        error_comparison(candidate, 0.2)


def test_running_unpublished_manifest_is_not_a_claim_of_a_live_job():
    result = publication_status({"status": "running", "outputs_published": False}, {})
    assert result == {"state": "not_published", "live_process_status": "not_checked"}


def test_published_flag_without_matching_bytes_is_insufficient():
    manifest = {
        "status": "complete",
        "outputs_published": True,
        "outputs": [{"path": "approved-summary.json", "sha256": "a" * 64}],
    }
    assert publication_status(manifest, {})["state"] == "identity_mismatch"
    assert (
        publication_status(manifest, {"approved-summary.json": "b" * 64})["state"]
        == "identity_mismatch"
    )
    assert publication_status(manifest, {"approved-summary.json": "a" * 64})["state"] == "published"


def test_all_links_present_allows_review_without_claiming_causal_proof():
    result = market_comparison_readiness(
        human_financial_improvement=True,
        independent_market_comparison=True,
        same_policy=True,
        same_market_rules=True,
        autonomous_prices=True,
    )
    assert result["ready_for_review"] is True
    assert result["scientific_claim_proven"] is False
