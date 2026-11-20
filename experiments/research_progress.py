"""Read allowlisted, already-opened result summaries and map them to the RP.

This module never imports experiment runners, opens raw/held-out data, follows
paths from a manifest, starts jobs, or calls an external model. Stored hashes
describe the inspected bytes; they do not certify continuous write exclusion.
"""

import hashlib
import json
import math
from pathlib import Path
from string import Template

from behavioral_market.evaluation.research_evidence import (
    error_comparison,
    execution_status,
    market_comparison_readiness,
    publication_status,
)

REPO = "behavioral-llm-market/"
SOURCES = {
    "persona": REPO + "outputs/human_benchmark/choices13k_confirmatory/evaluation/summary.json",
    "examples": REPO + "outputs/human_benchmark/choices13k_data_driven/evaluation/summary.json",
    "external": REPO
    + "outputs/human_benchmark/cpc18_external/aigc_relay/test/evaluation/summary.json",
    "financial": REPO + "outputs/financial_probability_development_20261004/analysis.json",
    "events": REPO
    + "outputs/two_event_20261004_v1/two_event_month_extension_v6_analysis/summary.json",
    "crypto_manifest": REPO
    + "outputs/crypto_execution_diagnostic_20261011/train_summary_20261013_v1/manifest.json",
    "crypto_summary": REPO
    + "outputs/crypto_execution_diagnostic_20261011/train_summary_20261013_v1/summary.json",
    "flow_manifest": REPO
    + "outputs/crypto_effective_flow_extraction_20261013/train_20261015_v1/manifest.json",
    "matrix": "experiments/final63_20261018T011504Z/batch_summary.json",
    "matrix_pairs": "experiments/final63_20261018T011504Z/method_comparisons.json",
    "preflight": "experiments/scit_preflight_20261018T015315Z/preflight_summary.json",
    "timing": "experiments/scit_timing_20261018T020924Z/timing_summary.json",
    "batch": "experiments/scit_batch_20261018T022314Z/batch_status.json",
}


def read_sources(workspace, sources=None):
    """Return one identified snapshot of explicitly allowlisted summary files."""
    root = Path(workspace).resolve()
    values, identities = {}, {}
    for key, relative in (SOURCES if sources is None else sources).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Source leaves workspace: {key}")
        data = path.read_bytes()
        if len(data) > 8 * 2**20:
            raise ValueError(f"Expected a small summary, not raw data: {key}")
        values[key] = json.loads(data)
        identities[key] = {
            "path": str(path),
            "relative_path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return values, identities


def verify_sources_unchanged(identities):
    """Endpoint content check only; not isolation or continuous immutability."""
    for key, identity in identities.items():
        data = Path(identity["path"]).read_bytes()
        if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError(f"Summary changed during report construction: {key}")


def build_findings(values, identities):
    persona = values["persona"]
    pc = persona["primary_contrast"]
    persona_result = error_comparison(
        persona["metrics"]["P1"]["mae"],
        persona["metrics"]["P0"]["mae"],
        pc["bootstrap_95_ci"],
    )
    examples = values["examples"]
    selected = examples["selected_method"]["selected_method"]
    example_result = error_comparison(
        examples["metrics"][selected]["mae"],
        examples["metrics"]["B0"]["mae"],
        examples["primary_contrasts"][selected + "_vs_B0"]["bootstrap_95_ci"],
    )
    external = values["external"]
    external_result = error_comparison(
        external["metrics"]["P_true"]["mae"],
        external["metrics"]["P0"]["mae"],
        external["contrasts"]["P_true_minus_P0"]["ci95"]["joint"],
    )
    external_control = error_comparison(
        external["metrics"]["P_true"]["mae"],
        external["metrics"]["P_shuffled"]["mae"],
        external["contrasts"]["P_true_minus_P_shuffled"]["ci95"]["joint"],
    )
    financial = values["financial"]
    methods = financial["overall"]["methods"]
    financial_result = error_comparison(
        methods["real_labels"]["brier_session_person_balanced"],
        methods["no_examples"]["brier_session_person_balanced"],
        financial["contrasts"]["real_labels-minus-no_examples"]["percentile_95"],
    )
    financial_statistical = error_comparison(
        methods["real_labels"]["brier_session_person_balanced"],
        methods["memory_ridge_logistic"]["brier_session_person_balanced"],
        financial["contrasts"]["real_labels-minus-memory_ridge_logistic"]["percentile_95"],
    )
    pairs = values["matrix_pairs"]["pairs"]
    pair_keys = [(p["ordering"], p["delay_ms"]) for p in pairs]
    if len(pair_keys) != len(set(pair_keys)) or set(pair_keys) != {
        (ordering, delay) for ordering in ("S1", "S2") for delay in (0, 1000, 5000)
    }:
        raise ValueError("Missing or duplicated fixed synthetic comparison")
    pair_results = []
    for pair in pairs:
        if not pair["both_complete"]:
            raise ValueError("Cannot report a full synthetic comparison from an incomplete pair")
        for method in ("M0", "M1"):
            if not pair["methods"][method]["complete"]:
                raise ValueError("Method receipt contradicts pair completeness")
            if pair["methods"][method]["received_paths"] != 20:
                raise ValueError("Unexpected synthetic path denominator")
        comparison = error_comparison(
            pair["methods"]["M1"]["energy_score"],
            pair["methods"]["M0"]["energy_score"],
        )
        if not math.isclose(
            comparison["difference"],
            pair["M1_minus_M0_energy_score"],
            abs_tol=1e-12,
        ):
            raise ValueError("Stored synthetic contrast differs from its component scores")
        pair_results.append(
            {"ordering": pair["ordering"], "delay_ms": pair["delay_ms"], **comparison}
        )
    matrix = values["matrix"]
    matrix_result = execution_status(
        planned=matrix["expected_full_matrix"],
        entered=matrix["cumulative_received"],
        returned=matrix["cumulative_complete"],
        verified=matrix["cumulative_complete"],
        exceptions=0,
    )
    pre = values["preflight"]
    preflight = execution_status(
        planned=pre["planned_calls"],
        entered=pre["actual_select_entries"],
        returned=pre["actual_select_natural_returns"],
        verified=pre["verified_draws"],
        exceptions=pre["caught_exceptions"],
    )
    batch = values["batch"]
    batch_result = execution_status(
        planned=batch["planned"],
        entered=batch["actual_entries"],
        returned=batch["returns"],
        verified=batch["verified_contracts"],
        exceptions=batch["exceptions"],
    )
    timing = values["timing"]
    if not timing["same_first100_preflight_return_bytes"]:
        raise ValueError("Timing prefix no longer matches its reviewed reference")
    crypto = publication_status(
        values["crypto_manifest"],
        {identities["crypto_summary"]["path"]: identities["crypto_summary"]["sha256"]},
    )
    if crypto["state"] != "published":
        raise ValueError("Crypto execution summary is not authenticated by its publication record")
    events = []
    for contrast in values["events"]["contrasts"]:
        actual = values["events"]["matching_window_actual_returns"][contrast["scenario"]]
        events.append(
            {
                "symbol": contrast["symbol"],
                "real_return": actual["return"],
                "news_return": contrast["news_return"]["mean"],
                "no_news_return": contrast["no_news_return"]["mean"],
                "baseline_date": actual["baseline_date"],
                "end_date": actual["end_date"],
                "seeds": len(contrast["by_seed"]),
            }
        )
    return {
        "scope": "Existing opened summaries only; no new model calls or market experiments.",
        "persona": {"items": persona["items"], **persona_result},
        "risk_examples": {"items": examples["items"], **example_result},
        "external_risk": {"items": external["common_items"], **external_result},
        "external_shuffled_control": external_control,
        "external_statistical_error": external["metrics"]["symmetric_fractional_logistic"]["mae"],
        "financial": {
            "decisions": financial["overall"]["records"],
            "responses": financial["responses"],
            "sessions": len(financial["per_session"]),
            **financial_result,
        },
        "financial_vs_statistical": financial_statistical,
        "financial_methods": {
            name: methods[name]["brier_session_person_balanced"]
            for name in ("no_examples", "real_labels", "shuffled_labels", "memory_ridge_logistic")
        },
        "events": events,
        "synthetic_matrix": matrix_result,
        "synthetic_pairs": pair_results,
        "synthetic_lower_error_pairs": sum(p["point_direction"] == "lower" for p in pair_results),
        "preflight": preflight,
        "batch": batch_result,
        "batch_current_decision": "paused_after_accepted_boundary_review",
        "batch_decision_source": (
            "Conversation review accepted the blocker and ended review waiting; "
            "not a new process probe."
        ),
        "timing_replayed_prefix_calls": timing["actual_entries"],
        "timing_additional_unique_observations": 0,
        "flow_extraction": publication_status(values["flow_manifest"], {}),
        "flow_recorded_status": values["flow_manifest"].get("status"),
        "crypto_execution_summary": crypto,
        "micro_to_market": market_comparison_readiness(
            human_financial_improvement=False,
            independent_market_comparison=False,
            same_policy=False,
            same_market_rules=False,
            autonomous_prices=True,
        ),
        "historical_full_extraction_status_is_not_live_process_evidence": True,
        "new_scientific_success_claimed": False,
        "inputs": identities,
    }


def validate_report_interpretation(findings):
    """The dated prose is not an evergreen monitor: changed outcomes need review."""
    expected_support = {
        "persona": "higher_error",
        "risk_examples": "lower_error",
        "external_risk": "lower_error",
        "external_shuffled_control": "not_established",
        "financial": "not_established",
        "financial_vs_statistical": "higher_error",
    }
    for name, expected in expected_support.items():
        if findings[name]["directional_support"] != expected:
            raise ValueError(f"Result changed; review the dated report narrative: {name}")
    if (
        findings["synthetic_matrix"]["verified"] != 240
        or findings["synthetic_matrix"]["state"] != "complete"
        or findings["synthetic_lower_error_pairs"] != 0
        or findings["preflight"]["planned"] != 4320
        or findings["preflight"]["verified"] != 1526
        or findings["batch"]["entered"] != 0
        or findings["batch"]["planned"] != 4320
        or findings["flow_extraction"]["state"] != "not_published"
        or findings["flow_recorded_status"] != "running"
        or findings["timing_replayed_prefix_calls"] != 100
        or findings["micro_to_market"]["ready_for_review"]
    ):
        raise ValueError("Execution evidence changed; review the dated report narrative")


def report_markdown(findings, as_of):
    validate_report_interpretation(findings)
    f = findings
    risk_reduction = 100 * (1 - f["risk_examples"]["candidate"] / f["risk_examples"]["reference"])
    finance_rows = "\n".join(
        f"| {label} | {f['financial_methods'][key]:.5f} |"
        for key, label in (
            ("no_examples", "普通 AI，不给真人示例"),
            ("real_labels", "给正确的真人交易示例"),
            ("shuffled_labels", "给打乱买卖标签的示例"),
            ("memory_ridge_logistic", "根据历史记录计算的简单统计模型"),
        )
    )
    event_rows = "\n".join(
        f"| {e['symbol']}，{e['baseline_date']}—{e['end_date']} | "
        f"{100 * e['real_return']:+.2f}% | {100 * e['news_return']:+.2f}% |"
        for e in f["events"]
    )
    template = Path(__file__).resolve().parents[1] / "docs/templates/rp_progress_zh.md"
    return Template(template.read_text()).substitute(
        {
            "as_of": f"{as_of}",
            "persona_items": f"{f['persona']['items']:,}",
            "risk_items": f"{f['risk_examples']['items']}",
            "risk_reference_error": f"{f['risk_examples']['reference']:.4f}",
            "risk_example_error": f"{f['risk_examples']['candidate']:.4f}",
            "risk_error_reduction": f"{risk_reduction:.1f}",
            "external_items": f"{f['external_risk']['items']}",
            "financial_sessions": f"{f['financial']['sessions']}",
            "financial_decisions": f"{f['financial']['decisions']}",
            "financial_responses": f"{f['financial']['responses']:,}",
            "financial_table": f"{finance_rows}",
            "event_table": f"{event_rows}",
            "synthetic_complete": f"{f['synthetic_matrix']['verified']}",
            "synthetic_improved_pairs": f"{f['synthetic_lower_error_pairs']}",
            "preflight_verified": f"{f['preflight']['verified']:,}",
            "preflight_unrun": f"{f['preflight']['not_entered']:,}",
        }
    )
