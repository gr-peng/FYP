#!/usr/bin/env python3
"""Build an offline RP progress report from existing opened summaries only."""

import argparse
import json
import os
import resource
import sys
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.research_progress import (  # noqa: E402
    build_findings,
    read_sources,
    report_markdown,
    verify_sources_unchanged,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 2**30, 8 * 2**30))
    values, identities = read_sources(args.workspace)
    findings = build_findings(values, identities)
    report = report_markdown(findings, args.as_of.isoformat())
    verify_sources_unchanged(identities)
    findings["as_of_date_Asia_Shanghai"] = args.as_of.isoformat()
    findings["built_utc"] = datetime.now(UTC).isoformat()
    findings["limitations"] = [
        "Summary reconciliation, not independent replay of all historic raw data.",
        "Endpoint equality is not continuous immutability or exclusive-write proof.",
        "No raw/sealed data, API, model, sampler or market-generation call.",
        "New software tests do not establish the RP scientific hypothesis.",
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "evidence_audit.json", output / f"进度报告_{args.as_of:%Y%m%d}.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Choose a new output directory; historical reports are preserved")
    targets[0].write_text(
        json.dumps(findings, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    targets[1].write_text(report)
    print(
        json.dumps(
            {
                "report": str(targets[1]),
                "evidence": str(targets[0]),
                "summary_files_checked": len(identities),
                "financial_improvement": findings["financial"]["directional_support"],
                "micro_to_market_ready": findings["micro_to_market"]["ready_for_review"],
                "paused_batch_new_calls": findings["batch"]["entered"],
                "new_model_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
