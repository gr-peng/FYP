#!/usr/bin/env python3
"""Check gross versus net period funding on already-opened train/development only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finance_data import load_development_rows, sha256  # noqa: E402
from experiments.scopic_states import pre_ranking_decisions  # noqa: E402


def audit():
    plan_path = ROOT / "config/frozen/financial_data_partitions_20260922.json"
    plan = json.loads(plan_path.read_text())
    report = {"partition_sha256": sha256(plan_path), "test_read": False, "splits": {}}
    for split in ("train", "development"):
        decisions = pre_ranking_decisions(load_development_rows(ROOT, plan, "scopic", split))
        exceptions = []
        for decision in decisions:
            state, action = decision["state"], decision["target"]
            gross_buy = sum(action["buy"][a] * p for a, p in state["current_prices"].items())
            gross_sale = sum(action["sell"][a] * p for a, p in state["current_prices"].items())
            if gross_buy > state["cash"]:
                exceptions.append(
                    {
                        **decision["provenance"],
                        "cash_before": state["cash"],
                        "gross_buy": gross_buy,
                        "gross_sale": gross_sale,
                        "cash_after": state["cash"] + gross_sale - gross_buy,
                    }
                )
        report["splits"][split] = {
            "decisions": len(decisions),
            "gross_buy_exceeds_cash_count": len(exceptions),
            "all_exceptions_funded_by_same_period_sales": all(
                x["cash_after"] >= 0 for x in exceptions
            ),
            "exceptions": exceptions,
        }
    output = ROOT / "outputs/financial_training_audit_20260922/instruction_constraint_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {k: v for k, v in x.items() if k != "exceptions"}
                for s, x in report["splits"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    audit()
