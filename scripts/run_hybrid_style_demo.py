"""Five scripted controls for the hybrid lifecycle; zero network or credentials."""

import argparse
import asyncio
import hashlib
import json

import pandas as pd

from behavioral_market.data.archive import write_json
from behavioral_market.llm.schemas import StyleDecision
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.hybrid_review import PROTOCOL
from scripts.audit_finite_lifecycle import audit, require
from scripts.run_multiagent_rule_comparison import NoModelClient, config_for


class ScriptedClient:
    hybrid_review_protocol = PROTOCOL
    review_provenance = "scripted"

    def __init__(self, switch_day=None, fail_at=None):
        self.calls = 0
        self.switch_day, self.fail_at = switch_day, fail_at

    async def decide(self, system, user, schema, identity):
        require(schema is StyleDecision and identity["kind"] == "style", "style-only interface")
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("expected scripted interruption on third review")
        state = json.loads(user)["agent_state"]
        current = state["current_style"]
        switch = identity["session_date"] == self.switch_day
        target = ("technical" if current == "fundamental" else "fundamental") if switch else current
        return StyleDecision(
            decision="switch" if switch else "stay",
            target_style=target,
            rationale="Scripted engineering control, not a real model response.",
        ), {
            "status": "ok",
            "fallback": False,
            "cache_hit": False,
            "repair_count": 0,
            "request_hash": hashlib.sha256((system + user).encode()).hexdigest(),
        }


def setup():
    config = config_for(20261106, "fixed")
    config["data"]["experiment_end"] = "2026-01-30"
    dates = pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d")
    bars = pd.DataFrame(
        {
            "symbol": "SYNTHETIC",
            "session_date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100,
        }
    )
    return config, bars


def compare(first, second, end=None):
    for name in ("market", "orders", "trades", "agents"):
        frames = [pd.read_parquet(folder / f"{name}.parquet") for folder in (first, second)]
        if end:
            frames = [frame[frame.session_date <= end].reset_index(drop=True) for frame in frames]
        pd.testing.assert_frame_equal(*frames)


async def execute(directory):
    directory.mkdir(parents=True, exist_ok=False)
    summary = {
        "new_model_POSTs": 0,
        "evidence_kind": "scripted_engineering_controls",
        "seed": 20261106,
        "accounts": 20,
        "configured_sessions": 30,
        "runs": {},
    }
    variants = {
        "fixed": None,
        "stay": ScriptedClient(),
        "switch_day5": ScriptedClient("2026-01-05"),
        "switch_day30": ScriptedClient("2026-01-30"),
        "failure_third": ScriptedClient("2026-01-05", fail_at=3),
    }
    for name, client in variants.items():
        config, bars = setup()
        if client:
            del config["behavior"]["rule_style_adaptation"]
        output = directory / name
        try:
            await event_runner.run_experiment(
                config,
                "endogenous",
                "hybrid_style_llm" if client else "rule_abm",
                "mock",
                output,
                bars=bars,
                events=[],
                client=client or NoModelClient(),
            )
        except RuntimeError as exc:
            if (
                name != "failure_third"
                or str(exc) != "expected scripted interruption on third review"
            ):
                raise
            metadata = json.loads((output / "run_metadata.json").read_text())
            require(
                metadata["completed_market_sessions"] == 5 and metadata["applied_review_rows"] == 0,
                "partial review was applied",
            )
            summary["runs"][name] = {
                "status": "expected_interruption",
                "scripted_calls": client.calls,
                "completed_market_sessions": 5,
                "applied_reviews": 0,
            }
        else:
            result = audit(output)
            write_json(output / "accounting_audit.json", result)
            summary["runs"][name] = {
                "status": "complete",
                "scripted_calls": client.calls if client else 0,
                "checked_reviews": result["checked_reviews"],
            }
    compare(directory / "fixed", directory / "stay")
    compare(directory / "fixed", directory / "switch_day5", "2026-01-05")
    compare(directory / "fixed", directory / "switch_day30")
    fixed = pd.read_parquet(directory / "fixed/orders.parquet")
    changed = pd.read_parquet(directory / "switch_day5/orders.parquet")
    summary["checks"] = {
        "stay_equals_fixed_all30": True,
        "day5_switch_first5_unchanged": True,
        "terminal_switch_all30_trading_unchanged": True,
        "day6_action_differences": int(
            ((fixed.session_date == "2026-01-06") & (fixed.action != changed.action)).sum()
        ),
    }
    summary["limitations"] = (
        "Scripted choices validate causal timing and accounting only. "
        "Not real model inference, human alignment, strategy superiority "
        "or realistic market evidence."
    )
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary))
    return summary


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.output))


if __name__ == "__main__":
    main()
