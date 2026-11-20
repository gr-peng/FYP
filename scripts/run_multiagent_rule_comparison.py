"""Predeclared 20-account, ten-session fixed/adaptive rule comparison; zero API."""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import pandas as pd

from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation import event_runner
from behavioral_market.simulation.finite_rule_pair import METHOD
from scripts.audit_finite_lifecycle import audit, describe, require

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
ARMS = ("fixed", "adaptive")


class NoModelClient:
    async def decide(self, *_args, **_kwargs):
        raise RuntimeError("This frozen rule comparison forbids model decisions")


def config_for(seed, arm):
    return {
        "experiment": {"seed": seed},
        "data": {
            "primary_symbol": "SYNTHETIC",
            "context_symbols": [],
            "experiment_start": "2026-01-01",
            "experiment_end": "2026-01-10",
            "calendar": "24/7",
        },
        "population": {
            "n_agents": 20,
            "initial_fundamental_share": 0.5,
            "initial_cash": "1000",
            "initial_shares": 2,
        },
        "behavior": {
            "memory_window": 20,
            "strategy_switch_interval": 5,
            "mean_field_visibility": "hidden",
            "strategy_return_method": METHOD,
            "rule_style_adaptation": arm,
            "finite_rule_pair": {
                "half_spread": "0.05",
                "counterparty_cash": "10000",
                "counterparty_shares": 100,
            },
        },
        "fundamental_shock": {"enabled": False, "magnitude": 0},
        "events": {"condition": "no_event"},
        "llm": {},
    }


def prepare(directory, design_path):
    directory.mkdir(parents=True, exist_ok=False)
    design = json.loads(design_path.read_text())
    require(
        design["schedule"]["seeds"] == list(SEEDS)
        and design["population"]["n_agents"] == 20
        and design["remote_POSTs_planned"] == 0,
        "unexpected preregistered design",
    )
    shutil.copyfile(design_path, directory / "design.json")
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
    bars.to_csv(directory / "warmup.csv", index=False)
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f"seed{seed}_{arm}"
            path = directory / "configs" / (name + ".json")
            write_json(path, config_for(seed, arm))
            runs.append(
                {"id": name, "seed": seed, "arm": arm, "config": str(path.relative_to(directory))}
            )
    sources = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/audit_finite_lifecycle.py",
        ROOT / "pyproject.toml",
    ]
    records = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = directory / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[str(relative)] = digest(source.read_bytes())
    inputs = [
        directory / "design.json",
        directory / "warmup.csv",
        *sorted((directory / "configs").glob("*.json")),
    ]
    plan = {
        "protocol": "multiagent-finite-rule-comparison-v1",
        "new_model_POSTs": 0,
        "runs": runs,
        "sources": records,
        "inputs": {str(p.relative_to(directory)): digest(p.read_bytes()) for p in inputs},
        "comparison": "same-seed first five sessions identical before review application; "
        "describe sessions six through ten; retain every seed and zero differences",
        "limits": "Programmed rules only. No human alignment, model inference, explicit "
        "population-information responsiveness or market-realism claim.",
    }
    write_json(directory / "plan.json", plan)
    (directory / "plan.sha256").write_text(digest((directory / "plan.json").read_bytes()) + "\n")
    return plan


def checked_plan(directory):
    require(
        digest((directory / "plan.json").read_bytes())
        == (directory / "plan.sha256").read_text().strip(),
        "plan hash mismatch",
    )
    plan = json.loads((directory / "plan.json").read_text())
    for path, expected in plan["sources"].items():
        require(
            digest((ROOT / path).read_bytes()) == expected
            and digest((directory / "sources" / path).read_bytes()) == expected,
            "execution source differs from frozen source",
        )
    for path, expected in plan["inputs"].items():
        require(digest((directory / path).read_bytes()) == expected, "frozen input changed")
    return plan


def compare_pair(output, seed):
    tables = {}
    for arm in ARMS:
        run = output / f"seed{seed}_{arm}"
        tables[arm] = {
            name: pd.read_parquet(run / (name + ".parquet"))
            for name in ["market", "orders", "trades", "agents", "style_switches"]
        }
    for name in ["market", "orders", "trades", "agents"]:
        first = [
            session_slice(tables[arm][name], end="2026-01-05").reset_index(drop=True)
            for arm in ARMS
        ]
        # An entirely untraded run is exported without trade columns. An arm
        # that only trades later has columns, but the same empty first block.
        if not all(frame.empty for frame in first):
            pd.testing.assert_frame_equal(*first)
    first_reviews = [
        tables[arm]["style_switches"]
        .query("session_date == '2026-01-05'")
        .sort_values("agent_id")
        .reset_index(drop=True)
        for arm in ARMS
    ]
    columns = [
        "agent_id",
        "from_style",
        "observed_account_return",
        "current_rule_block_return",
        "alternative_rule_block_return",
        "paired_return_gap",
    ]
    pd.testing.assert_frame_equal(first_reviews[0][columns], first_reviews[1][columns])
    results = {}
    for arm in ARMS:
        frame = tables[arm]
        late_market = frame["market"].query("session_date >= '2026-01-06'")
        late_trades = session_slice(frame["trades"], start="2026-01-06")
        close_accounts = (
            frame["agents"].query("session_date == '2026-01-10'").sort_values("agent_id")
        )
        shares = (
            frame["agents"]
            .query("session_date >= '2026-01-06'")
            .groupby("session_date")
            .current_style.apply(lambda values: float((values == "fundamental").mean()))
        )
        review = frame["style_switches"]
        results[arm] = {
            "first_review_switches": int(
                ((review.session_date == "2026-01-05") & (review.decision == "switch")).sum()
            ),
            "terminal_review_switches": int(
                ((review.session_date == "2026-01-10") & (review.decision == "switch")).sum()
            ),
            "second_block_closes": late_market.close.tolist(),
            "second_block_single_side_trade_quantity": (
                0 if late_trades.empty else int(late_trades.quantity.sum())
            ),
            "second_block_fundamental_share_before_daily_review": shares.to_dict(),
            "final_account_wealth": dict(
                zip(close_accounts.agent_id, close_accounts.total_equity, strict=True)
            ),
        }
        daily = frame["agents"].groupby("session_date")
        require(
            daily.cash.sum().sub(20000).abs().max() <= 0.00002, "actual cash conservation failed"
        )
        require(daily.position.sum().eq(40).all(), "actual shares conservation failed")
    gaps = [
        a - f
        for a, f in zip(
            results["adaptive"]["second_block_closes"],
            results["fixed"]["second_block_closes"],
            strict=True,
        )
    ]
    if results["adaptive"]["first_review_switches"] == 0:
        for name in ["market", "orders", "trades", "agents"]:
            later = [
                session_slice(tables[arm][name], start="2026-01-06").reset_index(drop=True)
                for arm in ARMS
            ]
            if not all(frame.empty for frame in later):
                pd.testing.assert_frame_equal(*later)
    return {
        "seed": seed,
        "first_block_equal": True,
        "no_first_switch_later_equality_checked": (
            results["adaptive"]["first_review_switches"] == 0
        ),
        "arms": results,
        "adaptive_minus_fixed_close": gaps,
        "max_absolute_close_difference": max(map(abs, gaps)),
        "interpretation": "No first-review switches means this setting did not activate the "
        "policy difference. Terminal-review switches cannot explain earlier prices.",
    }


def session_slice(frame, *, start=None, end=None):
    if frame.empty:
        return frame
    selected = pd.Series(True, index=frame.index)
    if start is not None:
        selected &= frame.session_date >= start
    if end is not None:
        selected &= frame.session_date <= end
    return frame.loc[selected]


async def execute(directory, output):
    plan = checked_plan(directory)
    output.mkdir(parents=True, exist_ok=False)
    bars = pd.read_csv(directory / "warmup.csv")
    for item in plan["runs"]:
        config = json.loads((directory / item["config"]).read_text())
        run = output / item["id"]
        await event_runner.run_experiment(
            config,
            "endogenous",
            "rule_abm",
            "mock",
            run,
            bars=bars.copy(deep=True),
            events=[],
            client=NoModelClient(),
            manifest_hash=plan["inputs"]["warmup.csv"],
        )
        checked = audit(run)
        write_json(run / "accounting_audit.json", checked)
        (run / "accounting_REPORT.txt").write_text(describe(checked))
    pairs = [compare_pair(output, seed) for seed in SEEDS]
    result = {
        "protocol": plan["protocol"],
        "new_model_POSTs": 0,
        "plan_sha256": digest((directory / "plan.json").read_bytes()),
        "pairs": pairs,
        "runs": 6,
        "agents_per_run": 20,
        "sessions_per_run": 10,
        "limits": plan["limits"],
    }
    write_json(output / "comparison.json", result)
    lines = [
        "20个模拟账户的策略对照（全部是本地规则，没有模型调用）",
        "",
        "同一随机起点比较两种设置：每五天照常复盘但保持策略；允许根据两份规则账的收益差换策略。",
        "3个预先固定的随机起点全部保留；每组前五天的订单、市场价格和账户完全相同。",
        "",
    ]
    for pair in pairs:
        arms = pair["arms"]
        lines.append(
            f"起点{pair['seed']}：允许换策略的一组，第五天有{arms['adaptive']['first_review_switches']}人换策略；"
            f"随后五天两组最大的价格差为{pair['max_absolute_close_difference']:.6f}。"
            f"后五天成交股数分别为保持组{arms['fixed']['second_block_single_side_trade_quantity']}、"
            f"允许更新组{arms['adaptive']['second_block_single_side_trade_quantity']}。"
        )
    lines += [
        "",
        "若第五天没有人换策略，后面的相同结果说明这套情境没有激活两组的区别。"
        "第十天的换策略发生在最后，不能拿来解释之前的价格。",
        "这些程序不读取群体比例，不能据此声称发现跟风，也没有证明真人行为或真实市场。"
        "20个账户达到RP规模下限；10天仍短于原计划30期，不能用来检验长期市场规律。",
        "",
    ]
    (output / "REPORT.txt").write_text("\n".join(lines))
    checked_plan(directory)
    files = [p for p in output.rglob("*") if p.is_file()]
    write_json(
        output / "artifact_manifest.json",
        {str(p.relative_to(output)): digest(p.read_bytes()) for p in files},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--prepare-design", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_design:
        require(args.output is None, "prepare and execute are separate operations")
        plan = prepare(args.packet, args.prepare_design)
        print(json.dumps({"prepared_runs": len(plan["runs"]), "new_model_POSTs": 0}))
    else:
        require(args.output is not None, "execution needs an exclusive output directory")
        result = asyncio.run(execute(args.packet, args.output))
        print(json.dumps({"completed_runs": result["runs"], "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
