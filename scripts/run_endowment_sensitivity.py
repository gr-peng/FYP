"""Frozen fixed-style cash/stock composition sensitivity; six runs, no model calls."""

import argparse
import asyncio
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from behavioral_market.data.archive import digest, write_json
from behavioral_market.simulation import event_runner
from scripts.audit_finite_lifecycle import audit, require
from scripts.diagnose_rule_market import analyze_run, number
from scripts.run_multiagent_rule_comparison import NoModelClient, config_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "fixed-style-endowment-sensitivity-v1"
SEEDS = (17, 29, 43)
ARMS = {
    "original": {"initial_cash": "1000", "initial_shares": 2},
    "equal_value": {"initial_cash": "600", "initial_shares": 6},
}


def configuration(seed, arm):
    config = config_for(seed, "fixed")
    config["data"]["experiment_end"] = "2026-01-30"
    config["population"].update(ARMS[arm])
    return config


def prepare(packet, design_path):
    design = json.loads(design_path.read_text())
    require(
        design["protocol"] == PROTOCOL
        and design["seeds"] == list(SEEDS)
        and design["accounts"] == 20
        and design["sessions"] == 30
        and design["arms"] == ARMS
        and design["new_model_POSTs"] == 0,
        "design differs from the declared six-run comparison",
    )
    packet.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(design_path, packet / "design.json")
    bars = pd.DataFrame(
        {
            "symbol": "SYNTHETIC",
            "session_date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100,
        }
    )
    bars.to_csv(packet / "warmup.csv", index=False)
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f"seed{seed}_{arm}"
            config_path = f"configs/{name}.json"
            write_json(packet / config_path, configuration(seed, arm))
            runs.append({"id": name, "seed": seed, "arm": arm, "config": config_path})
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/audit_finite_lifecycle.py",
        ROOT / "scripts/diagnose_rule_market.py",
        ROOT / "scripts/run_multiagent_rule_comparison.py",
        ROOT / "tests/test_endowment_sensitivity.py",
        ROOT / "pyproject.toml",
    ]
    sources = {}
    for path in source_paths:
        name = str(path.relative_to(ROOT))
        target = packet / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[name] = digest(path.read_bytes())
    inputs = ["design.json", "warmup.csv", *[r["config"] for r in runs]]
    plan = {
        "protocol": PROTOCOL,
        "new_model_POSTs": 0,
        "runs": runs,
        "sources": sources,
        "inputs": {name: digest((packet / name).read_bytes()) for name in inputs},
        "limits": design["analysis"],
    }
    write_json(packet / "plan.json", plan)
    (packet / "plan.sha256").write_text(digest((packet / "plan.json").read_bytes()) + "\n")
    return plan


def checked_plan(packet):
    sha = digest((packet / "plan.json").read_bytes())
    require(sha == (packet / "plan.sha256").read_text().strip(), "plan hash mismatch")
    plan = json.loads((packet / "plan.json").read_text())
    require(plan["protocol"] == PROTOCOL and plan["new_model_POSTs"] == 0, "wrong protocol")
    expected_runs = [
        {
            "id": f"seed{seed}_{arm}",
            "seed": seed,
            "arm": arm,
            "config": f"configs/seed{seed}_{arm}.json",
        }
        for seed in SEEDS
        for arm in ARMS
    ]
    require(plan["runs"] == expected_runs, "run identities differ")
    for name, expected in plan["sources"].items():
        require(
            digest((ROOT / name).read_bytes()) == expected
            and digest((packet / "sources" / name).read_bytes()) == expected,
            "frozen source changed: " + name,
        )
    for name, expected in plan["inputs"].items():
        require(digest((packet / name).read_bytes()) == expected, "frozen input changed: " + name)
    for row in plan["runs"]:
        require(
            json.loads((packet / row["config"]).read_text())
            == configuration(row["seed"], row["arm"]),
            "account composition or other configuration differs",
        )
    return plan


def summarize_run(run, arm):
    checked = audit(run)
    diagnostic = analyze_run(run)
    rows, daily = diagnostic["account_order_details"], diagnostic["daily"]
    require(len(rows) == 600 and len(daily) == 30, "incomplete 20-account/30-day run")
    reviews = pd.read_parquet(run / "style_switches.parquet")
    require(len(reviews) == 120 and reviews.decision.eq("stay").all(), "fixed styles changed")
    require(
        all(
            r["style"] == diagnostic["reconstructed_personas"][r["agent_id"]]["initial_style"]
            for r in rows
        ),
        "orders did not keep initial style",
    )
    actions = Counter(r["action"] for r in rows)
    clipped = Counter(r["requested_action"] for r in rows if r["known_active_size_clipped"])
    holds = Counter(r["hold_reason"] for r in rows if r["hold_reason"])
    last = daily[-1]
    accounts = pd.read_parquet(run / "agents.parquet")
    final = accounts[accounts.session_date == daily[-1]["session_date"]].sort_values("agent_id")
    summary = {
        "seed": diagnostic["seed"],
        "arm": arm,
        "initial_cash_each": ARMS[arm]["initial_cash"],
        "initial_shares_each": ARMS[arm]["initial_shares"],
        "initial_marked_wealth_each": "1200",
        "decisions": {side: actions[side] for side in ["buy", "sell", "hold"]},
        "active_size_clipping": {side: clipped[side] for side in ["buy", "sell"]},
        "hold_reasons": dict(holds),
        "quantity": {
            side: {
                "requested_after_policy": sum(
                    r["requested_quantity"] for r in rows if r["requested_action"] == side
                ),
                "accepted": sum(r["quantity"] for r in rows if r["action"] == side),
                "filled_order_side": sum(r["filled_quantity"] for r in rows if r["action"] == side),
                "unfilled": sum(r["unfilled_quantity"] for r in rows if r["action"] == side),
            }
            for side in ["buy", "sell"]
        },
        "single_side_traded_quantity": sum(d["single_side_traded_quantity"] for d in daily),
        "zero_trade_days": sum(d["single_side_traded_quantity"] == 0 for d in daily),
        "opening_zero_share_account_days": sum(d["opening_zero_share_accounts"] for d in daily),
        "opening_cash_cannot_buy_one_account_days": sum(
            d["opening_cash_cannot_buy_one"] for d in daily
        ),
        "final_zero_share_accounts": last["closing_zero_share_accounts"],
        "daily_closes": [d["close"] for d in daily],
        "final_valuation": {
            k: last[k]
            for k in [
                "total_cash",
                "total_shares",
                "total_marked_wealth",
                "wealth_change_from_price_mark",
                "sum_realized_pnl",
                "sum_unrealized_pnl",
                "net_cash_creation",
            ]
        },
        "final_individual_wealth": dict(zip(final.agent_id, final.total_equity, strict=True)),
        "checked_account_days": checked["checked_agent_days"],
        "checked_branch_days": checked["checked_branch_days"],
        "limitations": "Pre-clipping raw HOLD direction and policy noise were not logged; "
        "keep unknown. Counts are repeated account-days, not independent people/markets.",
    }
    require(
        summary["quantity"]["buy"]["filled_order_side"]
        == summary["quantity"]["sell"]["filled_order_side"]
        == summary["single_side_traded_quantity"],
        "volume counted more than once",
    )
    write_json(run / "accounting_audit.json", checked)
    write_json(run / "diagnostic.json", diagnostic)
    return summary, diagnostic["reconstructed_personas"]


def describe(result):
    lines = [
        "初始现金和股票比例会怎样影响程序市场？（零模型调用）",
        "",
        "每次20户、30天；每户起始财富都是1200。原组有1000现金和2股，另一组有600现金和6股。"
        "两组同一随机起点的投资者性格、规则和固定策略相同。三组随机起点全部保留。",
        "第二种设置让现金与股票的起始市值各占一半，但买卖报价不同，实际能买和能卖的数量仍不完全相等。",
        "",
    ]
    for pair in result["pairs"]:
        lines.append(f"随机起点{pair['seed']}：")
        for arm, label in [("original", "原组"), ("equal_value", "现金股票各半组")]:
            r = pair["arms"][arm]
            d = r["decisions"]
            lines.append(
                f"  {label}：买/卖/等待{d['buy']}/{d['sell']}/{d['hold']}次，"
                f"成交{r['single_side_traded_quantity']}股，"
                f"没有成交的天数{r['zero_trade_days']}，末价{r['daily_closes'][-1]}，期末没有股票{r['final_zero_share_accounts']}户。"
            )
            v = r["final_valuation"]
            lines.append(
                f"  全体现金始终{v['total_cash']}、股票始终{v['total_shares']}股；"
                f"期末总账面财富变化{v['wealth_change_from_price_mark']}来自股票重新估值，没有新现金产生。"
            )
        lines.append(
            f"  各半组减原组：成交量差{pair['equal_minus_original_volume']}股，"
            f"末价差{pair['equal_minus_original_final_close']}。"
        )
    lines += [
        "",
        "这次同时改变了市场总现金和总股票数，所以只能说明结果对初始配置有多敏感，"
        "不能单独证明库存造成某种价格走势。没有追求价格变平或成交增加，零成交同样保留。"
        "这仍是规则程序的检查，不能据此说AI更像真人、发现跟风或模拟市场更真实。",
        "",
    ]
    return "\n".join(lines)


async def execute(packet, output):
    plan = checked_plan(packet)
    preflight = json.loads((packet / "independent_preflight.json").read_text())
    require(
        preflight.get("accepted_for_execution") is True
        and preflight.get("mustfix") == []
        and preflight.get("plan_sha256") == digest((packet / "plan.json").read_bytes()),
        "independent preflight does not approve this plan",
    )
    output.mkdir(parents=True, exist_ok=False)
    bars = pd.read_csv(packet / "warmup.csv")
    summaries, personas = {}, {}
    for item in plan["runs"]:
        run = output / item["id"]
        await event_runner.run_experiment(
            json.loads((packet / item["config"]).read_text()),
            "endogenous",
            "rule_abm",
            "mock",
            run,
            bars=bars.copy(deep=True),
            events=[],
            client=NoModelClient(),
            manifest_hash=plan["inputs"]["warmup.csv"],
        )
        summaries[item["id"]], personas[item["id"]] = summarize_run(run, item["arm"])
    pairs = []
    for seed in SEEDS:
        first, second = [summaries[f"seed{seed}_{arm}"] for arm in ARMS]
        require(
            personas[f"seed{seed}_original"] == personas[f"seed{seed}_equal_value"],
            "paired personas differ",
        )
        pairs.append(
            {
                "seed": seed,
                "same_personas": True,
                "arms": {arm: summaries[f"seed{seed}_{arm}"] for arm in ARMS},
                "equal_minus_original_volume": second["single_side_traded_quantity"]
                - first["single_side_traded_quantity"],
                "equal_minus_original_final_close": str(
                    number(second["daily_closes"][-1]) - number(first["daily_closes"][-1])
                ),
            }
        )
    result = {
        "protocol": PROTOCOL,
        "new_model_POSTs": 0,
        "runs": 6,
        "pairs": pairs,
        "plan_sha256": digest((packet / "plan.json").read_bytes()),
        "limits": plan["limits"],
    }
    checked_plan(packet)
    write_json(output / "comparison.json", result)
    (output / "REPORT.txt").write_text(describe(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--prepare-design", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_design:
        require(args.output is None, "prepare and execute are separate")
        plan = prepare(args.packet, args.prepare_design)
        print(json.dumps({"prepared_runs": len(plan["runs"]), "new_model_POSTs": 0}))
    else:
        require(args.output is not None, "execution needs a new output directory")
        result = asyncio.run(execute(args.packet, args.output))
        print(json.dumps({"completed_runs": result["runs"], "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
