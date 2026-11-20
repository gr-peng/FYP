"""Run two review blocks with constructed prices and a scripted local client (zero API)."""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.schemas import OrderDecision, StyleDecision, hold
from behavioral_market.simulation.event_runner import run_experiment
from behavioral_market.simulation.finite_rule_pair import METHOD


def fictional_inputs():
    dates = pd.date_range("2025-12-01", "2026-01-10").strftime("%Y-%m-%d")
    bars = pd.DataFrame(
        {
            "symbol": "SYNTHETIC",
            "session_date": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 100,
        }
    )
    config = {
        "experiment": {"seed": 17},
        "data": {
            "primary_symbol": "SYNTHETIC",
            "context_symbols": [],
            "experiment_start": "2026-01-01",
            "experiment_end": "2026-01-10",
            "calendar": "24/7",
        },
        "population": {
            "n_agents": 1,
            "initial_fundamental_share": 1,
            "initial_cash": "1000",
            "initial_shares": 2,
        },
        "behavior": {
            "memory_window": 20,
            "strategy_switch_interval": 5,
            "mean_field_visibility": "hidden",
            "strategy_return_method": METHOD,
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
    return config, bars


class ScriptedLifecycleClient:
    """Predetermined answers exercise accounting and resetting, not model behavior."""

    def __init__(self):
        self.records = []

    async def decide(self, system, user, schema, identity):
        payload = json.loads(user)
        day = payload["observation"]["session_date"]
        if schema is StyleDecision:
            change = day == "2026-01-05"
            decision = StyleDecision(
                decision="switch" if change else "stay",
                target_style="technical",
                rationale="predetermined fixture",
            )
        elif day == "2026-01-01":
            decision = OrderDecision(
                action="buy",
                quantity=8,
                limit_price=105,
                sentiment="neutral",
                rationale="predetermined loss",
            )
        elif day == "2026-01-06":
            decision = OrderDecision(
                action="sell",
                quantity=2,
                limit_price=100,
                sentiment="neutral",
                rationale="predetermined sale",
            )
        else:
            decision = hold("predetermined waiting")
        self.records.append(
            {
                "identity": identity,
                "system": system,
                "user": payload,
                "decision": decision.model_dump(),
                "authority": "local_script",
            }
        )
        return decision, {"status": "mock"}


async def run_demo(output):
    output.mkdir(parents=True, exist_ok=False)
    config, bars = fictional_inputs()
    write_json(output / "fictional_config.json", config)
    bars.to_csv(output / "constructed_bars.csv", index=False)
    client = ScriptedLifecycleClient()
    await run_experiment(
        config,
        "historical",
        "vanilla_llm",
        "mock",
        output / "run",
        bars=bars,
        events=[],
        client=client,
        manifest_hash=digest((output / "constructed_bars.csv").read_bytes()),
    )
    write_json(output / "scripted_requests_and_answers.json", client.records)
    reviews = [
        row["user"]["agent_state"]["counterfactual"]
        for row in client.records
        if row["identity"]["kind"] == "style"
    ]
    write_json(
        output / "summary.json",
        {
            "evidence_kind": "constructed_two_block_lifecycle_demo",
            "model_POSTs": 0,
            "method": METHOD,
            "sessions": 10,
            "scripted_order_decisions": 10,
            "scripted_style_decisions": 2,
            "review_summaries": reviews,
            "limits": "Predetermined decisions and constructed constant prices; exercises the "
            "production lifecycle, not model intelligence, human realism or market realism.",
        },
    )
    first, second = reviews
    start = second["block_start_portfolio"]
    report = (
        "两次连续复盘演示（全部行情和选择均由本地程序设定，没有调用模型）\n\n"
        "起点是1000元现金和2股，每股100元；十天收盘价一直为100元。\n"
        "第一天故意用105元买8股，用来检验实际运行的模拟账户亏损会不会被算进策略比较。\n"
        f"第一次复盘：实际运行的模拟账户收益{first['observed_account_return']:.4%}；"
        f"两种规则账户收益分别为{first['current_rule_block_return']:.4%}和"
        f"{first['alternative_rule_block_return']:.4%}，差值{first['gap']:.4%}。\n"
        "第五天按预设答案切换策略。切换本身不买卖。\n"
        f"下一段从实际运行的模拟账户的{start['cash']}元现金、{start['position']}股重新开始；"
        f"每股成本{start['avg_cost']}元，按100元市价计总财富{start['total_equity']}元。\n"
        "第六天实际运行的模拟账户卖2股；两份规则账户仍按各自规则行动，持仓没有被这笔卖单改动。\n"
        f"第二次复盘的策略收益差为{second['gap']:.4%}。\n\n"
        "这说明两段计账、复盘、换策略和重新开始已经连通。选择是预设的，不能当成AI"
        "理解经济信息的证据。两份规则账户使用指定的报价和有限交易对手，也不能当成真实"
        "市场里的成交承诺。旧版汇总报告暂不接受新口径；本目录保留逐日记录及本报告。\n"
    )
    (output / "REPORT.txt").write_text(report, encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(
        output / "artifact_manifest.json",
        {
            "files": {str(p.relative_to(output)): digest(p.read_bytes()) for p in files},
            "driver_source_sha256": digest(Path(__file__).read_bytes()),
        },
    )
    return {"output": str(output), "model_POSTs": 0, "reviews": len(reviews)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
