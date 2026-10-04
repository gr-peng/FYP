"""Independent raw-output audit and Chinese report for the first two-event pilot."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavioral_market.data.archive import digest, write_json
from behavioral_market.llm.schemas import OrderDecision


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit_run(path):
    metadata = read_json(path / "run_metadata.json")
    assert metadata["status"] == "complete", path
    assert metadata["dirty_worktree"] is False, path
    config = read_json(path / "config.json")
    identity = {
        key: metadata[key]
        for key in ("scenario", "seed", "treatment", "event_condition", "liquidity", "provider")
    }
    assert (
        digest(json.dumps([config, identity], sort_keys=True).encode()) == metadata["config_hash"]
    )
    result = read_json(path / "metrics.json")
    decisions = records(path / "decisions.jsonl")
    responses = records(path / "transport/llm_responses.jsonl")
    response_by_hash = {
        r["request_hash"]: r["response"] for r in responses if r.get("response") is not None
    }
    proposed = Counter()
    for row in decisions:
        key = row["response_metadata"]["request_hash"]
        raw = response_by_hash[key]["choices"][0]["message"]["content"]
        assert not response_by_hash[key]["choices"][0]["message"].get("reasoning_content")
        assert OrderDecision.model_validate_json(raw).model_dump() == row["decision"]
        proposed[row["decision"]["action"]] += 1
        obs = row["payload"]["observation"]
        day = row["decision_at"][:10]
        assert obs["completed_history_through"] < day
        assert all(
            pd.Timestamp(n["published_at"]) <= pd.Timestamp(row["decision_at"]) for n in obs["news"]
        )
        if metadata["event_condition"] == "no_news":
            assert obs["news"] == []
    assert len(decisions) == result["decisions"]
    assert dict(proposed) == {k: v for k, v in result["proposed_actions"].items() if v}
    trades = read_json(path / "trades.json")
    assert len(trades) == result["trades"]
    participant_lots = sum(
        t["quantity"]
        * (int(t["buyer_id"] != "BACKGROUND_LP") + int(t["seller_id"] != "BACKGROUND_LP"))
        for t in trades
    )
    assert participant_lots == sum(r["execution"]["filled_lots"] for r in decisions)
    assert float(result["cash_conservation"]) == 0 and result["lot_conservation"] == 0
    daily = read_json(path / "daily.json")
    initial = read_json(path / "initial_state.json")["price_per_lot"]
    assert np.isclose(daily[-1]["close_per_lot"] / initial - 1, result["return"], atol=1e-12)
    return result, decisions, responses


def main(root):
    config = read_json(root / "config/two_event_market_pilot_v4.json")
    study = read_json(root / config["data_config"])
    base = root / "outputs" / study["study_id"]
    directory = base / config["experiment_id"] / "aigc_relay"
    result_rows, decisions_by_case, all_responses = [], {}, []
    for path in sorted(directory.rglob("run_metadata.json")):
        if "attempts" in path.parts:
            continue
        result, decisions, responses = audit_run(path.parent)
        result_rows.append(result)
        key = tuple(result[k] for k in ("scenario", "treatment", "liquidity", "event_condition"))
        decisions_by_case[key] = decisions
        all_responses.extend(responses)
    expected_runs = (
        len(config["scenarios"])
        * len(config["treatments"])
        * len(config["liquidity_conditions"])
        * len(config["event_conditions"])
    )
    assert len(result_rows) == expected_runs
    pre_news_pairs = 0
    real_returns = {}
    for spec in config["scenarios"]:
        first = pd.Timestamp(spec["events"][0]["at"])
        for treatment in config["treatments"]:
            for liquidity in config["liquidity_conditions"]:
                prefix = spec["scenario_id"], treatment, liquidity
                sides = [
                    decisions_by_case[(*prefix, condition)] for condition in ("news", "no_news")
                ]
                assert [(r["decision_at"], r["agent_id"]) for r in sides[0]] == [
                    (r["decision_at"], r["agent_id"]) for r in sides[1]
                ]
                for left, right in zip(*sides, strict=True):
                    if pd.Timestamp(left["decision_at"]) < first:
                        assert left["payload"] == right["payload"]
                        assert left["decision"] == right["decision"]
                        pre_news_pairs += 1
        bars = pd.read_parquet(
            root / "data/processed" / study["study_id"] / f"{spec['scenario_id']}_daily.parquet"
        )
        bars = bars[bars.symbol.eq(spec["symbol"])].sort_values("session_date")
        origin = bars[bars.session_date < spec["start"]].iloc[-1].close
        end = bars[bars.session_date.eq(spec["end"])].iloc[0].close
        real_returns[spec["scenario_id"]] = float(end / origin - 1)
    usage = Counter()
    status = Counter()
    for row in all_responses:
        status[str(row.get("http_status", row.get("status")))] += 1
        if not row.get("cached") and row.get("response") is not None:
            for key, value in row["response"].get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    summary = {
        "runs": result_rows,
        "raw_response_rows": len(all_responses),
        "response_statuses": dict(status),
        "cache_hits": sum(bool(r.get("cached")) for r in all_responses),
        "successful_uncached_usage": dict(usage),
        "pre_news_paired_decisions_checked": pre_news_pairs,
        "matching_window_real_returns": real_returns,
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
    }
    output = base / "pilot_analysis_v4"
    write_json(output / "summary.json", summary)
    lines = [
        "# 双事件模拟：首轮诊断报告",
        "",
        "这份报告回答的是系统为什么不成交、加入有限做市与不同投资偏好后有没有改善。"
        "它不是最终结论：每格只有6名代理和1个随机种子，尚未完成一个月的多种子主实验，也没有把金融真人行为标定接入新市场。",
        "",
        "## 现实中发生了什么",
        "",
        "Meta场景以2026年7月1日的可核验报道为主，而非已证实的6月本人帖子。"
        "从6月30日至7月31日，SOXX跌21.20%，MU跌28.70%，NVDA却涨0.33%。"
        "板块下跌属实，但不同股票差别很大；一条消息是否造成全部跌幅仍不能据此确定。",
        "",
        "大选场景中，BTC从11月5日UTC零点到AP正式确认前已经上涨8.66%，"
        "确认后24小时再涨1.44%。模型需要处理逐步出现的信息，不能在正式确认时才一次性注入全部上涨。",
        "",
        "## 这次具体改了什么",
        "",
        "股票仍按交易所开市时间，币按全天候交易。两种市场都在新闻实际出现时增加决策轮次。"
        "BTC明确按每手0.001 BTC计价，资金、持仓与委托单位统一。",
        "",
        "做市商有固定现金和库存，不能无限接盘；代理与做市商合计资金、份额必须守恒。"
        "私人目标组给不同代理不同的资产配置偏好，以形成分歧。这些偏好和做市参数都是假设，尚不是从真人或历史LOB估出的参数。",
        "",
        "## 真实模型跑出了什么",
        "",
        "MU覆盖6月30日至7月2日；BTC覆盖11月5日至7日。下表均为真实模型结果，"
        "收益从模拟开始前最后收盘计算，不能与一个月的真实收益直接比较。",
        "",
        "| 资产 | 代理设定 | 背景做市 | 有新闻收益 | 无新闻收益 | 新闻增量（百分点） | "
        "有新闻成交笔数 | 有新闻买/卖/不动 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    frame = pd.DataFrame(result_rows)
    for spec in config["scenarios"]:
        for treatment in config["treatments"]:
            for liquidity in config["liquidity_conditions"]:
                group = frame[
                    (frame.scenario == spec["scenario_id"])
                    & (frame.treatment == treatment)
                    & (frame.liquidity == liquidity)
                ].set_index("event_condition")
                yes, no = group.loc["news"], group.loc["no_news"]
                counts = yes.proposed_actions
                treatment_name = "普通" if treatment == "vanilla" else "私人配置目标"
                lines.append(
                    f"| {spec['symbol']} | {treatment_name} | "
                    f"{'无' if liquidity == 'none' else '有限'} | {yes['return']:+.2%} | "
                    f"{no['return']:+.2%} | {(yes['return'] - no['return']) * 100:+.2f} | "
                    f"{yes.trades} | {counts['buy']}/{counts['sell']}/{counts['hold']} |"
                )
    lines.extend(
        [
            "",
            "同期实际收益："
            + "；".join(
                f"{spec['symbol']} {real_returns[spec['scenario_id']]:+.2%}"
                for spec in config["scenarios"]
            )
            + "。实际收盘覆盖全天，模拟仅在规定轮次决策，仍不是完全等同的市场时间分辨率。",
            "",
            "## 这些数字是什么意思",
            "",
            "MU的8格共192次决策中有14次买入、178次观望、没有卖单，最终没有成交。"
            "逐笔核对发现，一部分买单把盘口卖价四舍五入到分，却略低于模拟保留六位小数的卖价，"
            "因此没有成交。这包含可修复的报价精度问题，不能全归因于投资者不愿交易。"
            "这轮没有复现芯片下跌。早期进度消息把零成交误写为全部观望，此处更正。",
            "",
            "BTC普通代理的有新闻/有限做市组涨约0.42%，共5笔成交，而且对手方全部是做市商。"
            "做市资金让一部分买单成交了，但这还不能证明代理之间产生了真实的人群分歧，"
            "也没有解释真实市场的涨幅。",
            "",
            "还发现明确的数字理解错误：MU某代理实际持仓价值占净值49.62%，"
            "其目标为20%，却在解释中说已经接近目标。下一步先由程序准确计算持仓比例、"
            "目标偏差与币的单位，再检验模型决策是否改善；当前结果完整保留。",
            "",
            "真人数据的另一个训练内检查显示，加入跨轮个人交易记忆后，统计模型的Brier误差"
            "从0.22750降至0.21571（越低越好）。这只涉及已经开放的51场训练实验、"
            "63,143个主动下单时点，不是新的独立验证，也不能据此推断观望概率或真实事件价格。",
            "",
            "## 审计与限制",
            "",
            f"16格均完成；{sum(r['decisions'] for r in result_rows)}个订单决策"
            "已与原始模型JSON逐条核对。"
            f"{pre_news_pairs}对消息前决策及观察完全一致，消息时间与历史行情截止时间未越界。"
            f"总拒单{sum(r['rejected_orders'] for r in result_rows)}笔；资金和份额守恒。",
            "",
            f"实际新调用成功用量：{dict(usage)}；缓存响应{summary['cache_hits']}条。"
            f"全部响应状态计数：{dict(status)}。这里记录实际token用量，未获取账户账单。",
            "",
            "接口尝试也保留：v1发现部分响应仍开启思考并截断，停止扩展；v2的原生开关探针失败；"
            "v3的网关透传组合返回HTTP500；v4的本地模板开关通过8/8探针，正式实验每次响应都检查。"
            "v1已知成功调用22,871输入/4,662输出token，3次在途中取消的调用最终用量未知。"
            "这些失败尝试不混入上表的市场结果。",
            "",
            "成交增加不等于价格变真实。尤其在有限做市组，报价围绕模拟日初价格展开，"
            "这可能把价格变化限制在人工价差附近。需要改变流动性参数和报价更新机制做敏感性分析，不能只选最像真实曲线的一组。",
            "",
            "2024大选可能已被模型记住：日期和币名保留时，无新闻组不等于完全不知情。"
            "本实验只能比较显式新闻信息的增量，不能证明对历史事件的独立预测。",
            "",
            "接下来仍需：金融真人行为的独立评价；匿名化记忆检查；更长窗口与多种子对照；"
            "流动性敏感性与固定历史环境比较。两次现实事件的因果解释需要额外证据。",
            "",
            "完整现实行情见[现实行情报告](../empirical/REPORT.md)；"
            "[冻结协议](../../../docs/two_event_protocol.md)。",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, data_spec in zip(axes, study["scenarios"], strict=True):
        daily = pd.read_parquet(
            root
            / "data/processed"
            / study["study_id"]
            / f"{data_spec['scenario_id']}_daily.parquet"
        )
        for symbol in data_spec["primary_outcomes"]:
            group = daily[daily.symbol.eq(symbol)].sort_values("session_date")
            event_day = data_spec["event_at"][:10]
            origin = group[group.session_date < event_day].iloc[-1]
            window = group[group.session_date >= origin.session_date]
            ax.plot(
                pd.to_datetime(window.session_date),
                (window.close / origin.close - 1) * 100,
                label=symbol,
            )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(pd.Timestamp(event_day), color="black", linestyle="--", linewidth=0.8)
        ax.set_title(data_spec["scenario_id"])
        ax.set_ylabel("Return from prior close (%)")
        ax.legend(frameon=False)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    fig.savefig(output / "real_price_paths.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "runs"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
