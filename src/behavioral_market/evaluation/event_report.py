# Report prose and source URLs intentionally retain readable sentence boundaries.
# ruff: noqa: E501
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from behavioral_market.data.archive import digest, write_json
from behavioral_market.evaluation.metrics import event_study, path_metrics
from behavioral_market.evaluation.usage import usage_report
from behavioral_market.simulation.event_runner import ROOT, llm_config, load_dataset


def markdown_table(frame):
    value = frame.copy()
    for column in value.select_dtypes("number"):
        value[column] = value[column].map(lambda x: f"{x:.5g}" if pd.notna(x) else "NA")
    lines = ["| " + " | ".join(value.columns) + " |", "|" + "---|" * len(value.columns)]
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in value.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def simulated_event_cars(market, bars, studies, initial_price):
    """Ex-post CAR diagnostics; benchmark future bars never enter agent observations."""
    dates = ["2026-06-12", *market.session_date.tolist()]
    simulated = pd.Series([initial_price, *market.close.astype(float)], index=dates).pct_change()
    qqq = bars[bars.symbol == "QQQ"].set_index("session_date").close.astype(float)
    benchmark_returns = qqq.sort_index().pct_change()
    result = {}
    for event, name in [("2026-07-01", "e1"), ("2026-07-09", "e2")]:
        row = studies.loc[
            (studies.symbol == "NVDA") & (studies.benchmark == "QQQ") & (studies.event == event)
        ].iloc[0]
        center = dates.index(event)
        for radius in (1, 3):
            window = dates[center - radius : center + radius + 1]
            abnormal = simulated.loc[window] - (
                row["alpha"] + row["beta"] * benchmark_returns.loc[window]
            )
            car = float(abnormal.sum())
            result[f"sim_CAR_{name}_{radius}"] = car
            result[f"CAR_difference_{name}_{radius}"] = car - row[f"CAR_{radius}"]
    return result


def evaluate(config, directory, destination):
    is_mock = directory.name == "mock"
    provider = directory.name
    bars, _, _, manifest_hash = load_dataset(config)
    destination.mkdir(parents=True, exist_ok=True)
    actual = bars[bars.symbol == config["data"]["primary_symbol"]].sort_values("session_date")
    actual = actual[actual.session_date >= "2026-06-12"]
    initial = float(actual.close.iloc[0])
    actual_normalized = actual.close.astype(float).to_numpy() / initial
    real_metrics = path_metrics(actual.close.astype(float))
    studies = pd.DataFrame(
        [
            event_study(bars, symbol, benchmark, event)
            for symbol in ["NVDA", "META", "SOXX", *config["data"]["robustness_symbols"]]
            for benchmark in ["QQQ", "SPY"]
            for event in ["2026-07-01", "2026-07-09"]
        ]
    )
    studies.to_csv(destination / "real_event_study.csv", index=False)
    studies.to_parquet(destination / "real_event_study.parquet", index=False)
    records, paths = [], {}
    for path in sorted(directory.rglob("run_metadata.json")):
        if "_superseded" in path.parts:
            continue
        metadata = json.loads(path.read_text())
        if (
            metadata["status"] != "complete"
            or metadata["n_agents"] != config["population"]["n_agents"]
        ):
            continue
        if metadata["data_manifest_hash"] != manifest_hash:
            raise ValueError("cannot aggregate runs from different data manifests")
        for name, expected in metadata["artifact_hashes"].items():
            if digest((path.parent / name).read_bytes()) != expected:
                raise ValueError("run artifact has been modified")
        metrics = json.loads((path.parent / "metrics.json").read_text())
        market = pd.read_parquet(path.parent / "market.parquet")
        normalized = np.r_[1.0, market.close.astype(float).to_numpy() / initial]
        metrics.update(
            normalized_path_rmse=float(np.sqrt(np.mean((normalized - actual_normalized) ** 2))),
            drawdown_difference=metrics["max_drawdown"] - real_metrics["max_drawdown"],
            volatility_difference=metrics["daily_volatility"] - real_metrics["daily_volatility"],
        )
        metrics.update(simulated_event_cars(market, bars, studies, initial))
        # A real float-turnover denominator is not available; do not compare it to toy float.
        record = {
            key: metadata[key]
            for key in [
                "mode",
                "treatment",
                "seed",
                "shock",
                "quality",
                "include_secondary",
                "source_hash",
            ]
        }
        accounts = pd.read_parquet(path.parent / "agents.parquet")
        initial_equity = (
            float(config["population"]["initial_cash"])
            + config["population"]["initial_shares"] * initial
        )
        last = accounts[accounts.session_date == accounts.session_date.max()]
        metrics["mean_agent_return"] = float(last.total_equity.mean() / initial_equity - 1)
        metrics["mean_agent_realized_pnl"] = float(last.realized_pnl.mean())
        record.update(metrics)
        records.append(record)
        paths[
            (
                record["mode"],
                record["treatment"],
                record["seed"],
                record["shock"],
                record["include_secondary"],
            )
        ] = normalized
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "run_metrics.csv", index=False)
    frame.to_parquet(destination / "run_metrics.parquet", index=False)
    pricing = (
        json.loads((ROOT / "config/siliconflow_pricing_20260901.json").read_text())
        if provider == "siliconflow"
        else None
    )
    usage = usage_report(directory, pricing)
    write_json(destination / "api_usage.json", usage)
    manifest = json.loads((ROOT / "data/manifests/meta_compute_2026_manifest.json").read_text())
    report = [
        "# Meta Compute 2026：真实数据与工程模拟校验"
        if is_mock
        else "# Meta Compute 2026：真实数据与 LLM 初轮实验报告",
        "",
        (
            "本报告中的代理决策由确定性 MockLLMClient 生成，仅用于端到端工程校验；不是硅基流动模型的行为结果。"
            if is_mock
            else "本报告由归档数据、逐单日志和已完成的运行自动生成。规则基线、mock 与真实 API 分目录保存。"
        ),
        "## 数据与事件时钟",
        "",
        f"行情：{len(bars)} 条日线、{bars.symbol.nunique()} 只标的。预热 2026-04-01 起，实验 2026-06-15 至 2026-07-17，共 23 个 XNYS 交易日。",
        "E1：7 月 1 日 10:39 ET 发布，7 月 2 日开盘首次可见；E2：7 月 9 日 14:04 ET 发布，7 月 10 日开盘首次可见。事件研究以发布日计算；agent 反应以首次可见交易日计算。",
        "[E1 原始报道](https://news.bloomberglaw.com/artificial-intelligence/meta-is-building-a-cloud-business-to-sell-excess-ai-compute-1)；[E2 原始报道](https://news.bloomberglaw.com/artificial-intelligence/metas-zuckerberg-says-exploring-ai-cloud-business-makes-sense)。",
        *["- " + issue for issue in manifest["issues"]],
        "",
        "SEC companyfacts 按 filed < 决策日期筛选，晚于时点的披露和更正不可见。GDELT 仅用于线索归档，seendate 不冒充发布时间。",
        "## 真实市场",
        "",
        f"NVDA 从 6 月 12 日收盘到 7 月 17 日收盘收益 {real_metrics['return']:.2%}，窗口最大回撤 {real_metrics['max_drawdown']:.2%}。",
        markdown_table(
            studies[(studies.symbol == "NVDA")][
                [
                    "event",
                    "benchmark",
                    "event_day_return",
                    "CAR_1",
                    "CAR_3",
                    "event_volume_ratio_previous20",
                ]
            ]
        ),
        "",
        "CAR 使用 4 月 1 日至 6 月 12 日的日收益估计截距和 beta，按交易日窗口求异常收益之和；使用价格收益，未计股息。",
    ]
    if not len(frame):
        report += ["", "当前还没有符合规模要求的完整模拟运行；上方仅为真实数据事件研究。"]
    else:
        expected = (
            len(config["suite"]["treatments"])
            * len(config["suite"]["seeds"])
            * (1 + len(config["suite"]["shocks"]))
        )
        if len(frame) < expected:
            report += [
                "",
                f"当前仅完成 {len(frame)}/{expected} 次计划运行；",
                "尚未完成的情景不参与下列均值和检验。此次部分结果不能代表完整矩阵。",
            ]
        groups = ["mode", "treatment", "shock", "include_secondary", "source_hash"]
        selected = [
            "return",
            "max_drawdown",
            "daily_volatility",
            "hold_rate",
            "switch_rate",
            "order_execution_rate",
            "fallback_rate",
            "mean_agent_return",
            "normalized_path_rmse",
            "CAR_difference_e1_1",
            "CAR_difference_e2_1",
            "trades",
        ]
        averages = frame.groupby(groups, dropna=False)[selected].mean().reset_index()
        averages["runs"] = frame.groupby(groups, dropna=False).size().to_numpy()
        averages.to_csv(destination / "group_means.csv", index=False)
        uncertainties = frame.groupby(groups, dropna=False)[selected].agg(
            ["mean", "std", "min", "max"]
        )
        uncertainties.to_csv(destination / "group_uncertainty.csv")
        comparisons = []
        for (mode, shock, secondary, code), group in frame.groupby(
            ["mode", "shock", "include_secondary", "source_hash"]
        ):
            low, high = (
                group[group.treatment == "low_herding"],
                group[group.treatment == "high_herding"],
            )
            for metric in [
                "return",
                "max_drawdown",
                "switch_rate",
                "hold_rate",
                "order_execution_rate",
                "mean_agent_return",
            ]:
                x, y = low[metric].to_numpy(), high[metric].to_numpy()
                if len(x) and len(y):
                    u = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                    pairs = low[["seed", metric]].merge(
                        high[["seed", metric]], on="seed", suffixes=("_low", "_high")
                    )
                    comparisons.append(
                        {
                            "mode": mode,
                            "shock": shock,
                            "include_secondary": secondary,
                            "metric": metric,
                            "n_low": len(x),
                            "n_high": len(y),
                            "MWU_p": float(u.pvalue),
                            "cliffs_delta_low_minus_high": float(
                                np.sign(x[:, None] - y[None, :]).mean()
                            ),
                            "paired_mean_high_minus_low": float(
                                (pairs[metric + "_high"] - pairs[metric + "_low"]).mean()
                            ),
                            "source_hash": code,
                        }
                    )
        pd.DataFrame(comparisons).to_csv(destination / "treatment_comparisons.csv", index=False)
        report += [
            "",
            "## 实际完成的运行",
            "",
            f"完成 {len(frame)} 次主规模运行，每次 {config['population']['n_agents']} agents、23 次交易决策及 2 次风格复盘/agent。",
            markdown_table(
                averages[
                    [
                        "mode",
                        "treatment",
                        "shock",
                        "runs",
                        "return",
                        "max_drawdown",
                        "order_execution_rate",
                        "switch_rate",
                        "mean_agent_return",
                    ]
                ]
            ),
            "",
            "表内为 seed 均值；各 seed 明细、标准差及范围见 run_metrics.csv、group_uncertainty.csv。历史回放价格固定，因此各组市场价格指标相同；该阶段比较的是交易、持仓和风格变化。",
            "内生 CDA 只有这批 agent 提供流动性，不补造市商、成交或外部价格。低成交量、价格停滞也是结果，需要与行为效果分开解释。",
            "CAR_difference_e1_1 / e2_1 是以真实 NVDA 预事件 beta 和真实 QQQ 事后基准计算的模拟 CAR 减真实 CAR；仅在评估时读取基准的未来日线，未传给 agent。历史回放差值应为零。E1/E2 的模型可见日分别比报道日晚一个交易日。",
            f"降级运行（订单 fallback >5%）：{int((frame.quality == 'degraded').sum())}。",
            "比较以运行/seed 为单位，禁止把 690 个 agent-day 当作 690 个独立样本。MWU 与 Cliff's delta 为探索性指标；同 seed 成对差异单独报告。3 个 seed 功效很低，多重比较未经校正，不能据此宣布稳健因果关系。",
        ]
        endogenous = frame[frame["mode"] == "endogenous"]
        if len(endogenous):
            shock_values = sorted(endogenous.shock.unique(), reverse=True)
            fig, axes = plt.subplots(
                len(shock_values), 1, figsize=(10, 4 * len(shock_values)), squeeze=False
            )
            for ax, shock in zip(axes[:, 0], shock_values, strict=True):
                ax.plot(actual.session_date, actual_normalized * 100, "k--", label="Real NVDA")
                for treatment in sorted(endogenous.treatment.unique()):
                    values = [
                        v
                        for (m, t, _, s, secondary), v in paths.items()
                        if m == "endogenous" and t == treatment and s == shock and secondary
                    ]
                    if values:
                        stack = np.array(values) * 100
                        ax.plot(actual.session_date, stack.mean(axis=0), label=treatment)
                        ax.fill_between(
                            actual.session_date, stack.min(axis=0), stack.max(axis=0), alpha=0.12
                        )
                for day in ["2026-07-02", "2026-07-10"]:
                    ax.axvline(day, color="grey", alpha=0.3)
                ax.set(
                    title=f"Fundamental shock {shock:.0%}; mean and seed range",
                    ylabel="June 12 close = 100",
                )
                ax.tick_params(axis="x", rotation=65)
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(destination / "normalized_paths.png", dpi=160)
            fig.savefig(destination / "normalized_paths.svg")
            plt.close(fig)
            report += ["", "![价格路径与 seed 范围](normalized_paths.png)"]
    report += [
        "",
        "## API 与复现",
        "",
        (
            "模型：确定性 MockLLMClient，无真实 LLM API 调用。"
            if is_mock
            else f"Provider：{provider}；模型：{llm_config(config, provider)['model']}；temperature={config['llm']['temperature']}，max_tokens={config['llm']['max_tokens']}，thinking 关闭，全局并发上限 {config['llm']['max_concurrency']}。"
        ),
        f"已记录的 chat/completions 响应 {usage['calls']} 次，HTTP 200 有 {usage['successful_http_calls']} 次，非 200/传输失败 {usage['failed_calls']} 次，其中重试请求 {usage['retry_calls']} 次；本地缓存命中 {usage['local_cache_hits']} 次。输入 {usage['input_tokens']:,} tokens，输出 {usage['output_tokens']:,} tokens。",
        (
            f"按调用时段与可见缓存 token 估算费用 ¥{usage['estimated_cost_cny']:.4f}，不等同到账单实扣。[定价公告](https://docs.siliconflow.cn/docs/release-notes/overview)。"
            if pricing
            else "中转站价格规则未归档，因此只报告 token 用量，不估算费用。"
        ),
        "单 agent 预检也计入 API 用量，但不混入 30-agent 主实验统计。请求/响应、trace、usage、配置、源码 hash、Git commit、数据 hash 及依赖版本均保存在 outputs 下；原始行情与新闻存 data/raw，模型响应存输出日志及 .cache。API key 仅存在本地忽略文件 .env。",
        "9 月 14 日内生并行批次曾触发 429 限流，随后出现 HTTP 402，未完成运行保留以供断点续跑。早期日志的 requested_at 是响应结束时刻；后续版本分别记录派发和接收时间。价格只是公告价估算，未返回 usage 的在途请求无法计费估算。",
        "## 结论边界与后续实验",
        "",
        "这是有限事件新闻、30-agent、多 seed 的首轮真实 API 实验。没有完成人类行为数据校准，因此不包含 Calibrated treatment，也不宣称人类行为有效性。基础价值是实验前收盘价锚加情景冲击，并非估值预测。",
        "首轮采用 E1+E2；E1-only、20/50/100 agents、20 seeds 与真实行为校准属于后续验证。真实市场的同期收益不能归因于单一新闻；归档输入无未来泄漏也不能证明模型预训练未记住历史事件。",
        "反事实风格收益采用确定性的无摩擦 long/cash 代理，风格复盘会暴露该代理误差；不是对另一个 LLM 账户的真实回测。",
        "",
    ]
    (destination / "REPORT.md").write_text("\n\n".join(report), encoding="utf-8")
    write_json(
        destination / "evaluation_metadata.json",
        {
            "data_manifest_hash": manifest_hash,
            "completed_runs": len(records),
            "real_metrics": real_metrics,
            "source_hashes": sorted({r["source_hash"] for r in records}),
        },
    )
    print(f"report: {destination / 'REPORT.md'}; complete runs={len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["siliconflow", "aigc_relay", "mock"], default="siliconflow"
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "config/meta_compute_2026.json").read_text())
    directory = ROOT / "outputs/meta_compute_2026" / args.provider
    evaluate(config, directory, directory / "evaluation")


if __name__ == "__main__":
    main()
