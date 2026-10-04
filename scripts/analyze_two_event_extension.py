# ruff: noqa: E402
"""Audit every frozen extension case and report all seeds and liquidity settings."""

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from experiments.two_event_pilot_analysis import audit_run


def read(path):
    return json.loads(path.read_text())


def interval(values):
    values = np.array(values, dtype=float)
    draws = np.random.default_rng(20261004).choice(values, size=(10000, len(values))).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "seed_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "seeds": len(values),
    }


def analyze(root, config_path):
    config = read(config_path)
    study = read(root / config["data_config"])
    base = root / "outputs" / study["study_id"]
    directory = base / config["experiment_id"] / "aigc_relay"
    output = base / (config["experiment_id"] + "_analysis")
    output.mkdir(parents=True, exist_ok=True)
    metrics, decisions, paths = {}, {}, {}
    all_responses = []
    commits = set()
    manifest_hashes = set()
    for metadata_path in sorted(directory.rglob("run_metadata.json")):
        if "attempts" in metadata_path.parts:
            continue
        result, rows, responses = audit_run(metadata_path.parent)
        metadata = read(metadata_path)
        key = tuple(
            result[name]
            for name in ("scenario", "seed", "treatment", "liquidity", "event_condition")
        )
        assert key not in metrics
        metrics[key] = result
        decisions[key] = rows
        paths[key] = metadata_path.parent
        commits.add(metadata["git_commit"])
        manifest_hashes.add(metadata["data_manifest_hash"])
        all_responses.extend(responses)
        for row in rows:
            assert not row["response_metadata"].get("fallback")
            decision = row["decision"]
            if decision["action"] != "hold":
                converted = Decimal(str(decision["limit_price"])) * Decimal(
                    str(row["payload"]["order_notional_multiplier"])
                )
                assert converted == Decimal(str(row["exchange_limit_price_per_lot"]))
                if row["rejection"] is None:
                    assert (
                        Decimal(str(decision["limit_price"]))
                        % Decimal(str(row["payload"]["constraints"]["price_tick"]))
                        == 0
                    )
    expected = {
        (s["scenario_id"], seed, treatment, liquidity, condition)
        for s in config["scenarios"]
        for seed in config["seeds"]
        for treatment in config["treatments"]
        for liquidity in config["liquidity_conditions"]
        for condition in config["event_conditions"]
    }
    assert set(metrics) == expected and len(commits) == 1
    assert manifest_hashes == {
        digest((root / "data/processed" / study["study_id"] / "manifest.json").read_bytes())
    }
    paired = 0
    contrasts = []
    actual = {}
    bars_by_symbol = {}
    for spec in config["scenarios"]:
        bars = pd.read_parquet(
            root / "data/processed" / study["study_id"] / f"{spec['scenario_id']}_daily.parquet"
        )
        bars = bars[bars.symbol.eq(spec["symbol"])].sort_values("session_date")
        origin = bars[bars.session_date < spec["start"]].iloc[-1]
        ending = bars[bars.session_date.eq(spec["end"])].iloc[0]
        actual[spec["scenario_id"]] = {
            "return": float(ending.close / origin.close - 1),
            "baseline_date": origin.session_date,
            "end_date": ending.session_date,
        }
        bars_by_symbol[spec["symbol"]] = (bars, origin)
        for treatment in config["treatments"]:
            for liquidity in config["liquidity_conditions"]:
                seed_rows = []
                for seed in config["seeds"]:
                    prefix = (spec["scenario_id"], seed, treatment, liquidity)
                    yes, no = [decisions[(*prefix, c)] for c in ("news", "no_news")]
                    assert [(r["decision_at"], r["agent_id"]) for r in yes] == [
                        (r["decision_at"], r["agent_id"]) for r in no
                    ]
                    for a, b in zip(yes, no, strict=True):
                        if pd.Timestamp(a["decision_at"]) < pd.Timestamp(spec["events"][0]["at"]):
                            assert a["payload"] == b["payload"] and a["decision"] == b["decision"]
                            paired += 1
                    news, control = [metrics[(*prefix, c)] for c in ("news", "no_news")]
                    seed_rows.append(
                        {
                            "seed": seed,
                            "news_return": news["return"],
                            "no_news_return": control["return"],
                            "news_increment_percentage_points": 100
                            * (news["return"] - control["return"]),
                            "news_trades": news["trades"],
                            "no_news_trades": control["trades"],
                            "news_background_trade_share": news["background_trade_share"],
                            "news_zero_volume_sessions": news["zero_volume_sessions"],
                            "news_actions": news["proposed_actions"],
                            "no_news_actions": control["proposed_actions"],
                        }
                    )
                contrasts.append(
                    {
                        "scenario": spec["scenario_id"],
                        "symbol": spec["symbol"],
                        "treatment": treatment,
                        "liquidity": liquidity,
                        "by_seed": seed_rows,
                        "news_increment_percentage_points": interval(
                            [r["news_increment_percentage_points"] for r in seed_rows]
                        ),
                        "news_return": interval([r["news_return"] for r in seed_rows]),
                        "no_news_return": interval([r["no_news_return"] for r in seed_rows]),
                    }
                )
    usage, status = Counter(), Counter()
    for row in all_responses:
        status[str(row.get("http_status", row.get("status")))] += 1
        if row.get("response") is not None:
            body = row["response"]
            assert body["choices"][0]["finish_reason"] == "stop"
            assert not body["choices"][0]["message"].get("reasoning_content")
            if not row.get("cached"):
                for key, value in body.get("usage", {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] += value
    result = {
        "config": str(config_path.relative_to(root)),
        "config_sha256": digest(config_path.read_bytes()),
        "git_commits": sorted(commits),
        "runs": list(metrics.values()),
        "contrasts": contrasts,
        "matching_window_actual_returns": actual,
        "pre_news_pairs_verified": paired,
        "response_statuses": dict(status),
        "cache_hits": sum(bool(r.get("cached")) for r in all_responses),
        "successful_uncached_usage": dict(usage),
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
        "interpretation": "Three seeds; synthetic 6-agent market. Seed intervals describe "
        "simulator repeat variation only, not human population uncertainty or event causality.",
    }
    write_json(output / "summary.json", result)
    lines = [
        "# 双事件扩展：全部种子与流动性结果",
        "",
        "每格6名代理，3组种子。私人配置目标仍是合成偏好，尚非真人标定。"
        "表中的新闻增量是有新闻减无新闻，不能把全部模拟涨跌当作新闻效应。"
        "区间仅是3个模拟种子的重采样描述，不能解释为现实市场的因果置信区间。",
        "",
        "| 资产 | 流动性 | 有新闻平均收益 | 无新闻平均收益 | "
        "平均新闻增量百分点 | 种子最小/最大 | 种子重采样95%区间 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for contrast in contrasts:
        d = contrast["news_increment_percentage_points"]
        lo, hi = d["seed_bootstrap_95"]
        lines.append(
            f"| {contrast['symbol']} | {contrast['liquidity']} | "
            f"{contrast['news_return']['mean']:+.3%} | {contrast['no_news_return']['mean']:+.3%} | "
            f"{d['mean']:+.3f} | [{d['min']:+.3f}, {d['max']:+.3f}] | [{lo:+.3f}, {hi:+.3f}] |"
        )
    lines.extend(["", "同期实际收益（严格使用各模拟开始前一天收盘为基准）："])
    for spec in config["scenarios"]:
        real = actual[spec["scenario_id"]]
        lines.append(
            f"- {spec['symbol']}：{real['baseline_date']}至{real['end_date']}，"
            f"{real['return']:+.3%}。"
        )
    lines.extend(
        [
            "",
            "每个种子的原始对照：",
            "",
            "| 资产 | 流动性 | 种子 | 有新闻收益 | 无新闻收益 | "
            "有新闻成交 | 无新闻成交 | 新闻组做市成交占比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for contrast in contrasts:
        for row in contrast["by_seed"]:
            share = row["news_background_trade_share"]
            share = "无成交" if share is None else f"{share:.1%}"
            lines.append(
                f"| {contrast['symbol']} | {contrast['liquidity']} | {row['seed']} | "
                f"{row['news_return']:+.3%} | {row['no_news_return']:+.3%} | "
                f"{row['news_trades']} | {row['no_news_trades']} | {share} |"
            )
    total = sum(r["decisions"] for r in metrics.values())
    rejects = sum(r["rejected_orders"] for r in metrics.values())
    lines.extend(
        [
            "",
            f"全部{len(metrics)}格、{total}条决策完成并与原始JSON逐笔核对；{paired}对消息前观察及决策相同。拒单{rejects}笔；现金和份额守恒。",
            f"成功新调用token用量：{dict(usage)}；响应状态：{dict(status)}；缓存{result['cache_hits']}条。",
            "",
            "日期和资产名称仍然可见，尤其2024大选可能属于模型已有知识；无新闻组不等于完全未知历史。没有未来实际价格输入内生市场，但这不能消除模型自身的历史记忆。实际日线覆盖全天，模拟只在固定轮次决策。",
            "",
            "![实际与模拟价格路径](price_paths.png)",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(
        len(config["scenarios"]),
        len(config["liquidity_conditions"]),
        figsize=(5 * len(config["liquidity_conditions"]), 7),
        squeeze=False,
    )
    for i, spec in enumerate(config["scenarios"]):
        bars, origin = bars_by_symbol[spec["symbol"]]
        real = bars[(bars.session_date >= spec["start"]) & (bars.session_date <= spec["end"])]
        for j, liquidity in enumerate(config["liquidity_conditions"]):
            ax = axes[i, j]
            for seed in config["seeds"]:
                for condition, color, style in [
                    ("news", "#176B87", "-"),
                    ("no_news", "#C27618", "--"),
                ]:
                    prefix = (
                        spec["scenario_id"],
                        seed,
                        config["treatments"][0],
                        liquidity,
                        condition,
                    )
                    path = paths[prefix]
                    daily = read(path / "daily.json")
                    start = read(path / "initial_state.json")["price_per_lot"]
                    series = pd.Series(
                        [d["close_per_lot"] / start * 100 for d in daily],
                        index=pd.to_datetime([d["session_date"] for d in daily]),
                    )
                    ax.plot(
                        series.index,
                        series.values,
                        color=color,
                        ls=style,
                        alpha=0.6,
                        lw=1.5,
                        label=condition if seed == config["seeds"][0] else None,
                    )
            ax.plot(
                pd.to_datetime(real.session_date),
                real.close / origin.close * 100,
                color="#282828",
                lw=2,
                label="actual close",
            )
            ax.axhline(100, color="#bbbbbb", lw=0.6)
            ax.set_title(f"{spec['symbol']} / {liquidity}")
            ax.set_ylabel("Price index (pre-start = 100)")
            span_days = (pd.Timestamp(spec["end"]) - pd.Timestamp(spec["start"])).days
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (span_days + 3) // 4)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.set_xlabel(str(spec["start"])[:4])
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "price_paths.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(metrics),
                "decisions": total,
                "contrasts": contrasts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    analyze(ROOT, args.config.resolve())
