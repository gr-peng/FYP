# Report prose and source URLs intentionally retain readable sentence boundaries.
# ruff: noqa: E501
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from behavioral_market.data.archive import digest, write_json
from behavioral_market.evaluation.metrics import event_study, path_metrics
from behavioral_market.evaluation.usage import usage_report
from behavioral_market.simulation.event_runner import ROOT, load_dataset


def markdown_table(frame):
    value = frame.copy()
    for column in value.select_dtypes("number"):
        value[column] = value[column].map(lambda x: f"{x:.5g}" if pd.notna(x) else "NA")
    lines = ["| " + " | ".join(value.columns) + " |", "|" + "---|" * len(value.columns)]
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in value.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def simulated_event_cars(market, bars, studies, initial_price, symbol="NVDA"):
    """Ex-post CAR diagnostics; benchmark future bars never enter agent observations."""
    dates = ["2026-06-12", *market.session_date.tolist()]
    simulated = pd.Series([initial_price, *market.close.astype(float)], index=dates).pct_change()
    qqq = bars[bars.symbol == "QQQ"].set_index("session_date").close.astype(float)
    benchmark_returns = qqq.sort_index().pct_change()
    result = {}
    for event, name in [("2026-07-01", "e1"), ("2026-07-09", "e2")]:
        row = studies.loc[
            (studies.symbol == symbol) & (studies.benchmark == "QQQ") & (studies.event == event)
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


def _population_visibility(metadata, directory):
    """Resolve recorded information availability without inferring remote exposure."""
    snapshot_path = directory / "config_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    if not isinstance(snapshot, dict):
        raise ValueError("config snapshot must be an object")
    sources = [("metadata", metadata)]
    for name in ("behavior", "run"):
        value = snapshot.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"snapshot {name} must be an object")
        sources.append((f"snapshot.{name}", value))
    values = []
    for name, source in sources:
        if "mean_field_visibility" in source:
            value = source["mean_field_visibility"]
            if not isinstance(value, str) or value not in ("visible", "hidden"):
                raise ValueError(f"invalid mean_field_visibility in {name}")
            values.append(value)
    explicit = "mean_field_visibility" in metadata
    visibility = metadata["mean_field_visibility"] if explicit else "visible"
    hidden_path = any(part.endswith("_mean_field_hidden") for part in directory.parts)
    if any(value != visibility for value in values) or (hidden_path and visibility != "hidden"):
        raise ValueError("contradictory or missing mean_field_visibility")
    expected_supplied = None if metadata["treatment"] == "rule_abm" else visibility == "visible"
    for name, source in sources:
        if "prompt_mean_field_supplied" in source:
            value = source["prompt_mean_field_supplied"]
            if value is not expected_supplied:
                raise ValueError(f"contradictory prompt_mean_field_supplied in {name}")
    return {
        "mean_field_visibility": visibility,
        "mean_field_visibility_origin": "metadata" if explicit else "legacy_default_visible",
        "prompt_mean_field_supplied": metadata.get("prompt_mean_field_supplied"),
    }


def _require_legacy_returns(metadata, directory):
    """Missing metadata cannot override an explicit method in the saved config."""
    sources = [metadata]
    snapshot_path = directory / "config_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if not isinstance(snapshot, dict):
            raise ValueError("config snapshot must be an object")
        sources.extend([snapshot.get("behavior", {}), snapshot.get("run", {})])
    methods = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("strategy return declarations must be objects")
        if "strategy_return_method" in source:
            method = source["strategy_return_method"]
            if not isinstance(method, str) or not method:
                raise ValueError("invalid strategy_return_method")
            methods.append(method)
        elif "finite_rule_pair" in source:
            raise ValueError("finite_rule_pair has no declared strategy_return_method")
        if "finite_rule_pair" in source and source.get("strategy_return_method") != (
            "finite_rule_pair_assigned_quotes_v1"
        ):
            raise ValueError("contradictory finite_rule_pair and strategy_return_method")
    if len(set(methods)) > 1:
        raise ValueError("contradictory strategy_return_method")
    if methods and methods[0] != "long_cash_signal_proxy":
        raise ValueError(
            "event_report does not yet support finite rule-pair runs; "
            "use their method-labelled per-run outputs"
        )


def _run_description(metadata, directory, market, metrics):
    """Keep descriptive evidence separate from the existing numerical tables."""
    snapshot_path = directory / "config_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    run = snapshot.get("run", {})

    def declaration(field):
        labels = sorted(
            {
                value
                for value in [metadata.get(field), run.get(field)]
                if isinstance(value, str) and value
            }
        )
        return {
            "value": labels[0] if len(labels) == 1 else None,
            "status": "declared"
            if len(labels) == 1
            else ("conflicting" if labels else "unrecorded"),
            "recorded_labels": labels,
        }

    return {
        "agents_declared": metadata["n_agents"],
        "sessions_declared": metadata.get("sessions"),
        "market_rows": len(market),
        "market_distinct_dates": int(market.session_date.nunique()),
        "market_first_date": str(market.session_date.min()),
        "market_last_date": str(market.session_date.max()),
        "provider": declaration("provider"),
        "model": declaration("model"),
        "quality_recorded": metadata.get("quality"),
        "order_fallback_rate_recorded": metrics.get("fallback_rate"),
        "invalid_style_rate_recorded": metrics.get("invalid_style_rate"),
    }


def evaluate(config, directory, destination, *, pricing_path=None):
    bars, _, _, manifest_hash = load_dataset(config)
    destination.mkdir(parents=True, exist_ok=True)
    actual = bars[bars.symbol == config["data"]["primary_symbol"]].sort_values("session_date")
    actual = actual[actual.session_date >= "2026-06-12"]
    initial = float(actual.close.iloc[0])
    actual_normalized = actual.close.astype(float).to_numpy() / initial
    real_metrics = path_metrics(actual.close.astype(float))
    primary_symbols = config["data"].get("primary_symbols", [config["data"]["primary_symbol"]])
    study_symbols = list(
        dict.fromkeys([*primary_symbols, "META", "SOXX", *config["data"]["robustness_symbols"]])
    )
    available = set(bars.symbol)
    studies = pd.DataFrame(
        [
            event_study(bars, symbol, benchmark, event)
            for symbol in study_symbols
            for benchmark in ["QQQ", "SPY"]
            for event in ["2026-07-01", "2026-07-09"]
            if symbol in available and benchmark in available
        ]
    )
    studies.to_csv(destination / "real_event_study.csv", index=False)
    studies.to_parquet(destination / "real_event_study.parquet", index=False)
    records, paths, path_rows, descriptions = [], {}, [], []
    for path in sorted(directory.rglob("run_metadata.json")):
        if "_superseded" in path.parts:
            continue
        metadata = json.loads(path.read_text())
        if (
            metadata["status"] != "complete"
            or metadata["n_agents"] != config["population"]["n_agents"]
        ):
            continue
        _require_legacy_returns(metadata, path.parent)
        if metadata["data_manifest_hash"] != manifest_hash:
            raise ValueError("cannot aggregate runs from different data manifests")
        visibility = _population_visibility(metadata, path.parent)
        if not isinstance(metadata["source_hash"], str) or not metadata["source_hash"]:
            raise ValueError("run source_hash must be recorded")
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
        symbol = metadata.get("symbol", config["data"]["primary_symbol"])
        metrics.update(simulated_event_cars(market, bars, studies, initial, symbol))
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
        record["symbol"] = symbol
        record["event_condition"] = metadata.get(
            "event_condition", "e1_e2" if metadata["include_secondary"] else "e1_only"
        )
        accounts = pd.read_parquet(path.parent / "agents.parquet")
        initial_equity = (
            float(config["population"]["initial_cash"])
            + config["population"]["initial_shares"] * initial
        )
        last = accounts[accounts.session_date == accounts.session_date.max()]
        metrics["mean_agent_return"] = float(last.total_equity.mean() / initial_equity - 1)
        metrics["mean_agent_realized_pnl"] = float(last.realized_pnl.mean())
        # Numerical metrics must not overwrite recorded run identities or exposure.
        record = {**metrics, **record, **visibility}
        identity_fields = (
            "mode",
            "treatment",
            "seed",
            "shock",
            "event_condition",
            "symbol",
            "source_hash",
            "mean_field_visibility",
        )
        key = tuple(record[field] for field in identity_fields)
        if key in paths:
            raise ValueError("duplicate run/seed identity within the same condition and source")
        records.append(record)
        descriptions.append(
            {
                **{field: record[field] for field in identity_fields},
                **_run_description(metadata, path.parent, market, metrics),
            }
        )
        paths[key] = normalized
        path_rows.extend(
            {
                **{field: record[field] for field in identity_fields},
                "session_date": session,
                "normalized_price": value * 100,
            }
            for session, value in zip(
                [actual.session_date.iloc[0], *market.session_date.tolist()],
                normalized,
                strict=True,
            )
        )
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "run_metrics.csv", index=False)
    frame.to_parquet(destination / "run_metrics.parquet", index=False)
    pd.DataFrame(path_rows).to_csv(destination / "normalized_paths.csv", index=False)
    pricing = json.loads(pricing_path.read_text()) if pricing_path is not None else None
    usage = usage_report(directory, pricing)
    write_json(destination / "api_usage.json", usage)
    manifest = json.loads((ROOT / config["data"]["manifest"]).read_text())
    context = {
        "included_runs": len(records),
        "agents_declared": sorted({d["agents_declared"] for d in descriptions}),
        "seed_values": sorted({r["seed"] for r in records}),
        "source_hashes": sorted({r["source_hash"] for r in records}),
        "mean_field_visibilities": sorted({r["mean_field_visibility"] for r in records}),
        "calendar_declared_in_evaluation_config": config["data"].get("calendar"),
        "planned_completion": None,
        "planned_completion_reason": "No complete frozen planned-key set has been verified.",
        "input_bars": {
            "rows": len(bars),
            "symbols": int(bars.symbol.nunique()),
            "first_date": str(bars.session_date.min()),
            "last_date": str(bars.session_date.max()),
        },
        "reference_prices": {
            "symbol": config["data"]["primary_symbol"],
            "first_date": str(actual.session_date.iloc[0]),
            "last_date": str(actual.session_date.iloc[-1]),
            "date_count": len(actual),
        },
        "runs": descriptions,
        "usage_scope": "All response logs under the supplied directory, including logs outside included runs.",
        "recorded_noncached_responses": usage["calls"],
        "recorded_local_cache_hits": usage["local_cache_hits"],
        "response_model_labels": usage["models_reported"],
        "pricing_assumption_source": pricing["source"] if pricing is not None else None,
    }
    write_json(destination / "report_context.json", context)
    seed_values = ", ".join(map(str, context["seed_values"])) or "未记录"
    agent_counts = ", ".join(map(str, context["agents_declared"])) or "未记录"
    calendar = context["calendar_declared_in_evaluation_config"] or "未记录"
    dates = context["reference_prices"]
    report = [
        "# 模拟研究记录报告",
        "",
        "本报告只描述本次纳入的记录。运行说明中的配置、响应记录中的标签和实际模型行为是不同证据；缺失内容保留未知。",
        "## 纳入范围与输入数据",
        "",
        f"本次纳入 {len(records)} 次完整状态运行；每次记录声明的模拟账户数：{agent_counts}。",
        f"随机种子：{len(context['seed_values'])} 个（{seed_values}）；源码版本：{len(context['source_hashes'])} 个；群体汇总条件：{', '.join(context['mean_field_visibilities']) or '未记录'}。",
        "计划完成度：未知。尚未核对包含全部条件的冻结运行键清单，不能用部分参数的乘积充当计划总数。",
        f"输入行情：{len(bars)} 条记录、{bars.symbol.nunique()} 只标的，日期 {context['input_bars']['first_date']} 至 {context['input_bars']['last_date']}。",
        f"评估配置声明的日历：{calendar}。下面按市场表实际记录的日期计数；这不等于已验证的逐账户下单或复盘次数。",
        *["- " + issue for issue in manifest["issues"]],
        "## 输入参考行情与计算约定",
        "",
        f"{config['data']['primary_symbol']} 的参考价格范围为 {dates['first_date']} 至 {dates['last_date']}，含 {dates['date_count']} 个记录日期；该范围收益 {real_metrics['return']:.2%}，最大回撤 {real_metrics['max_drawdown']:.2%}。",
        markdown_table(
            studies[(studies.symbol == config["data"]["primary_symbol"])][
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
        "上表事件日期和收益窗口沿用当前评估代码的固定计算约定，不是本批记录对消息发布时间或代理首次可见时间的证明。事件收益比较会用到事后参考行情；本报告没有逐条核验代理当时实际收到的信息。",
    ]
    if not len(frame):
        report += ["", "当前没有符合筛选条件且状态完整的模拟运行；上方仅描述输入参考行情。"]
    else:
        groups = [
            "symbol",
            "event_condition",
            "mode",
            "treatment",
            "shock",
            "source_hash",
            "mean_field_visibility",
        ]
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
        for (symbol, condition, mode, shock, code, visibility), group in frame.groupby(
            ["symbol", "event_condition", "mode", "shock", "source_hash", "mean_field_visibility"]
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
                pairs = low[["seed", metric]].merge(
                    high[["seed", metric]],
                    on="seed",
                    suffixes=("_low", "_high"),
                    validate="one_to_one",
                )
                available = bool(len(x) and len(y))
                u = (
                    mannwhitneyu(x, y, alternative="two-sided", method="auto")
                    if available
                    else None
                )
                comparisons.append(
                    {
                        "mode": mode,
                        "symbol": symbol,
                        "event_condition": condition,
                        "shock": shock,
                        "metric": metric,
                        "n_low": len(x),
                        "n_high": len(y),
                        "n_pairs": len(pairs),
                        "comparison_status": "paired_available"
                        if len(pairs)
                        else ("no_common_seed" if available else "missing_treatment"),
                        "mean_field_visibility": visibility,
                        "MWU_p": float(u.pvalue) if u is not None else None,
                        "cliffs_delta_low_minus_high": float(
                            np.sign(x[:, None] - y[None, :]).mean()
                        )
                        if available
                        else None,
                        "paired_mean_high_minus_low": float(
                            (pairs[metric + "_high"] - pairs[metric + "_low"]).mean()
                        )
                        if len(pairs)
                        else None,
                        "source_hash": code,
                    }
                )
        pd.DataFrame(comparisons).to_csv(destination / "treatment_comparisons.csv", index=False)
        report += [
            "",
            "## 纳入记录的数值比较",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {
                            "方式": d["mode"],
                            "设置": d["treatment"],
                            "种子": d["seed"],
                            "群体条件": d["mean_field_visibility"],
                            "记录日期数": d["market_distinct_dates"],
                            "起始日期": d["market_first_date"],
                            "末尾日期": d["market_last_date"],
                        }
                        for d in descriptions
                    ]
                )
            ),
            markdown_table(
                averages[
                    [
                        "mode",
                        "symbol",
                        "event_condition",
                        "treatment",
                        "shock",
                        "source_hash",
                        "mean_field_visibility",
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
            "表内为同条件、同源码版本的运行均值；各次明细、波动范围及配对数见 run_metrics.csv、group_uncertainty.csv 和 treatment_comparisons.csv。按输入行情回放的价格由输入给定；内部撮合方式的价格来自该运行的订单。",
            "mean_field_visibility 的 visible/hidden 是配置的滞后群体汇总可见/隐藏条件。提示中是否实际提供汇总，须结合 prompt_mean_field_supplied；规则程序不经客户端提示，后者为空，规则仍使用原观察。两字段都不证明远端调用或信息被使用。两种条件和不同源码版本分别统计、配对及画线。旧记录缺少条件字段时，只有没有矛盾信息才按旧默认 visible 解析，并在明细标为 legacy_default_visible；这只是兼容解析，不能证明实际曝光。提示记录缺失时也保留空值。",
            "事件窗口差值沿用评估代码的计算方法；参考行情在这里用于事后比较，不能单凭差值认定消息的真实影响或模型的行为改善。低成交和价格停滞需结合逐单记录解释。",
            f"记录标为降级的运行：{int((frame.quality == 'degraded').sum())} 次。降级标签本身不能说明是哪一种失败；下方只列已有的订单回退及策略答复失败比例，缺失记为未记录。",
            "比较以运行记录及相同随机种子的配对为单位，不能把账户每天的动作都当成互相独立的样本。检验和差异指标仅作探索；样本规模与配对数应逐组查看，不能据此宣布稳健因果关系。",
        ]
        degraded = [d for d in descriptions if d["quality_recorded"] == "degraded"]
        if degraded:
            report.append(
                markdown_table(
                    pd.DataFrame(
                        [
                            {
                                "设置": d["treatment"],
                                "种子": d["seed"],
                                "群体条件": d["mean_field_visibility"],
                                "订单回退比例（记录值）": "未记录"
                                if d["order_fallback_rate_recorded"] is None
                                else f"{d['order_fallback_rate_recorded']:.1%}",
                                "策略答复无效率（记录值）": "未记录"
                                if d["invalid_style_rate_recorded"] is None
                                else f"{d['invalid_style_rate_recorded']:.1%}",
                            }
                            for d in degraded
                        ]
                    )
                )
            )
        endogenous = frame[frame["mode"] == "endogenous"]
        if len(endogenous):
            shock_values = sorted(endogenous.shock.unique(), reverse=True)
            fig, axes = plt.subplots(
                len(shock_values), 1, figsize=(10, 4 * len(shock_values)), squeeze=False
            )
            for ax, shock in zip(axes[:, 0], shock_values, strict=True):
                ax.plot(
                    actual.session_date,
                    actual_normalized * 100,
                    "k--",
                    label=f"Input reference {config['data']['primary_symbol']}",
                )
                identities = (
                    endogenous[["treatment", "source_hash", "mean_field_visibility"]]
                    .drop_duplicates()
                    .sort_values(["treatment", "source_hash", "mean_field_visibility"])
                )
                for treatment, code, visibility in identities.itertuples(index=False, name=None):
                    values = [
                        v
                        for (m, t, _, s, condition, symbol, source, exposure), v in paths.items()
                        if m == "endogenous"
                        and t == treatment
                        and s == shock
                        and condition == "e1_e2"
                        and symbol == config["data"]["primary_symbol"]
                        and source == code
                        and exposure == visibility
                    ]
                    if values:
                        stack = np.array(values) * 100
                        ax.plot(
                            actual.session_date,
                            stack.mean(axis=0),
                            label=f"{treatment} / {visibility} / {code}",
                        )
                        ax.fill_between(
                            actual.session_date, stack.min(axis=0), stack.max(axis=0), alpha=0.12
                        )
                ax.set(
                    title=f"Declared shock {shock:.0%}; mean and seed range",
                    ylabel=f"{dates['first_date']} close = 100",
                )
                ax.tick_params(axis="x", rotation=65)
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(destination / "normalized_paths.png", dpi=160)
            fig.savefig(destination / "normalized_paths.svg")
            plt.close(fig)
            report += ["", "![价格路径与 seed 范围](normalized_paths.png)"]
    for field, label in [("provider", "服务来源声明"), ("model", "模型名称声明")]:
        declared = sorted(
            {d[field]["value"] for d in descriptions if d[field]["value"] is not None}
        )
        missing = sum(d[field]["status"] == "unrecorded" for d in descriptions)
        conflicting = sum(d[field]["status"] == "conflicting" for d in descriptions)
        report.append(
            f"{label}：{', '.join(declared) or '未记录'}；未记录 {missing} 次，声明冲突 {conflicting} 次。"
        )
    report += [
        "",
        "## 响应记录与复现范围",
        "",
        "上面的运行统计只含符合筛选条件的完整记录；这里的用量汇总覆盖整个指定目录下的响应日志，可能包括预检、未完成或未纳入的运行。两者不能直接混同。",
        f"目录内非缓存响应记录 {usage['calls']} 条，记录为 HTTP 200 的有 {usage['successful_http_calls']} 条，其他状态或传输失败记录 {usage['failed_calls']} 条，其中重试标记 {usage['retry_calls']} 条；本地缓存命中记录 {usage['local_cache_hits']} 条。记录的输入 {usage['input_tokens']:,} tokens，输出 {usage['output_tokens']:,} tokens。",
        f"非缓存响应中的模型标签：{', '.join(usage['models_reported']) or '此范围未记录标签'}。标签和运行声明都不构成模型身份或真实远端调用的独立证明；没有日志也不能反推从未调用。请求配置与实际返回模式需分别核实，本报告没有验证实际思考模式。",
        (
            f"按显式指定的归档价格统一套价估算 ¥{usage['estimated_cost_cny']:.4f}；这是整个用量目录的价格假设，不证明服务商匹配，也不等同到账单实扣。归档来源：{pricing['source']}。"
            if pricing
            else "未显式指定归档价格，只报告用量记录，不估算费用。"
        ),
        "本报告没有核验旧批次的故障经过、限流历史或实际账单。逐次运行的说明来源、记录日期与已知失败比例见 report_context.json；原数值表和价格路径另行保存，便于复查。",
        "## 结论边界",
        "",
        "本报告描述当前记录及其数值比较，尚未核验真人行为校准、模型是否使用群体信息或模拟市场是否接近真实市场。跟风设置的名称不能代替行为证据。",
        "研究结论仍需独立的行为和市场验证；输入记录的时间顺序检查也不能排除模型预训练记忆。不能将参考行情的同期变化直接归因于某一条消息。",
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
            "mean_field_visibilities": sorted({r["mean_field_visibility"] for r in records}),
            "legacy_default_visible_runs": sum(
                r["mean_field_visibility_origin"] == "legacy_default_visible" for r in records
            ),
        },
    )
    print(f"report: {destination / 'REPORT.md'}; complete runs={len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["siliconflow", "aigc_relay", "mock"], default="siliconflow"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--pricing", type=Path, help="Explicit archived rates; omitted means no cost estimate"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = args.directory or ROOT / "outputs/meta_compute_2026" / args.provider
    evaluate(
        config, directory, args.destination or directory / "evaluation", pricing_path=args.pricing
    )


if __name__ == "__main__":
    main()
