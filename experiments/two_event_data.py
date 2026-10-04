"""Auditable empirical baselines for the two requested event episodes.

These are retrospective price/volume observations, not causal estimates or LOB data.
No credentials, model calls, or human-benchmark labels are used here.
"""

import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_market.data.archive import Archive, digest, write_json
from behavioral_market.data.calendar import sessions
from behavioral_market.data.yahoo import YahooProvider

ROOT = Path(__file__).resolve().parents[1]
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def parse_binance_klines(body, checksum, symbol, interval):
    """Verify the publisher's checksum and parse one ZIP without extracting paths."""
    expected = checksum.decode("ascii").split()[0]
    if len(expected) != 64 or digest(body) != expected:
        raise ValueError("Binance publisher checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        files = archive.namelist()
        if len(files) != 1 or not files[0].endswith(".csv"):
            raise ValueError("expected one CSV in Binance archive")
        frame = pd.read_csv(archive.open(files[0]), header=None, names=KLINE_COLUMNS)
    if frame.empty:
        raise ValueError("empty Binance archive")
    frame = frame.apply(pd.to_numeric, errors="raise")
    raw = frame.open_time.to_numpy()
    milliseconds = (raw >= 10**12) & (raw < 10**14)
    microseconds = (raw >= 10**15) & (raw < 10**17)
    if not (milliseconds.all() or microseconds.all()):
        raise ValueError("unknown or mixed Binance timestamp unit")
    unit = "ms" if milliseconds.all() else "us"
    frame["bar_open_utc"] = pd.to_datetime(frame.open_time, unit=unit, utc=True)
    close_time = pd.to_datetime(frame.close_time, unit=unit, utc=True)
    step = {"1d": pd.Timedelta(days=1), "1m": pd.Timedelta(minutes=1)}[interval]
    frame["bar_end_utc"] = frame.bar_open_utc + step
    tick = pd.Timedelta(1, unit=unit)
    if not (close_time == frame.bar_end_utc - tick).all():
        raise ValueError("inconsistent kline closing timestamp")
    if (raw % (step // tick) != 0).any():
        raise ValueError("unaligned kline opening timestamp")
    frame["session_date"] = frame.bar_open_utc.dt.strftime("%Y-%m-%d")
    frame["symbol"] = symbol
    frame["source"] = "binance_spot_archive"
    frame["source_timezone"] = "UTC"
    if ((frame.taker_buy_volume < 0) | (frame.taker_buy_volume > frame.volume)).any():
        raise ValueError("invalid taker buy volume")
    if (frame.number_of_trades < 0).any():
        raise ValueError("negative trade count")
    return frame.drop(columns=["open_time", "close_time", "ignore"])


def binance_file(archive, symbol, interval, period, frequency):
    url = (
        f"https://data.binance.vision/data/spot/{frequency}/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{period}.zip"
    )
    body, _ = archive.get("binance/two_event", url)
    checksum, _ = archive.get("binance/two_event", url + ".CHECKSUM")
    return parse_binance_klines(body, checksum, symbol, interval)


def validate_bars(frame, spec, intraday=False):
    """Reject missing intervals rather than filling or inventing observations."""
    if set(frame.symbol) != set(spec["symbols"]):
        raise ValueError("symbol coverage mismatch")
    time = "bar_open_utc" if intraday else "session_date"
    if frame.duplicated(["symbol", time]).any():
        raise ValueError("duplicate market intervals")
    prices = frame[["open", "high", "low", "close"]].astype(float)
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("nonpositive or nonfinite prices")
    if (prices.low > prices[["open", "close"]].min(axis=1)).any() or (
        prices.high < prices[["open", "close"]].max(axis=1)
    ).any():
        raise ValueError("inconsistent OHLC")
    if not np.isfinite(frame.volume.astype(float)).all() or (frame.volume < 0).any():
        raise ValueError("invalid volume")
    if intraday:
        start = pd.Timestamp(spec["intraday"]["start"], tz="UTC")
        end = pd.Timestamp(spec["intraday"]["end"], tz="UTC") + pd.Timedelta(days=1)
        expected = set(pd.date_range(start, end, freq="1min", inclusive="left"))
    else:
        expected = set(sessions(spec["history_start"], spec["history_end"], spec["calendar"]).index)
    for symbol, group in frame.groupby("symbol"):
        if set(group[time]) != expected:
            raise ValueError(f"{symbol}: market time coverage mismatch")


def prepare(config, root=ROOT):
    directory = root / "data/processed" / config["study_id"]
    manifest_path = directory / "manifest.json"
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_hash"] != config_hash:
            raise ValueError("existing study has different config; use a new study_id")
        verify_manifest(manifest, root)
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    archive = Archive(root / "data/raw")
    records, corporate_actions = [], {}
    for spec in config["scenarios"]:
        if spec["provider"] == "yahoo":

            def fetch_equity(symbol, spec=spec):
                provider = YahooProvider(archive)
                values = provider.daily(symbol, spec["history_start"], spec["history_end"])
                rows = [item.model_dump(mode="json") for item in values]
                result = pd.DataFrame(rows)
                for col in ("open", "high", "low", "close", "volume"):
                    result[col] = pd.to_numeric(result[col])
                return result, provider.actions

            with ThreadPoolExecutor(max_workers=4) as executor:
                fetched = list(executor.map(fetch_equity, spec["symbols"]))
            daily = pd.concat([item[0] for item in fetched], ignore_index=True)
            for _, actions in fetched:
                corporate_actions.update(actions)
            # Keep explicitly audited, split-adjusted prices. Never silently divide twice.
            # New splits still stop ingestion. Simulation needs a separate unit audit.
            for symbol, actions in corporate_actions.items():
                for split in actions.get("splits", {}).values():
                    day = datetime.fromtimestamp(split["date"], UTC).date().isoformat()
                    if spec["history_start"] <= day <= spec["history_end"]:
                        allowed = [
                            item
                            for item in spec.get("audited_splits", [])
                            if item["symbol"] == symbol
                            and item["effective_session"] == day
                            and item["ratio"] == split["numerator"] / split["denominator"]
                        ]
                        if len(allowed) != 1:
                            raise ValueError(
                                f"{symbol}: split requires an explicit adjustment audit"
                            )
            timing = sessions(spec["history_start"], spec["history_end"], spec["calendar"])
            daily["bar_open_utc"] = daily.session_date.map(timing.market_open)
            daily["bar_end_utc"] = daily.session_date.map(timing.market_close)
        elif spec["provider"] == "binance_spot_archive":
            periods = pd.period_range(spec["history_start"], spec["history_end"], freq="M")
            jobs = [(symbol, str(period)) for symbol in spec["symbols"] for period in periods]
            with ThreadPoolExecutor(max_workers=4) as executor:
                parts = list(
                    executor.map(
                        lambda job: binance_file(archive, job[0], "1d", job[1], "monthly"), jobs
                    )
                )
            daily = pd.concat(parts, ignore_index=True)
            daily = daily[daily.session_date.between(spec["history_start"], spec["history_end"])]
        else:
            raise ValueError("unknown market data provider")
        validate_bars(daily, spec)
        daily = daily.sort_values(["symbol", "session_date"]).reset_index(drop=True)
        path = directory / f"{spec['scenario_id']}_daily.parquet"
        daily.to_parquet(path, index=False)
        records.append(file_record(path, root, len(daily)))
        daily.to_csv(path.with_suffix(".csv"), index=False)
        records.append(file_record(path.with_suffix(".csv"), root, len(daily)))
        print(f"{spec['scenario_id']}: {len(daily)} daily bars verified", flush=True)
        if spec["intraday"]:
            days = pd.date_range(spec["intraday"]["start"], spec["intraday"]["end"])
            jobs = [
                (symbol, day.strftime("%Y-%m-%d")) for symbol in spec["symbols"] for day in days
            ]
            with ThreadPoolExecutor(max_workers=4) as executor:
                parts = list(
                    executor.map(
                        lambda job: binance_file(archive, job[0], "1m", job[1], "daily"), jobs
                    )
                )
            intraday = pd.concat(parts, ignore_index=True).sort_values(["symbol", "bar_open_utc"])
            validate_bars(intraday, spec, intraday=True)
            path = directory / f"{spec['scenario_id']}_minute.parquet"
            intraday.to_parquet(path, index=False)
            records.append(file_record(path, root, len(intraday)))
            print(f"{spec['scenario_id']}: {len(intraday)} minute bars verified", flush=True)
    actions_path = directory / "corporate_actions.json"
    write_json(actions_path, corporate_actions)
    records.append(file_record(actions_path, root))
    manifest = {
        "study_id": config["study_id"],
        "config_hash": config_hash,
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_hash": digest(Path(__file__).read_bytes()),
        "raw_files": sorted(archive.entries, key=lambda item: item["path"]),
        "processed_files": records,
        "limitations": [
            "Retrospective selected episodes, not causal identification or untouched test events.",
            "OHLCV and taker-side volume are not order-book observations.",
            "Equity raw-price returns exclude dividends; crypto is USDT-denominated spot.",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def file_record(path, root, rows=None):
    return {"path": str(path.relative_to(root)), "sha256": digest(path.read_bytes()), "rows": rows}


def verify_manifest(manifest, root=ROOT):
    for record in [*manifest["processed_files"], *manifest["raw_files"]]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        if digest(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"market artifact hash mismatch: {path.name}")


def date_window(prices, start, end):
    """Return close-to-close outcomes from the last session before start."""
    pre = prices.loc[prices.index < start]
    after = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if pre.empty or after.empty:
        raise ValueError("event window lacks baseline or outcomes")
    values = np.r_[float(pre.iloc[-1]), after.to_numpy(dtype=float)]
    return {
        "baseline_date": str(pre.index[-1]),
        "last_date": str(after.index[-1]),
        "return": float(values[-1] / values[0] - 1),
        "max_drawdown": float((1 - values / np.maximum.accumulate(values)).max()),
        "minimum_return_from_baseline": float(values.min() / values[0] - 1),
        "maximum_return_from_baseline": float(values.max() / values[0] - 1),
        "sessions": len(after),
    }


def summarize_daily(frame, spec):
    validate_bars(frame, spec)
    event_day = pd.Timestamp(spec["event_at"]).strftime("%Y-%m-%d")
    closes = frame.pivot(index="session_date", columns="symbol", values="close").astype(float)
    closes = closes.sort_index()
    if event_day not in closes.index:
        raise ValueError("event date is not a trading session; specify its alignment")
    if spec["estimation_end"] >= spec["anticipation_start"]:
        raise ValueError("estimation overlaps anticipated event window")
    returns = closes.pct_change(fill_method=None)
    result = []
    for symbol in spec["symbols"]:
        row = {"symbol": symbol, "event_day": event_day, "windows": {}}
        for offset in (0, 7, 30):
            end = (pd.Timestamp(event_day) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            if end > spec["history_end"]:
                raise ValueError("requested horizon exceeds source coverage")
            row["windows"][f"event_plus_{offset}d"] = date_window(closes[symbol], event_day, end)
        row["windows"]["anticipation"] = date_window(
            closes[symbol],
            spec["anticipation_start"],
            (pd.Timestamp(event_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        row["windows"]["whole_episode"] = date_window(
            closes[symbol], spec["anticipation_start"], spec["history_end"]
        )
        series = frame[frame.symbol.eq(symbol)].set_index("session_date").volume.astype(float)
        prior_volume = series.loc[series.index < event_day].tail(20).mean()
        row["event_volume_ratio_previous20"] = (
            float(series.loc[event_day] / prior_volume) if prior_volume > 0 else None
        )
        benchmark = spec["benchmark"]
        if benchmark is not None and symbol != benchmark:
            pre = returns.loc[returns.index <= spec["estimation_end"], [symbol, benchmark]].dropna()
            if len(pre) < 30 or pre[benchmark].std() <= 0:
                raise ValueError("insufficient market-model estimation data")
            design = np.column_stack([np.ones(len(pre)), pre[benchmark].to_numpy()])
            alpha, beta = np.linalg.lstsq(design, pre[symbol].to_numpy(), rcond=None)[0]
            abnormal = returns[symbol] - alpha - beta * returns[benchmark]
            row["market_model"] = {
                "benchmark": benchmark,
                "alpha": float(alpha),
                "beta": float(beta),
                "estimation_first_return": str(pre.index[0]),
                "estimation_last_return": str(pre.index[-1]),
                "n": len(pre),
                "interpretation": "descriptive benchmark adjustment, not a causal effect",
                "CAR": {
                    key: float(abnormal.loc[event_day : window["last_date"]].sum())
                    for key, window in row["windows"].items()
                    if key.startswith("event_plus")
                },
            }
        result.append(row)
    return result


def summarize_confirmation(frame, event_at, episode_start):
    """Use only bars COMPLETED at each cutoff; a bar straddling news stays out."""
    event = pd.Timestamp(event_at)
    start = pd.Timestamp(episode_start)
    if event.tzinfo is None or start.tzinfo is None or start >= event:
        raise ValueError("aware, ordered event and start timestamps required")
    rows = []
    for symbol, group in frame.groupby("symbol"):
        group = group.sort_values("bar_open_utc")
        # First opening is observed at start, unlike that minute's eventual close.
        origin = group.loc[group.bar_open_utc.eq(start), "open"]
        prior = group.loc[group.bar_end_utc <= event]
        if len(origin) != 1 or prior.empty:
            raise ValueError("intraday baseline missing")
        last = prior.iloc[-1]
        if event - last.bar_end_utc >= pd.Timedelta(minutes=1):
            raise ValueError("intraday pre-event data gap")
        base = float(last.close)
        row = {
            "symbol": symbol,
            "confirmation_at": event.isoformat(),
            "pre_confirmation_price": base,
            "pre_confirmation_bar_end": last.bar_end_utc.isoformat(),
            "return_episode_start_to_confirmation": base / float(origin.iloc[0]) - 1,
        }
        for minutes in (60, 1440):
            cutoff = event + pd.Timedelta(minutes=minutes)
            eligible = group.loc[group.bar_end_utc <= cutoff]
            if eligible.empty or cutoff - eligible.iloc[-1].bar_end_utc >= pd.Timedelta(minutes=1):
                raise ValueError("intraday post-event data gap")
            end = eligible.iloc[-1]
            row[f"return_after_{minutes}m"] = float(end.close) / base - 1
            row[f"end_after_{minutes}m"] = end.bar_end_utc.isoformat()
        rows.append(row)
    return rows


def evaluate(config, root=ROOT):
    directory = root / "data/processed" / config["study_id"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_hash"] != digest(json.dumps(config, sort_keys=True).encode()):
        raise ValueError("evaluation configuration differs from prepared data")
    verify_manifest(manifest, root)
    output = root / "outputs" / config["study_id"] / "empirical"
    summary = {"study_id": config["study_id"], "scenarios": []}
    for spec in config["scenarios"]:
        bars = pd.read_parquet(directory / f"{spec['scenario_id']}_daily.parquet")
        record = {"scenario_id": spec["scenario_id"], "daily": summarize_daily(bars, spec)}
        if spec["intraday"]:
            minute = pd.read_parquet(directory / f"{spec['scenario_id']}_minute.parquet")
            validate_bars(minute, spec, intraday=True)
            record["confirmation_windows"] = summarize_confirmation(
                minute, spec["event_at"], spec["intraday"]["start"] + "T00:00:00Z"
            )
        summary["scenarios"].append(record)
    summary["manifest_sha256"] = digest(manifest_path.read_bytes())
    summary["analysis_source_sha256"] = digest(Path(__file__).read_bytes())
    summary["completed_at"] = datetime.now(UTC).isoformat()
    write_json(output / "summary.json", summary)
    write_report(summary, config, output / "REPORT.md")
    return summary


def write_report(summary, config, path):
    lines = [
        "# 两个事件的现实行情核验（阶段报告）",
        "",
        "这是公开行情的回顾性分析，还不是新的 LLM 模拟实验报告。"
        "涨跌与事件同处一个时间窗口，不能据此全部归因于事件。",
        "",
        "日线收益以事件前一交易日收盘为基准；+7/+30日指事件日期加7/30个自然日，"
        "休市时使用此前最近收盘。最大回撤按日收盘计算，与期末收益是不同指标。",
        "",
    ]
    for spec, result in zip(config["scenarios"], summary["scenarios"], strict=True):
        lines.extend(
            [
                f"## {spec['title_zh']}",
                "",
                {
                    "meta_compute_2026": (
                        "用户确认研究今年6—7月这次消息。可核验主事件为7月1日的Bloomberg报道，"
                        "尚未核实所谓6月本人发帖。以下检查实际行情，不预设所有芯片股票都跌，"
                        "也不把全部跌幅归因于一条消息。"
                    ),
                    "trump_election_2024": (
                        "AP于11月6日美东05:34（UTC 10:34）确认胜选。计票信息与市场反应"
                        "在此前已经出现；这个时间只是正式确认节点。此处价格单位为USDT，"
                        "不假定它与美元始终完全相同。"
                    ),
                }[spec["scenario_id"]],
                "",
                "| 资产 | 消息前阶段收益 | 消息当天收益 | +7日收益 | +30日收益 | "
                "事件后窗口最大回撤 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["daily"]:
            windows = row["windows"]
            numbers = [
                windows[name]["return"]
                for name in ("anticipation", "event_plus_0d", "event_plus_7d", "event_plus_30d")
            ]
            text = " | ".join(f"{value:+.2%}" for value in numbers)
            lines.append(
                f"| {row['symbol']} | {text} | {windows['event_plus_30d']['max_drawdown']:.2%} |"
            )
        if "confirmation_windows" in result:
            lines.extend(
                [
                    "",
                    "胜选信息逐步出现，因此把确认前和确认后分开：",
                    "",
                    "| 资产 | 11月5日UTC零点至确认前 | 确认后1小时 | 确认后24小时 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for row in result["confirmation_windows"]:
                lines.append(
                    f"| {row['symbol']} | {row['return_episode_start_to_confirmation']:+.2%} | "
                    f"{row['return_after_60m']:+.2%} | {row['return_after_1440m']:+.2%} |"
                )
        lines.extend(
            [
                "",
                "来源："
                + "；".join(f"[来源{i + 1}]({url})" for i, url in enumerate(spec["source_urls"])),
                "",
            ]
        )
    lines.extend(
        [
            "## 这些数据还不能回答什么",
            "",
            "- 日线/分钟线不是订单簿，不能验证排队、撤单、真实可成交数量。",
            "- 股票为价格收益，不含现金股息；币为Binance现货USDT报价。",
            "- 一个整月还包含其他新闻与市场变化；市场模型CAR也不等于事件的因果效应。",
            "- 两个场景本来就是看过结果后选定的案例，不能称为未见事件上的预测成功。",
            "- 尚待金融行为比较、有限流动性对照、真实模型和微观到宏观实验。",
            "",
            f"输入manifest SHA256：`{summary['manifest_sha256']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
