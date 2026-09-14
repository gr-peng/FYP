import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .alpha_vantage import AlphaVantageProvider
from .archive import Archive, digest, write_json
from .calendar import ET, sessions
from .gdelt import discover
from .news_dedup import deduplicate
from .schemas import MarketEvent
from .yahoo import YahooProvider

ROOT = Path(__file__).resolve().parents[3]


def download(config, root=ROOT):
    load_dotenv(root / ".env")
    spec = config["data"]
    directory = root / spec["dataset_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    archive = Archive(root / "data/raw")
    issues = []
    provider = (
        YahooProvider(archive)
        if spec["market_provider"] == "yahoo"
        else AlphaVantageProvider(archive, os.getenv("ALPHA_VANTAGE_API_KEY"))
    )
    symbols = [spec["primary_symbol"], *spec["context_symbols"], *spec["robustness_symbols"]]
    schedule = sessions(spec["history_start"], spec["experiment_end"])
    rows = []
    for symbol in symbols:
        bars = (
            provider.daily(symbol, spec["history_start"], spec["experiment_end"])
            if isinstance(provider, YahooProvider)
            else provider.daily(symbol)
        )
        selected = [bar for bar in bars if bar.session_date.isoformat() in schedule.index]
        present = {bar.session_date.isoformat() for bar in selected}
        missing = set(schedule.index) - present
        if missing:
            raise RuntimeError(f"{symbol}: missing sessions {sorted(missing)}")
        if len(selected) != len(present):
            raise RuntimeError(f"{symbol}: duplicate sessions")
        for bar in selected:
            row = bar.model_dump(mode="json")
            session = schedule.loc[row["session_date"]]
            row.update(
                market_open_utc=session.market_open.isoformat(),
                market_close_utc=session.market_close.isoformat(),
                market_open_et=session.market_open.tz_convert(ET).isoformat(),
                market_close_et=session.market_close.tz_convert(ET).isoformat(),
            )
            rows.append(row)
        print(f"archived {symbol}: {len(selected)} verified XNYS sessions", flush=True)
    frame = pd.DataFrame(rows).sort_values(["symbol", "session_date"])
    frame.to_parquet(directory / "market_daily.parquet", index=False)
    frame.to_csv(directory / "market_daily.csv", index=False)
    if isinstance(provider, YahooProvider):
        write_json(directory / "corporate_actions.json", provider.actions)
        # No silent split adjustment inside a fixed-share simulator.
        for split in provider.actions.get(spec["primary_symbol"], {}).get("splits", {}).values():
            if (
                datetime.fromtimestamp(split["date"], UTC).date().isoformat()
                >= spec["experiment_start"]
            ):
                raise RuntimeError(
                    "primary asset split during simulation requires explicit adjustment"
                )
        issues.append(
            "Yahoo chart fallback used: no Alpha Vantage key supplied; OHLC as returned, "
            "without dividend adjustment. Corporate actions archived; "
            "no primary-asset split during simulation."
        )
    events = [
        MarketEvent.model_validate(x)
        for x in json.loads((root / "config/meta_compute_2026_events.json").read_text())
    ]
    registry = [e.model_dump(mode="json") for e in events]
    write_json(directory / "events.json", registry)
    write_json(root / "data/raw/events/meta_compute_2026/source_registry.json", registry)
    pd.DataFrame(registry).to_parquet(directory / "event_registry.parquet", index=False)
    pd.DataFrame(
        [
            {
                "news_id": e.event_id,
                "published_at_utc": e.first_seen_at.isoformat(),
                "published_at_et": e.published_at_et.isoformat(),
                "title": e.headline,
                "summary": e.neutral_summary,
                "url": e.source_urls[0],
                "source": "Bloomberg",
            }
            for e in events
        ]
    ).to_parquet(directory / "news.parquet", index=False)
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        news_provider = AlphaVantageProvider(archive, os.environ["ALPHA_VANTAGE_API_KEY"])
        try:
            news = deduplicate(
                [
                    item
                    for symbol in ["META", "NVDA"]
                    for item in news_provider.news(symbol, "20260615T0000", "20260718T0000")
                ]
            )
            write_json(directory / "news_discovery.json", [n.model_dump(mode="json") for n in news])
        except (RuntimeError, ValueError) as exc:
            issues.append(str(exc))
    else:
        issues.append(
            "No Alpha Vantage NEWS_SENTIMENT feed: prompts use two source-verified "
            "canonical events only."
        )
    try:
        articles = discover(archive)
        write_json(directory / "gdelt_discovery.json", articles)
        print(f"GDELT discovered {len(articles)} source URLs", flush=True)
    except (RuntimeError, ValueError) as exc:
        issues.append(f"GDELT unavailable: {exc}")
    try:
        body, _ = archive.get(
            "sec",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
            headers={"User-Agent": "FYP academic research gpeng615@connect.hkust-gz.edu.cn"},
        )
        facts = json.loads(body)
        write_json(directory / "sec_companyfacts.json", facts)
    except (RuntimeError, ValueError) as exc:
        issues.append(
            f"Optional SEC facts unavailable: {exc}; financial statement fields remain null."
        )
    manifest = {
        "dataset": "meta_compute_2026",
        "created_at": datetime.now(UTC).isoformat(),
        "market_provider": spec["market_provider"],
        "issues": issues,
        "raw_files": archive.entries,
        "processed_files": [
            {"path": str(p.relative_to(root)), "sha256": digest(p.read_bytes())}
            for p in sorted(directory.iterdir())
            if p.is_file()
        ],
    }
    write_json(root / "data/manifests/meta_compute_2026_manifest.json", manifest)
    print(json.dumps({"rows": len(frame), "symbols": len(symbols), "issues": issues}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/meta_compute_2026.json")
    args = parser.parse_args()
    download(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
