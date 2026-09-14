import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from .schemas import DailyBar, NewsItem


def parse_daily_csv(raw: str, symbol: str, fetched_at: str) -> list[DailyBar]:
    if raw.lstrip().startswith("{"):
        raise ValueError("Alpha Vantage returned an API error or quota message")
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows or "timestamp" not in rows[0]:
        raise ValueError("missing Alpha Vantage daily CSV fields")
    return [
        DailyBar(
            symbol=symbol,
            session_date=r["timestamp"],
            source="alpha_vantage",
            fetched_at=fetched_at,
            **{k: r[k] for k in ("open", "high", "low", "close", "volume")},
        )
        for r in rows
    ]


class AlphaVantageProvider:
    URL = "https://www.alphavantage.co/query"

    def __init__(self, archive, api_key):
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not configured")
        self.archive, self.api_key = archive, api_key

    def daily(self, symbol):
        body, meta = self.archive.get(
            "alpha_vantage/kline",
            self.URL,
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "datatype": "csv",
                "apikey": self.api_key,
            },
        )
        return parse_daily_csv(body.decode(), symbol, meta["fetched_at"])

    def news(self, symbol, start, end):
        body, meta = self.archive.get(
            "alpha_vantage/news",
            self.URL,
            {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "time_from": start,
                "time_to": end,
                "sort": "EARLIEST",
                "limit": 1000,
                "apikey": self.api_key,
            },
        )
        data = json.loads(body)
        if "feed" not in data:
            raise ValueError("Alpha Vantage news response lacks feed")
        items = []
        for row in data["feed"]:
            published = datetime.strptime(row["time_published"], "%Y%m%dT%H%M%S").replace(
                tzinfo=UTC
            )
            items.append(
                NewsItem(
                    news_id=hashlib.sha256(row["url"].encode()).hexdigest(),
                    published_at_utc=published,
                    published_at_et=published.astimezone(ZoneInfo("America/New_York")),
                    source=row["source"],
                    title=row["title"],
                    summary=row.get("summary"),
                    url=row["url"],
                    symbols=[x["ticker"] for x in row.get("ticker_sentiment", [])],
                    fetched_at=meta["fetched_at"],
                )
            )
        return items
