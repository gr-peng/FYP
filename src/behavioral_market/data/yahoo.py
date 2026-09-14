import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .schemas import DailyBar


class YahooProvider:
    def __init__(self, archive):
        self.archive = archive
        self.actions = {}

    def daily(self, symbol, start, end):
        body, meta = self.archive.get(
            "yahoo/kline",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            {
                "period1": int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp()),
                "period2": int(
                    (
                        datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1)
                    ).timestamp()
                ),
                "interval": "1d",
                "events": "div,splits",
            },
            headers={"User-Agent": "Mozilla/5.0"},
        )
        chart = json.loads(body)["chart"]
        if chart.get("error") or not chart.get("result"):
            raise ValueError(f"Yahoo returned no data for {symbol}")
        data = chart["result"][0]
        quote = data["indicators"]["quote"][0]
        self.actions[symbol] = data.get("events", {})
        return [
            DailyBar(
                symbol=symbol,
                session_date=datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).date(),
                source="yahoo_chart",
                fetched_at=meta["fetched_at"],
                **{
                    k: (
                        int(quote[k][i])
                        if k == "volume"
                        else Decimal(str(quote[k][i])).quantize(Decimal("0.000001"))
                    )
                    for k in ("open", "high", "low", "close", "volume")
                },
            )
            for i, ts in enumerate(data["timestamp"])
            if quote["close"][i] is not None
        ]
