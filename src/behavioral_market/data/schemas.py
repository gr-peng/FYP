from datetime import date, datetime
from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class DailyBar(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str
    session_date: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    source: str
    source_timezone: str = "America/New_York"
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def price_range(self):
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("inconsistent OHLC range")
        return self


class MarketEvent(BaseModel):
    event_id: str
    first_seen_at: AwareDatetime
    published_at_et: AwareDatetime
    headline: str
    neutral_summary: str
    related_symbols: list[str]
    source_urls: list[str]
    confidence: str = "reported"


class NewsItem(BaseModel):
    news_id: str
    published_at_utc: AwareDatetime
    published_at_et: AwareDatetime
    source: str
    title: str
    summary: str | None = None
    url: str
    symbols: list[str] = Field(default_factory=list)
    fetched_at: datetime
