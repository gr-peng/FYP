from datetime import date

import pandas as pd
import pandas_market_calendars as mcal

ET = "America/New_York"


def sessions(start: str | date, end: str | date) -> pd.DataFrame:
    schedule = mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)
    schedule.index = pd.Index([x.date().isoformat() for x in schedule.index], name="session_date")
    return schedule


def visible_news(events, previous_cutoff, current_cutoff):
    if previous_cutoff.tzinfo is None or current_cutoff.tzinfo is None:
        raise ValueError("aware cutoff timestamps required")
    return [event for event in events if previous_cutoff < event.first_seen_at <= current_cutoff]
