from behavioral_market.data.calendar import ET, visible_news
from behavioral_market.data.features import features


def build_observation(
    history, session, cutoff, previous_cutoff, events, fundamental, mean_field, context_returns=None
):
    if (history.session_date >= session).any():
        raise ValueError("observation history contains uncompleted sessions")
    return {
        "session_date": session,
        "decision_at_utc": cutoff.isoformat(),
        "decision_at_et": cutoff.tz_convert(ET).isoformat(),
        **features(history),
        "fundamental_value": fundamental,
        "fundamental_method": "pre_start_close_anchor_scenario",
        "current_ratio": None,
        "leverage": None,
        "free_cash_flow": None,
        "context_returns": context_returns or {},
        "visible_events": [
            e.model_dump(mode="json") for e in visible_news(events, previous_cutoff, cutoff)
        ],
        "mean_field": mean_field,
        "history_last_session": history.session_date.iloc[-1],
    }


def available_fundamentals(facts, session):
    """Use filing DATE + next session availability, excluding later restatements."""
    if not facts:
        return {}
    tags = facts.get("facts", {}).get("us-gaap", {})
    chosen = {}
    for tag in (
        "AssetsCurrent",
        "LiabilitiesCurrent",
        "Liabilities",
        "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ):
        values = tags.get(tag, {}).get("units", {}).get("USD", [])
        eligible = [
            x for x in values if x.get("filed", "9999") < session and x.get("end", "9999") < session
        ]
        if eligible:
            # Most recent disclosed fiscal period; freshest filing available by cutoff.
            chosen[tag] = max(eligible, key=lambda x: (x["end"], x["filed"], x.get("start", "")))

    def ratio(a, b):
        return (
            chosen[a]["val"] / chosen[b]["val"]
            if a in chosen and b in chosen and chosen[b]["val"]
            else None
        )

    ocf = chosen.get("NetCashProvidedByUsedInOperatingActivities")
    capex = chosen.get("PaymentsToAcquirePropertyPlantAndEquipment")
    fcf = (
        ocf["val"] - capex["val"]
        if ocf and capex and (ocf.get("start"), ocf["end"]) == (capex.get("start"), capex["end"])
        else None
    )
    return {
        "current_ratio": ratio("AssetsCurrent", "LiabilitiesCurrent"),
        "leverage": ratio("Liabilities", "StockholdersEquity"),
        "free_cash_flow": fcf,
        "fundamentals_filing_dates": sorted({x["filed"] for x in chosen.values()}),
    }
