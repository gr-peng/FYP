"""Reconstruct pre-ranking financial decisions from the scopic training archive.

Only periods 1--5 of the main task are exposed. Later decisions need the actual
peer-ranking display, including its cost-basis convention, which is not yet audited.
Raw rounds 1--3 are the pre-reset phase; prices remain visible history, portfolios reset.
"""

from collections import defaultdict
from fractions import Fraction

ASSETS = tuple("abcdef")


def integer(row: dict, field: str) -> int:
    value = row[field]
    result = int(value)
    if float(value) != result:
        raise ValueError(f"noninteger financial field: {field}")
    return result


def pre_ranking_decisions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["participant.code"]].append(row)
    result = []
    for participant, records in sorted(grouped.items()):
        # No future behavioral field is accessed, even during validation.
        past = {
            integer(row, "subsession.round_number"): row
            for row in records
            if integer(row, "subsession.round_number") <= 8
        }
        if (
            len(past) != 8
            or len([r for r in records if integer(r, "subsession.round_number") <= 8]) != 8
        ):
            raise ValueError("expected exactly one record per raw round 1--8")
        if set(past) != set(range(1, 9)):
            raise ValueError("missing pre-ranking history")
        prices = {
            t: {a: integer(row, f"player.price_{a}") for a in ASSETS} for t, row in past.items()
        }
        if any(p <= 0 for path in prices.values() for p in path.values()):
            raise ValueError("prices must be positive")
        cash = 10000
        holdings = dict.fromkeys(ASSETS, 0)
        basis = dict.fromkeys(ASSETS, Fraction(0))
        history = []
        for t in range(4, 9):
            row = past[t]
            pre = {a: integer(row, f"player.quantity_pre_{a}") for a in ASSETS}
            if pre != holdings:
                raise ValueError("pre-trade holdings disagree with preceding ledger/reset")
            state = {
                "main_period": t - 3,
                "total_main_periods": 14,
                "initial_cash": 10000,
                "cash": cash,
                "holdings": dict(holdings),
                "current_prices": dict(prices[t]),
                "price_history": [dict(prices[i]) for i in range(1, t + 1)],
                "average_acquisition_prices": {
                    a: float(basis[a] / holdings[a]) if holdings[a] else None for a in ASSETS
                },
                "past_main_trades": list(history),
                "peer_rankings_seen": [],
            }
            buys = {a: integer(row, f"player.quantity_buy_{a}") for a in ASSETS}
            sells = {a: integer(row, f"player.quantity_sell_{a}") for a in ASSETS}
            post = {a: integer(row, f"player.quantity_{a}") for a in ASSETS}
            for a in ASSETS:
                if buys[a] < 0 or sells[a] < 0 or sells[a] > holdings[a]:
                    raise ValueError("invalid purchase/sale quantity")
                if buys[a] and sells[a]:
                    raise ValueError(
                        "same-asset simultaneous buying and selling needs an order rule"
                    )
                if post[a] != holdings[a] + buys[a] - sells[a]:
                    raise ValueError("trade does not conserve shares")
                if sells[a]:
                    basis[a] *= Fraction(holdings[a] - sells[a], holdings[a])
                basis[a] += buys[a] * prices[t][a]
                cash += (sells[a] - buys[a]) * prices[t][a]
            if cash < 0:
                raise ValueError("trade violates cash constraint")
            holdings = post
            action = {"buy": buys, "sell": sells}
            # Anonymous source IDs and current labels are provenance/targets, not model inputs.
            result.append(
                {
                    "provenance": {
                        "participant": participant,
                        "session": row["session.code"],
                        "peer_group": row["group.id_in_subsession"],
                        "treatment": row["participant._current_app_name"],
                        "raw_round": t,
                    },
                    "state": state,
                    "target": action,
                }
            )
            history.append({"main_period": t - 3, "prices": dict(prices[t]), **action})
    return result


def sell_opportunities(decisions: list[dict]) -> list[dict]:
    """A conditional liquidation target, not a complete six-asset trading policy."""
    return [
        {
            "provenance": item["provenance"],
            "state": item["state"],
            "asset": asset,
            "sell_any": int(item["target"]["sell"][asset] > 0),
            "fraction_sold": item["target"]["sell"][asset] / item["state"]["holdings"][asset],
        }
        for item in decisions
        for asset in ASSETS
        if item["state"]["holdings"][asset] > 0
    ]
