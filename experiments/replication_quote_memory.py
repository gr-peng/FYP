"""Draft: past-only participant history across periods and account resets.

Extends, rather than alters, the frozen partial-observation quote schema. No
current or future target action is used. Historical orders remain recorded human
actions; this is not autonomous generated memory or an unconditional HOLD model.
"""

import copy
from collections import defaultdict

import numpy as np

from experiments.replication_quote_states import GROUP


def extend_memory(records, orders, ledger):
    mapping = ledger[GROUP + ["player", "participant"]].drop_duplicates()
    if mapping.duplicated(GROUP + ["player"]).any():
        raise ValueError("ambiguous participant mapping")
    actual = orders[orders.is_replacement.eq(0)].merge(
        mapping, on=GROUP + ["player"], how="left", validate="many_to_one"
    )
    if actual.participant.isna().any():
        raise ValueError("unmapped human order")
    # The repeated market may reset holdings, but the same subject retains experience.
    histories = defaultdict(list)
    for row in actual.itertuples(index=False):
        histories[(row.session, row.participant)].append(
            (int(row.repeat), int(row.period), float(row.created), row.side)
        )
    indexed = {}
    for key, rows in histories.items():
        rows.sort(key=lambda r: r[:3])
        periods = defaultdict(lambda: [0, 0])
        for repeat, period, _, side in rows:
            if side not in {"bid", "ask"}:
                raise ValueError("unknown order side")
            periods[(repeat, period)][0 if side == "bid" else 1] += 1
        keys = sorted(periods)
        values = np.array([periods[key] for key in keys], dtype=int)
        indexed[key] = keys, values
    results = []
    for record in records:
        updated = copy.deepcopy(record)
        p = record["provenance"]
        target_key = p["repeat"], p["period"]
        keys, values = indexed[(p["session"], p["participant"])]
        earlier = [
            (key, value) for key, value in zip(keys, values, strict=True) if key < target_key
        ]
        same = [value for key, value in earlier if key[0] == p["repeat"]]
        previous = [value for key, value in earlier if key[0] < p["repeat"]]

        def counts(rows):
            total = np.sum(rows, axis=0) if rows else [0, 0]
            return {"buy": int(total[0]), "sell": int(total[1])}

        updated["state"]["own_completed_period_memory"] = {
            "scope": "Earlier completed periods only; accounts reset between repetitions.",
            "earlier_periods_this_repetition": counts(same),
            "previous_repetitions": counts(previous),
            "last_completed_period_with_own_orders": (
                {
                    "repetition": earlier[-1][0][0],
                    "period": earlier[-1][0][1],
                    "buy": int(earlier[-1][1][0]),
                    "sell": int(earlier[-1][1][1]),
                }
                if earlier
                else None
            ),
        }
        results.append(updated)
    return results
