import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.scopic_states import ASSETS, pre_ranking_decisions  # noqa: E402


def sample_rows():
    rows = []
    for t in range(1, 18):
        row = {
            "participant.code": "person",
            "session.code": "session",
            "group.id_in_subsession": "group",
            "participant._current_app_name": "spt1",
            "subsession.round_number": t,
            "player.result_final": "forbidden",
            "player.holt_laury1": "forbidden",
        }
        for a in ASSETS:
            before = 0 if t <= 4 else (2 if t == 5 else 1)
            buy = 2 if t == 4 and a == "a" else 0
            sell = 1 if t == 5 and a == "a" else 0
            before = before if a == "a" else 0
            row.update(
                {
                    f"player.price_{a}": 100 + t,
                    f"player.quantity_pre_{a}": before,
                    f"player.quantity_buy_{a}": buy,
                    f"player.quantity_sell_{a}": sell,
                    f"player.quantity_{a}": before + buy - sell,
                }
            )
        rows.append(row)
    return rows


def test_state_uses_past_actions_and_current_prices_only():
    rows = sample_rows()
    decisions = pre_ranking_decisions(rows)
    assert len(decisions) == 5
    assert decisions[0]["state"]["cash"] == 10000
    assert len(decisions[0]["state"]["price_history"]) == 4
    assert decisions[1]["state"]["cash"] == 9792
    assert decisions[1]["state"]["holdings"]["a"] == 2
    assert decisions[2]["state"]["cash"] == 9897
    assert decisions[2]["state"]["average_acquisition_prices"]["a"] == 104
    changed = copy.deepcopy(rows)
    # Poison every future field except the selector. Neither outcome validation nor
    # model state assembly may inspect it in the pre-ranking benchmark.
    for row in changed[8:]:
        for key in row:
            if key not in {"participant.code", "subsession.round_number"}:
                row[key] = "unreadable_future"
    for row in changed:
        row["player.result_final"] = object()
        row["player.holt_laury1"] = object()
    assert pre_ranking_decisions(changed) == decisions
    changed = copy.deepcopy(rows)
    changed[7]["player.quantity_buy_b"] = 1
    changed[7]["player.quantity_b"] = 1
    after = pre_ranking_decisions(changed)
    assert [x["state"] for x in after] == [x["state"] for x in decisions]
    assert after[-1]["target"] != decisions[-1]["target"]


def test_history_does_not_include_future_prices_or_trades():
    rows = sample_rows()
    before = pre_ranking_decisions(rows)
    rows[7]["player.price_b"] = 999
    after = pre_ranking_decisions(rows)
    assert before[:4] == after[:4]
    assert before[4]["state"] != after[4]["state"]
    assert len(before[0]["state"]["past_main_trades"]) == 0
    assert len(before[4]["state"]["past_main_trades"]) == 4


def test_inconsistent_accounting_is_rejected():
    rows = sample_rows()
    rows[4]["player.quantity_pre_a"] = 3
    with pytest.raises(ValueError, match="preceding ledger"):
        pre_ranking_decisions(rows)
    rows = sample_rows()
    rows[3]["player.quantity_buy_a"] = 1000
    rows[3]["player.quantity_a"] = 1000
    with pytest.raises(ValueError, match="cash constraint"):
        pre_ranking_decisions(rows)
