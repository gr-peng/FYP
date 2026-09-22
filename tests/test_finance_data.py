import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.finance_data import cda_split, scopic_split, selected_rows  # noqa: E402


def test_cda_split_keeps_reused_numeric_sessions_separate_and_ignores_row_order():
    rows = [
        {
            "SessionID": f"Session {i}-{condition}",
            "Condition (BuyANDSell / BuyORSell)": condition,
            "Round": r,
        }
        for condition, count in (("BuyANDSell", 9), ("BuyORSell", 12))
        for i in range(1, count + 1)
        for r in (1, 12)
    ]
    frame = pd.DataFrame(rows)
    plan = cda_split(frame)
    assert plan == cda_split(frame.sample(frac=1, random_state=5))
    assert [len(plan["assignments"][s]) for s in ("train", "development", "test")] == [11, 5, 5]
    assert len({x for values in plan["assignments"].values() for x in values}) == 21


def test_scopic_split_prevents_session_and_peer_leakage():
    rows = []
    for sid, treatments in enumerate(
        [("spt1",), ("spt1",), ("spt1",), ("spt2",), ("spt1", "spt2"), ("spt1", "spt2")]
    ):
        for group, treatment in enumerate(treatments):
            for person in range(3):
                for round_number in (1, 17):
                    rows.append(
                        {
                            "session.code": str(sid),
                            "group.id_in_subsession": group,
                            "participant.code": f"{sid}-{group}-{person}",
                            "participant._current_app_name": treatment,
                            "subsession.round_number": round_number,
                        }
                    )
    frame = pd.DataFrame(rows)
    plan = scopic_split(frame)
    assert plan == scopic_split(frame.sample(frac=1, random_state=4))
    for ids in plan["assignments"].values():
        assert len(ids) == 2
        assert set().union(*(set(plan["treatments"][x]) for x in ids)) == {"spt1", "spt2"}
    assert len({x for values in plan["assignments"].values() for x in values}) == 6


def test_closed_holdout_and_filter_before_behavior_validation():
    assignments = {"train": ["a"], "development": ["b"], "test": ["c"]}

    class ForbiddenBehavior(dict):
        def __getitem__(self, key):
            if key != "sid":
                raise AssertionError("held-out behavior accessed")
            return super().__getitem__(key)

    rows = [{"sid": "a", "action": 1}, ForbiddenBehavior(sid="c"), {"sid": "b", "action": 2}]
    assert list(selected_rows(rows, "sid", assignments, "train")) == [rows[0]]
    with pytest.raises(ValueError, match="closed"):
        list(selected_rows(rows, "sid", assignments, "test"))
    with pytest.raises(ValueError, match="overlapping"):
        list(selected_rows(rows, "sid", {**assignments, "test": ["a"]}, "train"))
