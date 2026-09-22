import pandas as pd
import pytest

from experiments.replication_training_data import deduplicate, selected_training_rows


def test_training_filter_never_accesses_other_rows_behavior():
    class ClosedRow(dict):
        def __getitem__(self, key):
            if key != "session":
                raise AssertionError("held-out behavior accessed")
            return super().__getitem__(key)

    assignments = {"train": ["a"], "development": ["b"], "test": ["c"]}
    rows = [
        {"session": "a", "price": "12"},
        ClosedRow(session="b"),
        ClosedRow(session="c"),
        ClosedRow(session="excluded"),
    ]
    assert list(selected_training_rows(rows, assignments, ["session", "price"], "session")) == [
        {"session": "a", "price": "12"}
    ]
    with pytest.raises(ValueError, match="only training"):
        list(selected_training_rows(rows, assignments, ["price"], "session", "test"))


def test_identical_exports_are_deduplicated_but_conflicting_orders_are_not():
    rows = pd.DataFrame(
        [
            {"session": "a", "uuid": "one", "price": 12, "source_member": "first.csv"},
            {"session": "a", "uuid": "one", "price": 12, "source_member": "second.csv"},
        ]
    )
    result, audit = deduplicate(rows, ["session", "uuid"])
    assert len(result) == 1 and audit["identical_copies_removed"] == 1
    assert result.iloc[0].source_members == ["first.csv", "second.csv"]
    rows.loc[1, "price"] = 13
    with pytest.raises(ValueError, match="conflicting cumulative"):
        deduplicate(rows, ["session", "uuid"])
