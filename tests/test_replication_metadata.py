import pandas as pd
import pytest

from experiments.replication_metadata import partitions


def metadata():
    sessions, markets, people = [], [], []
    for i in range(83):
        study, slab = 1 if i < 31 else 2, f"session{i}"
        sessions.append({"slab": slab, "sid": i, "study": study})
        for mid in (0, 1):
            market_index = 2 * i + mid
            mlab, cond = f"market{market_index}", 2 * (study - 1) + mid + 1
            base = {"slab": slab, "mlab": mlab, "mid": mid, "study": study, "cond": cond}
            markets.append({**base, "sid": i})
            for pid in range(10 if market_index < 108 else 8):
                people.append({**base, "pid": pid, "plab": f"person{len(people)}"})
    return tuple(pd.DataFrame(rows) for rows in (sessions, markets, people))


def test_session_split_keeps_paired_markets_and_people_together():
    frames = metadata()
    assignments, counts = partitions(*frames)
    assert [counts[k]["sessions"] for k in assignments] == [51, 16, 16]
    assert [counts[k]["markets"] for k in assignments] == [102, 32, 32]
    assert sum(c["participants"] for c in counts.values()) == 1544
    all_ids = [sid for ids in assignments.values() for sid in ids]
    assert len(all_ids) == len(set(all_ids)) == 83
    # File/row ordering and outcome-like fields cannot change metadata assignment.
    permuted = [f.sample(frac=1, random_state=7).assign(price="unreadable") for f in frames]
    assert partitions(*permuted) == (assignments, counts)


@pytest.mark.parametrize("duplicate_person", [False, True])
def test_cross_session_or_duplicate_person_metadata_is_rejected(duplicate_person):
    sessions, markets, people = metadata()
    if duplicate_person:
        people.loc[1, "plab"] = people.loc[0, "plab"]
    else:
        people.loc[0, "slab"] = "session82"
    with pytest.raises(ValueError, match="metadata"):
        partitions(sessions, markets, people)
