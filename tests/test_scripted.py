import json
from pathlib import Path

from behavioral_market.simulation.scripted import run


def test_scripted_run_is_reproducible(tmp_path: Path) -> None:
    config = {
        "experiment": {"name": "test", "seed": 42},
        "market": {"asset": "TEST", "initial_price": "100", "periods": 5},
        "population": {"n_agents": 10, "initial_cash": "10000", "initial_shares": 10},
    }
    first = run(config, tmp_path / "first")
    second = run(config, tmp_path / "second")
    first.pop("generated_at_utc")
    second.pop("generated_at_utc")
    assert first == second
    assert first["trades"] > 0
    for name in ("orders.csv", "trades.csv", "market.csv", "agents.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert json.loads((tmp_path / "first" / "config_snapshot.json").read_text()) == config
