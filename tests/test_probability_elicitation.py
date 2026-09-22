import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.probability_elicitation import (  # noqa: E402
    LLMProbabilityClient,
    validate_probabilities,
)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"A": True, "B": 0},
        {"A": "0.5", "B": 0.5},
        {"A": float("nan"), "B": 0.5},
        {"A": 0.7, "B": 0.4},
        {"A": -0.1, "B": 1.1},
        {"A": 0.5, "C": 0.5},
    ],
)
def test_reject_unusable_human_probabilities(probabilities):
    with pytest.raises(ValueError):
        validate_probabilities(json.dumps({"probabilities": probabilities}), {"A": "a", "B": "b"})


def test_roundoff_is_explicitly_recorded():
    p, meta = validate_probabilities('{"probabilities":{"A":0.5,"B":0.5001}}', {"A": "a", "B": "b"})
    assert abs(sum(p.values()) - 1) < 1e-12
    assert meta["normalized"] is True
    assert meta["raw_probability_mass"] == 1.0001


def test_invalid_output_never_falls_back_to_a_trading_action_and_usage_is_preserved():
    class Transport:
        def __init__(self, valid_second):
            self.identities = []
            self.valid_second = valid_second

        async def completion(self, messages, identity):
            self.identities.append(identity)
            raw = '{"choice":"A"}'
            if len(self.identities) == 2 and self.valid_second:
                raw = '{"probabilities":{"A":0.2,"B":0.8}}'
            return {
                "choices": [{"message": {"content": raw}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }, {"cache_hit": False}

    for valid in (True, False):
        transport = Transport(valid)
        p, meta = asyncio.run(
            LLMProbabilityClient(transport).evaluate_choice(
                "visible", "predict participant", {"A": "hold", "B": "sell"}, {"task": "test"}
            )
        )
        assert len(meta["calls"]) == 2
        assert sum(c["usage"]["prompt_tokens"] for c in meta["calls"]) == 40
        assert [x["repair"] for x in transport.identities] == [0, 1]
        if valid:
            assert p == {"A": 0.2, "B": 0.8}
        else:
            assert p is None
            assert meta["status"] == "invalid_probability_schema"
