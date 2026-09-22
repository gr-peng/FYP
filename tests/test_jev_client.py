import asyncio
import json

import httpx
import pytest

from experiments.jev_client import JevClient, JevContractError, sample_action, validate_answer


def response():
    return {
        "model": "jev-1.13.0",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "B",
                "probabilities": {"A": 0.3, "B": 0.7},
                "confidence": 0.4,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def test_api_argmax_is_distinct_from_stochastic_action():
    answer = validate_answer(response(), "decision", {"A": None, "B": None}, "jev-1.13.0")
    assert answer["choice"] == "B"
    assert sample_action(answer["probabilities"], 0.1) == "A"
    assert sample_action(answer["probabilities"], 0.9) == "B"


@pytest.mark.parametrize(
    "probabilities",
    [
        {"A": 0.3, "C": 0.7},
        {"A": -0.1, "B": 1.1},
        {"A": 0.7, "B": 0.7},
        {"A": float("nan"), "B": 0.7},
        {"A": "0.3", "B": 0.7},
        {"A": False, "B": 1.0},
    ],
)
def test_invalid_distributions_are_rejected(probabilities):
    body = response()
    body["answers"]["decision"]["probabilities"] = probabilities
    with pytest.raises(JevContractError):
        validate_answer(body, "decision", {"A": None, "B": None}, "jev-1.13.0")


def test_model_drift_and_non_argmax_responses_are_rejected():
    with pytest.raises(JevContractError, match="pinned"):
        validate_answer(response(), "decision", {"A": None, "B": None}, "jev-1.14.0")
    body = response()
    body["answers"]["decision"]["choice"] = "A"
    with pytest.raises(JevContractError, match="maximum"):
        validate_answer(body, "decision", {"A": None, "B": None}, "jev-1.13.0")


def test_credential_destination_is_checked_before_http_client(tmp_path):
    with pytest.raises(ValueError, match="official endpoint"):
        JevClient("secret", tmp_path, tmp_path, endpoint="https://example.org/v1/systemone")
    with pytest.raises(ValueError, match="official endpoint"):
        JevClient(
            "secret",
            tmp_path,
            tmp_path,
            endpoint="https://api.typesafe.ai@other.example/v1/systemone",
        )


def test_jev_wire_contract_cache_and_redaction(tmp_path):
    async def execute():
        calls = []

        def handle(request):
            calls.append(json.loads(request.content))
            assert request.headers["authorization"] == "Bearer unit-secret"
            return httpx.Response(200, json=response(), headers={"x-typesafe-request-id": "test"})

        client = JevClient(
            "unit-secret",
            tmp_path / "log",
            tmp_path / "cache",
            transport=httpx.MockTransport(handle),
        )
        try:
            a = await client.evaluate_choice("state", "question", {"A": None, "B": None}, {"id": 1})
            b = await client.evaluate_choice("state", "question", {"A": None, "B": None}, {"id": 1})
            assert not a["cache_hit"] and b["cache_hit"]
            assert len(calls) == 1
            assert calls[0]["questions"]["decision"]["type"] == "choice"
            client.log("redaction.jsonl", {"value": "unit-secret"})
            assert all(
                "unit-secret" not in p.read_text() for p in (tmp_path / "log").glob("*.jsonl")
            )
        finally:
            await client.close()

    asyncio.run(execute())
