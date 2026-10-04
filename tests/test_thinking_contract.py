import asyncio
import json

import httpx
import pytest

from behavioral_market.llm.client import FatalAPIError, OpenAICompatibleClient


def client_at(tmp_path):
    return OpenAICompatibleClient(
        {
            "model": "test",
            "temperature": 0.2,
            "max_tokens": 512,
            "enable_thinking": False,
            "thinking": {"type": "disabled"},
            "require_no_reasoning": True,
            "api_key_env": "TEST_KEY",
            "allowed_host": "example.test",
            "chat_completions_url": "https://example.test/chat",
            "models_url": "https://example.test/models",
            "max_concurrency": 1,
            "timeout_seconds": 1,
            "transport_retries": 0,
        },
        "test-key",
        tmp_path / "logs",
        tmp_path / "cache",
    )


def test_native_mode_is_in_request_and_reasoning_response_is_not_cached(tmp_path):
    async def scenario():
        client = client_at(tmp_path)
        client.config.update(
            extra_body={"thinking": {"type": "disabled"}, "enable_thinking": False},
            allowed_openai_params=["thinking"],
            reasoning_effort="none",
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        await client.http.aclose()

        def respond(request):
            payload = json.loads(request.content)
            assert payload["thinking"] == {"type": "disabled"}
            assert payload["enable_thinking"] is False
            assert payload["extra_body"]["thinking"] == {"type": "disabled"}
            assert payload["reasoning_effort"] == "none"
            assert payload["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
            return httpx.Response(
                200,
                json={
                    "model": "test",
                    "choices": [
                        {"message": {"content": "{}", "reasoning_content": "unexpected reasoning"}}
                    ],
                },
            )

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with pytest.raises(FatalAPIError, match="non-thinking contract"):
            await client.completion([{"role": "user", "content": "test"}], {})
        assert not list((tmp_path / "cache").glob("*.json"))
        assert (tmp_path / "logs/llm_responses.jsonl").exists()
        await client.close()

    asyncio.run(scenario())


def test_reasoning_validation_is_opt_in_for_legacy_clients(tmp_path):
    client = client_at(tmp_path)
    response = {"model": "test", "choices": [{"message": {"reasoning_content": "unexpected"}}]}
    with pytest.raises(FatalAPIError):
        client.validate_response_contract(response)
    client.config.pop("require_no_reasoning")
    client.validate_response_contract(response)
    asyncio.run(client.close())


def test_passthrough_cannot_override_model_or_destination(tmp_path):
    async def scenario():
        client = client_at(tmp_path)
        client.config["extra_body"] = {"model": "other-model"}
        with pytest.raises(ValueError, match="only contain thinking-mode"):
            await client.completion([], {})
        await client.close()

    asyncio.run(scenario())
