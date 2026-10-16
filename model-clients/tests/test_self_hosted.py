"""Independent TypeSafe HTTP services and explicit credential handling."""

import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import Noul, RetryPolicy

from thesis_model_clients import TypeSafeHTTPDecisionClient


@pytest.mark.parametrize("authenticated", [False, True])
def test_standard_endpoint_needs_no_health_metadata_or_gateway_credentials(
    monkeypatch, authenticated
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "must-not-leak")
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-leak-either")
    monkeypatch.setenv("LOCAL_DECISION_KEY", "chosen-local-key")
    requests = []
    payload = {
        "model": "custom-v1",
        "usage": {"input_tokens": 9, "output_tokens": 0},
        "answers": {"q": {"type": "noul", "noul": 0.7}},
    }

    def reply(request):
        requests.append(request)
        return httpx2.Response(200, json=payload)

    async def run():
        async with TypeSafeHTTPDecisionClient(
            "custom-v1",
            "my-local-service",
            base_url="http://localhost:9999/api/",
            api_key_env="LOCAL_DECISION_KEY" if authenticated else None,
            _transport=httpx2.MockTransport(reply),
        ) as client:
            assert client._timeout_seconds == 120
            return await client.evaluate({"message": "unchanged"}, {"q": Noul()})

    result = asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://localhost:9999/api/v1/systemone"
    assert request.headers.get("Authorization") == (
        "Bearer chosen-local-key" if authenticated else None
    )
    assert "must-not-leak" not in str(request.headers)
    assert "providerOptions" not in json.loads(request.content)
    assert result.resolved_providers == ()
    assert result.requested_provider == "my-local-service"
    assert result.usage.cost_usd is None
    assert result.calls[0].raw == payload


def test_failed_local_response_preserves_usage_raw_diagnostics_without_retry():
    payload = {
        "model": "custom-v1",
        "provider": "local",
        "usage": {"input_tokens": 123, "output_tokens": 0},
        "error": {"category": "ModelInputLimit", "message": "State was truncated"},
        "server_metadata": {"state_tokens_dropped": 5},
    }
    attempts = []

    async def run():
        async with TypeSafeHTTPDecisionClient(
            "custom-v1",
            "local",
            base_url="http://localhost:9999",
            retry=RetryPolicy(max_retries=2),
            _transport=httpx2.MockTransport(
                lambda request: attempts.append(request) or httpx2.Response(422, json=payload)
            ),
        ) as client:
            return await client.evaluate("state", {"q": Noul()})

    result = asyncio.run(run())
    assert len(attempts) == 1 and result.usage.retries == 0
    assert result.usage.input_tokens == 123 and result.usage.cost_usd is None
    assert result.calls[0].raw["body"] == payload
    assert result.resolved_providers == ("local",)
    assert set(result.errors) == {"q"}


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "file:///tmp/model",
        "http://user:key@host",
        "http://host?key=value",
        "http://host#key",
        "http://host:99999",
    ],
)
def test_invalid_api_roots_fail_before_network(base_url):
    with pytest.raises(ValueError):
        TypeSafeHTTPDecisionClient("model", "local", base_url=base_url)


@pytest.mark.parametrize("api_key_env", ["AI_GATEWAY_API_KEY", "", "contains spaces"])
def test_invalid_or_gateway_auth_variable_is_rejected(api_key_env):
    with pytest.raises(ValueError, match="api_key_env"):
        TypeSafeHTTPDecisionClient(
            "model", "local", base_url="http://localhost", api_key_env=api_key_env
        )


def test_missing_explicit_local_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("LOCAL_DECISION_KEY", raising=False)
    with pytest.raises(ValueError, match="LOCAL_DECISION_KEY"):
        TypeSafeHTTPDecisionClient(
            "model", "local", base_url="http://localhost", api_key_env="LOCAL_DECISION_KEY"
        )
