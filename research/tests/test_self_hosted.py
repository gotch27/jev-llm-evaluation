import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import Noul, RetryPolicy

from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.types import ModelSpec
from thesis_research.clients import TypeSafeHTTPDecisionClient

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/boolean_question_answering/benchmarks/boolq-local-smoke.toml"


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


def test_old_model_serialization_is_unchanged_and_local_settings_roundtrip():
    assert ModelSpec("jev", "system_one", "typesafe-ai/jev", "typesafe-ai").as_dict() == {
        "id": "jev",
        "backend": "system_one",
        "model": "typesafe-ai/jev",
        "provider": "typesafe-ai",
    }
    spec = load_benchmark_plan(PLAN).models[0]
    assert spec.transport == "self_hosted"
    assert spec.as_dict()["base_url"] == "http://127.0.0.1:8000"
    assert "api_key_env" not in spec.as_dict() and "timeout_seconds" not in spec.as_dict()


@pytest.mark.parametrize(
    "replacement",
    [
        ('transport = "self_hosted"', 'transport = "unknown"'),
        ('transport = "self_hosted"', 'transport = "vercel"'),
        ('backend = "system_one"', 'backend = "llm"'),
        ('base_url = "http://127.0.0.1:8000"', 'base_url = "file:///tmp/model"'),
        ('base_url = "http://127.0.0.1:8000"', "timeout_seconds = -1"),
    ],
)
def test_invalid_local_plan_settings_are_rejected(tmp_path, replacement):
    path = tmp_path / "plan.toml"
    path.write_text(PLAN.read_text().replace(*replacement))
    with pytest.raises(ValueError):
        load_benchmark_plan(path)


def test_three_local_plans_preserve_original_cohorts_and_task_settings():
    for hosted_path in (ROOT / "experiments").rglob("*-system-one-smoke.toml"):
        hosted = load_benchmark_plan(hosted_path)
        local = load_benchmark_plan(
            hosted_path.with_name(hosted_path.name.replace("-system-one-smoke", "-local-smoke"))
        )
        assert local.cohort == hosted.cohort
        assert local.task_config == hosted.task_config
        assert local.execution.example_concurrency == local.execution.model_concurrency == 1
        assert local.baseline_model_id == "laya-local"
        assert [model.id for model in local.models] == ["laya-local"]
        assert local.models[0].transport == "self_hosted"
