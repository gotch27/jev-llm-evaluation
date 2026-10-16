import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest
from system_one_adapter.providers import Message, ProviderResult
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeError,
    Usage,
)

from thesis_model_clients import (
    TypeSafeHTTPDecisionClient,
    VercelLLMDecisionClient,
    VercelSystemOneDecisionClient,
)
from thesis_model_clients import system_one as system_one_module
from thesis_model_clients import vercel_provider as vercel_provider_module
from thesis_model_clients.gateway_metadata import reported_cost, resolved_provider
from thesis_model_clients.vercel_provider import (
    VercelGatewayProvider,
    VercelGatewayProviderResult,
)

QUESTIONS = {
    "relevant": Noul(instructions="Is the message relevant?"),
    "intent": Choice(
        instructions="What is the intent?",
        criteria={"card": "A card problem", "cash": "A cash problem"},
    ),
    "urgency": Score(
        instructions="How urgent is it?",
        criteria=["Not urgent", "Urgent", "Critical"],
    ),
}

SYSTEM_ONE_ROUTES = (
    ("typesafe-ai/jev", "typesafe-ai"),
    ("convaiinnovations/laya", "boundless"),
    ("liquid/d1", "liquid"),
    ("openai/gpt-6-luna-decisions", "openai"),
    ("microsoft/microsoft-decision-1", "azure"),
    ("scaledown/scaledown-classify-1", "scaledown"),
    ("perplexity/pplx-decider-v1.1-27b", "perplexity"),
    ("liquid/d1-ultra", "liquid"),
    ("typesafe-ai/jev-1.14.0", "typesafe-ai"),
)


class FakeJevClient:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.closed = False

    async def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.error:
            raise self.error
        payload = SystemOneResponse(
            model="typesafe-ai/jev",
            usage=Usage(input_tokens=12, output_tokens=0),
            answers={
                "relevant": NoulAnswer(noul=0.9),
                "intent": ChoiceAnswer(
                    choice="card",
                    confidence=0.8,
                    probabilities={"card": 0.9, "cash": 0.1},
                ),
                "urgency": {
                    "type": "score",
                    "score": 1.8,
                    "confidence": 0.7,
                    "legend": {0: "Not urgent", 1: "Urgent", 2: "Critical"},
                    "probabilities": {0: 0.0, 1: 0.2, 2: 0.8},
                },
            },
        ).model_dump(mode="json")
        payload["provider_metadata"] = {
            "gateway": {
                "cost": "0.0001",
                "generationId": "offline-generation",
                "routing": {"resolvedProvider": "typesafe-ai"},
            }
        }
        return SystemOneResponse.from_http_response(
            httpx2.Response(
                200,
                json=payload,
                request=httpx2.Request("POST", "https://example.test/systemone"),
            )
        )

    async def aclose(self):
        self.closed = True


class FakeAdapterProvider:
    model_name = "openai/exact-model-2026-09-01"

    def __init__(
        self,
        *,
        delay=0.01,
        fail_question=None,
        fail_once=False,
        malformed_question=None,
        output_mode="probabilities",
    ):
        self.delay = delay
        self.fail_question = fail_question
        self.fail_once = fail_once
        self.malformed_question = malformed_question
        self.output_mode = output_mode
        self.failed_once = False
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.closed = False

    def translate_error(self, error):
        return error if isinstance(error, TypeSafeError) else TypeSafeError(str(error))

    async def request(self, messages, *, schema, structured):
        question_id = next(iter(schema["$defs"]["TypeSafeAnswers"]["properties"]))
        self.calls.append((question_id, messages, schema, structured))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if question_id == self.fail_question and (not self.fail_once or not self.failed_once):
                self.failed_once = True
                if self.fail_once:
                    raise TypeSafeAPIConnectionError("temporary failure")
                raise TypeSafeError("permanent failure")
            if question_id == self.malformed_question:
                return ProviderResult(text="not JSON", input_tokens=10, output_tokens=2)
            answers = (
                {"relevant": True, "intent": "card", "urgency": 2}
                if self.output_mode == "discrete"
                else {
                    "relevant": 0.8,
                    "intent": {"card": 0.6, "cash": 0.2},
                    "urgency": {"0": 0.1, "1": 0.2, "2": 0.7},
                }
            )
            answer = answers[question_id]
            return VercelGatewayProviderResult(
                text=json.dumps({"answers": {question_id: answer}}),
                input_tokens=10,
                output_tokens=4,
                request={"question_id": question_id},
                raw_response={"model": self.model_name, "provider": "openai"},
                resolved_model=self.model_name,
                resolved_provider="openai",
                cost_usd=0.01,
            )
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True


def test_jev_sends_all_questions_in_one_call_and_closes():
    async def run():
        fake = FakeJevClient()
        async with VercelSystemOneDecisionClient(
            "typesafe-ai/jev", "typesafe-ai", _client=fake
        ) as client:
            result = await client.evaluate("A message", QUESTIONS)
        return fake, result

    fake, result = asyncio.run(run())
    assert len(fake.calls) == 1
    assert fake.calls[0][1] is QUESTIONS
    assert fake.calls[0][2]["extra_body"] == {
        "providerOptions": {"gateway": {"only": ["typesafe-ai"]}}
    }
    assert result.answers.keys() == QUESTIONS.keys()
    assert result.errors == {}
    assert result.requested_model == result.resolved_models[0] == "typesafe-ai/jev"
    assert result.resolved_providers == ("typesafe-ai",)
    assert result.usage.input_tokens == 12
    assert result.usage.cost_usd == pytest.approx(0.0001)
    assert result.usage.retries == 0
    assert result.calls[0].raw["provider_metadata"]["gateway"]["generationId"] == (
        "offline-generation"
    )
    assert fake.closed


def test_jev_failure_is_recorded_for_every_question():
    async def run():
        fake = FakeJevClient(TypeSafeError("unavailable"))
        client = VercelSystemOneDecisionClient("typesafe-ai/jev", "typesafe-ai", _client=fake)
        result = await client.evaluate("A message", QUESTIONS)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.answers == {}
    assert result.errors.keys() == QUESTIONS.keys()
    assert len({id(error) for error in result.errors.values()}) == 1
    assert result.calls[0].error.message == "unavailable"


def test_vercel_isolates_questions_runs_concurrently_and_normalizes():
    async def run():
        provider = FakeAdapterProvider()
        async with VercelLLMDecisionClient(
            provider.model_name,
            "openai",
            max_concurrency=2,
            retry=RetryPolicy(max_retries=0),
            _provider=provider,
        ) as client:
            result = await client.evaluate("A message", QUESTIONS)
        return provider, result

    provider, result = asyncio.run(run())
    assert len(provider.calls) == 3
    assert provider.max_active == 2
    assert all(len(call.question_ids) == 1 for call in result.calls)
    assert {question_id for question_id, *_ in provider.calls} == set(QUESTIONS)
    assert all(structured for *_, structured in provider.calls)
    assert isinstance(result.answers["relevant"], NoulAnswer)
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == "card"
    assert result.answers["intent"].probabilities == pytest.approx({"card": 0.75, "cash": 0.25})
    intent_call = next(call for call in result.calls if call.question_ids == ("intent",))
    assert intent_call.raw["debug"]["original_probabilities"]["intent"] == {
        "card": 0.6,
        "cash": 0.2,
    }
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 12
    assert result.usage.cost_usd == pytest.approx(0.03)
    assert result.usage.retries == 0
    assert result.resolved_models == (provider.model_name,)
    assert result.resolved_providers == ("openai",)
    assert provider.closed


def test_vercel_preserves_partial_failure():
    async def run():
        provider = FakeAdapterProvider(fail_question="intent")
        client = VercelLLMDecisionClient(
            provider.model_name,
            "openai",
            retry=RetryPolicy(max_retries=0),
            _provider=provider,
        )
        result = await client.evaluate("A message", QUESTIONS)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.answers.keys() == {"relevant", "urgency"}
    assert result.errors.keys() == {"intent"}
    assert result.errors["intent"].message == "permanent failure"
    assert result.usage.input_tokens is None
    assert result.usage.cost_usd is None


def test_vercel_discrete_mode_uses_discrete_schemas():
    output_mode = "discrete"

    async def run():
        provider = FakeAdapterProvider(output_mode=output_mode)
        client = VercelLLMDecisionClient(
            provider.model_name,
            "openai",
            output_mode=output_mode,
            retry=RetryPolicy(max_retries=0),
            _provider=provider,
        )
        result = await client.evaluate("A message", QUESTIONS)
        await client.aclose()
        return provider, result

    provider, result = asyncio.run(run())
    schemas = {
        question_id: schema["$defs"]["TypeSafeAnswers"]["properties"][question_id]
        for question_id, _, schema, _ in provider.calls
    }
    intent_schema = schemas["intent"]
    assert intent_schema["enum"] == ["card", "cash"]
    assert schemas["relevant"]["type"] == "boolean"
    assert schemas["urgency"]["type"] == "integer"
    assert result.errors == {}
    assert result.answers["intent"].choice == "card"
    assert result.answers["intent"].probabilities == {"card": 1.0, "cash": 0.0}
    assert result.answers["relevant"].noul == 1.0
    assert result.answers["urgency"].score == 2.0
    assert result.answers["urgency"].probabilities == {0: 0.0, 1: 0.0, 2: 1.0}


def test_vercel_does_not_retry_malformed_output():
    async def run():
        provider = FakeAdapterProvider(malformed_question="relevant", delay=0)
        client = VercelLLMDecisionClient(
            provider.model_name,
            "openai",
            retry=RetryPolicy(max_retries=2, backoff_initial=0),
            _provider=provider,
        )
        result = await client.evaluate("A message", {"relevant": QUESTIONS["relevant"]})
        await client.aclose()
        return provider, result, client

    provider, result, client = asyncio.run(run())
    assert len(provider.calls) == 1
    assert result.answers == {}
    assert result.errors.keys() == {"relevant"}
    assert "Validation" in result.errors["relevant"].error_type
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(client.evaluate("A message", {"relevant": QUESTIONS["relevant"]}))


def test_vercel_retries_transient_failure_and_records_it():
    async def run():
        provider = FakeAdapterProvider(fail_question="relevant", fail_once=True, delay=0)
        client = VercelLLMDecisionClient(
            provider.model_name,
            "openai",
            retry=RetryPolicy(
                max_retries=1,
                backoff_initial=0,
                backoff_jitter=0,
                timeout=None,
            ),
            _provider=provider,
        )
        result = await client.evaluate("A message", {"relevant": QUESTIONS["relevant"]})
        await client.aclose()
        return provider, result

    provider, result = asyncio.run(run())
    assert len(provider.calls) == 2
    assert result.errors == {}
    assert result.usage.retries == 1
    assert result.calls[0].raw["debug"]["retry_reasons"][0][0] == "provider_error"


def test_vercel_evaluation_can_be_cancelled():
    async def run():
        provider = FakeAdapterProvider(delay=10)
        client = VercelLLMDecisionClient(provider.model_name, "openai", _provider=provider)
        task = asyncio.create_task(client.evaluate("A message", QUESTIONS))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.aclose()
        return provider

    provider = asyncio.run(run())
    assert provider.active == 0
    assert provider.closed


def test_missing_credentials_and_aliases_fail_before_network(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        VercelSystemOneDecisionClient("typesafe-ai/jev", "typesafe-ai")
    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        VercelLLMDecisionClient("openai/exact-model", "openai")
    with pytest.raises(ValueError, match="canonical"):
        VercelLLMDecisionClient("~openai/latest", "openai")
    with pytest.raises(ValueError, match="canonical"):
        VercelLLMDecisionClient("vercel/auto", "openai")
    with pytest.raises(ValueError, match="output_mode"):
        VercelLLMDecisionClient("openai/exact-model", "openai", output_mode="confidence")
    with pytest.raises(ValueError, match="output_mode"):
        VercelLLMDecisionClient("openai/exact-model", "openai", output_mode="label")
    with pytest.raises(ValueError, match="reasoning effort"):
        VercelLLMDecisionClient(
            "openai/exact-model",
            "openai",
            reasoning_effort="extreme",
            _provider=FakeAdapterProvider(),
        )


def test_jev_constructs_typesafe_client_for_vercel(monkeypatch):
    captured = {}

    class FakeConstructedClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def aclose(self):
            captured["closed"] = True

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    monkeypatch.setattr(system_one_module, "AsyncTypeSafeClient", FakeConstructedClient)
    client = VercelSystemOneDecisionClient("typesafe-ai/jev", "typesafe-ai")
    asyncio.run(client.aclose())

    assert captured["api_key"] == "unused-test-key"
    assert captured["model"] == "typesafe-ai/jev"
    assert captured["base_url"] == "https://ai-gateway.vercel.sh/typesafe"
    assert captured["closed"] is True


def test_llm_provider_constructs_openai_client_for_vercel(monkeypatch):
    captured = {}

    class FakeConstructedClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(vercel_provider_module.openai, "AsyncOpenAI", FakeConstructedClient)
    provider = VercelGatewayProvider(
        "openai/exact-model-2026-09-01",
        "openai",
        "unused-test-key",
    )
    asyncio.run(provider.aclose())

    assert captured == {
        "api_key": "unused-test-key",
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "timeout": 60.0,
        "max_retries": 0,
        "closed": True,
    }


class FakeCompletions:
    def __init__(self):
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"answers":{"q":0.8}}'),
                    finish_reason="stop",
                )
            ],
            model_dump=lambda **kwargs: {
                "id": "generation-1",
                "model": "openai/exact-model-2026-09-01",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.004},
                "provider_metadata": {
                    "gateway": {
                        "routing": {
                            "canonicalSlug": "openai/exact-model-2026-09-01",
                            "resolvedProvider": "openai",
                            "resolvedProviderApiModelId": "exact-model-2026-09-01",
                        },
                    }
                },
            },
        )


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())
        self.closed = False

    async def close(self):
        self.closed = True


def test_vercel_provider_sends_strict_pinned_request_and_keeps_raw_response():
    async def run():
        fake = FakeOpenAIClient()
        provider = VercelGatewayProvider(
            "openai/exact-model-2026-09-01",
            "openai",
            "unused-test-key",
            reasoning_effort="low",
            _client=fake,
        )
        result = await provider.request(
            [Message(role="user", content="state")],
            schema={"type": "object"},
            structured=True,
        )
        await provider.aclose()
        return fake, result

    fake, result = asyncio.run(run())
    request = fake.chat.completions.request
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["extra_body"] == {
        "providerOptions": {"gateway": {"only": ["openai"]}},
        "reasoning": {"effort": "low"},
    }
    assert "models" not in json.dumps(request) and "api_key" not in str(request)
    assert result.resolved_model == "openai/exact-model-2026-09-01"
    assert result.resolved_provider == "openai"
    assert result.cost_usd == 0.004
    assert result.raw_response["id"] == "generation-1"
    assert fake.closed


def test_vercel_provider_rejects_unstructured_mode():
    async def run():
        provider = VercelGatewayProvider(
            "openai/exact-model-2026-09-01",
            "openai",
            "unused-test-key",
            _client=FakeOpenAIClient(),
        )
        with pytest.raises(ValueError, match="structured"):
            await provider.request([], schema={}, structured=False)
        await provider.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("cost", [0, "0", "0.0025", None, True, -1, "nan", "inf", "bad"])
def test_gateway_cost_keeps_unknown_distinct_from_zero(cost):
    raw = {"provider_metadata": {"gateway": {"cost": cost}}}
    expected = float(cost) if cost in (0, "0", "0.0025") and cost is not True else None
    assert reported_cost(raw) == expected


def test_gateway_routing_requires_reported_metadata():
    assert resolved_provider({}) is None
    assert (
        resolved_provider(
            {"providerMetadata": {"gateway": {"routing": {"finalProvider": "typesafe-ai"}}}}
        )
        == "typesafe-ai"
    )


HTTP_SYSTEM_ONE_ROUTES = (*SYSTEM_ONE_ROUTES, ("laya-base", "local"))


def http_system_one_client(model, provider, **kwargs):
    if provider == "local":
        return TypeSafeHTTPDecisionClient(
            model, provider, base_url="http://127.0.0.1:8000", **kwargs
        )
    return VercelSystemOneDecisionClient(model, provider, **kwargs)


def _http_system_one_payload(model="typesafe-ai/jev", provider="typesafe-ai"):
    return {
        "model": model,
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "answers": {"relevant": {"type": "noul", "noul": 0.9}},
        "provider_metadata": {
            "gateway": {
                "cost": "0.0001",
                "generationId": "offline-generation",
                "routing": {"resolvedProvider": provider},
            }
        },
    }


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
def test_system_one_http_retries_are_isolated_between_concurrent_examples(
    monkeypatch, model, provider
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    attempts = {}
    active = 0
    max_active = 0

    async def reply(request):
        nonlocal active, max_active
        state = json.loads(request.content)["state"]
        attempts[state] = attempts.get(state, 0) + 1
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0)
            if state == "retry" and attempts[state] == 1:
                return httpx2.Response(503, json={"message": "offline transient failure"})
            return httpx2.Response(200, json=_http_system_one_payload(model, provider))
        finally:
            active -= 1

    async def run():
        async with http_system_one_client(
            model,
            provider,
            max_concurrency=2,
            retry=RetryPolicy(backoff_initial=0, backoff_max=0),
            _transport=httpx2.MockTransport(reply),
        ) as client:
            return await asyncio.gather(
                client.evaluate("retry", {"relevant": QUESTIONS["relevant"]}),
                client.evaluate("first-attempt", {"relevant": QUESTIONS["relevant"]}),
                client.evaluate("third-example", {"relevant": QUESTIONS["relevant"]}),
            )

    retried, first_attempt, third = asyncio.run(run())
    assert attempts == {"retry": 2, "first-attempt": 1, "third-example": 1}
    assert max_active == 2
    assert [result.usage.retries for result in (retried, first_attempt, third)] == [1, 0, 0]
    assert all(result.usage.cost_usd == 0.0001 for result in (retried, first_attempt, third))
    assert "unused-test-key" not in json.dumps(retried.calls[0].raw)


@pytest.mark.parametrize("failure", ["http", "connection"])
@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
def test_system_one_exhausted_retries_are_recorded(monkeypatch, failure, model, provider):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    attempts = 0

    def reply(request):
        nonlocal attempts
        attempts += 1
        if failure == "connection":
            raise httpx2.ConnectError("offline connection failure", request=request)
        return httpx2.Response(
            503,
            json={"message": "offline unavailable"},
            headers={"x-typesafe-request-id": "offline-request"},
        )

    async def run():
        async with http_system_one_client(
            model,
            provider,
            retry=RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0),
            _transport=httpx2.MockTransport(reply),
        ) as client:
            return await client.evaluate("A message", QUESTIONS)

    result = asyncio.run(run())
    assert attempts == 3
    assert result.usage.retries == 2
    assert result.usage.cost_usd is None
    assert result.usage.input_tokens is None
    assert result.errors.keys() == QUESTIONS.keys()
    if failure == "http":
        assert result.calls[0].raw["status"] == 503
        assert result.calls[0].raw["request_id"] == "offline-request"
        assert result.calls[0].raw["body"]["message"] == "offline unavailable"


@pytest.mark.parametrize("backend", ["system_one", "llm"])
def test_provider_call_latency_excludes_queue_and_decision_latency_includes_it(
    monkeypatch, backend
):
    clock = [0.0]
    monkeypatch.setattr(system_one_module.time, "perf_counter", lambda: clock[0])

    class TimedJevClient(FakeJevClient):
        async def system_one(self, *args, **kwargs):
            clock[0] += 3
            return await super().system_one(*args, **kwargs)

    class TimedProvider(FakeAdapterProvider):
        async def request(self, *args, **kwargs):
            result = await super().request(*args, **kwargs)
            clock[0] += 3
            return result

    async def run():
        provider = TimedProvider(delay=0)
        client = (
            VercelSystemOneDecisionClient(
                "typesafe-ai/jev", "typesafe-ai", max_concurrency=1, _client=TimedJevClient()
            )
            if backend == "system_one"
            else VercelLLMDecisionClient(
                provider.model_name, "openai", max_concurrency=1, _provider=provider
            )
        )
        async with client:
            async with client._semaphore:
                pending = asyncio.create_task(
                    client.evaluate("A message", {"relevant": QUESTIONS["relevant"]})
                )
                await asyncio.sleep(0)
                clock[0] = 20
            return await pending

    result = asyncio.run(run())
    assert result.latency_seconds == 23
    assert result.calls[0].latency_seconds == 3


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
@pytest.mark.parametrize("cost", [None, "0", "0.0025"])
def test_system_one_http_routes_preserve_shared_questions_answers_and_metadata(
    monkeypatch, model, provider, cost
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    requests = []
    # Large Choice tasks must reach every provider without altering the option set.
    questions = {**QUESTIONS, "intent": Choice(criteria={f"intent-{i}": None for i in range(77)})}
    state = {"message": "A card problem", "context": ["same", "frozen", "state"]}
    payload = _http_system_one_payload(model, provider)
    payload["answers"].update(
        intent={
            "type": "choice",
            "choice": "intent-0",
            "confidence": 0.8,
            "probabilities": {f"intent-{i}": 1.0 if i == 0 else 0.0 for i in range(77)},
        },
        urgency={
            "type": "score",
            "score": 1.8,
            "confidence": 0.7,
            "legend": {"0": "Not urgent", "1": "Urgent", "2": "Critical"},
            "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
        },
    )
    gateway = payload["provider_metadata"]["gateway"]
    if cost is None:
        gateway.pop("cost")
    else:
        gateway["cost"] = cost

    def reply(request):
        requests.append(request)
        return httpx2.Response(200, json=payload)

    async def run():
        async with http_system_one_client(
            model, provider, _transport=httpx2.MockTransport(reply)
        ) as client:
            return await client.evaluate(state, questions)

    result = asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == (
        "http://127.0.0.1:8000/v1/systemone"
        if provider == "local"
        else "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    )
    assert body["model"] == model
    if provider == "local":
        assert "providerOptions" not in body
        assert "Authorization" not in request.headers
    else:
        assert body["providerOptions"] == {"gateway": {"only": [provider]}}
    assert body["state"] == state
    assert body["questions"] == {
        key: question.model_dump(mode="json", exclude_none=True)
        for key, question in questions.items()
    }
    assert len(body["questions"]["intent"]["criteria"]) == 77
    assert result.answers["intent"].choice == "intent-0"
    assert result.answers["relevant"].noul == 0.9
    assert result.answers["urgency"].score == 1.8
    assert result.answers["urgency"].probabilities == {0: 0.0, 1: 0.2, 2: 0.8}
    assert result.requested_model == model
    assert result.resolved_models == (model,)
    assert result.requested_provider == provider
    assert result.resolved_providers == (provider,)
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == (None if cost is None else float(cost))
    assert result.usage.retries == 0
    assert result.usage.malformed_response_retries == 0
    assert result.latency_seconds >= result.calls[0].latency_seconds >= 0
    assert result.calls[0].raw == payload
    assert "unused-test-key" not in json.dumps(result.calls[0].raw)


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
@pytest.mark.parametrize("refused_ids", [("intent",), tuple(QUESTIONS)])
def test_system_one_refusals_keep_successful_answers_usage_and_raw_without_retries(
    monkeypatch, model, provider, refused_ids
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    payload = _http_system_one_payload(model, provider)
    payload["answers"].update(
        intent={
            "type": "choice",
            "choice": "card",
            "confidence": 0.9,
            "probabilities": {"card": 0.9, "cash": 0.1},
        },
        urgency={
            "type": "score",
            "score": 1.8,
            "confidence": 0.8,
            "legend": {"0": "Not urgent", "1": "Urgent", "2": "Critical"},
            "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
        },
    )
    for question_id in refused_ids:
        payload["answers"][question_id] = {"type": "refusal", "name": question_id}
    requests = []

    def reply(request):
        requests.append(request)
        return httpx2.Response(200, json=payload)

    async def run():
        async with http_system_one_client(
            model, provider, _transport=httpx2.MockTransport(reply)
        ) as client:
            return await client.evaluate("A message", QUESTIONS)

    result = asyncio.run(run())
    assert len(requests) == 1
    assert set(result.answers) == set(QUESTIONS) - set(refused_ids)
    assert set(result.errors) == set(refused_ids)
    for question_id, error in result.errors.items():
        assert error.error_type == "ModelRefusal"
        assert question_id in error.message
        assert error.diagnostics["answer"] == payload["answers"][question_id]
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == 0.0001
    assert result.usage.retries == 0
    assert result.calls[0].error is None  # HTTP succeeded, even if a question was refused.
    assert result.calls[0].raw == payload
    assert result.resolved_models == (model,)
    assert result.resolved_providers == (provider,)


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
def test_system_one_http_evaluation_cancellation_releases_concurrency(monkeypatch, model, provider):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        attempts = 0

        async def reply(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                entered.set()
                await release.wait()
            return httpx2.Response(200, json=_http_system_one_payload(model, provider))

        async with http_system_one_client(
            model, provider, max_concurrency=1, _transport=httpx2.MockTransport(reply)
        ) as client:
            task = asyncio.create_task(client.evaluate("cancel", QUESTIONS))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            result = await asyncio.wait_for(
                client.evaluate("next", {"relevant": QUESTIONS["relevant"]}), timeout=1
            )
            assert result.usage.retries == 0
            assert not result.errors
        assert client._closed
        assert attempts == 2

    asyncio.run(run())


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
def test_factory_selects_generic_system_one_client_without_llm_output(monkeypatch, model, provider):
    from thesis_model_clients import create_decision_client

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    settings = (
        {"transport": "self_hosted", "base_url": "http://127.0.0.1:8000"}
        if provider == "local"
        else {}
    )
    client = create_decision_client("system_one", model, provider, max_concurrency=1, **settings)
    expected = TypeSafeHTTPDecisionClient if provider == "local" else VercelSystemOneDecisionClient
    assert type(client) is expected
    assert client.requested_model == model
    assert client.requested_provider == provider
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    "backend,provider,mode,reasoning,error",
    [
        ("jev", "typesafe-ai", None, None, "backend must be"),
        ("system_one", "", None, None, "provider"),
        ("llm", "", "discrete", None, "provider"),
        ("llm", "openai", None, None, "llm_output_mode"),
        ("system_one", "typesafe-ai", None, "low", "only valid with the llm"),
    ],
)
def test_factory_rejects_missing_or_retired_settings(backend, provider, mode, reasoning, error):
    from thesis_model_clients import create_decision_client

    with pytest.raises(ValueError, match=error):
        create_decision_client(
            backend,
            "typesafe-ai/jev",
            provider,
            max_concurrency=1,
            llm_output_mode=mode,
            reasoning_effort=reasoning,
        )


@pytest.mark.parametrize("model,provider", HTTP_SYSTEM_ONE_ROUTES)
@pytest.mark.parametrize("cost", [None, 0.0])
def test_system_one_nonretryable_failure_preserves_reported_routing_cost_and_usage(
    monkeypatch, model, provider, cost
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    requests = []
    payload = {
        "error": {"message": "Unsupported inputs", "isRetryable": False},
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "providerMetadata": {"gateway": {"routing": {"resolvedProvider": provider}}},
    }
    if cost is not None:
        payload["providerMetadata"]["gateway"]["cost"] = cost

    def reply(request):
        requests.append(request)
        return httpx2.Response(422, json=payload)

    async def run():
        async with http_system_one_client(
            model, provider, _transport=httpx2.MockTransport(reply)
        ) as client:
            return await client.evaluate("A message", QUESTIONS)

    result = asyncio.run(run())
    assert len(requests) == 1
    assert not result.answers
    assert result.errors.keys() == QUESTIONS.keys()
    assert result.requested_model == model
    assert result.resolved_models == ()
    assert result.requested_provider == provider
    assert result.resolved_providers == (provider,)
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == cost
    assert result.usage.retries == 0
    assert result.calls[0].raw["body"] == payload
