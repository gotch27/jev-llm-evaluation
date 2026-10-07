"""System One implementation of the shared structured-decision interface."""

from __future__ import annotations

import asyncio
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, JSONContent, Questions, RetryPolicy, TypeSafeError

from thesis_research.clients.contracts import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TIMEOUT_SECONDS,
    CallRecord,
    DecisionClient,
    DecisionError,
    DecisionResult,
    UsageTotals,
    aggregate_usage,
    unique_non_null,
)
from thesis_research.clients.gateway_metadata import reported_cost, resolved_provider

VERCEL_TYPESAFE_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"


@dataclass
class _RequestAttempts:
    count: int = 0


class VercelSystemOneDecisionClient(DecisionClient):
    """Evaluate structured questions through Vercel's System One endpoint.

    All questions for one state are sent in a single System One request. A semaphore
    bounds concurrent requests when the same client is shared across dataset examples.

    Args:
        model: Canonical Vercel System One model identifier.
        provider: Upstream provider slug that must serve every request.
        max_concurrency: Maximum active System One requests for this client.
        timeout_seconds: HTTP timeout applied to each request.
        retry: Optional TypeSafe transient retry policy.
        _client: Optional SDK client injected by offline tests.
        _transport: Optional HTTP transport injected by offline tests.

    Raises:
        ValueError: If settings are invalid or ``AI_GATEWAY_API_KEY`` is absent.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry: RetryPolicy | None = None,
        _client: AsyncTypeSafeClient | Any | None = None,
        _transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        if not provider.strip():
            raise ValueError("provider must be nonempty")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if _client is not None and _transport is not None:
            raise ValueError("Supply only one of _client or _transport")
        retry = retry or RetryPolicy()
        self._attempts: ContextVar[_RequestAttempts | None] = ContextVar(
            "system_one_request_attempts", default=None
        )
        self._tracks_attempts = _client is None
        if _client is None:
            api_key = os.environ.get("AI_GATEWAY_API_KEY")
            if not api_key:
                raise ValueError("AI_GATEWAY_API_KEY is required")
            _client = AsyncTypeSafeClient(
                api_key=api_key,
                model=model,
                retry=retry,
                timeout=timeout_seconds,
                base_url=VERCEL_TYPESAFE_BASE_URL,
                http_client=httpx2.AsyncClient(
                    timeout=timeout_seconds,
                    transport=_transport,
                    event_hooks={"request": [self._record_attempt]},
                ),
            )
        self.requested_model = model
        # Pin and record the requested route; resolved routing comes from the response.
        self.requested_provider = provider
        # This policy controls retries. Actual retry counts are measured per evaluation.
        self._retry = retry
        self._timeout_seconds = timeout_seconds
        self._client = _client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._closed = False

    async def evaluate(self, state: JSONContent, questions: Questions) -> DecisionResult:
        """Evaluate all supplied questions in one bounded System One request.

        Provider failures become per-question errors in the returned result so
        prediction runs can preserve the failed example and continue.

        Args:
            state: JSON-compatible information the questions refer to.
            questions: TypeSafe questions keyed by stable question ID.

        Returns:
            Typed answers, errors, usage, latency, and the raw call record.

        Raises:
            RuntimeError: If the client has already been closed.
            ValueError: If no questions are supplied.
        """
        if self._closed:
            raise RuntimeError("The System One client is closed")
        if not questions:
            raise ValueError("At least one question is required")
        started = time.perf_counter()
        async with self._semaphore:
            call_started = time.perf_counter()
            attempts = _RequestAttempts()
            token = self._attempts.set(attempts)
            try:
                response = await self._client.system_one(
                    state,
                    questions,
                    model=self.requested_model,
                    retry=self._retry,
                    timeout=self._timeout_seconds,
                    extra_body={
                        "providerOptions": {
                            "gateway": {"only": [self.requested_provider]},
                        }
                    },
                )
            except TypeSafeError as error:
                latency = time.perf_counter() - call_started
                decision_error = _decision_error(error)
                body = getattr(error, "body", None)
                raw_error = body if isinstance(body, dict) else {}
                reported_model = raw_error.get("model")
                call = CallRecord(
                    question_ids=tuple(questions),
                    requested_model=self.requested_model,
                    resolved_model=reported_model if isinstance(reported_model, str) else None,
                    requested_provider=self.requested_provider,
                    resolved_provider=resolved_provider(raw_error),
                    latency_seconds=latency,
                    usage=UsageTotals(
                        _reported_tokens(raw_error, "input_tokens"),
                        _reported_tokens(raw_error, "output_tokens"),
                        reported_cost(raw_error),
                        max(0, attempts.count - 1) if self._tracks_attempts else None,
                        0,
                    ),
                    error=decision_error,
                    raw=decision_error.diagnostics,
                )
                errors = dict.fromkeys(questions, decision_error)
                return _result(self, {}, errors, [call], time.perf_counter() - started)
            finally:
                self._attempts.reset(token)
            call_latency = time.perf_counter() - call_started

        # SDK model_dump ignores gateway extensions, including billed cost and routing.
        raw = response.raw_http_response.json()
        usage = UsageTotals(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=reported_cost(raw),
            retries=(
                max(0, attempts.count - 1)
                if self._tracks_attempts
                else _response_retries(response.raw_http_response)
            ),
            malformed_response_retries=0,
        )
        call = CallRecord(
            question_ids=tuple(questions),
            requested_model=self.requested_model,
            resolved_model=response.model,
            requested_provider=self.requested_provider,
            resolved_provider=resolved_provider(raw),
            latency_seconds=call_latency,
            usage=usage,
            error=None,
            raw=raw,
        )
        return _result(
            self,
            dict(response.answers),
            _refusal_errors(raw, questions),
            [call],
            time.perf_counter() - started,
        )

    async def _record_attempt(self, request: httpx2.Request) -> None:
        """Count each HTTP attempt in its evaluation's isolated async context."""
        attempts = self._attempts.get()
        if attempts is not None:
            attempts.count += 1

    async def aclose(self) -> None:
        """Close the underlying TypeSafe HTTP client once."""
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("The System One client is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _refusal_errors(raw: dict[str, Any], questions: Questions) -> dict[str, DecisionError]:
    """Recover per-question refusals omitted by the SDK's typed answer decoder."""
    raw_answers = raw.get("answers")
    if not isinstance(raw_answers, dict):
        return {}
    errors = {}
    for question_id in questions:
        answer = raw_answers.get(question_id)
        if isinstance(answer, dict) and answer.get("type") == "refusal":
            errors[question_id] = DecisionError(
                "ModelRefusal",
                f"Model declined to answer question {question_id!r}",
                {"answer": answer},
            )
    return errors


def _decision_error(error: TypeSafeError) -> DecisionError:
    debug = getattr(error, "debug", None)
    diagnostics = dict(debug) if isinstance(debug, dict) else {}
    for attribute in ("status", "body", "endpoint"):
        value = getattr(error, attribute, None)
        if value is not None:
            diagnostics[attribute] = value
    headers = getattr(error, "headers", {})
    request_id = headers.get("x-typesafe-request-id")
    if request_id is not None:
        diagnostics["request_id"] = request_id
    return DecisionError(type(error).__name__, str(error), diagnostics)


def _response_retries(response: httpx2.Response) -> int | None:
    """Read the final SDK attempt when an externally supplied client was used."""
    try:
        count = int(response.request.headers.get("X-TypeSafe-Retry-Count", "0"))
    except (RuntimeError, ValueError):
        return None
    return count if count >= 0 else None


def _reported_tokens(raw: dict[str, Any], name: str) -> int | None:
    """Retain valid token counts if a failed HTTP response reports usage."""
    usage = raw.get("usage")
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _result(
    client: VercelSystemOneDecisionClient,
    answers: dict,
    errors: dict,
    calls: list[CallRecord],
    latency: float,
) -> DecisionResult:
    return DecisionResult(
        answers=answers,
        errors=errors,
        requested_model=client.requested_model,
        requested_provider=client.requested_provider,
        resolved_models=unique_non_null([call.resolved_model for call in calls]),
        resolved_providers=unique_non_null([call.resolved_provider for call in calls]),
        usage=aggregate_usage(calls),
        latency_seconds=latency,
        calls=tuple(calls),
    )
