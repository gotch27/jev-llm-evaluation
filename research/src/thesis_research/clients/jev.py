"""Jev implementation of the shared structured-decision interface."""

from __future__ import annotations

import asyncio
import os
import time
from types import TracebackType
from typing import Any, Self

from typesafe_sdk import AsyncTypeSafeClient, JSONContent, Questions, RetryPolicy, TypeSafeError

from thesis_research.clients.contracts import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TIMEOUT_SECONDS,
    CallRecord,
    DecisionError,
    DecisionResult,
    UsageTotals,
    aggregate_usage,
    unique_non_null,
)

VERCEL_TYPESAFE_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
VERCEL_JEV_PROVIDER = "typesafe-ai"


class JevDecisionClient:
    """Evaluate structured questions through Jev's System One endpoint.

    All questions for one state are sent in a single Jev request because Jev
    evaluates them independently. A semaphore bounds concurrent requests when
    the same client is shared across dataset examples.

    Args:
        model: Vercel's Jev model identifier, currently ``typesafe-ai/jev``.
        max_concurrency: Maximum active Jev requests for this client.
        timeout_seconds: HTTP timeout applied to each request.
        retry: Optional TypeSafe transient retry policy.
        _client: Optional SDK client injected by offline tests.

    Raises:
        ValueError: If settings are invalid or ``AI_GATEWAY_API_KEY`` is absent.
    """

    def __init__(
        self,
        model: str,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry: RetryPolicy | None = None,
        _client: AsyncTypeSafeClient | Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        retry = retry or RetryPolicy()
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
            )
        self.requested_model = model
        self.requested_provider = VERCEL_JEV_PROVIDER
        self._retry = retry
        self._timeout_seconds = timeout_seconds
        self._client = _client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._closed = False

    async def evaluate(self, state: JSONContent, questions: Questions) -> DecisionResult:
        """Evaluate all supplied questions in one bounded Jev request.

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
            raise RuntimeError("The Jev client is closed")
        if not questions:
            raise ValueError("At least one question is required")
        started = time.perf_counter()
        async with self._semaphore:
            call_started = time.perf_counter()
            try:
                response = await self._client.system_one(
                    state,
                    questions,
                    model=self.requested_model,
                    retry=self._retry,
                    timeout=self._timeout_seconds,
                    extra_body={
                        "providerOptions": {
                            "gateway": {"only": [VERCEL_JEV_PROVIDER]},
                        }
                    },
                )
            except TypeSafeError as error:
                latency = time.perf_counter() - call_started
                decision_error = _decision_error(error)
                call = CallRecord(
                    question_ids=tuple(questions),
                    requested_model=self.requested_model,
                    resolved_model=None,
                    requested_provider=self.requested_provider,
                    resolved_provider=None,
                    latency_seconds=latency,
                    usage=_unknown_usage(),
                    error=decision_error,
                    raw=decision_error.diagnostics,
                )
                errors = dict.fromkeys(questions, decision_error)
                return _result(self, {}, errors, [call], time.perf_counter() - started)

        raw = response.model_dump(mode="json")
        usage = UsageTotals(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=None,
            retries=None,
            malformed_response_retries=0,
        )
        call = CallRecord(
            question_ids=tuple(questions),
            requested_model=self.requested_model,
            resolved_model=response.model,
            requested_provider=self.requested_provider,
            resolved_provider=self.requested_provider,
            latency_seconds=time.perf_counter() - call_started,
            usage=usage,
            error=None,
            raw=raw,
        )
        return _result(
            self,
            dict(response.answers),
            {},
            [call],
            time.perf_counter() - started,
        )

    async def aclose(self) -> None:
        """Close the underlying TypeSafe HTTP client once."""
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("The Jev client is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _decision_error(error: TypeSafeError) -> DecisionError:
    debug = getattr(error, "debug", None)
    diagnostics = debug if isinstance(debug, dict) else {}
    return DecisionError(type(error).__name__, str(error), diagnostics)


def _unknown_usage() -> UsageTotals:
    return UsageTotals(None, None, None, None, None)


def _result(
    client: JevDecisionClient,
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
