"""System One implementation of the shared structured-decision interface."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

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


class TypeSafeHTTPDecisionClient(DecisionClient):
    """Evaluate native decisions at any TypeSafe-compatible HTTP API root.

    No health endpoint or server-specific metadata is required. Authentication
    is opt-in through an explicitly named environment variable; cloud credentials
    are never selected implicitly. All questions share one bounded HTTP request.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        base_url: str,
        api_key_env: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: float = 120.0,
        retry: RetryPolicy | None = None,
        _client: AsyncTypeSafeClient | Any | None = None,
        _transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        validate_http_settings(base_url, api_key_env, timeout_seconds)
        api_key = None
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is required")
        self._configure(
            model,
            provider,
            base_url=base_url,
            api_key=api_key,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            retry=retry or RetryPolicy(timeout=timeout_seconds),
            extra_body=None,
            _client=_client,
            _transport=_transport,
        )

    def _configure(
        self,
        model: str,
        provider: str,
        *,
        base_url: str,
        api_key: str | None,
        max_concurrency: int,
        timeout_seconds: float,
        retry: RetryPolicy,
        extra_body: dict[str, Any] | None,
        _client: AsyncTypeSafeClient | Any | None,
        _transport: httpx2.AsyncBaseTransport | None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        if not provider.strip():
            raise ValueError("provider must be nonempty")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if _client is not None and _transport is not None:
            raise ValueError("Supply only one of _client or _transport")
        self._authenticated = api_key is not None
        self._extra_body = extra_body
        self._attempts: ContextVar[_RequestAttempts | None] = ContextVar(
            "system_one_request_attempts", default=None
        )
        self._tracks_attempts = _client is None
        if _client is None:
            _client = AsyncTypeSafeClient(
                # The SDK requires a key. The request hook removes its placeholder
                # Authorization header entirely for unauthenticated endpoints.
                api_key=api_key if api_key is not None else "unauthenticated",
                model=model,
                retry=retry,
                timeout=timeout_seconds,
                base_url=base_url,
                http_client=httpx2.AsyncClient(
                    timeout=timeout_seconds,
                    transport=_transport,
                    event_hooks={"request": [self._record_attempt]},
                    follow_redirects=False,
                ),
            )
        self.requested_model = model
        self.requested_provider = provider
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
                    extra_body=self._extra_body,
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
        if not self._authenticated:
            request.headers.pop("authorization", None)
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


class VercelSystemOneDecisionClient(TypeSafeHTTPDecisionClient):
    """Use the shared TypeSafe transport with Vercel credentials and pinned routing."""

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
        api_key = os.environ.get("AI_GATEWAY_API_KEY")
        if _client is None and not api_key:
            raise ValueError("AI_GATEWAY_API_KEY is required")
        self._configure(
            model,
            provider,
            base_url=VERCEL_TYPESAFE_BASE_URL,
            api_key=api_key,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            retry=retry or RetryPolicy(),
            extra_body={"providerOptions": {"gateway": {"only": [provider]}}},
            _client=_client,
            _transport=_transport,
        )


def validate_http_settings(base_url: str, api_key_env: str | None, timeout_seconds: float) -> None:
    """Validate endpoint settings without reading credentials or opening connections."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a nonempty HTTP API root")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base_url must be an HTTP(S) API root without credentials, query, or fragment"
        )
    # Accessing the parsed port also validates its range and syntax.
    parsed.port
    if api_key_env is not None and (
        not isinstance(api_key_env, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env) is None
        or api_key_env == "AI_GATEWAY_API_KEY"
    ):
        raise ValueError(
            "api_key_env must name a separate environment variable, not AI_GATEWAY_API_KEY"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or (not math.isfinite(timeout_seconds) or timeout_seconds <= 0)
    ):
        raise ValueError("timeout_seconds must be positive and finite")


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
    client: TypeSafeHTTPDecisionClient,
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
