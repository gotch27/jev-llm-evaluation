"""Vercel AI Gateway-backed LLM decisions using TypeSafe's System One adapter."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self

from system_one_adapter import AsyncSystemOneAdapterClient
from typesafe_sdk import Answer, JSONContent, Questions, RetryPolicy, TypeSafeError

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
from thesis_research.clients.vercel_provider import (
    REASONING_EFFORTS,
    VercelGatewayProvider,
)


@dataclass(frozen=True)
class _QuestionEvaluation:
    call: CallRecord
    answer: Answer | None


class VercelLLMDecisionClient(DecisionClient):
    """Evaluate questions through isolated, concurrent Vercel AI Gateway requests.

    TypeSafe's adapter creates and validates the JSON schema for each question.
    One provider request is made per question so every answer remains isolated;
    successful answers are retained if another question fails.

    Args:
        model: Canonical Vercel model identifier, excluding routing aliases.
        provider: Upstream provider slug that must serve every request.
        output_mode: Return discrete answers or question-specific probabilities.
        reasoning_effort: Explicit provider reasoning effort, or ``None`` to
            leave the setting unspecified.
        max_concurrency: Maximum active provider requests for this client.
        timeout_seconds: HTTP timeout applied to each request.
        retry: Optional TypeSafe transient retry policy.
        _provider: Optional adapter provider injected by offline tests.

    Raises:
        ValueError: If settings are invalid or ``AI_GATEWAY_API_KEY`` is absent.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        output_mode: Literal["discrete", "probabilities"] = "probabilities",
        reasoning_effort: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry: RetryPolicy | None = None,
        _provider: Any | None = None,
    ) -> None:
        _validate_exact_model(model)
        if not provider.strip():
            raise ValueError("provider must be nonempty")
        if output_mode not in ("discrete", "probabilities"):
            raise ValueError("output_mode must be discrete or probabilities")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if _provider is None:
            api_key = os.environ.get("AI_GATEWAY_API_KEY")
            if not api_key:
                raise ValueError("AI_GATEWAY_API_KEY is required")
            _provider = VercelGatewayProvider(
                model,
                provider,
                api_key,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
        self.requested_model = model
        self.requested_provider = provider
        self.output_mode = output_mode
        self.reasoning_effort = reasoning_effort
        self._provider = _provider
        self._retry = retry or RetryPolicy()
        self._adapter = AsyncSystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode=output_mode,
            normalize_probabilities=output_mode == "probabilities",
            n_retry_malformed_structure=0,
            retry=self._retry,
            model=self._provider,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._closed = False

    async def evaluate(self, state: JSONContent, questions: Questions) -> DecisionResult:
        """Evaluate each question concurrently and combine their outcomes.

        Args:
            state: JSON-compatible information each question refers to.
            questions: TypeSafe questions keyed by stable question ID.

        Returns:
            Successful typed answers, per-question errors, aggregate usage,
            end-to-end latency, and individual provider-call records.

        Raises:
            RuntimeError: If the client has already been closed.
            ValueError: If no questions are supplied.
            asyncio.CancelledError: If the caller cancels the evaluation.
        """
        if self._closed:
            raise RuntimeError("The Vercel AI Gateway client is closed")
        if not questions:
            raise ValueError("At least one question is required")
        started = time.perf_counter()
        tasks = [
            asyncio.create_task(self._evaluate_one(state, question_id, question))
            for question_id, question in questions.items()
        ]
        evaluations = list(await asyncio.gather(*tasks))
        calls = [evaluation.call for evaluation in evaluations]
        answers = {}
        errors = {}
        for evaluation in evaluations:
            call = evaluation.call
            question_id = call.question_ids[0]
            if call.error is not None:
                errors[question_id] = call.error
            else:
                assert evaluation.answer is not None
                answers[question_id] = evaluation.answer
        return DecisionResult(
            answers=answers,
            errors=errors,
            requested_model=self.requested_model,
            requested_provider=self.requested_provider,
            resolved_models=unique_non_null([call.resolved_model for call in calls]),
            resolved_providers=unique_non_null([call.resolved_provider for call in calls]),
            usage=aggregate_usage(calls),
            latency_seconds=time.perf_counter() - started,
            calls=tuple(calls),
        )

    async def _evaluate_one(
        self,
        state: JSONContent,
        question_id: str,
        question: Any,
    ) -> _QuestionEvaluation:
        try:
            async with self._semaphore:
                started = time.perf_counter()
                response = await self._adapter.system_one(
                    state,
                    {question_id: question},
                    model=self._provider,
                    retry=self._retry,
                )
                call_latency = time.perf_counter() - started
        except (TypeSafeError, ValueError) as error:
            decision_error = _decision_error(error)
            return _QuestionEvaluation(
                call=CallRecord(
                    question_ids=(question_id,),
                    requested_model=self.requested_model,
                    resolved_model=None,
                    requested_provider=self.requested_provider,
                    resolved_provider=None,
                    latency_seconds=time.perf_counter() - started,
                    usage=UsageTotals(None, None, None, None, None),
                    error=decision_error,
                    raw=decision_error.diagnostics,
                ),
                answer=None,
            )
        raw = response.model_dump(mode="json")
        provider_result = _provider_result(raw)
        return _QuestionEvaluation(
            call=CallRecord(
                question_ids=(question_id,),
                requested_model=self.requested_model,
                resolved_model=provider_result.get("resolved_model", response.model),
                requested_provider=self.requested_provider,
                resolved_provider=provider_result.get("resolved_provider"),
                latency_seconds=call_latency,
                usage=UsageTotals(
                    input_tokens=response.usage.input_tokens_total,
                    output_tokens=response.usage.output_tokens_total,
                    cost_usd=provider_result.get("cost_usd"),
                    retries=response.usage.n_retries,
                    malformed_response_retries=response.usage.n_retries_malformed_structure,
                ),
                error=None,
                raw=raw,
            ),
            answer=response.answers[question_id],
        )

    async def aclose(self) -> None:
        """Close the adapter and its underlying Vercel AI Gateway HTTP resources."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._adapter.aclose()
        finally:
            close = getattr(self._provider, "aclose", None)
            if close is not None:
                await close()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("The Vercel AI Gateway client is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _validate_exact_model(model: str) -> None:
    model = model.strip()
    if not model:
        raise ValueError("model must be nonempty")
    if (
        "/" not in model
        or model.startswith("~")
        or model.startswith("vmc/")
        or model in {"vercel/auto", "vercel/free"}
    ):
        raise ValueError("Use a canonical Vercel model ID in creator/model form")


def _decision_error(error: Exception) -> DecisionError:
    debug = getattr(error, "debug", None)
    diagnostics = debug if isinstance(debug, dict) else {}
    return DecisionError(type(error).__name__, str(error), diagnostics)


def _provider_result(response: dict[str, Any]) -> dict[str, Any]:
    attempts = response.get("debug", {}).get("llm_attempts", [])
    if not attempts:
        return {}
    result = attempts[-1].get("llm_response")
    return result if isinstance(result, dict) else {}
