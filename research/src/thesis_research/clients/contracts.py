"""Provider-neutral types for structured model decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self

from typesafe_sdk import Answer, JSONContent, Questions

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_CONCURRENCY = 5


@dataclass(frozen=True)
class DecisionError:
    """Describe one failed question evaluation without raising it immediately.

    Attributes:
        error_type: Exception or provider error category.
        message: Human-readable failure explanation.
        diagnostics: Structured provider or retry details when available.
    """

    error_type: str
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsageTotals:
    """Store aggregate usage across all calls for one decision.

    A field is ``None`` when at least one contributing call did not report it,
    preventing an incomplete total from being mistaken for a complete value.

    Attributes:
        input_tokens: Total provider input tokens, when completely reported.
        output_tokens: Total provider output tokens, when completely reported.
        cost_usd: Total reported cost in US dollars.
        retries: Total transient provider retries.
        malformed_response_retries: Total retries caused by malformed output.
    """

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    retries: int | None
    malformed_response_retries: int | None


@dataclass(frozen=True)
class CallRecord:
    """Record the routing, usage, timing, and diagnostics of one API call.

    Attributes:
        question_ids: Questions included in this provider request.
        requested_model: Model identifier supplied by the experiment.
        resolved_model: Model reported by the provider, when known.
        requested_provider: Provider requested by the experiment.
        resolved_provider: Provider that served the request, when known.
        latency_seconds: Duration after acquiring a concurrency slot, including
            SDK retries and their backoff, excluding local semaphore queue time.
        usage: Token, cost, and retry information for this call.
        error: Structured failure information, or ``None`` after success.
        raw: JSON-safe request, response, and diagnostic information.
    """

    question_ids: tuple[str, ...]
    requested_model: str
    resolved_model: str | None
    requested_provider: str
    resolved_provider: str | None
    latency_seconds: float
    usage: UsageTotals
    error: DecisionError | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    """Combine typed answers and diagnostics for one evaluated state.

    Attributes:
        answers: Successful typed answers keyed by question ID.
        errors: Per-question failures keyed by question ID.
        requested_model: Model requested by the caller.
        requested_provider: Provider requested by the caller.
        resolved_models: Distinct models reported across underlying calls.
        resolved_providers: Distinct providers reported across underlying calls.
        usage: Usage aggregated across all underlying calls.
        latency_seconds: End-to-end duration, including semaphore queue time.
        calls: Individual provider-call records in question order.
    """

    answers: dict[str, Answer]
    errors: dict[str, DecisionError]
    requested_model: str
    requested_provider: str
    resolved_models: tuple[str, ...]
    resolved_providers: tuple[str, ...]
    usage: UsageTotals
    latency_seconds: float
    calls: tuple[CallRecord, ...]


class DecisionClient(Protocol):
    """Define the asynchronous interface shared by Jev and LLM clients.

    Implementations are async context managers so callers can reuse and close
    their underlying HTTP connections reliably.
    """

    async def evaluate(self, state: JSONContent, questions: Questions) -> DecisionResult:
        """Evaluate typed questions against a state."""
        ...

    async def aclose(self) -> None:
        """Close owned network resources."""
        ...

    async def __aenter__(self) -> Self:
        """Enter the client context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the client context."""
        ...


def unique_non_null(values: list[str | None]) -> tuple[str, ...]:
    """Return each non-null value once, preserving provider-call order."""
    return tuple(dict.fromkeys(value for value in values if value is not None))


def aggregate_usage(calls: list[CallRecord]) -> UsageTotals:
    """Aggregate call usage while marking every incomplete total as unknown.

    Args:
        calls: Provider-call records belonging to one decision.

    Returns:
        Totals whose fields are ``None`` when any contributing call omitted
        the corresponding measurement.
    """

    def sum_optional(values: list[int | None]) -> int | None:
        return (
            None
            if any(value is None for value in values)
            else sum(value for value in values if value is not None)
        )

    usages = [call.usage for call in calls]
    costs = [usage.cost_usd for usage in usages]
    return UsageTotals(
        input_tokens=sum_optional([usage.input_tokens for usage in usages]),
        output_tokens=sum_optional([usage.output_tokens for usage in usages]),
        cost_usd=None
        if any(cost is None for cost in costs)
        else sum(cost for cost in costs if cost is not None),
        retries=sum_optional([usage.retries for usage in usages]),
        malformed_response_retries=sum_optional(
            [usage.malformed_response_retries for usage in usages]
        ),
    )
