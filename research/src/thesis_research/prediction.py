"""Run a structured intent task over dataset examples and stream durable records."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

from typesafe_sdk import ChoiceAnswer

from thesis_research.clients import DecisionClient, DecisionError, DecisionResult, UsageTotals
from thesis_research.datasets import Banking77Example
from thesis_research.tasks import IntentClassificationTask

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "retries",
    "malformed_response_retries",
)


@dataclass
class _PredictionSummary:
    """Accumulate progress, routing, and usage for one prediction run."""

    expected: int
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    resolved_models: list[str] = field(default_factory=list)
    resolved_providers: list[str] = field(default_factory=list)
    _usage_values: dict[str, int | float] = field(
        default_factory=lambda: dict.fromkeys(_USAGE_FIELDS, 0)
    )
    _usage_complete: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(_USAGE_FIELDS, True)
    )

    def add(self, result: DecisionResult, successful: bool) -> None:
        """Add one decision result without overstating incomplete usage."""
        self.attempted += 1
        if successful:
            self.successful += 1
        else:
            self.failed += 1
        _extend_unique(self.resolved_models, result.resolved_models)
        _extend_unique(self.resolved_providers, result.resolved_providers)
        for name in _USAGE_FIELDS:
            value = getattr(result.usage, name)
            if value is None:
                self._usage_complete[name] = False
            else:
                self._usage_values[name] += value

    def as_dict(self) -> dict[str, Any]:
        """Return the current progress as a JSON-safe metadata object."""
        return {
            "expected": self.expected,
            "attempted": self.attempted,
            "successful": self.successful,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
            "resolved_models": self.resolved_models,
            "resolved_providers": self.resolved_providers,
            "usage": {
                name: self._usage_values[name] if self._usage_complete[name] else None
                for name in _USAGE_FIELDS
            },
        }


async def predict_examples(
    client: DecisionClient,
    examples: Sequence[Banking77Example],
    task: IntentClassificationTask,
    labels: Sequence[str],
    predictions_file: TextIO,
    decisions_file: TextIO,
    *,
    max_concurrency: int,
    record_choice_details: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_prediction: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate examples in bounded batches and stream ordered JSONL records.

    Each completed batch is flushed before the next begins. This preserves
    partial progress if a later batch is interrupted. Provider-level failures
    become explicit error predictions rather than stopping the dataset run.

    Args:
        client: Open asynchronous client used for structured decisions.
        examples: Dataset examples in the order they should be written.
        task: Validated state builder and Choice question.
        labels: Complete set of allowed prediction labels.
        predictions_file: Writable stream receiving evaluator-ready records.
        decisions_file: Writable stream receiving full diagnostic records.
        max_concurrency: Maximum examples evaluated in one active batch.
        record_choice_details: Preserve Choice confidence and probabilities.
            Set to false for label-only LLM output so adapter-derived one-hot
            values are not presented as measured model confidence.
        on_progress: Optional callback invoked after each flushed batch.
        on_prediction: Optional callback invoked for each completed prediction.

    Returns:
        JSON-safe counts, elapsed time, routing, and aggregate usage.

    Raises:
        ValueError: If ``max_concurrency`` is less than one.
        asyncio.CancelledError: If the caller cancels prediction generation.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    allowed_labels = set(labels)
    summary = _PredictionSummary(expected=len(examples))
    started = time.perf_counter()

    for offset in range(0, len(examples), max_concurrency):
        batch = examples[offset : offset + max_concurrency]
        states = [task.build_state(example.text) for example in batch]
        tasks = [asyncio.create_task(client.evaluate(state, task.questions())) for state in states]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for pending in tasks:
                pending.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for example, state, result in zip(batch, states, results, strict=True):
            prediction, successful = _prediction_record(
                example.id,
                result,
                task.question_id,
                allowed_labels,
                record_choice_details=record_choice_details,
            )
            _write_json_line(predictions_file, prediction)
            _write_json_line(
                decisions_file,
                {
                    "id": example.id,
                    "state": state,
                    "prediction": prediction,
                    "decision": decision_result_to_dict(
                        result,
                        record_choice_details=record_choice_details,
                    ),
                },
            )
            summary.add(result, successful)
            if on_prediction is not None:
                on_prediction(prediction)
        predictions_file.flush()
        decisions_file.flush()
        summary.elapsed_seconds = time.perf_counter() - started
        if on_progress is not None:
            on_progress(summary.as_dict())

    summary.elapsed_seconds = time.perf_counter() - started
    return summary.as_dict()


def decision_result_to_dict(
    result: DecisionResult,
    *,
    record_choice_details: bool = True,
) -> dict[str, Any]:
    """Convert a provider-neutral decision result into a JSON-safe record.

    Typed TypeSafe answers are serialized while raw provider diagnostics are
    retained for later inspection and reproducibility analysis.

    Args:
        result: Decision result returned by the Jev or Vercel LLM client.
        record_choice_details: Preserve Choice confidence and probabilities.

    Returns:
        Dictionary suitable for one ``decisions.jsonl`` record.
    """
    return {
        "answers": {
            question_id: _answer_to_dict(answer, record_choice_details)
            for question_id, answer in result.answers.items()
        },
        "choice_output_mode": "probabilities" if record_choice_details else "label",
        "errors": {
            question_id: _error_to_dict(error) for question_id, error in result.errors.items()
        },
        "requested_model": result.requested_model,
        "requested_provider": result.requested_provider,
        "resolved_models": list(result.resolved_models),
        "resolved_providers": list(result.resolved_providers),
        "usage": asdict(result.usage),
        "latency_seconds": result.latency_seconds,
        "calls": [
            {
                "question_ids": list(call.question_ids),
                "requested_model": call.requested_model,
                "resolved_model": call.resolved_model,
                "requested_provider": call.requested_provider,
                "resolved_provider": call.resolved_provider,
                "latency_seconds": call.latency_seconds,
                "usage": asdict(call.usage),
                "error": _error_to_dict(call.error) if call.error is not None else None,
                "raw": call.raw,
            }
            for call in result.calls
        ],
    }


def _prediction_record(
    example_id: str,
    result: DecisionResult,
    question_id: str,
    allowed_labels: set[str],
    *,
    record_choice_details: bool,
) -> tuple[dict[str, Any], bool]:
    answer = result.answers.get(question_id)
    error = result.errors.get(question_id)
    if answer is not None and error is not None:
        return {"id": example_id, "error": "Decision returned both an answer and an error"}, False
    if error is not None:
        return {"id": example_id, "error": _prediction_error(error)}, False
    if not isinstance(answer, ChoiceAnswer):
        return {"id": example_id, "error": "Decision did not return a Choice answer"}, False
    if answer.choice not in allowed_labels:
        return {
            "id": example_id,
            "error": f"Decision returned an unknown label: {answer.choice}",
        }, False
    prediction: dict[str, Any] = {"id": example_id, "label": answer.choice}
    if record_choice_details:
        prediction["confidence"] = answer.confidence
        prediction["probabilities"] = dict(answer.probabilities)
    return prediction, True


def _answer_to_dict(answer: object, record_choice_details: bool) -> object:
    if isinstance(answer, ChoiceAnswer) and not record_choice_details:
        return {"type": "choice", "choice": answer.choice}
    model_dump = getattr(answer, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else answer


def _prediction_error(error: DecisionError) -> str:
    message = error.message.strip() or "No error message was supplied"
    return f"{error.error_type}: {message}"


def _error_to_dict(error: DecisionError) -> dict[str, Any]:
    return {
        "error_type": error.error_type,
        "message": error.message,
        "diagnostics": error.diagnostics,
    }


def _write_json_line(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, UsageTotals):
        return asdict(value)
    return repr(value)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)
