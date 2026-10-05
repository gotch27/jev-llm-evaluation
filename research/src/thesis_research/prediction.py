"""Run one structured task over labeled examples and stream durable records."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from thesis_research.clients import DecisionClient, DecisionError, DecisionResult, UsageTotals
from thesis_research.datasets import (
    ChoiceReferenceSchema,
    DatasetExample,
    NoulReferenceSchema,
    ReferenceSchema,
    ScoreReferenceSchema,
)
from thesis_research.evaluation import prediction_status
from thesis_research.tasks import StructuredTask

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
        self.attempted += 1
        self.successful += int(successful)
        self.failed += int(not successful)
        _extend_unique(self.resolved_models, result.resolved_models)
        _extend_unique(self.resolved_providers, result.resolved_providers)
        for name in _USAGE_FIELDS:
            value = getattr(result.usage, name)
            if value is None:
                self._usage_complete[name] = False
            else:
                self._usage_values[name] += value

    def as_dict(self) -> dict[str, Any]:
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
    examples: Sequence[DatasetExample],
    task: StructuredTask,
    schema: ReferenceSchema,
    predictions_file: TextIO,
    decisions_file: TextIO,
    *,
    max_concurrency: int,
    record_answer_details: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_prediction: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate states, evaluate bounded batches, and stream ordered records."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    task.validate_state_fields(examples)
    summary = _PredictionSummary(expected=len(examples))
    started = time.perf_counter()

    for offset in range(0, len(examples), max_concurrency):
        batch = examples[offset : offset + max_concurrency]
        states = [task.build_state(example.state) for example in batch]
        pending = [
            asyncio.create_task(client.evaluate(state, task.questions())) for state in states
        ]
        try:
            results = await asyncio.gather(*pending)
        except BaseException:
            for request in pending:
                request.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        for example, state, result in zip(batch, states, results, strict=True):
            prediction, successful = _prediction_record(
                example.id,
                result,
                task.question_id,
                schema,
                record_answer_details=record_answer_details,
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
                        record_answer_details=record_answer_details,
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
    record_answer_details: bool = True,
) -> dict[str, Any]:
    """Convert a provider-neutral decision result into a JSON-safe record."""
    mode = "probabilities" if record_answer_details else "discrete"
    return {
        "answers": {
            question_id: _answer_to_dict(answer, record_answer_details)
            for question_id, answer in result.answers.items()
        },
        "answer_output_mode": mode,
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
    schema: ReferenceSchema,
    *,
    record_answer_details: bool,
) -> tuple[dict[str, Any], bool]:
    answer = result.answers.get(question_id)
    error = result.errors.get(question_id)
    if answer is not None and error is not None:
        return {"id": example_id, "error": "Decision returned both an answer and an error"}, False
    if error is not None:
        return {"id": example_id, "error": _prediction_error(error)}, False
    if isinstance(schema, ChoiceReferenceSchema):
        prediction, successful = _choice_prediction(
            example_id, answer, schema, record_answer_details
        )
    elif isinstance(schema, NoulReferenceSchema):
        prediction, successful = _noul_prediction(example_id, answer, record_answer_details)
    else:
        prediction, successful = _score_prediction(
            example_id, answer, schema, record_answer_details
        )
    if successful and prediction_status(prediction, schema)[0] != "valid":
        return {
            "id": example_id,
            "error": f"Decision returned invalid {schema.question_type} answer details",
        }, False
    return prediction, successful


def _choice_prediction(
    example_id: str,
    answer: object,
    schema: ChoiceReferenceSchema,
    details: bool,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(answer, ChoiceAnswer):
        return {"id": example_id, "error": "Decision did not return a Choice answer"}, False
    if answer.choice not in schema.labels:
        return {
            "id": example_id,
            "error": f"Decision returned an unknown label: {answer.choice}",
        }, False
    prediction: dict[str, Any] = {"id": example_id, "label": answer.choice}
    if details:
        prediction["confidence"] = answer.confidence
        prediction["probabilities"] = dict(answer.probabilities)
    return prediction, True


def _noul_prediction(example_id: str, answer: object, details: bool) -> tuple[dict[str, Any], bool]:
    if not isinstance(answer, NoulAnswer):
        return {"id": example_id, "error": "Decision did not return a Noul answer"}, False
    probability = answer.noul
    if (
        not isinstance(probability, int | float)
        or isinstance(probability, bool)
        or not math.isfinite(probability)
    ):
        return {"id": example_id, "error": "Decision returned an invalid Noul probability"}, False
    probability = float(probability)
    if not 0 <= probability <= 1:
        return {"id": example_id, "error": "Decision returned an invalid Noul probability"}, False
    prediction: dict[str, Any] = {"id": example_id, "value": probability >= 0.5}
    if details:
        prediction["probability"] = probability
    return prediction, True


def _score_prediction(
    example_id: str,
    answer: object,
    schema: ScoreReferenceSchema,
    details: bool,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(answer, ScoreAnswer):
        return {"id": example_id, "error": "Decision did not return a Score answer"}, False
    score = answer.score
    if (
        not isinstance(score, int | float)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= schema.level_count - 1
    ):
        return {"id": example_id, "error": "Decision returned an invalid Score value"}, False
    probabilities = dict(answer.probabilities)
    expected_levels = set(range(schema.level_count))
    if (
        set(probabilities) != expected_levels
        or any(type(level) is not int for level in probabilities)
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or value > 1
            for value in probabilities.values()
        )
        or not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    ):
        return {"id": example_id, "error": "Decision returned invalid Score probabilities"}, False
    highest = max(probabilities.values())
    level = min(level for level, probability in probabilities.items() if probability == highest)
    prediction: dict[str, Any] = {
        "id": example_id,
        "level": level,
        "score": float(score),
    }
    if details:
        prediction["confidence"] = answer.confidence
        prediction["probabilities"] = {
            str(level): probability for level, probability in probabilities.items()
        }
    return prediction, True


def _answer_to_dict(answer: object, details: bool) -> object:
    if not details:
        if isinstance(answer, ChoiceAnswer):
            return {"type": "choice", "choice": answer.choice}
        if isinstance(answer, NoulAnswer):
            return {"type": "noul", "value": answer.noul >= 0.5}
        if isinstance(answer, ScoreAnswer):
            return {"type": "score", "score": answer.score}
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
