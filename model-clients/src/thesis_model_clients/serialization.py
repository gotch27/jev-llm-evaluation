"""JSON-safe decision records shared by benchmarks and standalone use cases."""

from dataclasses import asdict
from typing import Any, Literal

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from thesis_model_clients.contracts import DecisionError, DecisionResult

AnswerOutputMode = Literal["discrete", "probabilities", "native_probabilities"]


def decision_result_to_dict(
    result: DecisionResult,
    *,
    answer_output_mode: AnswerOutputMode = "probabilities",
) -> dict[str, Any]:
    """Convert a provider-neutral decision result into a JSON-safe record."""
    validate_answer_output_mode(answer_output_mode)
    record_answer_details = answer_output_mode != "discrete"
    return {
        "answers": {
            question_id: _answer_to_dict(answer, record_answer_details)
            for question_id, answer in result.answers.items()
        },
        "answer_output_mode": answer_output_mode,
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


def validate_answer_output_mode(mode: str) -> None:
    """Reject unknown modes before evaluating or serializing a decision."""
    if mode not in ("discrete", "probabilities", "native_probabilities"):
        raise ValueError(
            "answer_output_mode must be discrete, probabilities, or native_probabilities"
        )


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


def _error_to_dict(error: DecisionError) -> dict[str, Any]:
    return {
        "error_type": error.error_type,
        "message": error.message,
        "diagnostics": error.diagnostics,
    }
