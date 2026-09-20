"""Execute and resume one model inside a coordinated benchmark."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from thesis_research.benchmark.progress import (
    BenchmarkObserver,
    ModelProgress,
    NullBenchmarkObserver,
)
from thesis_research.benchmark.types import ModelSpec
from thesis_research.clients import DecisionClient, create_decision_client
from thesis_research.datasets import Banking77Example
from thesis_research.evaluation import read_prediction_jsonl
from thesis_research.prediction import predict_examples
from thesis_research.run_storage import artifact_metadata, write_json
from thesis_research.tasks import IntentClassificationTask

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "retries",
    "malformed_response_retries",
)


async def run_model(
    spec: ModelSpec,
    directory: Path,
    examples: Sequence[Banking77Example],
    labels: Sequence[str],
    task: IntentClassificationTask,
    *,
    example_concurrency: int,
    llm_output_mode: str,
    client: DecisionClient | None = None,
    observer: BenchmarkObserver | None = None,
) -> dict[str, Any]:
    """Complete the missing predictions for one benchmark model.

    Existing label and error records are treated as completed attempts. A
    resumed run calls the provider only for IDs absent from both durable JSONL
    files. The function validates the two files before appending anything.
    """
    observer = observer if observer is not None else NullBenchmarkObserver()
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    recorded_output_mode = "native_probabilities" if spec.backend == "jev" else llm_output_mode
    metadata = _load_or_create_metadata(
        metadata_path,
        spec,
        example_concurrency,
        recorded_output_mode,
    )
    predictions, decisions = _load_records(directory)
    expected_ids = {example.id for example in examples}
    _validate_completed_records(predictions, decisions, expected_ids)
    previous_summary = _summarize(directory, len(examples), labels, metadata)
    observer.model_started(spec.id, ModelProgress.from_summary(previous_summary))
    remaining = [example for example in examples if example.id not in predictions]
    if not remaining:
        summary = previous_summary
        write_json(directory / "summary.json", summary)
        metadata["progress"] = summary
        metadata["status"] = "completed"
        metadata["artifacts"] = _artifacts(directory)
        write_json(metadata_path, metadata)
        observer.model_completed(spec.id, ModelProgress.from_summary(summary))
        return summary

    attempt = {
        "started_at": datetime.now(UTC).isoformat(),
        "completed_before": len(predictions),
        "status": "running",
    }
    metadata.setdefault("attempts", []).append(attempt)
    metadata["status"] = "running"
    metadata.pop("error", None)
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    try:
        decision_client = client or create_decision_client(
            spec.backend,
            spec.model,
            spec.provider,
            max_concurrency=example_concurrency,
            llm_output_mode=llm_output_mode,
        )

        def record_progress(progress: dict[str, Any]) -> None:
            merged = _merge_progress(previous_summary, progress)
            metadata["progress"] = merged
            write_json(metadata_path, metadata)
            observer.model_progress(spec.id, ModelProgress.from_summary(merged))

        def record_prediction(prediction: dict[str, Any]) -> None:
            error = prediction.get("error")
            if isinstance(error, str):
                observer.prediction_failed(spec.id, prediction["id"], error)

        async with decision_client:
            with (directory / "predictions.jsonl").open("a", encoding="utf-8") as output:
                with (directory / "decisions.jsonl").open("a", encoding="utf-8") as diagnostics:
                    await predict_examples(
                        decision_client,
                        remaining,
                        task,
                        labels,
                        output,
                        diagnostics,
                        max_concurrency=example_concurrency,
                        record_choice_details=recorded_output_mode != "label",
                        on_progress=record_progress,
                        on_prediction=record_prediction,
                    )
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["finished_at"] = datetime.now(UTC).isoformat()
        attempt["status"] = "completed"
        summary = _summarize(directory, len(examples), labels, metadata)
        write_json(directory / "summary.json", summary)
        metadata["progress"] = summary
        metadata["status"] = "completed"
        observer.model_completed(spec.id, ModelProgress.from_summary(summary))
        return summary
    except Exception as error:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(error).__name__}: {error}"
        metadata["status"] = "failed"
        metadata["error"] = attempt["error"]
        raise
    except BaseException as error:
        attempt["status"] = "interrupted"
        attempt["error"] = f"{type(error).__name__}: {error}"
        metadata["status"] = "interrupted"
        metadata["error"] = attempt["error"]
        raise
    finally:
        attempt.setdefault("finished_at", datetime.now(UTC).isoformat())
        attempt.setdefault("elapsed_seconds", time.perf_counter() - started)
        try:
            current_predictions, current_decisions = _load_records(directory)
            _validate_completed_records(current_predictions, current_decisions, expected_ids)
            attempt["completed_after"] = len(current_predictions)
        except Exception as record_error:
            metadata["record_error"] = f"{type(record_error).__name__}: {record_error}"
        metadata["artifacts"] = _artifacts(directory)
        write_json(metadata_path, metadata)


def _load_or_create_metadata(
    path: Path,
    spec: ModelSpec,
    example_concurrency: int,
    output_mode: str,
) -> dict[str, Any]:
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("model") != spec.as_dict():
            raise ValueError(f"Saved model metadata does not match plan for {spec.id}")
        if metadata.get("example_concurrency") != example_concurrency:
            raise ValueError(f"Saved concurrency does not match plan for {spec.id}")
        if metadata.get("output_mode") != output_mode:
            raise ValueError(f"Saved output mode does not match plan for {spec.id}")
        return metadata
    metadata = {
        "model": spec.as_dict(),
        "gateway": "vercel",
        "output_mode": output_mode,
        "example_concurrency": example_concurrency,
        "status": "pending",
        "attempts": [],
        "progress": None,
    }
    write_json(path, metadata)
    return metadata


def _load_records(directory: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    predictions_path = directory / "predictions.jsonl"
    decisions_path = directory / "decisions.jsonl"
    if predictions_path.exists() != decisions_path.exists():
        raise ValueError(f"Model record is incomplete in {directory}")
    if not predictions_path.exists():
        return {}, {}
    predictions = read_prediction_jsonl(predictions_path.read_bytes())
    decisions: dict[str, dict] = {}
    for number, line in enumerate(decisions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"Decision line {number} has no string ID")
        if record["id"] in decisions:
            raise ValueError(f"Duplicate decision ID: {record['id']}")
        decisions[record["id"]] = record
    return predictions, decisions


def _validate_completed_records(
    predictions: dict[str, dict],
    decisions: dict[str, dict],
    expected_ids: set[str],
) -> None:
    if set(predictions) != set(decisions):
        raise ValueError("Prediction and decision records contain different IDs")
    unexpected = set(predictions) - expected_ids
    if unexpected:
        raise ValueError(f"Model records contain unexpected IDs: {sorted(unexpected)[:5]}")
    for example_id, decision in decisions.items():
        if decision.get("prediction") != predictions[example_id]:
            raise ValueError(f"Prediction differs from decision record for {example_id}")


def _summarize(
    directory: Path,
    expected: int,
    labels: Sequence[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    predictions, decisions = _load_records(directory)
    allowed = set(labels)
    successful = sum(record.get("label") in allowed for record in predictions.values())
    usage_values: dict[str, list[int | float | None]] = {field: [] for field in _USAGE_FIELDS}
    resolved_models: list[str] = []
    resolved_providers: list[str] = []
    for record in decisions.values():
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError(f"Malformed decision record for {record.get('id')}")
        usage = decision.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"Missing decision usage for {record.get('id')}")
        for field in _USAGE_FIELDS:
            usage_values[field].append(usage.get(field))
        _extend_unique(resolved_models, decision.get("resolved_models", []))
        _extend_unique(resolved_providers, decision.get("resolved_providers", []))
    attempts = metadata.get("attempts", [])
    return {
        "expected": expected,
        "attempted": len(predictions),
        "successful": successful,
        "failed": len(predictions) - successful,
        "elapsed_seconds": sum(
            attempt.get("elapsed_seconds", 0.0) for attempt in attempts if isinstance(attempt, dict)
        ),
        "resolved_models": resolved_models,
        "resolved_providers": resolved_providers,
        "usage": {field: _complete_sum(values) for field, values in usage_values.items()},
    }


def _complete_sum(values: list[int | float | None]) -> int | float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _merge_progress(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    usage = {}
    for field in _USAGE_FIELDS:
        before = previous["usage"][field]
        now = current["usage"][field]
        usage[field] = None if before is None or now is None else before + now
    return {
        "expected": previous["expected"],
        "attempted": previous["attempted"] + current["attempted"],
        "successful": previous["successful"] + current["successful"],
        "failed": previous["failed"] + current["failed"],
        "elapsed_seconds": previous["elapsed_seconds"] + current["elapsed_seconds"],
        "resolved_models": list(
            dict.fromkeys([*previous["resolved_models"], *current["resolved_models"]])
        ),
        "resolved_providers": list(
            dict.fromkeys([*previous["resolved_providers"], *current["resolved_providers"]])
        ),
        "usage": usage,
    }


def _extend_unique(target: list[str], values: object) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        if isinstance(value, str) and value not in target:
            target.append(value)


def _artifacts(directory: Path) -> dict[str, dict[str, str]]:
    return artifact_metadata(
        directory,
        ("predictions.jsonl", "decisions.jsonl", "summary.json"),
    )
