"""Execute and resume one model inside a coordinated benchmark."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from thesis_research.benchmark.model_records import (
    load_model_records,
    load_or_create_model_metadata,
    merge_model_progress,
    model_artifacts,
    summarize_model_records,
    validate_model_records,
)
from thesis_research.benchmark.progress import (
    BenchmarkObserver,
    ModelProgress,
    NullBenchmarkObserver,
)
from thesis_research.benchmark.types import ModelSpec
from thesis_research.clients import DecisionClient, create_decision_client
from thesis_research.datasets import Banking77Example
from thesis_research.prediction import predict_examples
from thesis_research.run_storage import write_json
from thesis_research.tasks import IntentClassificationTask


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
    metadata = load_or_create_model_metadata(
        metadata_path,
        spec,
        example_concurrency,
        recorded_output_mode,
    )
    predictions, decisions = load_model_records(directory)
    expected_ids = {example.id for example in examples}
    validate_model_records(predictions, decisions, expected_ids)
    previous_summary = summarize_model_records(directory, len(examples), labels, metadata)
    observer.model_started(spec.id, ModelProgress.from_summary(previous_summary))
    remaining = [example for example in examples if example.id not in predictions]
    if not remaining:
        summary = previous_summary
        write_json(directory / "summary.json", summary)
        metadata["progress"] = summary
        metadata["status"] = "completed"
        metadata["artifacts"] = model_artifacts(directory)
        write_json(metadata_path, metadata)
        observer.model_completed(spec.id, ModelProgress.from_summary(summary))
        return summary

    attempt = {
        "number": len(metadata.get("attempts", [])) + 1,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_before": len(predictions),
        "status": "running",
    }
    metadata.setdefault("attempts", []).append(attempt)
    metadata["status"] = "running"
    metadata.pop("error", None)
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    measurement_started: float | None = None
    try:
        decision_client = client or create_decision_client(
            spec.backend,
            spec.model,
            spec.provider,
            max_concurrency=example_concurrency,
            llm_output_mode=llm_output_mode,
        )

        def record_progress(progress: dict[str, Any]) -> None:
            merged = merge_model_progress(previous_summary, progress)
            metadata["progress"] = merged
            write_json(metadata_path, metadata)
            observer.model_progress(spec.id, ModelProgress.from_summary(merged))

        def record_prediction(prediction: dict[str, Any]) -> None:
            error = prediction.get("error")
            if isinstance(error, str):
                observer.prediction_failed(spec.id, prediction["id"], error)

        async with decision_client:
            measurement_started = time.perf_counter()
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
            attempt["measurement_elapsed_seconds"] = time.perf_counter() - measurement_started
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["finished_at"] = datetime.now(UTC).isoformat()
        attempt["status"] = "completed"
        summary = summarize_model_records(directory, len(examples), labels, metadata)
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
        if measurement_started is not None:
            attempt.setdefault(
                "measurement_elapsed_seconds",
                time.perf_counter() - measurement_started,
            )
        try:
            current_predictions, current_decisions = load_model_records(directory)
            validate_model_records(current_predictions, current_decisions, expected_ids)
            attempt["completed_after"] = len(current_predictions)
        except Exception as record_error:
            metadata["record_error"] = f"{type(record_error).__name__}: {record_error}"
        metadata["artifacts"] = model_artifacts(directory)
        write_json(metadata_path, metadata)
