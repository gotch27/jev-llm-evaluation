"""Validate, summarize, and describe one model's durable benchmark records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from thesis_research.benchmark.timing import latency_statistics
from thesis_research.benchmark.types import ModelSpec
from thesis_research.evaluation import read_prediction_jsonl
from thesis_research.run_storage import artifact_metadata, write_json

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "retries",
    "malformed_response_retries",
)


def load_or_create_model_metadata(
    path: Path,
    spec: ModelSpec,
    example_concurrency: int,
    output_mode: str,
) -> dict[str, Any]:
    """Load compatible model metadata or initialize it for a new run."""
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


def load_model_records(directory: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load paired prediction and decision JSONL files by example ID."""
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


def validate_model_records(
    predictions: dict[str, dict],
    decisions: dict[str, dict],
    expected_ids: set[str],
) -> None:
    """Reject unpaired, unexpected, or inconsistent saved model records."""
    if set(predictions) != set(decisions):
        raise ValueError("Prediction and decision records contain different IDs")
    unexpected = set(predictions) - expected_ids
    if unexpected:
        raise ValueError(f"Model records contain unexpected IDs: {sorted(unexpected)[:5]}")
    for example_id, decision in decisions.items():
        if decision.get("prediction") != predictions[example_id]:
            raise ValueError(f"Prediction differs from decision record for {example_id}")


def archive_error_records_for_retry(directory: Path) -> tuple[str, ...]:
    """Archive failed provider attempts and remove them from active model records.

    Successful records remain untouched, so a resumed benchmark calls the
    provider only for previously failed and unfinished example IDs. The full
    failed prediction and decision records are retained in ``retry_history.jsonl``.
    """
    predictions, decisions = load_model_records(directory)
    failed_ids = tuple(
        example_id
        for example_id, prediction in predictions.items()
        if isinstance(prediction.get("error"), str)
    )
    if not failed_ids:
        return ()

    archived_at = datetime.now(UTC).isoformat()
    history_path = directory / "retry_history.jsonl"
    history = history_path.read_text(encoding="utf-8") if history_path.exists() else ""
    history += "".join(
        json.dumps(
            {
                "archived_at": archived_at,
                "id": example_id,
                "prediction": predictions[example_id],
                "decision": decisions[example_id],
            },
            ensure_ascii=False,
        )
        + "\n"
        for example_id in failed_ids
    )
    _replace_text(history_path, history)

    failed = set(failed_ids)
    _replace_jsonl(
        directory / "predictions.jsonl",
        (record for example_id, record in predictions.items() if example_id not in failed),
    )
    _replace_jsonl(
        directory / "decisions.jsonl",
        (record for example_id, record in decisions.items() if example_id not in failed),
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("error_retries", []).append(
        {
            "prepared_at": archived_at,
            "count": len(failed_ids),
            "example_ids": list(failed_ids),
        }
    )
    metadata["status"] = "pending"
    metadata["progress"] = None
    write_json(metadata_path, metadata)
    return failed_ids


def _replace_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _replace_text(path, content)


def _replace_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def summarize_model_records(
    directory: Path,
    expected: int,
    labels: Sequence[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build progress, usage, and latency metrics from durable records."""
    predictions, decisions = load_model_records(directory)
    allowed = set(labels)
    successful = sum(record.get("label") in allowed for record in predictions.values())
    usage_values: dict[str, list[int | float | None]] = {field: [] for field in USAGE_FIELDS}
    resolved_models: list[str] = []
    resolved_providers: list[str] = []
    all_decision_latencies: list[float] = []
    successful_decision_latencies: list[float] = []
    provider_call_latencies: list[float] = []
    for example_id, record in decisions.items():
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError(f"Malformed decision record for {record.get('id')}")
        usage = decision.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"Missing decision usage for {record.get('id')}")
        for field in USAGE_FIELDS:
            usage_values[field].append(usage.get(field))
        _extend_unique(resolved_models, decision.get("resolved_models", []))
        _extend_unique(resolved_providers, decision.get("resolved_providers", []))
        latency = _latency_value(decision.get("latency_seconds"), example_id)
        all_decision_latencies.append(latency)
        if predictions[example_id].get("label") in allowed:
            successful_decision_latencies.append(latency)
        calls = decision.get("calls")
        if not isinstance(calls, list):
            raise ValueError(f"Missing decision calls for {example_id}")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError(f"Malformed provider call for {example_id}")
            provider_call_latencies.append(_latency_value(call.get("latency_seconds"), example_id))
    attempts = metadata.get("attempts", [])
    measurement_elapsed = sum(
        attempt.get("measurement_elapsed_seconds", attempt.get("elapsed_seconds", 0.0))
        for attempt in attempts
        if isinstance(attempt, dict)
    )
    return {
        "expected": expected,
        "attempted": len(predictions),
        "successful": successful,
        "failed": len(predictions) - successful,
        "elapsed_seconds": measurement_elapsed,
        "total_elapsed_seconds": sum(
            attempt.get("elapsed_seconds", 0.0) for attempt in attempts if isinstance(attempt, dict)
        ),
        "resolved_models": resolved_models,
        "resolved_providers": resolved_providers,
        "usage": {field: _complete_sum(values) for field, values in usage_values.items()},
        "latency": {
            "successful_decisions": latency_statistics(successful_decision_latencies),
            "all_decisions": latency_statistics(all_decision_latencies),
            "provider_calls": latency_statistics(provider_call_latencies),
        },
    }


def merge_model_progress(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Merge an existing run summary with progress from a resumed attempt."""
    usage = {}
    for field in USAGE_FIELDS:
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


def model_artifacts(directory: Path) -> dict[str, dict[str, str]]:
    """Return checksums for all model artifacts currently present."""
    return artifact_metadata(
        directory,
        ("predictions.jsonl", "decisions.jsonl", "retry_history.jsonl", "summary.json"),
    )


def _latency_value(value: object, example_id: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Missing latency for {example_id}")
    return float(value)


def _complete_sum(values: list[int | float | None]) -> int | float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _extend_unique(target: list[str], values: object) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        if isinstance(value, str) and value not in target:
            target.append(value)
