"""Build machine-readable metrics for Jev-to-LLM comparisons."""

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

from thesis_research.benchmark.types import ModelSpec
from thesis_research.datasets import Banking77Example
from thesis_research.evaluation import evaluate_classification, read_prediction_jsonl
from thesis_research.run_storage import artifact_metadata, write_json


def generate_report(
    run: Path,
    models: Sequence[ModelSpec],
    examples: list[Banking77Example],
    labels: list[str],
) -> dict[str, Any]:
    """Evaluate every model and compare each LLM with the Jev baseline.

    The JSONL tables are deliberately tidy: each row is one model, label,
    pair, or paired example. They can later be loaded directly into pandas or
    another visualization tool without rerunning model calls.
    """
    report = run / "report"
    report.mkdir(exist_ok=True)
    model_summaries: dict[str, dict[str, Any]] = {}
    model_outcomes: dict[str, list[dict[str, Any]]] = {}
    model_execution: dict[str, dict[str, Any]] = {}
    for model in models:
        predictions = read_prediction_jsonl(
            (run / "models" / model.id / "predictions.jsonl").read_bytes()
        )
        if set(predictions) != {example.id for example in examples}:
            raise ValueError(f"Model {model.id} does not have a complete cohort")
        summary, outcomes = evaluate_classification(examples, labels, predictions)
        model_summaries[model.id] = summary
        model_outcomes[model.id] = outcomes
        model_execution[model.id] = json.loads(
            (run / "models" / model.id / "summary.json").read_text(encoding="utf-8")
        )

    baseline = next(model for model in models if model.backend == "jev")
    comparisons = [
        _compare_outcomes(
            baseline,
            candidate,
            model_summaries,
            model_outcomes,
        )
        for candidate in models
        if candidate.id != baseline.id
    ]
    summary = {
        "baseline_model_id": baseline.id,
        "models": model_summaries,
        "execution": model_execution,
        "comparisons": comparisons,
    }
    write_json(report / "summary.json", summary)
    _write_jsonl(
        report / "model_metrics.jsonl",
        (
            {
                "model_id": model.id,
                "backend": model.backend,
                "model": model.model,
                "provider": model.provider,
                "total": model_summaries[model.id]["total"],
                "correct": model_summaries[model.id]["correct"],
                "accuracy": model_summaries[model.id]["accuracy"],
                "macro_f1": model_summaries[model.id]["macro_f1"],
                **model_summaries[model.id]["prediction_counts"],
                "elapsed_seconds": model_execution[model.id]["elapsed_seconds"],
                **model_execution[model.id]["usage"],
            }
            for model in models
        ),
    )
    _write_jsonl(
        report / "per_label_metrics.jsonl",
        (
            {
                "model_id": model.id,
                "label": label,
                **model_summaries[model.id]["per_label"][label],
            }
            for model in models
            for label in labels
        ),
    )
    _write_jsonl(report / "pairwise_statistics.jsonl", iter(comparisons))
    _write_jsonl(
        report / "model_outcomes.jsonl",
        (
            _flat_outcome(model.id, outcome)
            for model in models
            for outcome in model_outcomes[model.id]
        ),
    )
    return summary


def exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    """Return the two-sided exact McNemar p-value for paired correctness."""
    if left_only < 0 or right_only < 0:
        raise ValueError("McNemar counts cannot be negative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, lower_tail / (2 ** (discordant - 1)))


def report_artifacts(run: Path) -> dict[str, dict[str, str]]:
    """Return checksums for every generated report table."""
    return artifact_metadata(
        run / "report",
        (
            "summary.json",
            "model_metrics.jsonl",
            "per_label_metrics.jsonl",
            "pairwise_statistics.jsonl",
            "model_outcomes.jsonl",
        ),
    )


def _compare_outcomes(
    baseline: ModelSpec,
    candidate: ModelSpec,
    summaries: dict[str, dict[str, Any]],
    outcomes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    both_correct = baseline_only = candidate_only = neither = 0
    both_valid = agreement = 0
    for left, right in zip(outcomes[baseline.id], outcomes[candidate.id], strict=True):
        if left["id"] != right["id"]:
            raise ValueError("Model outcomes are not aligned by cohort ID")
        left_correct = left["correct"]
        right_correct = right["correct"]
        both_correct += int(left_correct and right_correct)
        baseline_only += int(left_correct and not right_correct)
        candidate_only += int(right_correct and not left_correct)
        neither += int(not left_correct and not right_correct)
        if left["status"] == right["status"] == "valid":
            both_valid += 1
            agreement += int(left["prediction"]["label"] == right["prediction"]["label"])
    baseline_accuracy = summaries[baseline.id]["accuracy"]
    candidate_accuracy = summaries[candidate.id]["accuracy"]
    return {
        "baseline_model_id": baseline.id,
        "comparison_model_id": candidate.id,
        "total": len(outcomes[baseline.id]),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "comparison_only_correct": candidate_only,
        "neither_correct": neither,
        "baseline_accuracy": baseline_accuracy,
        "comparison_accuracy": candidate_accuracy,
        "accuracy_difference": candidate_accuracy - baseline_accuracy,
        "both_valid": both_valid,
        "valid_label_agreement": agreement,
        "valid_label_agreement_rate": agreement / both_valid if both_valid else None,
        "mcnemar_exact_two_sided_p_value": exact_mcnemar_p_value(
            baseline_only,
            candidate_only,
        ),
    }


def _flat_outcome(model_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
    prediction = outcome["prediction"]
    return {
        "model_id": model_id,
        "id": outcome["id"],
        "reference_label": outcome["reference_label"],
        "predicted_label": prediction.get("label") if prediction else None,
        "confidence": prediction.get("confidence") if prediction else None,
        "error": prediction.get("error") if prediction else None,
        "status": outcome["status"],
        "correct": outcome["correct"],
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
