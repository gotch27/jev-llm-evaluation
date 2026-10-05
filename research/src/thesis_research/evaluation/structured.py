"""Provider-independent evaluation for Choice, Noul, and Score datasets."""

import json
import math
from collections import Counter

from thesis_research.datasets import (
    ChoiceReference,
    ChoiceReferenceSchema,
    DatasetExample,
    NoulReference,
    NoulReferenceSchema,
    ReferenceSchema,
    ScoreReference,
    ScoreReferenceSchema,
    reference_class,
)
from thesis_research.tasks import class_labels

_ANSWER_FIELDS = {"label", "value", "level", "score"}


def read_prediction_jsonl(data: bytes) -> dict[str, dict]:
    """Parse prediction records by ID without depending on file order."""
    predictions: dict[str, dict] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"Line {number}: expected an object with a string id")
        has_error = "error" in record
        answer_fields = _ANSWER_FIELDS & set(record)
        has_answer = bool(answer_fields)
        if has_error == has_answer:
            raise ValueError(f"Line {number}: supply an answer or an error, but not both")
        kinds = sum(
            (
                "label" in answer_fields,
                "value" in answer_fields,
                bool({"level", "score"} & answer_fields),
            )
        )
        if has_answer and kinds != 1:
            raise ValueError(f"Line {number}: prediction contains mixed answer types")
        if bool({"level", "score"} & answer_fields) and not {"level", "score"} <= answer_fields:
            raise ValueError(f"Line {number}: Score predictions require level and score")
        if has_error and (not isinstance(record["error"], str) or not record["error"].strip()):
            raise ValueError(f"Line {number}: error must be a nonempty string")
        if record["id"] in predictions:
            raise ValueError(f"Duplicate prediction ID: {record['id']}")
        predictions[record["id"]] = record
    return predictions


def evaluate_structured(
    examples: list[DatasetExample],
    schema: ReferenceSchema,
    predictions: dict[str, dict],
) -> tuple[dict, list[dict]]:
    """Score one typed prediction set against every expected example."""
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset")
    expected = {example.id for example in examples}
    if len(expected) != len(examples):
        raise ValueError("Duplicate reference IDs")
    unknown = set(predictions) - expected
    if unknown:
        raise ValueError(f"Unexpected prediction IDs: {sorted(unknown)[:5]}")

    labels = class_labels(schema)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique and nonempty")
    matrix = {label: dict.fromkeys(labels, 0) for label in labels}
    unsuccessful = dict.fromkeys(labels, 0)
    counts = Counter()
    outcomes: list[dict] = []
    probabilities: list[tuple[bool, float]] = []
    scores: list[tuple[int, int, float]] = []

    for example in examples:
        reference_label = reference_class(example.reference)
        if reference_label not in matrix:
            raise ValueError(f"Unknown reference label: {reference_label}")
        record = predictions.get(example.id)
        status, predicted_label = prediction_status(record, schema)
        correct = status == "valid" and predicted_label == reference_label
        counts[status] += 1
        counts["correct"] += int(correct)
        if status == "valid":
            assert predicted_label is not None
            matrix[reference_label][predicted_label] += 1
            if isinstance(schema, NoulReferenceSchema) and "probability" in record:
                probabilities.append((reference_label == "true", float(record["probability"])))
            if isinstance(schema, ScoreReferenceSchema):
                assert isinstance(example.reference, ScoreReference)
                scores.append(
                    (example.reference.level, int(record["level"]), float(record["score"]))
                )
        else:
            unsuccessful[reference_label] += 1
        outcomes.append(
            {
                "id": example.id,
                "state_fields": example.state,
                "reference_label": reference_label,
                "reference_value": _reference_value(example),
                "predicted_class": predicted_label,
                "prediction": record,
                "status": status,
                "correct": correct,
            }
        )

    per_label = _per_label_metrics(labels, matrix, unsuccessful)
    summary = {
        "question_type": schema.question_type,
        "total": len(examples),
        "correct": counts["correct"],
        "accuracy": counts["correct"] / len(examples),
        "macro_f1": sum(value["f1"] for value in per_label.values()) / len(labels),
        "prediction_counts": {
            key: counts[key] for key in ("valid", "missing", "invalid", "failed")
        },
        "per_label": per_label,
        "confusion_matrix": {
            "rows": labels,
            "columns": [*labels, "<unsuccessful>"],
            "values": [[*matrix[label].values(), unsuccessful[label]] for label in labels],
        },
    }
    if isinstance(schema, NoulReferenceSchema):
        summary.update(_noul_metrics(probabilities, len(examples)))
    if isinstance(schema, ScoreReferenceSchema):
        summary.update(_score_metrics(scores, schema.level_count, len(examples)))
    return summary, outcomes


def prediction_status(record: dict | None, schema: ReferenceSchema) -> tuple[str, str | None]:
    """Classify one saved prediction and return its discrete class key."""
    if record is None:
        return "missing", None
    if "error" in record:
        return "failed", None
    if isinstance(schema, ChoiceReferenceSchema):
        return _choice_prediction_status(record, schema)
    if isinstance(schema, NoulReferenceSchema):
        return _noul_prediction_status(record)
    return _score_prediction_status(record, schema)


def _choice_prediction_status(
    record: dict, schema: ChoiceReferenceSchema
) -> tuple[str, str | None]:
    if not set(record) <= {"id", "label", "confidence", "probabilities"}:
        return "invalid", None
    label = record.get("label")
    if not isinstance(label, str) or label not in schema.labels:
        return "invalid", None
    if "confidence" in record and not _probability(record["confidence"]):
        return "invalid", None
    if "probabilities" in record and not _distribution(record["probabilities"], set(schema.labels)):
        return "invalid", None
    return "valid", label


def _noul_prediction_status(record: dict) -> tuple[str, str | None]:
    if not set(record) <= {"id", "value", "probability"}:
        return "invalid", None
    value = record.get("value")
    if not isinstance(value, bool):
        return "invalid", None
    if "probability" in record:
        probability = record["probability"]
        if not _probability(probability) or value != (float(probability) >= 0.5):
            return "invalid", None
    return "valid", "true" if value else "false"


def _score_prediction_status(record: dict, schema: ScoreReferenceSchema) -> tuple[str, str | None]:
    if not set(record) <= {"id", "level", "score", "confidence", "probabilities"}:
        return "invalid", None
    level = record.get("level")
    score = record.get("score")
    if (
        not isinstance(level, int)
        or isinstance(level, bool)
        or not 0 <= level < schema.level_count
        or not isinstance(score, int | float)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= schema.level_count - 1
    ):
        return "invalid", None
    if "confidence" in record and not _probability(record["confidence"]):
        return "invalid", None
    if "probabilities" in record:
        probabilities = record["probabilities"]
        if not _distribution(
            probabilities,
            {str(index) for index in range(schema.level_count)},
            require_all=True,
            require_sum=True,
        ):
            return "invalid", None
        highest = max(probabilities.values())
        derived_level = min(
            int(key) for key, probability in probabilities.items() if probability == highest
        )
        if level != derived_level:
            return "invalid", None
    return "valid", str(level)


def _per_label_metrics(
    labels: list[str], matrix: dict[str, dict[str, int]], unsuccessful: dict[str, int]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values()) + unsuccessful[label]
        predicted_count = sum(matrix[actual][label] for actual in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def _noul_metrics(probabilities: list[tuple[bool, float]], total: int) -> dict:
    if not probabilities:
        return {
            "probability_count": 0,
            "probability_coverage": 0.0,
            "brier_score": None,
            "log_loss": None,
        }
    epsilon = 1e-15
    brier = sum((probability - float(reference)) ** 2 for reference, probability in probabilities)
    log_loss = -sum(
        math.log(min(1 - epsilon, max(epsilon, probability)))
        if reference
        else math.log(min(1 - epsilon, max(epsilon, 1 - probability)))
        for reference, probability in probabilities
    )
    return {
        "probability_count": len(probabilities),
        "probability_coverage": len(probabilities) / total,
        "brier_score": brier / len(probabilities),
        "log_loss": log_loss / len(probabilities),
    }


def _score_metrics(scores: list[tuple[int, int, float]], level_count: int, total: int) -> dict:
    if not scores:
        return {
            "numeric_count": 0,
            "numeric_coverage": 0.0,
            "mean_absolute_error": None,
            "root_mean_squared_error": None,
            "quadratic_weighted_kappa": None,
        }
    absolute = [abs(score - reference) for reference, _, score in scores]
    squared = [(score - reference) ** 2 for reference, _, score in scores]
    return {
        "numeric_count": len(scores),
        "numeric_coverage": len(scores) / total,
        "mean_absolute_error": sum(absolute) / len(absolute),
        "root_mean_squared_error": math.sqrt(sum(squared) / len(squared)),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(
            [(reference, predicted) for reference, predicted, _ in scores], level_count
        ),
    }


def _quadratic_weighted_kappa(pairs: list[tuple[int, int]], levels: int) -> float:
    observed = [[0 for _ in range(levels)] for _ in range(levels)]
    actual = [0 for _ in range(levels)]
    predicted = [0 for _ in range(levels)]
    for reference, prediction in pairs:
        observed[reference][prediction] += 1
        actual[reference] += 1
        predicted[prediction] += 1
    denominator_scale = (levels - 1) ** 2
    observed_weight = 0.0
    expected_weight = 0.0
    count = len(pairs)
    for left in range(levels):
        for right in range(levels):
            weight = (left - right) ** 2 / denominator_scale
            observed_weight += weight * observed[left][right]
            expected_weight += weight * actual[left] * predicted[right] / count
    if expected_weight == 0:
        return 1.0 if observed_weight == 0 else 0.0
    return 1 - observed_weight / expected_weight


def _reference_value(example: DatasetExample) -> str | bool | int:
    reference = example.reference
    if isinstance(reference, ChoiceReference):
        return reference.label
    if isinstance(reference, NoulReference):
        return reference.value
    return reference.level


def _probability(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _distribution(
    value: object,
    labels: set[str],
    *,
    require_all: bool = False,
    require_sum: bool = False,
) -> bool:
    if not isinstance(value, dict) or not value or not set(value) <= labels:
        return False
    if require_all and set(value) != labels:
        return False
    if not all(_probability(probability) for probability in value.values()):
        return False
    return not require_sum or math.isclose(sum(value.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
