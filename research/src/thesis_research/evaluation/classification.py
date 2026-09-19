"""Classification scoring independent of model providers."""

import json
from collections import Counter

from thesis_research.datasets import Banking77Example


def read_prediction_jsonl(data: bytes) -> dict[str, dict]:
    """Parse and validate prediction records indexed by example ID.

    Each nonblank JSONL line must contain a string ``id`` and exactly one of
    ``label`` or ``error``. Record order is intentionally discarded because
    evaluation joins predictions to references by ID.

    Args:
        data: UTF-8 encoded JSONL prediction content.

    Returns:
        Prediction records keyed by their example IDs.

    Raises:
        UnicodeDecodeError: If the input is not valid UTF-8.
        json.JSONDecodeError: If a nonblank line is not valid JSON.
        ValueError: If a record is malformed or an ID appears more than once.
    """
    predictions = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"Line {number}: expected an object with a string id")
        if ("label" in record) == ("error" in record):
            raise ValueError(f"Line {number}: supply exactly one of label or error")
        if "error" in record and (
            not isinstance(record["error"], str) or not record["error"].strip()
        ):
            raise ValueError(f"Line {number}: error must be a nonempty string")
        if record["id"] in predictions:
            raise ValueError(f"Duplicate prediction ID: {record['id']}")
        predictions[record["id"]] = record
    return predictions


def evaluate_classification(
    examples: list[Banking77Example], labels: list[str], predictions: dict
) -> tuple[dict, list]:
    """Score saved predictions against all expected classification examples.

    Missing, invalid, and explicitly failed predictions remain in the accuracy
    denominator and reduce recall for their reference labels.

    Args:
        examples: Reference examples for the configured dataset split.
        labels: Complete label inventory in the desired report order.
        predictions: Prediction records keyed by stable example ID.

    Returns:
        A pair containing the aggregate metric summary and ordered per-example
        outcomes. The summary includes accuracy, macro-F1, per-label metrics,
        status counts, and a confusion matrix.

    Raises:
        ValueError: If references or labels are empty or duplicated, or if a
            prediction ID does not belong to the evaluated split.
    """
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset")
    expected = {example.id for example in examples}
    if len(expected) != len(examples):
        raise ValueError("Duplicate reference IDs")
    unknown = set(predictions) - expected
    if unknown:
        raise ValueError(f"Unexpected prediction IDs: {sorted(unknown)[:5]}")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique and nonempty")
    # The final column captures unsuccessful predictions without inventing an intent.
    matrix = {label: {prediction: 0 for prediction in labels} for label in labels}
    unsuccessful = dict.fromkeys(labels, 0)
    counts = Counter()
    outcomes = []
    for example in examples:
        if example.label not in matrix:
            raise ValueError(f"Unknown reference label: {example.label}")
        record = predictions.get(example.id)
        predicted = record.get("label") if record else None
        if record is None:
            status = "missing"
        elif "error" in record:
            status = "failed"
        elif not isinstance(predicted, str) or predicted not in matrix:
            status = "invalid"
        else:
            status = "valid"
        correct = status == "valid" and predicted == example.label
        counts[status] += 1
        counts["correct"] += int(correct)
        if status == "valid":
            matrix[example.label][predicted] += 1
        else:
            unsuccessful[example.label] += 1
        outcomes.append(
            {
                "id": example.id,
                "text": example.text,
                "reference_label": example.label,
                "prediction": record,
                "status": status,
                "correct": correct,
            }
        )
    per_label = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values()) + unsuccessful[label]
        predicted_count = sum(matrix[actual][label] for actual in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
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
    }, outcomes
