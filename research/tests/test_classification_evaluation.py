import json

import pytest

from thesis_research.datasets import Banking77Example
from thesis_research.evaluation import evaluate_classification, read_prediction_jsonl


def predictions(records):
    return read_prediction_jsonl("\n".join(json.dumps(record) for record in records).encode())


def test_metrics_and_order():
    examples = [
        Banking77Example(f"test:{i}", "message", label)
        for i, label in enumerate(["a", "a", "b", "b"])
    ]
    records = [{"id": f"test:{i}", "label": label} for i, label in enumerate(["a", "b", "b", "b"])]
    summary, outcomes = evaluate_classification(examples, ["a", "b"], predictions(records))
    assert summary["accuracy"] == 0.75
    assert summary["per_label"]["a"] == {
        "precision": 1,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert summary["per_label"]["b"]["f1"] == pytest.approx(0.8)
    assert summary["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert summary["confusion_matrix"]["values"] == [[1, 1, 0], [0, 2, 0]]
    assert evaluate_classification(examples, ["a", "b"], predictions(records[::-1])) == (
        summary,
        outcomes,
    )


def test_unsuccessful_predictions_remain_in_denominator():
    examples = [Banking77Example(str(i), "message", "a") for i in range(4)]
    records = [
        {"id": "0", "label": "a"},
        {"id": "1", "label": "unknown"},
        {"id": "2", "error": "timeout"},
    ]
    summary, _ = evaluate_classification(examples, ["a", "b"], predictions(records))
    assert summary["accuracy"] == 0.25
    assert summary["prediction_counts"] == {"valid": 1, "invalid": 1, "failed": 1, "missing": 1}
    assert summary["per_label"]["a"]["recall"] == 0.25
    assert summary["macro_f1"] == pytest.approx(0.2)
    assert summary["confusion_matrix"]["values"] == [[1, 0, 3], [0, 0, 0]]


@pytest.mark.parametrize("label", [None, 12, [], {}])
def test_non_string_labels_are_invalid(label):
    summary, _ = evaluate_classification(
        [Banking77Example("1", "message", "a")], ["a"], predictions([{"id": "1", "label": label}])
    )
    assert summary["prediction_counts"]["invalid"] == 1


@pytest.mark.parametrize(
    "records",
    [
        [{"id": "1", "label": "a"}, {"id": "1", "label": "a"}],
        [{"id": "1"}],
        [{"id": "1", "label": "a", "error": "failed"}],
        [{"id": "1", "error": ""}],
        [{"label": "a"}],
        [None],
    ],
)
def test_reject_malformed_or_duplicate_records(records):
    with pytest.raises(ValueError):
        predictions(records)


def test_reject_unknown_ids():
    with pytest.raises(ValueError, match="Unexpected"):
        evaluate_classification(
            [Banking77Example("1", "message", "a")], ["a"], predictions([{"id": "2", "label": "a"}])
        )


def test_all_missing_and_empty_dataset():
    summary, _ = evaluate_classification([Banking77Example("1", "message", "a")], ["a"], {})
    assert summary["accuracy"] == summary["macro_f1"] == 0
    with pytest.raises(ValueError, match="empty"):
        evaluate_classification([], ["a"], {})
