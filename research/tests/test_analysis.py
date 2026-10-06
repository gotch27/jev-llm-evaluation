"""Analysis rejects mismatched reports and preserves unavailable measurements."""

import json
import math

import pytest

pytest.importorskip("pandas", reason="Install the analysis dependency group")
pytest.importorskip("matplotlib", reason="Install the analysis dependency group")

from thesis_research.analysis import (  # noqa: E402
    export_analysis,
    load_benchmark,
    model_table,
    pairwise_table,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "benchmark"
    report = run / "report"
    report.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"status": "completed"}))
    examples = [
        {"id": f"train:{index}", "reference": {"label": label}}
        for index, label in enumerate(("a", "b", "a", "b"), start=1)
    ]
    (run / "cohort.json").write_text(
        json.dumps(
            {
                "dataset_id": "fixture",
                "question_type": "choice",
                "split": "train",
                "count": 4,
                "examples": examples,
            }
        )
    )
    labels = {"a": {}, "b": {}, "unused": {}}
    (report / "summary.json").write_text(
        json.dumps(
            {
                "question_type": "choice",
                "baseline_model_id": "jev",
                "models": {model: {"per_label": labels} for model in ("jev", "llm")},
            }
        )
    )
    metrics = []
    outcomes = []
    per_label = []
    for model, predictions in (("jev", ["a", None, "a", "a"]), ("llm", ["a", "b", "b", "a"])):
        metrics.append(
            {
                "model_id": model,
                "model": model,
                "provider": None,
                "reasoning_effort": None,
                "total": 4,
                "correct": 2,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "valid": 3 if model == "jev" else 4,
                "failed": 1 if model == "jev" else 0,
                "missing": 0,
                "invalid": 0,
                "cost_usd": None if model == "jev" else 0.012,
                "all_decision_latency_p50_seconds": 0.4,
                "all_decision_latency_p95_seconds": 0.7,
            }
        )
        for example, prediction in zip(examples, predictions, strict=True):
            label = example["reference"]["label"]
            outcomes.append(
                {
                    "model_id": model,
                    "id": example["id"],
                    "reference_label": label,
                    "predicted_label": prediction,
                    "status": "valid" if prediction else "failed",
                    "correct": prediction == label,
                    "latency_seconds": 0.4,
                }
            )
        per_label.extend(
            {"model_id": model, "label": label, "support": 0 if label == "unused" else 2, "f1": 0.5}
            for label in labels
        )
    pairs = [
        {
            "baseline_model_id": "jev",
            "comparison_model_id": "llm",
            "total": 4,
            "both_correct": 1,
            "baseline_only_correct": 1,
            "comparison_only_correct": 1,
            "neither_correct": 1,
        }
    ]
    for name, rows in (
        ("model_metrics", metrics),
        ("model_outcomes", outcomes),
        ("per_label_metrics", per_label),
        ("pairwise_statistics", pairs),
    ):
        write_rows(report / f"{name}.jsonl", rows)
    return run


def test_failures_stay_in_denominator_and_unknown_cost_stays_missing(completed_run):
    data = load_benchmark(completed_run)
    table = model_table(data).set_index("model_id")
    assert table.loc["jev", "accuracy"] == 0.5
    assert table.loc["jev", "failed"] == 1
    assert math.isnan(table.loc["jev", "cost_usd_per_1000_examples"])
    assert table.loc["llm", "cost_usd_per_1000_examples"] == 3
    assert pairwise_table(data).iloc[0]["oracle_accuracy_upper_bound"] == 0.75


@pytest.mark.parametrize("change", ["drop", "duplicate", "reference", "correct"])
def test_rejects_mismatched_or_inconsistent_outcomes(completed_run, change):
    path = completed_run / "report/model_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "reference":
        rows[0]["reference_label"] = "b"
    else:
        rows[0]["correct"] = False
    write_rows(path, rows)
    with pytest.raises(ValueError):
        load_benchmark(completed_run)


def test_rejects_incomplete_benchmark(completed_run):
    (completed_run / "metadata.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="completed benchmark"):
        load_benchmark(completed_run)


def test_rejects_inconsistent_paired_counts(completed_run):
    path = completed_run / "report/pairwise_statistics.jsonl"
    pair = json.loads(path.read_text())
    pair["both_correct"] = 2
    write_rows(path, [pair])
    with pytest.raises(ValueError, match="Paired counts"):
        load_benchmark(completed_run)


def test_cannot_export_into_original_run(completed_run):
    data = load_benchmark(completed_run)
    with pytest.raises(ValueError, match="outside the source"):
        export_analysis(data, completed_run / "analysis")
