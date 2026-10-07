"""Automatic presentation preserves saved measurements and completed experiments."""

import json

import pytest

from thesis_research.reporting import format_number, generate_dashboard, load_saved_report
from thesis_research.run_storage import file_checksum


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def saved_report(tmp_path):
    def build(question_type="choice", model_ids=("jev", "laya", "liquid-d1"), baseline="laya"):
        run = tmp_path / "run"
        report = run / "report"
        report.mkdir(parents=True)
        metadata = {
            "status": "completed",
            "name": "Fixture <script>alert('injection')</script>",
            "timestamp": "2026-10-06T10:00:00+00:00",
            "code": {"commit": "original-code", "dirty": False},
        }
        (run / "metadata.json").write_text(json.dumps(metadata))
        (run / "plan.toml").write_text('backend = "jev"\n')  # Reports also read retired plans.
        (run / "task.toml").write_text('name = "frozen task"\n')
        references = {
            "choice": [{"label": "a"}, {"label": "b"}, {"label": "a"}],
            "noul": [{"value": True}, {"value": False}, {"value": True}],
            "score": [{"level": 0}, {"level": 1}, {"level": 2}],
        }[question_type]
        cohort = {
            "dataset_id": "fixture",
            "question_type": question_type,
            "split": "train",
            "count": 3,
            "examples": [
                {"id": f"train:{index}", "reference": reference}
                for index, reference in enumerate(references)
            ],
        }
        (run / "cohort.json").write_text(json.dumps(cohort))
        (report / "summary.json").write_text(
            json.dumps(
                {
                    "question_type": question_type,
                    "baseline_model_id": baseline,
                    "models": {model: {} for model in model_ids},
                }
            )
        )
        metrics, outcomes, classes = [], [], []
        for model in model_ids:
            metrics.append(
                {
                    "model_id": model,
                    "model": model,
                    "provider": "fixture",
                    "question_type": question_type,
                    "reasoning_effort": None,
                    "total": 3,
                    "correct": 1,
                    "accuracy": 1 / 3,
                    "macro_f1": 0.25,
                    "failed": 1,
                    "invalid": 0,
                    "missing": 0,
                    "valid": 2,
                    "cost_usd": {
                        "jev": None,
                        "laya": 0.0,
                        "liquid-d1": 0.0015,
                        "gpt-6-luna-decisions": 0.0002,
                    }[model],
                    "all_decision_latency_p50_seconds": 0.25,
                    "all_decision_latency_p95_seconds": 0.565,
                    "brier_score": 0.12,
                    "log_loss": 0.3,
                    "probability_coverage": 2 / 3,
                    "mean_absolute_error": 0.5,
                    "root_mean_squared_error": 0.7,
                    "quadratic_weighted_kappa": 0.4,
                    "numeric_coverage": 2 / 3,
                }
            )
            for index, latency in enumerate((0.0, 0.25, 0.6)):
                outcomes.append(
                    {
                        "model_id": model,
                        "id": f"train:{index}",
                        "latency_seconds": latency,
                        "status": "failed" if index == 2 else "valid",
                        "correct": index == 0,
                        "error": "Provider error <img src=x onerror=alert(1)>"
                        if index == 2
                        else None,
                    }
                )
            classes += [
                {"model_id": model, "label": label, "support": support, "f1": f1}
                for label, support, f1 in (("a", 2, 0.5), ("b", 1, 0), ("unused", 0, 0))
            ]
        pairs = [
            {
                "baseline_model_id": baseline,
                "comparison_model_id": model,
                "total": 3,
                "both_correct": 1,
                "baseline_only_correct": 0,
                "comparison_only_correct": 0,
                "neither_correct": 2,
                "accuracy_difference": 0.0,
                "mcnemar_exact_two_sided_p_value": 1.0,
            }
            for model in model_ids
            if model != baseline
        ]
        for name, rows in (
            ("model_metrics", metrics),
            ("model_outcomes", outcomes),
            ("per_label_metrics", classes),
            ("pairwise_statistics", pairs),
        ):
            write_rows(report / f"{name}.jsonl", rows)
        return run

    return build


@pytest.mark.parametrize("question_type", ["choice", "noul", "score"])
@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("baseline", ["laya", "gpt-6-luna-decisions"])
def test_standalone_dashboard_supports_all_tasks_and_arbitrary_baselines(
    saved_report, tmp_path, question_type, single, baseline
):
    models = (baseline,) if single else ("jev", "laya", "liquid-d1", "gpt-6-luna-decisions")
    run = saved_report(question_type, models, baseline)
    before = {str(path): file_checksum(path) for path in run.rglob("*") if path.is_file()}
    path = generate_dashboard(run, tmp_path / "export/index.html")
    document = path.read_text()
    assert "33.3%" in document
    if "laya" in models:
        assert "0.00000000" in document  # Reported zero is visible.
    assert "unknown" in document if not single else "Single-model run" in document
    assert "&lt;script&gt;" in document and "<script>" not in document
    assert "&lt;img" in document and "<img" not in document
    assert "<svg" in document and "https://" not in document
    assert "Decision latency distribution" in document
    assert "(n=3)" in document  # Includes zero latency and failed decisions.
    name = "Laya" if baseline == "laya" else "GPT-6 Luna Decisions"
    assert (f"Paired correctness · baseline: {name}" in document) != single
    if "gpt-6-luna-decisions" in models:
        assert "GPT-6 Luna Decisions" in document
        assert "stroke='#882255'" in document
    assert ("Brier score" in document) == (question_type == "noul")
    assert ("Weighted kappa" in document) == (question_type == "score")
    assert {str(path): file_checksum(path) for path in run.rglob("*") if path.is_file()} == before
    manifest = json.loads((path.parent / "visualization.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["baseline_model_id"] == baseline
    assert manifest["dashboard_sha256"] == file_checksum(path)
    assert manifest["source_sha256"]["report/model_metrics.jsonl"] == file_checksum(
        run / "report/model_metrics.jsonl"
    )
    assert manifest["benchmark_code"]["commit"] == "original-code"


def test_dashboard_rejects_incomplete_source_before_writing(saved_report, tmp_path):
    run = saved_report()
    (run / "metadata.json").write_text(json.dumps({"status": "interrupted"}))
    output = tmp_path / "export/index.html"
    with pytest.raises(ValueError, match="completed benchmark"):
        generate_dashboard(run, output)
    assert not output.parent.exists()


def test_report_rejects_mismatched_comparisons(saved_report):
    run = saved_report()
    write_rows(run / "report/pairwise_statistics.jsonl", [])
    with pytest.raises(ValueError, match="Paired comparisons"):
        load_saved_report(run)


def test_dashboard_preserves_small_positive_p_values(saved_report, tmp_path):
    run = saved_report()
    path = run / "report/pairwise_statistics.jsonl"
    pairs = [json.loads(line) for line in path.read_text().splitlines()]
    pairs[0]["mcnemar_exact_two_sided_p_value"] = 1e-12
    write_rows(path, pairs)
    document = generate_dashboard(run, tmp_path / "export/index.html").read_text()
    assert "<td>1e-12</td>" in document


def test_dashboard_handles_unreported_and_zero_latencies(saved_report, tmp_path):
    run = saved_report(model_ids=("laya",))
    rows = [
        json.loads(line) for line in (run / "report/model_outcomes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["latency_seconds"] = None
    write_rows(run / "report/model_outcomes.jsonl", rows)
    path = generate_dashboard(run, tmp_path / "export/index.html")
    assert "(n=0)" in path.read_text()
    assert "<polyline" not in path.read_text()


@pytest.mark.parametrize(
    "value,expected",
    [(None, "unknown"), (0, "0.000"), (float("nan"), "unknown"), (False, "unknown")],
)
def test_unavailable_measurements_remain_distinct_from_zero(value, expected):
    assert format_number(value) == expected


def test_report_command_exports_history_without_credentials_or_model_calls(
    saved_report, tmp_path, monkeypatch, capsys
):
    import sys

    from thesis_research import cli
    from thesis_research.benchmark import model_run

    run = saved_report("noul", ("jev", "laya", "liquid-d1", "gpt-6-luna-decisions"))
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)

    def reject_model_call(*args, **kwargs):
        raise AssertionError("Reporting must never construct a provider client")

    monkeypatch.setattr(model_run, "create_decision_client", reject_model_call)
    output = tmp_path / "view"
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        sys, "argv", ["thesis-research", "report", str(run), "--output-dir", str(output)]
    )
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == str(output / "index.html")
    assert "Brier score" in captured.err
    assert "GPT-6 Luna Decisions" in captured.err
    assert "0.00000000" in captured.err and "unknown" in captured.err
    assert {
        path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()
    } == before


def test_report_command_rejects_output_inside_historical_run(saved_report, monkeypatch, capsys):
    import sys

    from thesis_research import cli

    run = saved_report()
    monkeypatch.setattr(
        sys, "argv", ["thesis-research", "report", str(run), "--output-dir", str(run / "report")]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "outside the source" in capsys.readouterr().err
    assert not (run / "report/index.html").exists()


@pytest.mark.parametrize("browser_available", [True, False])
def test_open_report_uses_local_file_url_and_browser_failure_keeps_saved_report(
    saved_report, tmp_path, monkeypatch, capsys, browser_available
):
    import sys

    from thesis_research import cli

    run = saved_report(model_ids=("laya",))
    output = tmp_path / "view"
    opened = []

    def open_browser(url):
        opened.append(url)
        return browser_available

    monkeypatch.setattr(cli.webbrowser, "open", open_browser)
    monkeypatch.setattr(
        sys,
        "argv",
        ["thesis-research", "report", str(run), "--output-dir", str(output), "--open-report"],
    )
    cli.main()
    captured = capsys.readouterr()
    assert opened == [(output / "index.html").as_uri()]
    assert (output / "index.html").is_file()
    assert ("Could not open dashboard automatically" in captured.err) != browser_available
    assert json.loads((run / "metadata.json").read_text())["status"] == "completed"
