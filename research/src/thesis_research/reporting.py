"""Readable summaries and standalone dashboards from completed benchmark records.

This module uses only the standard library. It reads the same saved metric tables
as notebook analysis, without recalculating predictions or calling providers.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Callable

from thesis_research.run_storage import file_checksum, git_state, write_json

MODEL_NAMES = {
    "jev": "Jev",
    "laya": "Laya",
    "liquid-d1": "Liquid d1",
    "gpt-5-6-luna": "GPT-5.6 Luna",
    "gemini-2-5-flash-lite": "Gemini 2.5 Flash Lite",
    "qwen3-5-flash": "Qwen 3.5 Flash",
}
MODEL_COLORS = {
    "jev": "#0072B2",
    "laya": "#56B4E9",
    "liquid-d1": "#E69F00",
    "gpt-5-6-luna": "#D55E00",
    "gemini-2-5-flash-lite": "#009E73",
    "qwen3-5-flash": "#CC79A7",
}
REPORT_TABLES = (
    "model_metrics",
    "per_label_metrics",
    "pairwise_statistics",
    "model_outcomes",
)
TASK_METRICS = {
    "choice": (),
    "noul": (
        ("brier_score", "Brier score"),
        ("log_loss", "Log loss"),
        ("probability_coverage", "Probability coverage"),
    ),
    "score": (
        ("mean_absolute_error", "MAE"),
        ("root_mean_squared_error", "RMSE"),
        ("quadratic_weighted_kappa", "Weighted kappa"),
        ("numeric_coverage", "Numeric coverage"),
    ),
}


@dataclass(frozen=True)
class SavedBenchmarkReport:
    """Completed benchmark tables, shared by terminal, dashboard and notebooks."""

    run: Path
    metadata: dict[str, Any]
    cohort: dict[str, Any]
    summary: dict[str, Any]
    metrics: list[dict[str, Any]]
    per_label: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    outcomes: list[dict[str, Any]]

    @property
    def question_type(self) -> str:
        return self.summary["question_type"]

    @property
    def caption(self) -> str:
        return (
            f"{self.cohort['dataset_id'].upper()} · {self.cohort['split']} · "
            f"{self.cohort['count']:,} "
            f"{'example' if self.cohort['count'] == 1 else 'examples'} · {self.question_type}"
        )


def load_saved_report(run: Path) -> SavedBenchmarkReport:
    """Read completed reports, including historical plans, without changing them."""
    run = run.resolve()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("Reporting requires a completed benchmark with a final report")
    cohort = json.loads((run / "cohort.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "report/summary.json").read_text(encoding="utf-8"))
    tables = [
        [
            json.loads(line)
            for line in (run / f"report/{name}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for name in REPORT_TABLES
    ]
    data = SavedBenchmarkReport(run, metadata, cohort, summary, *tables)
    model_ids = [row["model_id"] for row in data.metrics]
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("Report has missing or duplicate model IDs")
    if data.question_type not in TASK_METRICS or cohort.get("question_type") != data.question_type:
        raise ValueError("Report has an unsupported or inconsistent question type")
    baseline = summary["baseline_model_id"]
    if baseline not in model_ids:
        raise ValueError("Report baseline does not identify a configured model")
    comparison_ids = [row["comparison_model_id"] for row in data.pairs]
    if (
        set(comparison_ids) != set(model_ids) - {baseline}
        or len(set(comparison_ids)) != len(comparison_ids)
        or any(row["baseline_model_id"] != baseline for row in data.pairs)
    ):
        raise ValueError("Paired comparisons do not match the metric report")
    return data


def model_name(model_id: str) -> str:
    return MODEL_NAMES.get(model_id, model_id)


def format_number(value: object, *, percentage: bool = False, digits: int = 3) -> str:
    """Render unavailable measurements explicitly, preserving reported zero."""
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:.1%}" if percentage else f"{number:.{digits}f}"


def _format_p_value(value: object) -> str:
    number = _number(value)
    return "unknown" if number is None else f"{number:.4g}"


def generate_dashboard(run: Path, destination: Path | None = None) -> Path:
    """Write one HTML file and its source checksums after measurement completes.

    A destination outside the source run supports immutable historical outputs.
    New benchmarks use their own ``report/index.html`` by default.
    """
    data = load_saved_report(run)
    destination = destination.resolve() if destination else data.run / "report/index.html"
    if destination.is_relative_to(data.run) and destination != data.run / "report/index.html":
        raise ValueError("A dashboard inside the source run must use report/index.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = _document(data)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)
    sources = ["metadata.json", "cohort.json", "report/summary.json"]
    sources += [f"report/{name}.jsonl" for name in REPORT_TABLES]
    for name in ("plan.toml", "task.toml"):
        if (data.run / name).exists():
            sources.append(name)
    write_json(
        destination.with_name("visualization.json"),
        {
            "status": "completed",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_run": str(data.run),
            "source_sha256": {name: file_checksum(data.run / name) for name in sources},
            "benchmark_code": data.metadata.get("code"),
            "reporting_code": git_state(),
            "dashboard": destination.name,
            "dashboard_sha256": file_checksum(destination),
            "model_order": [row["model_id"] for row in data.metrics],
            "baseline_model_id": data.summary["baseline_model_id"],
        },
    )
    return destination


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    heading = "".join(f"<th scope='col'>{_escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        f"<div class='table-scroll'><table><thead><tr>{heading}</tr></thead><tbody>"
        f"{body}</tbody></table></div>"
    )


def _metrics_table(data: SavedBenchmarkReport) -> str:
    headers = [
        "Model",
        "N",
        "Accuracy",
        "Macro-F1",
        "Failed",
        "Invalid",
        "Missing",
        "Cost (USD)",
        "p50 (s)",
        "p95 (s)",
    ]
    rows = []
    for row in data.metrics:
        model_id = row["model_id"]
        baseline = (
            " <span class='badge'>baseline</span>"
            if model_id == data.summary["baseline_model_id"]
            else ""
        )
        cost = format_number(row.get("cost_usd"), digits=8)
        cells = [
            _escape(model_name(model_id)) + baseline,
            _escape(row["total"]),
            format_number(row.get("accuracy"), percentage=True),
            format_number(row.get("macro_f1")),
            *[_escape(row.get(field, 0)) for field in ("failed", "invalid", "missing")],
            cost,
            format_number(row.get("all_decision_latency_p50_seconds")),
            format_number(row.get("all_decision_latency_p95_seconds")),
        ]
        rows.append(cells)
    return _table(headers, rows)


def _svg(content: str, height: int, title: str) -> str:
    return (
        f"<svg viewBox='0 0 820 {height}' role='img' aria-label='{_escape(title)}'>"
        f"<title>{_escape(title)}</title>{content}</svg>"
    )


def _bar_chart(
    rows: list[dict[str, Any]],
    fields: list[tuple[str, str, str]],
    title: str,
    formatter: Callable[[object], str],
    maximum: float | None = None,
) -> str:
    known = [_number(row.get(field)) for row in rows for field, _, _ in fields]
    limit = maximum or max((value for value in known if value is not None), default=0) or 1
    group_height = 26 * len(fields) + 18
    content = ""
    for index, row in enumerate(rows):
        y = 26 + index * group_height
        name = model_name(row["model_id"])
        content += f"<text x='0' y='{y + 13}' class='model-label'>{_escape(name)}</text>"
        for offset, (field, label, color) in enumerate(fields):
            value = _number(row.get(field))
            position = y + offset * 26
            width = max(0, value / limit * 470) if value is not None else 0
            content += (
                f"<rect x='205' y='{position}' width='470' height='16' rx='3' fill='#edf1f6'/>"
            )
            if value is not None:
                content += (
                    f"<rect x='205' y='{position}' width='{width:.2f}' height='16' rx='3' "
                    f"fill='{color}'>"
                    f"<title>{_escape(name)} · {_escape(label)}: "
                    f"{_escape(formatter(value))}</title></rect>"
                )
            content += f"<text x='690' y='{position + 13}'>{_escape(formatter(value))}</text>"
    content += f"<text x='205' y='{len(rows) * group_height + 34}'>0</text>"
    if maximum is not None or any(value is not None and value > 0 for value in known):
        content += (
            f"<text x='675' y='{len(rows) * group_height + 34}' text-anchor='end'>"
            f"{_escape(formatter(limit))}</text>"
        )
    legend = "".join(
        f"<span><i style='background:{color}'></i>{_escape(label)}</span>"
        for _, label, color in fields
    )
    return (
        _svg(content, len(rows) * group_height + 58, title) + f"<div class='legend'>{legend}</div>"
    )


def _stacked_chart(
    rows: list[dict[str, Any]], fields: list[tuple[str, str, str]], title: str
) -> str:
    content = ""
    for index, row in enumerate(rows):
        y = 22 + index * 48
        name = model_name(row["model_id"])
        content += f"<text x='0' y='{y + 18}' class='model-label'>{_escape(name)}</text>"
        left = 205.0
        total = row["total"] or 1
        for field, label, color in fields:
            value = row.get(field, 0)
            width = value / total * 470
            content += (
                f"<rect x='{left:.2f}' y='{y}' width='{width:.2f}' height='26' fill='{color}'>"
                f"<title>{_escape(name)} · {_escape(label)}: {_escape(value)} / "
                f"{_escape(total)}</title></rect>"
            )
            if width > 28:
                content += (
                    f"<text x='{left + width / 2:.2f}' y='{y + 18}' text-anchor='middle' "
                    f"fill='white'>{_escape(value)}</text>"
                )
            left += width
        content += f"<text x='690' y='{y + 18}'>n = {_escape(row['total'])}</text>"
    legend = "".join(
        f"<span><i style='background:{color}'></i>{_escape(label)}</span>"
        for _, label, color in fields
    )
    return _svg(content, len(rows) * 48 + 25, title) + f"<div class='legend'>{legend}</div>"


def _latency_chart(data: SavedBenchmarkReport) -> str:
    series = {}
    for row in data.metrics:
        values = [
            _number(outcome.get("latency_seconds"))
            for outcome in data.outcomes
            if outcome["model_id"] == row["model_id"]
        ]
        series[row["model_id"]] = sorted(
            value for value in values if value is not None and value >= 0
        )
    limit = max((value for values in series.values() for value in values), default=0) or 1
    content = ""
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = 270 - fraction * 240
        x = 65 + fraction * 685
        content += f"<line x1='65' y1='{y}' x2='750' y2='{y}' stroke='#e2e8f0'/>"
        content += f"<text x='52' y='{y + 5}' text-anchor='end'>{fraction:.0%}</text>"
        content += f"<text x='{x}' y='298' text-anchor='middle'>{limit * fraction:.3f}s</text>"
    legend = ""
    for index, (model_id, values) in enumerate(series.items()):
        color = MODEL_COLORS.get(model_id, ("#475569", "#7c3aed", "#059669")[index % 3])
        legend += (
            f"<span><i style='background:{color}'></i>{_escape(model_name(model_id))} "
            f"(n={len(values)})</span>"
        )
        if not values:
            continue
        previous = count = 0
        points = []
        for value, group in groupby(values):
            count += len(list(group))
            x = 65 + value / limit * 685
            points += [
                f"{x:.2f},{270 - previous / len(values) * 240:.2f}",
                f"{x:.2f},{270 - count / len(values) * 240:.2f}",
            ]
            previous = count
        points.append("750,30")
        content += (
            f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' "
            f"stroke-width='2.5'>"
            f"<title>{_escape(model_name(model_id))}: {len(values)} recorded "
            f"latencies</title></polyline>"
        )
    return (
        _svg(content, 320, "Decision latency distribution: cumulative fraction of examples")
        + f"<div class='legend'>{legend}</div>"
    )


def _class_table(data: SavedBenchmarkReport) -> str:
    models = [row["model_id"] for row in data.metrics]
    grouped: dict[str, dict[str, Any]] = {}
    for row in data.per_label:
        grouped.setdefault(str(row["label"]), {})[row["model_id"]] = row
    rows = []
    for label, metrics in grouped.items():
        support = next(iter(metrics.values()))["support"]
        cells = [_escape(label), _escape(support)]
        for model in models:
            row = metrics.get(model, {})
            value = _number(row.get("f1"))
            if not row.get("support") or value is None:
                cells.append("<span class='muted'>—</span>")
            else:
                opacity = 0.08 + 0.5 * value
                cells.append(
                    f"<span class='heat-cell' style='background:rgba(0,114,178,{opacity:.3f})'>"
                    f"{value:.3f}</span>"
                )
        rows.append(cells)
    return _table(["Class", "Support", *[model_name(model) + " F1" for model in models]], rows)


def _document(data: SavedBenchmarkReport) -> str:
    def card(title: str, content: str, full: bool = False) -> str:
        return (
            f"<section class='card{' full' if full else ''}'><h2>{_escape(title)}</h2>"
            f"{content}</section>"
        )

    def percent(value: object) -> str:
        return format_number(value, percentage=True)

    cards = card("Model statistics", _metrics_table(data), full=True)
    extra_metrics = TASK_METRICS[data.question_type]
    if extra_metrics:
        rows = [
            [
                _escape(model_name(row["model_id"])),
                *[
                    format_number(row.get(field), percentage=field.endswith("coverage"))
                    for field, _ in extra_metrics
                ],
            ]
            for row in data.metrics
        ]
        cards += card(
            "Task-specific statistics",
            _table(["Model", *[label for _, label in extra_metrics]], rows),
            full=True,
        )
    cards += card(
        "Accuracy and macro-F1",
        _bar_chart(
            data.metrics,
            [("accuracy", "Accuracy", "#0072B2"), ("macro_f1", "Macro-F1", "#E69F00")],
            "Accuracy and macro-F1",
            percent,
            1,
        ),
    )
    cards += card(
        "Prediction status",
        _stacked_chart(
            data.metrics,
            [
                ("valid", "Valid", "#0072B2"),
                ("failed", "Failed", "#b45309"),
                ("invalid", "Invalid", "#be123c"),
                ("missing", "Missing", "#64748b"),
            ],
            "Prediction status",
        ),
    )
    cards += card(
        "Decision latency distribution",
        "<p class='muted'>Fraction of examples completed within each duration; "
        "includes failed decisions with recorded latency.</p>" + _latency_chart(data),
    )
    cards += card(
        "Reported cost",
        _bar_chart(
            data.metrics,
            [("cost_usd", "USD for this cohort", "#0072B2")],
            "Reported cost",
            lambda value: format_number(value, digits=8),
        )
        + "<p class='muted'>Unknown means unreported; an explicitly reported zero stays zero.</p>",
    )
    if data.pairs:
        rows = [{**row, "model_id": row["comparison_model_id"]} for row in data.pairs]
        baseline = model_name(data.summary["baseline_model_id"])
        cards += card(
            f"Paired correctness · baseline: {baseline}",
            _stacked_chart(
                rows,
                [
                    ("both_correct", "Both correct", "#0072B2"),
                    ("baseline_only_correct", "Only baseline correct", "#059669"),
                    ("comparison_only_correct", "Only comparison correct", "#b45309"),
                    ("neither_correct", "Neither correct", "#64748b"),
                ],
                "Paired correctness",
            ),
            full=True,
        )
        cells = [
            [
                _escape(model_name(row["comparison_model_id"])),
                format_number(row.get("accuracy_difference"), percentage=True),
                _format_p_value(row.get("mcnemar_exact_two_sided_p_value")),
            ]
            for row in data.pairs
        ]
        cards += card(
            "Paired comparison statistics",
            _table(["Comparison model", "Accuracy difference", "McNemar p-value"], cells),
            full=True,
        )
    else:
        cards += card(
            "Single-model run",
            "<p>No paired comparisons. The sole configured model is the baseline.</p>",
            full=True,
        )
    cards += card(
        "F1 by reference class",
        "<p class='muted'>Classes without reference examples show —. Macro-F1 uses "
        "the configured class set.</p>" + _class_table(data),
        full=True,
    )
    unsuccessful = [row for row in data.outcomes if row["status"] != "valid"]
    if unsuccessful:
        rows = [
            [
                _escape(model_name(row["model_id"])),
                _escape(row["id"]),
                _escape(row["status"]),
                _escape(row.get("error") or "No valid prediction recorded"),
            ]
            for row in unsuccessful
        ]
        cards += card(
            f"Unsuccessful predictions · {len(unsuccessful)}",
            _table(["Model", "Example", "Status", "Error"], rows),
            full=True,
        )
    name = data.metadata.get("name", data.run.name)
    benchmark_code = data.metadata.get("code", {})
    code = benchmark_code.get("commit") or "unknown"
    if benchmark_code.get("dirty"):
        code += " (working tree had changes)"
    style = Path(__file__).with_name("reporting.css").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>{_escape(name)} · Benchmark report</title>
<style>{style}</style></head><body><main><header>
<div class="eyebrow">Benchmark results</div><h1>{_escape(name)}</h1>
<p>{_escape(data.caption)}</p>
<p class="muted">Baseline: {_escape(model_name(data.summary["baseline_model_id"]))}
· {len(data.metrics)} models
· Started {_escape(data.metadata.get("timestamp", "unknown"))}</p>
<p class="muted">Descriptive statistics for this frozen cohort.
All examples, including unsuccessful predictions, remain in accuracy totals.</p></header>
<div class="grid">{cards}</div>
<footer>Source run: {_escape(data.run.name)}<br>Benchmark code: {_escape(code)}<br>
Generated from saved metric tables; no inference calls.
Source checksums and reporting provenance: visualization.json.</footer>
</main></body></html>"""
