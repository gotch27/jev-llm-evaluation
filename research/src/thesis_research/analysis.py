"""Read completed Choice benchmarks and export descriptive tables and figures.

Install the optional ``analysis`` dependency group. This module reads saved
reports and never calls a model or changes the benchmark that produced them.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from thesis_research.run_storage import file_checksum, git_state, write_json

MODEL_NAMES = {
    "jev": "Jev",
    "gpt-5-6-luna": "GPT-5.6 Luna",
    "gemini-2-5-flash-lite": "Gemini 2.5 Flash Lite",
    "qwen3-5-flash": "Qwen 3.5 Flash",
}
MODEL_COLORS = {
    "jev": "#0072B2",
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


@dataclass
class BenchmarkAnalysis:
    """One completed run, with model order taken from its frozen report."""

    run: Path
    metadata: dict[str, Any]
    cohort: dict[str, Any]
    summary: dict[str, Any]
    metrics: pd.DataFrame
    per_label: pd.DataFrame
    pairs: pd.DataFrame
    outcomes: pd.DataFrame

    @property
    def model_ids(self) -> list[str]:
        return self.metrics["model_id"].tolist()

    @property
    def caption(self) -> str:
        split = self.cohort["split"]
        purpose = "development / model selection" if split == "train" else "held-out split"
        return (
            f"{self.cohort['dataset_id'].upper()} | {split} | "
            f"n = {self.cohort['count']:,} | {purpose}"
        )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark(run: Path) -> BenchmarkAnalysis:
    """Load completed reports, rejecting partial or mismatched Choice cohorts."""
    run = run.resolve()
    metadata = _json(run / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("Analysis requires a completed benchmark with a final report")
    cohort = _json(run / "cohort.json")
    summary = _json(run / "report" / "summary.json")
    if cohort.get("question_type") != "choice" or summary.get("question_type") != "choice":
        raise ValueError("This first analysis workflow supports Choice benchmarks")
    tables = [
        pd.DataFrame(
            json.loads(line)
            for line in (run / "report" / f"{name}.jsonl").read_text().splitlines()
            if line.strip()
        )
        for name in REPORT_TABLES
    ]
    data = BenchmarkAnalysis(run, metadata, cohort, summary, *tables)
    _validate_cohort(data)
    return data


def _validate_cohort(data: BenchmarkAnalysis) -> None:
    examples = data.cohort["examples"]
    references = {row["id"]: row["reference"]["label"] for row in examples}
    if not references or len(references) != len(examples) or len(examples) != data.cohort["count"]:
        raise ValueError("Saved cohort has duplicate IDs or an inconsistent count")
    models = data.model_ids
    if not models or len(set(models)) != len(models):
        raise ValueError("Report has missing or duplicate model IDs")
    if set(data.outcomes["model_id"]) != set(models):
        raise ValueError("Outcome models do not match the metric report")
    if set(data.summary["models"]) != set(models):
        raise ValueError("Summary models do not match the metric report")
    if data.outcomes.duplicated(["model_id", "id"]).any():
        raise ValueError("Report has duplicate model/example outcomes")
    expected_support = pd.Series(list(references.values())).value_counts().to_dict()
    if set(data.per_label["model_id"]) != set(models):
        raise ValueError("Per-label models do not match the metric report")
    for model in models:
        rows = data.outcomes.loc[data.outcomes["model_id"] == model]
        if set(rows["id"]) != set(references):
            raise ValueError(f"Model {model} does not cover exactly the frozen cohort")
        if not rows["reference_label"].equals(rows["id"].map(references)):
            raise ValueError(f"Model {model} has references that differ from the cohort")
        expected_correct = rows["status"].eq("valid") & rows["predicted_label"].eq(
            rows["reference_label"]
        )
        if not rows["correct"].equals(expected_correct):
            raise ValueError(f"Model {model} has inconsistent correctness flags")
        metric = data.metrics.loc[data.metrics["model_id"] == model].iloc[0]
        if (
            metric["total"] != len(references)
            or metric["correct"] != expected_correct.sum()
            or not np.isclose(metric["accuracy"], expected_correct.mean())
        ):
            raise ValueError(f"Model {model} has metrics inconsistent with its outcomes")
        per_label = data.per_label.loc[data.per_label["model_id"] == model]
        reported_support = dict(zip(per_label["label"], per_label["support"], strict=True))
        expected_labels = set(data.summary["models"][model]["per_label"])
        if (
            per_label["label"].duplicated().any()
            or set(reported_support) != expected_labels
            or not set(expected_support).issubset(expected_labels)
            or reported_support
            != {label: expected_support.get(label, 0) for label in expected_labels}
        ):
            raise ValueError(f"Model {model} has inconsistent per-label support")
    baseline = data.summary["baseline_model_id"]
    if baseline not in models or set(data.pairs["comparison_model_id"]) != set(models) - {baseline}:
        raise ValueError("Paired comparisons do not match the metric report")
    if data.pairs["comparison_model_id"].duplicated().any():
        raise ValueError("Report has duplicate paired comparisons")
    correct = data.outcomes.pivot(index="id", columns="model_id", values="correct")
    for row in data.pairs.to_dict("records"):
        a, b = correct[baseline], correct[row["comparison_model_id"]]
        counts = {
            "both_correct": (a & b).sum(),
            "baseline_only_correct": (a & ~b).sum(),
            "comparison_only_correct": (~a & b).sum(),
            "neither_correct": (~a & ~b).sum(),
        }
        if (
            row["baseline_model_id"] != baseline
            or row["total"] != len(references)
            or any(row[field] != count for field, count in counts.items())
        ):
            raise ValueError("Paired counts disagree with the saved outcomes")


def model_table(data: BenchmarkAnalysis) -> pd.DataFrame:
    """Keep unavailable costs missing, and include failures in accuracy totals."""
    table = data.metrics[
        [
            "model_id",
            "model",
            "provider",
            "reasoning_effort",
            "total",
            "correct",
            "accuracy",
            "macro_f1",
            "valid",
            "missing",
            "invalid",
            "failed",
            "all_decision_latency_p50_seconds",
            "all_decision_latency_p95_seconds",
            "cost_usd",
        ]
    ].copy()
    table["cost_usd_per_1000_examples"] = table["cost_usd"] / table["total"] * 1000
    return table


def pairwise_table(data: BenchmarkAnalysis) -> pd.DataFrame:
    """Add a reference-informed upper bound, not a measured combined system."""
    table = data.pairs.copy()
    table["oracle_accuracy_upper_bound"] = 1 - table["neither_correct"] / table["total"]
    return table


def confusion_pairs(data: BenchmarkAnalysis) -> pd.DataFrame:
    """Count valid but incorrect label pairs; failed predictions stay separate."""
    wrong = data.outcomes.loc[data.outcomes["status"].eq("valid") & ~data.outcomes["correct"]]
    return (
        wrong.groupby(["model_id", "reference_label", "predicted_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "model_id", "reference_label"], ascending=[False, True, True])
    )


def plot_figures(data: BenchmarkAnalysis) -> dict[str, Figure]:
    """Create descriptive figures with consistent order, labels, and scope."""
    names = [MODEL_NAMES.get(model, model) for model in data.model_ids]
    colors = [MODEL_COLORS.get(model, f"C{i}") for i, model in enumerate(data.model_ids)]
    metrics = data.metrics.set_index("model_id").loc[data.model_ids]
    figures = {}
    with plt.rc_context(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        positions = np.arange(len(names))
        for offset, field, label, color in (
            (-0.19, "accuracy", "Accuracy", "#0072B2"),
            (0.19, "macro_f1", "Macro-F1", "#E69F00"),
        ):
            bars = ax.bar(positions + offset, metrics[field] * 100, 0.38, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(
            xticks=positions,
            xticklabels=names,
            ylim=(0, 108),
            ylabel="Score (%)",
            title=f"Classification quality\n{data.caption}",
        )
        ax.legend(loc="lower right")
        figures["classification_quality"] = fig

        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for model, name, color in zip(data.model_ids, names, colors, strict=True):
            latency = data.outcomes.loc[data.outcomes["model_id"] == model, "latency_seconds"]
            latency = latency.dropna().sort_values().to_numpy()
            if not len(latency) or not np.isfinite(latency).all() or (latency <= 0).any():
                raise ValueError(f"Model {model} has missing or nonpositive measured latency")
            ax.step(
                latency,
                np.arange(1, len(latency) + 1) / len(latency),
                where="post",
                label=f"{name} (n={len(latency)})",
                color=color,
            )
        ax.set(
            xscale="log",
            xlabel="Decision latency (seconds, log scale)",
            ylabel="Fraction of measured decisions",
            ylim=(0, 1.02),
            title=f"Latency distribution, including failed decisions\n{data.caption}",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        figures["latency_ecdf"] = fig

        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for model, name, color in zip(data.model_ids, names, colors, strict=True):
            row = metrics.loc[model]
            x, y = row["all_decision_latency_p50_seconds"], row["accuracy"] * 100
            ax.scatter(x, y, color=color, s=80)
            ax.annotate(name, (x, y), xytext=(8, 6), textcoords="offset points")
        ax.set(
            xlabel="Median decision latency (seconds)",
            ylabel="Accuracy (%)",
            ylim=(0, 100),
            title=f"Accuracy and latency\n{data.caption}",
        )
        ax.margins(x=0.28)
        ax.grid(alpha=0.2)
        figures["accuracy_latency"] = fig

        heatmap = data.per_label.pivot(index="label", columns="model_id", values="f1")
        heatmap = heatmap.reindex(columns=data.model_ids).sort_index()
        fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap) * 0.23)), layout="constrained")
        image = ax.imshow(heatmap, vmin=0, vmax=1, cmap="cividis", aspect="auto")
        ax.set(
            xticks=range(len(names)),
            xticklabels=names,
            yticks=range(len(heatmap)),
            yticklabels=heatmap.index,
            title=f"F1 by intent\n{data.caption}",
        )
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(image, ax=ax, shrink=0.5, label="F1")
        figures["intent_f1"] = fig

        pairs = data.pairs.set_index("comparison_model_id")
        pair_names = [MODEL_NAMES.get(model, model) for model in pairs.index]
        baseline = MODEL_NAMES.get(
            data.summary["baseline_model_id"], data.summary["baseline_model_id"]
        )
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        left = np.zeros(len(pairs))
        for field, label, color in (
            ("both_correct", "Both correct", "#009E73"),
            ("baseline_only_correct", f"Only {baseline} correct", "#0072B2"),
            ("comparison_only_correct", "Only LLM correct", "#E69F00"),
            ("neither_correct", "Neither correct", "#999999"),
        ):
            values = pairs[field].to_numpy()
            bars = ax.barh(pair_names, values, left=left, label=label, color=color)
            ax.bar_label(
                bars,
                labels=[str(value) if value else "" for value in values],
                label_type="center",
                fontsize=9,
            )
            left += values
        ax.set(xlabel="Examples", title=f"Paired correctness with {baseline}\n{data.caption}")
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
        figures["paired_correctness"] = fig
    return figures


def export_analysis(data: BenchmarkAnalysis, destination: Path) -> Path:
    """Save CSVs, vector PDFs, 300-dpi PNGs, and source/software provenance."""
    destination = destination.resolve()
    if destination.is_relative_to(data.run):
        raise ValueError("Analysis outputs must be outside the source benchmark directory")
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "model_comparison": model_table(data),
        "per_label_metrics": data.per_label,
        "paired_comparison": pairwise_table(data),
        "confusion_pairs": confusion_pairs(data),
    }
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    for name, figure in plot_figures(data).items():
        for suffix in ("pdf", "png"):
            figure.savefig(destination / f"{name}.{suffix}", dpi=300)
        plt.close(figure)
    sources = ["metadata.json", "plan.toml", "task.toml", "cohort.json", "report/summary.json"]
    sources += [f"report/{name}.jsonl" for name in REPORT_TABLES]
    write_json(
        destination / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_run": str(data.run),
            "caption": data.caption,
            "source_sha256": {name: file_checksum(data.run / name) for name in sources},
            "analysis_code": git_state(),
            "analysis_module_sha256": file_checksum(Path(__file__)),
            "software": {name: version(name) for name in ("pandas", "numpy", "matplotlib")},
            "notes": [
                "Descriptive point estimates; no confidence intervals are plotted.",
                "Training runs are development results, not held-out evaluation.",
                "Missing provider costs remain unavailable, never zero.",
                "McNemar p-values are exploratory and unadjusted for multiple comparisons.",
                "Oracle accuracy uses reference labels; it is not a measured combined system.",
                "Latency reflects the recorded runner, gateway, execution settings, "
                "and model order.",
            ],
        },
    )
    return destination


def main() -> None:
    """Export the same analysis without opening Jupyter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed benchmark directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = load_benchmark(args.run)
    output = args.output_dir or Path("outputs/analysis") / data.run.name
    print(export_analysis(data, output))


if __name__ == "__main__":
    main()
