"""Rich terminal presentation for coordinated benchmark progress."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self, Sequence

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Column

from thesis_research.benchmark.progress import BenchmarkObserver, ModelProgress


class BenchmarkTerminal(BenchmarkObserver):
    """Render model progress while keeping event logs in terminal scrollback."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        show_progress: bool | None = None,
    ) -> None:
        self.console = console or Console(stderr=True)
        self.show_progress = self.console.is_terminal if show_progress is None else show_progress
        self._tasks: dict[str, int] = {}
        self._progress = Progress(
            TextColumn(
                "{task.description}",
                table_column=Column(width=22, no_wrap=True, overflow="ellipsis"),
            ),
            TextColumn(
                "{task.fields[status]}",
                table_column=Column(width=9, no_wrap=True),
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("valid={task.fields[valid]:>4}"),
            TextColumn("errors={task.fields[failed]:>3}"),
            TextColumn("{task.fields[rate]:>9}"),
            TextColumn("elapsed={task.fields[elapsed]:>8}"),
            TextColumn("ETA={task.fields[eta]:>8}"),
            console=self.console,
            disable=not self.show_progress,
            expand=True,
            refresh_per_second=4,
        )

    def __enter__(self) -> Self:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def benchmark_started(
        self,
        name: str,
        run: Path,
        model_ids: Sequence[str],
        total_examples: int,
    ) -> None:
        self._log("cyan", f"Starting {name}: {total_examples} examples, {len(model_ids)} models")
        self._log("dim", f"Run record: {run}")
        initial = ModelProgress(total_examples, 0, 0, 0, 0.0)
        for model_id in model_ids:
            self._tasks[model_id] = self._progress.add_task(
                model_id,
                total=total_examples,
                start=False,
                status="[yellow]waiting[/yellow]",
                **_progress_fields(initial),
            )

    def model_started(self, model_id: str, progress: ModelProgress) -> None:
        task_id = self._tasks[model_id]
        self._update(model_id, progress, "[cyan]running[/cyan]")
        self._progress.start_task(task_id)
        self._log("cyan", f"{model_id} started at {progress.completed}/{progress.total}")

    def model_progress(self, model_id: str, progress: ModelProgress) -> None:
        self._update(model_id, progress, "[cyan]running[/cyan]")

    def prediction_failed(self, model_id: str, example_id: str, error: str) -> None:
        message = _truncate(error.replace("\n", " "), 180)
        self._log("yellow", f"{model_id} · {example_id} · {message}")

    def model_completed(self, model_id: str, progress: ModelProgress) -> None:
        self._update(model_id, progress, "[green]done[/green]")
        self._progress.stop_task(self._tasks[model_id])
        self._log(
            "green",
            f"{model_id} completed {progress.completed}/{progress.total} · "
            f"valid={progress.valid} · errors={progress.failed} · "
            f"elapsed={_format_duration(progress.elapsed_seconds)}",
        )

    def model_failed(self, model_id: str, error: BaseException) -> None:
        if model_id in self._tasks:
            self._progress.update(self._tasks[model_id], status="[red]failed[/red]")
            self._progress.stop_task(self._tasks[model_id])
        self._log("red", f"{model_id} stopped: {type(error).__name__}: {error}")

    def report_started(self) -> None:
        self._log("cyan", "All model runs completed; generating comparison report")

    def benchmark_completed(self, run: Path) -> None:
        self._log("green", f"Benchmark completed: {run}")

    def benchmark_failed(self, error: BaseException) -> None:
        self._log("red", f"Benchmark stopped: {type(error).__name__}: {error}")

    def _update(self, model_id: str, progress: ModelProgress, status: str) -> None:
        self._progress.update(
            self._tasks[model_id],
            total=progress.total,
            completed=progress.completed,
            status=status,
            **_progress_fields(progress),
        )

    def _log(self, style: str, message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [{style}]{escape(message)}[/{style}]")


def _progress_fields(progress: ModelProgress) -> dict[str, str | int]:
    rate = progress.completed / progress.elapsed_seconds if progress.elapsed_seconds > 0 else 0.0
    remaining = max(0, progress.total - progress.completed)
    eta = remaining / rate if rate > 0 else None
    return {
        "valid": progress.valid,
        "failed": progress.failed,
        "rate": f"{rate:.2f}/s" if rate else "--/s",
        "elapsed": _format_duration(progress.elapsed_seconds),
        "eta": _format_duration(eta) if eta is not None else "--:--",
    }


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{remainder:02.0f}s"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"
