"""Human-readable terminal progress without provider or benchmark side effects."""

from io import StringIO
from pathlib import Path

from rich.console import Console

from thesis_research.benchmark.progress import ModelProgress
from thesis_research.terminal import BenchmarkTerminal


def test_terminal_renders_model_progress_and_failure_logs():
    output = StringIO()
    console = Console(file=output, color_system=None, force_terminal=False, width=160)

    with BenchmarkTerminal(console=console, show_progress=True) as terminal:
        terminal.benchmark_started(
            "Synthetic benchmark",
            Path("outputs/run"),
            ["jev", "llm-a"],
            10,
        )
        terminal.model_started("jev", ModelProgress(10, 0, 0, 0, 0.0))
        terminal.model_progress("jev", ModelProgress(10, 6, 5, 1, 3.0))
        terminal.prediction_failed("jev", "train:7", "Bad [structured] output")
        terminal.model_completed("jev", ModelProgress(10, 10, 9, 1, 5.0))
        terminal.model_started("llm-a", ModelProgress(10, 0, 0, 0, 0.0))
        terminal.model_failed("llm-a", RuntimeError("provider unavailable"))
        terminal.benchmark_failed(RuntimeError("one model failed"))

    rendered = output.getvalue()
    assert "Starting Synthetic benchmark: 10 examples, 2 models" in rendered
    assert "jev · train:7 · Bad [structured] output" in rendered
    assert "jev completed 10/10 · valid=9 · errors=1 · elapsed=5.00s" in rendered
    assert "llm-a stopped: RuntimeError: provider unavailable" in rendered
    assert "Benchmark stopped: RuntimeError: one model failed" in rendered
    assert "valid=" in rendered
    assert "errors=" in rendered
    assert "elapsed=" in rendered
    assert "2.00/s" in rendered
