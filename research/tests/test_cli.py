"""Command-line application behavior."""

import os
import sys
from pathlib import Path

import pytest

from thesis_research import cli


@pytest.mark.parametrize(
    ("existing_value", "expected_value"),
    [(None, "file-key"), ("external-key", "external-key")],
)
def test_cli_loads_dotenv_without_overriding_environment(
    tmp_path,
    monkeypatch,
    existing_value,
    expected_value,
):
    """Load the local .env before parsing while respecting an external value."""
    (tmp_path / ".env").write_text("AI_GATEWAY_API_KEY=file-key\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    if existing_value is not None:
        monkeypatch.setenv("AI_GATEWAY_API_KEY", existing_value)

    class EnvironmentLoaded(Exception):
        pass

    def verify_environment():
        assert os.environ["AI_GATEWAY_API_KEY"] == expected_value
        raise EnvironmentLoaded

    monkeypatch.setattr(cli, "_build_parser", verify_environment)
    with pytest.raises(EnvironmentLoaded):
        cli.main()


def test_cli_passes_terminal_observer_to_benchmark(monkeypatch, capsys):
    terminal_observer = object()

    class FakeTerminal:
        def __enter__(self):
            return terminal_observer

        def __exit__(self, exc_type, exc, traceback):
            pass

    async def fake_run_benchmark(plan, data, outputs, *, observer, runner_location):
        assert plan == Path("plan.toml")
        assert data == Path("data")
        assert outputs == Path("outputs/benchmarks")
        assert observer is terminal_observer
        assert runner_location == "local-mac-oslo"
        return Path("outputs/benchmarks/run")

    monkeypatch.setattr(cli, "BenchmarkTerminal", FakeTerminal)
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "thesis-research",
            "benchmark",
            "--plan",
            "plan.toml",
            "--runner-location",
            "local-mac-oslo",
        ],
    )

    cli.main()

    assert capsys.readouterr().out.strip() == "outputs/benchmarks/run"


def test_cli_passes_retry_errors_to_resume(monkeypatch, capsys):
    terminal_observer = object()

    class FakeTerminal:
        def __enter__(self):
            return terminal_observer

        def __exit__(self, exc_type, exc, traceback):
            pass

    async def fake_resume(run, data, *, observer, runner_location, retry_errors):
        assert run == Path("outputs/benchmarks/run")
        assert data == Path("data")
        assert observer is terminal_observer
        assert runner_location == "local-mac-oslo"
        assert retry_errors is True
        return run

    monkeypatch.setattr(cli, "BenchmarkTerminal", FakeTerminal)
    monkeypatch.setattr(cli, "resume_benchmark", fake_resume)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "thesis-research",
            "benchmark",
            "--resume",
            "outputs/benchmarks/run",
            "--retry-errors",
            "--runner-location",
            "local-mac-oslo",
        ],
    )

    cli.main()

    assert capsys.readouterr().out.strip() == "outputs/benchmarks/run"
