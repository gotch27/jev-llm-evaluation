"""Coordinate benchmark creation, model execution, recovery, and reporting."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thesis_research.benchmark.cohort import (
    cohort_record,
    examples_from_cohort,
    select_banking77_cohort,
)
from thesis_research.benchmark.model_run import run_model
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.progress import BenchmarkObserver, NullBenchmarkObserver
from thesis_research.benchmark.report import generate_report, report_artifacts
from thesis_research.benchmark.types import BenchmarkPlan
from thesis_research.clients import DecisionClient
from thesis_research.config import load_task_config
from thesis_research.datasets import Banking77Example, load_prepared_banking77, sha256_bytes
from thesis_research.run_storage import create_run_directory, file_checksum, git_state, write_json
from thesis_research.tasks import IntentClassificationTask, build_banking77_task


async def run_benchmark(
    plan_path: Path,
    data: Path,
    outputs: Path,
    *,
    observer: BenchmarkObserver | None = None,
    _clients: dict[str, DecisionClient] | None = None,
) -> Path:
    """Create and execute one Jev-to-LLM benchmark.

    The plan and task are validated before a run directory is created. After
    that point, every failure is recorded inside the unique benchmark
    directory so the same run can be resumed without repeating completed IDs.
    """
    observer = observer if observer is not None else NullBenchmarkObserver()
    plan = load_benchmark_plan(plan_path)
    plan_bytes = plan_path.read_bytes()
    task_bytes = plan.task_config.read_bytes()
    load_task_config(plan.task_config)

    run, timestamp = create_run_directory(outputs)
    (run / "plan.toml").write_bytes(plan_bytes)
    (run / "task.toml").write_bytes(task_bytes)
    metadata: dict[str, Any] = {
        "name": plan.name,
        "timestamp": timestamp.isoformat(),
        "working_directory": str(Path.cwd()),
        "code": git_state(),
        "status": "started",
        "source_plan_path": str(plan_path.resolve()),
        "plan": plan.as_dict(),
        "dataset": None,
        "attempts": [],
        "models": {},
        "inputs": {
            "plan.toml": {"sha256": sha256_bytes(plan_bytes)},
            "task.toml": {"sha256": sha256_bytes(task_bytes)},
        },
    }
    write_json(run / "metadata.json", metadata)
    try:
        config = load_task_config(run / "task.toml")
        labels, split_examples, manifest = load_prepared_banking77(
            data,
            config.dataset_revision,
            plan.cohort.split,
        )
        task = build_banking77_task(config, labels)
        examples = select_banking77_cohort(split_examples, labels, plan.cohort)
        write_json(
            run / "cohort.json",
            cohort_record(
                examples,
                dataset_revision=config.dataset_revision,
                split=plan.cohort.split,
                spec=plan.cohort,
            ),
        )
        metadata["inputs"]["cohort.json"] = {"sha256": file_checksum(run / "cohort.json")}
        metadata["dataset"] = manifest
        metadata["split"] = plan.cohort.split
        metadata["split_examples"] = len(split_examples)
        metadata["cohort_examples"] = len(examples)
        await _execute(run, plan, task, labels, examples, metadata, observer, _clients)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(run / "metadata.json", metadata)
        raise ValueError(f"Benchmark failed; preserved record at {run}: {error}") from error
    except BaseException as error:
        metadata["status"] = "interrupted"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(run / "metadata.json", metadata)
        raise
    return run


async def resume_benchmark(
    run: Path,
    data: Path,
    *,
    observer: BenchmarkObserver | None = None,
    _clients: dict[str, DecisionClient] | None = None,
) -> Path:
    """Resume only the missing model predictions in a benchmark directory."""
    observer = observer if observer is not None else NullBenchmarkObserver()
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _verify_saved_inputs(run, metadata)
    if metadata.get("status") == "completed":
        return run
    plan = load_benchmark_plan(run / "plan.toml", task_config_override=run / "task.toml")
    config = load_task_config(run / "task.toml")
    labels, split_examples, manifest = load_prepared_banking77(
        data,
        config.dataset_revision,
        plan.cohort.split,
    )
    examples = examples_from_cohort(
        json.loads((run / "cohort.json").read_text(encoding="utf-8")),
        split_examples,
        dataset_revision=config.dataset_revision,
        split=plan.cohort.split,
    )
    task = build_banking77_task(config, labels)
    metadata["dataset"] = manifest
    metadata.pop("error", None)
    try:
        await _execute(run, plan, task, labels, examples, metadata, observer, _clients)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(metadata_path, metadata)
        raise ValueError(f"Benchmark resume failed at {run}: {error}") from error
    except BaseException as error:
        metadata["status"] = "interrupted"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(metadata_path, metadata)
        raise
    return run


async def _execute(
    run: Path,
    plan: BenchmarkPlan,
    task: IntentClassificationTask,
    labels: list[str],
    examples: list[Banking77Example],
    metadata: dict[str, Any],
    observer: BenchmarkObserver,
    clients: dict[str, DecisionClient] | None,
) -> None:
    attempt = {
        "started_at": datetime.now(UTC).isoformat(),
        "code": git_state(),
        "status": "running",
        "command": _command(run, metadata),
    }
    metadata.setdefault("attempts", []).append(attempt)
    metadata["status"] = "running"
    write_json(run / "metadata.json", metadata)
    observer.benchmark_started(
        plan.name,
        run,
        [model.id for model in plan.models],
        len(examples),
    )
    semaphore = asyncio.Semaphore(plan.execution.model_concurrency)

    async def execute_model(model_index: int) -> dict[str, Any]:
        model = plan.models[model_index]
        async with semaphore:
            try:
                return await run_model(
                    model,
                    run / "models" / model.id,
                    examples,
                    labels,
                    task,
                    example_concurrency=plan.execution.example_concurrency,
                    llm_output_mode=plan.llm_output.mode,
                    client=clients.get(model.id) if clients is not None else None,
                    observer=observer,
                )
            except BaseException as error:
                observer.model_failed(model.id, error)
                raise

    try:
        results = await asyncio.gather(
            *(execute_model(index) for index in range(len(plan.models))),
            return_exceptions=True,
        )
        failures = {
            model.id: f"{type(result).__name__}: {result}"
            for model, result in zip(plan.models, results, strict=True)
            if isinstance(result, BaseException)
        }
        metadata["models"] = _model_statuses(run, plan)
        if failures:
            attempt["status"] = "failed"
            attempt["model_errors"] = failures
            raise ValueError(f"Model runs failed: {failures}")
        observer.report_started()
        generate_report(run, plan.models, examples, labels)
        metadata["report_artifacts"] = report_artifacts(run)
        metadata["status"] = "completed"
        attempt["status"] = "completed"
        observer.benchmark_completed(run)
    except Exception as error:
        if attempt["status"] == "running":
            attempt["status"] = "failed"
        observer.benchmark_failed(error)
        raise
    except BaseException as error:
        if attempt["status"] == "running":
            attempt["status"] = "interrupted"
        observer.benchmark_failed(error)
        raise
    finally:
        attempt["finished_at"] = datetime.now(UTC).isoformat()
        metadata["models"] = _model_statuses(run, plan)
        write_json(run / "metadata.json", metadata)


def _verify_saved_inputs(run: Path, metadata: dict[str, Any]) -> None:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Benchmark metadata has no saved input checksums")
    for name in ("plan.toml", "task.toml", "cohort.json"):
        expected = inputs.get(name, {}).get("sha256")
        if not isinstance(expected, str) or file_checksum(run / name) != expected:
            raise ValueError(f"Saved benchmark input changed: {name}")


def _model_statuses(run: Path, plan: BenchmarkPlan) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for model in plan.models:
        path = run / "models" / model.id / "metadata.json"
        statuses[model.id] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "pending"}
        )
    return statuses


def _command(run: Path, metadata: dict[str, Any]) -> list[str]:
    if len(metadata.get("attempts", [])) > 1:
        return ["uv", "run", "thesis-research", "benchmark", "--resume", str(run)]
    source = metadata.get("source_plan_path")
    return ["uv", "run", "thesis-research", "benchmark", "--plan", str(source)]
