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
    select_cohort,
)
from thesis_research.benchmark.model_records import (
    archive_error_records_for_retry,
    load_model_records,
    summarize_model_records,
    validate_model_metadata,
    validate_model_records,
)
from thesis_research.benchmark.model_run import run_model
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.progress import BenchmarkObserver, NullBenchmarkObserver
from thesis_research.benchmark.report import generate_report, report_artifacts
from thesis_research.benchmark.timing import runner_environment
from thesis_research.benchmark.types import BenchmarkPlan
from thesis_research.clients import DecisionClient
from thesis_research.config import load_task_config
from thesis_research.datasets import (
    DatasetExample,
    ReferenceSchema,
    load_prepared_dataset,
    sha256_bytes,
)
from thesis_research.run_storage import create_run_directory, file_checksum, git_state, write_json
from thesis_research.tasks import StructuredTask, build_structured_task


async def run_benchmark(
    plan_path: Path,
    data: Path,
    outputs: Path,
    *,
    observer: BenchmarkObserver | None = None,
    runner_location: str | None = None,
    _clients: dict[str, DecisionClient] | None = None,
) -> Path:
    """Create and execute one Jev-to-LLM benchmark.

    The plan and task are validated before a run directory is created. After
    that point, every failure is recorded inside the unique benchmark
    directory so the same run can be resumed without repeating completed IDs.
    """
    observer = observer if observer is not None else NullBenchmarkObserver()
    environment = runner_environment(runner_location)
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
        "runner": environment,
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
        prepared = load_prepared_dataset(
            data,
            config.dataset.id,
            config.dataset.revision,
            plan.cohort.split,
        )
        task = build_structured_task(config, prepared.reference_schema)
        task.validate_state_fields(prepared.examples)
        examples = select_cohort(prepared.examples, prepared.reference_schema, plan.cohort)
        write_json(
            run / "cohort.json",
            cohort_record(
                examples,
                dataset_id=config.dataset.id,
                dataset_revision=config.dataset.revision,
                split=plan.cohort.split,
                schema=prepared.reference_schema,
                spec=plan.cohort,
                task=task,
            ),
        )
        metadata["inputs"]["cohort.json"] = {"sha256": file_checksum(run / "cohort.json")}
        metadata["dataset"] = prepared.manifest
        metadata["dataset_id"] = config.dataset.id
        metadata["question_type"] = prepared.reference_schema.question_type
        metadata["split"] = plan.cohort.split
        metadata["split_examples"] = len(prepared.examples)
        metadata["cohort_examples"] = len(examples)
        await _execute(
            run,
            plan,
            task,
            prepared.reference_schema,
            examples,
            metadata,
            observer,
            environment,
            _clients,
        )
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
    runner_location: str | None = None,
    retry_errors: bool = False,
    _clients: dict[str, DecisionClient] | None = None,
) -> Path:
    """Resume missing predictions, optionally retrying archived error records."""
    observer = observer if observer is not None else NullBenchmarkObserver()
    environment = runner_environment(runner_location)
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _verify_saved_inputs(run, metadata)
    plan = load_benchmark_plan(run / "plan.toml", task_config_override=run / "task.toml")
    config = load_task_config(run / "task.toml")
    if metadata.get("status") == "completed":
        if retry_errors:
            raise ValueError("A completed benchmark report cannot retry error records")
        return run
    prepared = load_prepared_dataset(
        data,
        config.dataset.id,
        config.dataset.revision,
        plan.cohort.split,
    )
    task = build_structured_task(config, prepared.reference_schema)
    task.validate_state_fields(prepared.examples)
    examples = examples_from_cohort(
        json.loads((run / "cohort.json").read_text(encoding="utf-8")),
        prepared.examples,
        dataset_id=config.dataset.id,
        dataset_revision=config.dataset.revision,
        split=plan.cohort.split,
        task=task,
    )
    # Validate every model before retry preparation or any new provider calls.
    expected_ids = {example.id for example in examples}
    for model in plan.models:
        directory = run / "models" / model.id
        model_metadata_path = directory / "metadata.json"
        model_metadata = {}
        if model_metadata_path.exists():
            model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
            output_mode = "native_probabilities" if model.backend == "jev" else plan.llm_output.mode
            validate_model_metadata(
                model_metadata, model, plan.execution.example_concurrency, output_mode
            )
        predictions, decisions = load_model_records(directory)
        validate_model_records(predictions, decisions, expected_ids)
        if predictions and not model_metadata_path.exists():
            raise ValueError(f"Saved model records have no metadata for {model.id}")
        summarize_model_records(directory, len(examples), prepared.reference_schema, model_metadata)
    if retry_errors:
        retried = {
            model.id: list(archive_error_records_for_retry(run / "models" / model.id))
            for model in plan.models
            if (run / "models" / model.id).is_dir()
        }
        metadata.setdefault("error_retry_preparations", []).append(
            {
                "prepared_at": datetime.now(UTC).isoformat(),
                "models": {model: ids for model, ids in retried.items() if ids},
            }
        )
        write_json(metadata_path, metadata)
    metadata["dataset"] = prepared.manifest
    metadata["dataset_id"] = config.dataset.id
    metadata["question_type"] = prepared.reference_schema.question_type
    metadata.pop("error", None)
    try:
        await _execute(
            run,
            plan,
            task,
            prepared.reference_schema,
            examples,
            metadata,
            observer,
            environment,
            _clients,
        )
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
    task: StructuredTask,
    schema: ReferenceSchema,
    examples: list[DatasetExample],
    metadata: dict[str, Any],
    observer: BenchmarkObserver,
    environment: dict[str, str],
    clients: dict[str, DecisionClient] | None,
) -> None:
    attempt = {
        "started_at": datetime.now(UTC).isoformat(),
        "code": git_state(),
        "status": "running",
        "command": _command(run, metadata, environment["location"]),
        "runner": environment,
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
                    schema,
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
        generate_report(run, plan.models, examples, schema)
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
    required = ["plan.toml", "task.toml", "cohort.json"]
    for name in required:
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


def _command(run: Path, metadata: dict[str, Any], runner_location: str) -> list[str]:
    if len(metadata.get("attempts", [])) > 1:
        command = ["uv", "run", "thesis-research", "benchmark", "--resume", str(run)]
    else:
        source = metadata.get("source_plan_path")
        command = ["uv", "run", "thesis-research", "benchmark", "--plan", str(source)]
    if runner_location != "unspecified":
        command.extend(["--runner-location", runner_location])
    return command
