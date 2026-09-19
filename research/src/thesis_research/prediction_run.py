"""Create durable run records while generating dataset predictions."""

from pathlib import Path
from typing import Any

from thesis_research.clients import DecisionClient, create_decision_client
from thesis_research.config import load_experiment_config
from thesis_research.datasets import load_prepared_banking77, sha256_bytes
from thesis_research.prediction import predict_examples
from thesis_research.run_storage import (
    artifact_metadata,
    create_run_directory,
    git_state,
    write_json,
)
from thesis_research.tasks import build_banking77_task


async def run_prediction_experiment(
    config_path: Path,
    data: Path,
    outputs: Path,
    *,
    backend: str,
    model: str,
    provider: str | None,
    max_concurrency: int = 5,
    limit: int | None = None,
    _client: DecisionClient | None = None,
) -> Path:
    """Run one configured model over a BANKING77 dataset split.

    Loads and validates the experiment configuration and dataset, constructs
    the shared intent-classification task, and generates predictions in
    bounded concurrent batches.

    After initial argument validation, every attempt receives a unique output
    directory. The directory preserves the configuration, predictions,
    decision diagnostics, progress, usage, provenance, and failure information.

    Args:
        config_path: Path to the versioned experiment TOML configuration.
        data: Root directory containing the prepared datasets.
        outputs: Root directory in which to create the unique run directory.
        backend: Model path to use: ``"jev"`` or ``"llm"``.
        model: Model identifier requested from the backend.
        provider: Pinned Vercel provider, or ``None`` for Jev.
        max_concurrency: Maximum number of examples processed concurrently.
        limit: Optional number of examples to process from the start of the
            configured split.
        _client: Optional preconstructed client used by offline tests.

    Returns:
        Path to the newly created prediction-run directory.

    Raises:
        ValueError: If an argument, configuration, dataset, or backend setting
            is invalid, or if the prediction run fails. Failures occurring
            after the run directory is created are preserved in its metadata.
        asyncio.CancelledError: If the run is cancelled. Available progress is
            marked as interrupted and preserved before cancellation propagates.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    run, timestamp = create_run_directory(outputs)
    metadata = {
        "timestamp": timestamp.isoformat(),
        "working_directory": str(Path.cwd()),
        "code": git_state(),
        "status": "started",
        "config_path": str(config_path.resolve()),
        "configuration": None,
        "dataset": None,
        "split": None,
        "selection": {"limit": limit, "split_examples": None, "selected_examples": None},
        "model": {
            "backend": backend,
            "requested_model": model,
            "gateway": "vercel",
            "requested_provider": "typesafe-ai" if backend == "jev" else provider,
            "max_concurrency": max_concurrency,
        },
        "progress": None,
        "command": _prediction_command(
            config_path,
            data,
            outputs,
            backend=backend,
            model=model,
            provider=provider,
            max_concurrency=max_concurrency,
            limit=limit,
        ),
    }
    write_json(run / "metadata.json", metadata)
    try:
        config_bytes = config_path.read_bytes()
        (run / "config.toml").write_bytes(config_bytes)
        metadata["config_sha256"] = sha256_bytes(config_bytes)
        config = load_experiment_config(run / "config.toml")
        metadata["configuration"] = config.as_dict()

        labels, examples, manifest = load_prepared_banking77(
            data,
            config.dataset_revision,
            config.split,
        )
        task = build_banking77_task(config, labels)
        selected_examples = examples if limit is None else examples[:limit]
        metadata["dataset"] = manifest
        metadata["split"] = config.split
        metadata["selection"] = {
            "limit": limit,
            "split_examples": len(examples),
            "selected_examples": len(selected_examples),
        }
        metadata["status"] = "running"
        write_json(run / "metadata.json", metadata)

        decision_client = _client or create_decision_client(
            backend,
            model,
            provider,
            max_concurrency=max_concurrency,
        )

        def record_progress(progress: dict[str, Any]) -> None:
            metadata["progress"] = progress
            write_json(run / "metadata.json", metadata)

        async with decision_client:
            with (run / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file:
                with (run / "decisions.jsonl").open("w", encoding="utf-8") as decisions_file:
                    summary = await predict_examples(
                        decision_client,
                        selected_examples,
                        task,
                        labels,
                        predictions_file,
                        decisions_file,
                        max_concurrency=max_concurrency,
                        on_progress=record_progress,
                    )
        write_json(run / "summary.json", summary)
        metadata["progress"] = summary
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise ValueError(f"Prediction run failed; preserved record at {run}: {error}") from error
    except BaseException as error:
        metadata["status"] = "interrupted"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["artifacts"] = artifact_metadata(
            run,
            ("config.toml", "predictions.jsonl", "decisions.jsonl", "summary.json"),
        )
        write_json(run / "metadata.json", metadata)
    return run


def _prediction_command(
    config_path: Path,
    data: Path,
    outputs: Path,
    *,
    backend: str,
    model: str,
    provider: str | None,
    max_concurrency: int,
    limit: int | None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "thesis-research",
        "predict",
        "--config",
        str(config_path),
        "--data-dir",
        str(data),
        "--output-dir",
        str(outputs),
        "--backend",
        backend,
        "--model",
        model,
        "--max-concurrency",
        str(max_concurrency),
    ]
    if provider is not None:
        command.extend(("--provider", provider))
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command
