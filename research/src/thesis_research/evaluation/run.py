"""Create durable evaluation records from saved classification predictions."""

import json
from pathlib import Path

from thesis_research.config import load_experiment_config
from thesis_research.datasets import load_prepared_banking77, sha256_bytes
from thesis_research.evaluation.classification import (
    evaluate_classification,
    read_prediction_jsonl,
)
from thesis_research.run_storage import create_run_directory, git_state, write_json


def run_classification_evaluation(
    config_path: Path,
    predictions_path: Path,
    data: Path,
    outputs: Path,
) -> Path:
    """Evaluate a prediction file and preserve the complete attempt.

    A unique output directory is created before inputs are read. Successful
    runs contain the copied inputs, summary, outcomes, and metadata. Failures
    keep their available inputs and error metadata for diagnosis.

    Args:
        config_path: Versioned experiment TOML used to select the dataset split.
        predictions_path: JSONL file containing labels or explicit errors.
        data: Root directory containing the prepared datasets.
        outputs: Root directory in which to create the evaluation directory.

    Returns:
        Path to the newly created evaluation directory.

    Raises:
        ValueError: If configuration, dataset, predictions, or scoring fails.
            The exception identifies the preserved run directory.
    """
    run, timestamp = create_run_directory(outputs)
    metadata = {
        "timestamp": timestamp.isoformat(),
        "working_directory": str(Path.cwd()),
        "code": git_state(),
        "status": "started",
        "config_path": str(config_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "command": [
            "uv",
            "run",
            "thesis-research",
            "evaluate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
            "--data-dir",
            str(data),
            "--output-dir",
            str(outputs),
        ],
    }
    write_json(run / "metadata.json", metadata)
    try:
        config_bytes = config_path.read_bytes()
        (run / "config.toml").write_bytes(config_bytes)
        prediction_bytes = predictions_path.read_bytes()
        (run / "predictions.jsonl").write_bytes(prediction_bytes)
        metadata["prediction_sha256"] = sha256_bytes(prediction_bytes)

        config = load_experiment_config(run / "config.toml")
        metadata["configuration"] = config.as_dict()
        labels, examples, manifest = load_prepared_banking77(
            data,
            config.dataset_revision,
            config.split,
        )
        metadata["dataset"] = manifest
        metadata["split"] = config.split

        predictions = read_prediction_jsonl(prediction_bytes)
        summary, outcomes = evaluate_classification(examples, labels, predictions)
        write_json(run / "summary.json", summary)
        with (run / "outcomes.jsonl").open("w", encoding="utf-8") as handle:
            for outcome in outcomes:
                handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise ValueError(f"Evaluation failed; preserved record at {run}: {error}") from error
    finally:
        write_json(run / "metadata.json", metadata)
    return run
