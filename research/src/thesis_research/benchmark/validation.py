"""Check task inputs and image requirements before constructing clients."""

from collections.abc import Sequence
from pathlib import Path

from thesis_model_clients.images import image_capability, validate_image_request

from thesis_research.benchmark.cohort import select_cohort
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.types import BenchmarkPlan
from thesis_research.config import TaskConfig, load_task_config
from thesis_research.datasets import DatasetExample, load_prepared_dataset
from thesis_research.tasks import StructuredTask, build_structured_task


def validate_benchmark(plan_path: Path, data: Path) -> dict:
    """Validate a complete selected cohort without credentials or inference calls."""
    plan = load_benchmark_plan(plan_path)
    config = load_task_config(plan.task_config)
    profiles = validate_image_models(plan, config)
    prepared = load_prepared_dataset(
        data,
        config.dataset.id,
        config.dataset.revision,
        plan.cohort.split,
        version=config.dataset.version,
    )
    task = build_structured_task(
        config, prepared.reference_schema, image_assets=prepared.image_assets
    )
    task.validate_state_fields(prepared.examples)
    validate_task_images(task, prepared.examples)
    examples = select_cohort(prepared.examples, prepared.reference_schema, plan.cohort)
    return {
        "name": plan.name,
        "dataset": config.dataset.as_dict(),
        "split": plan.cohort.split,
        "cohort_examples": len(examples),
        "models": [model.id for model in plan.models],
        "question_type": task.question_type,
        "image_fields": [name for name, _ in task.image_fields],
        "image_capabilities": profiles,
    }


def validate_image_models(plan: BenchmarkPlan, config: TaskConfig) -> dict:
    """Collect all incompatible model routes; a mixed plan fails as a whole."""
    if not config.state.images:
        return {}
    errors, profiles = [], {}
    for model in plan.models:
        try:
            profiles[model.id] = image_capability(
                model.backend, model.model, model.provider, model.transport
            )
        except ValueError as error:
            errors.append(f"{model.id}: {error}")
    if errors:
        raise ValueError("Incompatible image benchmark; no model calls made:\n" + "\n".join(errors))
    return profiles


def validate_task_images(task: StructuredTask, examples: Sequence[DatasetExample]) -> None:
    """Check every image-bearing input before any participant starts."""
    if not task.image_fields:
        return
    for example in examples:
        try:
            state = task.build_state(example.state)
            validate_image_request(state, task.questions(), task.image_inputs(state))
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid image input at {example.id}: {error}") from error
