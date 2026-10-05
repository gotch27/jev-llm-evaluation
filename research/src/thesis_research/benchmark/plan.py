"""Load and strictly validate benchmark TOML plans."""

import re
import tomllib
from pathlib import Path
from typing import Any, cast

from thesis_research.benchmark.types import (
    Backend,
    BenchmarkPlan,
    CohortSpec,
    CohortStrategy,
    DatasetSplit,
    ExecutionSpec,
    LLMOutputMode,
    LLMOutputSpec,
    ModelSpec,
    ReasoningEffort,
)

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def load_benchmark_plan(
    path: Path,
    *,
    task_config_override: Path | None = None,
) -> BenchmarkPlan:
    """Load a benchmark plan and resolve its task configuration path.

    Args:
        path: Versioned benchmark TOML file.
        task_config_override: Copied task file to use when resuming a run.

    Returns:
        A fully validated immutable benchmark plan.

    Raises:
        OSError: If the plan cannot be read.
        ValueError: If fields are missing, unexpected, or inconsistent.
    """
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    _require_fields(
        value,
        {"name", "task_config", "cohort", "llm_output", "execution", "models"},
        {"name", "task_config", "cohort", "llm_output", "execution", "models"},
        "Plan",
    )
    task_config_value = _nonempty_string(value["task_config"], "task_config")
    task_config = task_config_override or (path.parent / task_config_value).resolve()
    cohort = _parse_cohort(value["cohort"])
    llm_output = _parse_llm_output(value["llm_output"])
    execution = _parse_execution(value["execution"])
    raw_models = value["models"]
    if not isinstance(raw_models, list):
        raise ValueError("models must be an array of tables")
    models = tuple(_parse_model(model, index) for index, model in enumerate(raw_models, 1))
    if len(models) < 2:
        raise ValueError("A benchmark requires Jev and at least one LLM")
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("Model IDs must be unique")
    if sum(model.backend == "jev" for model in models) != 1:
        raise ValueError("A benchmark requires exactly one Jev model")
    if not any(model.backend == "llm" for model in models):
        raise ValueError("A benchmark requires at least one LLM model")
    return BenchmarkPlan(
        name=_nonempty_string(value["name"], "name"),
        task_config=task_config,
        task_config_value=task_config_value,
        cohort=cohort,
        llm_output=llm_output,
        execution=execution,
        models=models,
    )


def _parse_cohort(value: object) -> CohortSpec:
    if not isinstance(value, dict):
        raise ValueError("cohort must be a table")
    _require_fields(
        value,
        {"split", "strategy"},
        {"split", "strategy", "size", "seed"},
        "cohort",
    )
    split = _nonempty_string(value["split"], "cohort split")
    strategy = value["strategy"]
    if strategy not in ("all", "random", "stratified_random"):
        raise ValueError("cohort strategy must be all, random, or stratified_random")
    size = value.get("size")
    seed = value.get("seed")
    if strategy == "all":
        if size is not None or seed is not None:
            raise ValueError("An all cohort cannot set size or seed")
    else:
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"A {strategy} cohort requires a positive integer size")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"A {strategy} cohort requires an integer seed")
    return CohortSpec(
        split=cast(DatasetSplit, split),
        strategy=cast(CohortStrategy, strategy),
        size=size,
        seed=seed,
    )


def _parse_llm_output(value: object) -> LLMOutputSpec:
    if not isinstance(value, dict):
        raise ValueError("llm_output must be a table")
    _require_fields(value, {"mode"}, {"mode"}, "llm_output")
    mode = value["mode"]
    if mode not in ("discrete", "probabilities"):
        raise ValueError("llm_output mode must be discrete or probabilities")
    return LLMOutputSpec(cast(LLMOutputMode, mode))


def _parse_execution(value: object) -> ExecutionSpec:
    if not isinstance(value, dict):
        raise ValueError("execution must be a table")
    fields = {"example_concurrency", "model_concurrency"}
    _require_fields(value, fields, fields, "execution")
    example_concurrency = _positive_integer(value["example_concurrency"], "example_concurrency")
    model_concurrency = _positive_integer(value["model_concurrency"], "model_concurrency")
    return ExecutionSpec(example_concurrency, model_concurrency)


def _parse_model(value: object, index: int) -> ModelSpec:
    if not isinstance(value, dict):
        raise ValueError(f"Model {index} must be a table")
    _require_fields(
        value,
        {"id", "backend", "model"},
        {"id", "backend", "model", "provider", "reasoning_effort"},
        f"Model {index}",
    )
    model_id = _nonempty_string(value["id"], f"Model {index} id")
    if _MODEL_ID.fullmatch(model_id) is None:
        raise ValueError(f"Model {index} id must contain lowercase letters, digits, _ or -")
    backend = value["backend"]
    if backend not in ("jev", "llm"):
        raise ValueError(f"Model {index} backend must be jev or llm")
    provider_value = value.get("provider")
    provider = (
        _nonempty_string(provider_value, f"Model {index} provider")
        if provider_value is not None
        else None
    )
    if backend == "jev" and provider is not None:
        raise ValueError(f"Model {index} cannot set a provider for Jev")
    if backend == "llm" and provider is None:
        raise ValueError(f"Model {index} requires a pinned provider")
    reasoning_value = value.get("reasoning_effort")
    if reasoning_value is not None and (
        not isinstance(reasoning_value, str) or reasoning_value not in _REASONING_EFFORTS
    ):
        raise ValueError(
            f"Model {index} reasoning_effort must be one of {sorted(_REASONING_EFFORTS)}"
        )
    if backend == "jev" and reasoning_value is not None:
        raise ValueError(f"Model {index} cannot set reasoning_effort for Jev")
    return ModelSpec(
        id=model_id,
        backend=cast(Backend, backend),
        model=_nonempty_string(value["model"], f"Model {index} model"),
        provider=provider,
        reasoning_effort=cast(ReasoningEffort | None, reasoning_value),
    )


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_fields(
    value: dict[str, Any], required: set[str], allowed: set[str], name: str
) -> None:
    missing = required - set(value)
    unexpected = set(value) - allowed
    if missing or unexpected:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
