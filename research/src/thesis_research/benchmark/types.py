"""Immutable configuration and cohort types for coordinated benchmarks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["jev", "llm"]
DatasetSplit = Literal["train", "test"]
CohortStrategy = Literal["all", "random", "stratified_random"]
LLMOutputMode = Literal["label", "probabilities"]


@dataclass(frozen=True)
class ModelSpec:
    """Identify one independently evaluated model."""

    id: str
    backend: Backend
    model: str
    provider: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Return the model settings in a JSON-safe form."""
        return {
            "id": self.id,
            "backend": self.backend,
            "model": self.model,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class CohortSpec:
    """Define the dataset split and examples shared by every model."""

    split: DatasetSplit
    strategy: CohortStrategy
    size: int | None
    seed: int | None

    def as_dict(self) -> dict[str, str | int | None]:
        """Return the cohort settings in a JSON-safe form."""
        return {
            "split": self.split,
            "strategy": self.strategy,
            "size": self.size,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class LLMOutputSpec:
    """Select the common structured answer format used by every LLM."""

    mode: LLMOutputMode

    def as_dict(self) -> dict[str, str]:
        """Return the LLM output settings in a JSON-safe form."""
        return {"mode": self.mode}


@dataclass(frozen=True)
class ExecutionSpec:
    """Bound example-level and model-level concurrency."""

    example_concurrency: int
    model_concurrency: int

    def as_dict(self) -> dict[str, int]:
        """Return the execution settings in a JSON-safe form."""
        return {
            "example_concurrency": self.example_concurrency,
            "model_concurrency": self.model_concurrency,
        }


@dataclass(frozen=True)
class BenchmarkPlan:
    """Describe one comparison of Jev with one or more LLMs."""

    name: str
    task_config: Path
    task_config_value: str
    cohort: CohortSpec
    llm_output: LLMOutputSpec
    execution: ExecutionSpec
    models: tuple[ModelSpec, ...]

    @property
    def jev(self) -> ModelSpec:
        """Return the sole Jev model, which is the comparison baseline."""
        return next(model for model in self.models if model.backend == "jev")

    def as_dict(self) -> dict[str, object]:
        """Return the complete validated plan in a JSON-safe form."""
        return {
            "name": self.name,
            "task_config": self.task_config_value,
            "cohort": self.cohort.as_dict(),
            "llm_output": self.llm_output.as_dict(),
            "execution": self.execution.as_dict(),
            "models": [model.as_dict() for model in self.models],
        }
