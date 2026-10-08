"""Immutable configuration and cohort types for coordinated benchmarks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["system_one", "llm"]
Transport = Literal["vercel", "self_hosted"]
DatasetSplit = str
CohortStrategy = Literal["all", "random", "stratified_random"]
LLMOutputMode = Literal["discrete", "probabilities"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class ModelSpec:
    """Identify one independently evaluated model."""

    id: str
    backend: Backend
    model: str
    provider: str
    reasoning_effort: ReasoningEffort | None = None
    transport: Transport = "vercel"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float | None = None

    def as_dict(self) -> dict[str, str | float | None]:
        """Return the model settings in a JSON-safe form."""
        values: dict[str, str | float | None] = {
            "id": self.id,
            "backend": self.backend,
            "model": self.model,
            "provider": self.provider,
        }
        if self.reasoning_effort is not None:
            values["reasoning_effort"] = self.reasoning_effort
        if self.transport != "vercel":
            values["transport"] = self.transport
        for name in ("base_url", "api_key_env", "timeout_seconds"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return values


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
    """Describe a shared-cohort run of System One models and/or LLMs."""

    name: str
    task_config: Path
    task_config_value: str
    baseline_model_id: str
    cohort: CohortSpec
    llm_output: LLMOutputSpec | None
    execution: ExecutionSpec
    models: tuple[ModelSpec, ...]

    @property
    def baseline(self) -> ModelSpec:
        """Return the explicitly selected comparison baseline."""
        return next(model for model in self.models if model.id == self.baseline_model_id)

    @property
    def llm_output_mode(self) -> LLMOutputMode | None:
        """Return the configured LLM mode, if this plan specifies one."""
        return self.llm_output.mode if self.llm_output is not None else None

    def as_dict(self) -> dict[str, object]:
        """Return the complete validated plan in a JSON-safe form."""
        return {
            "name": self.name,
            "task_config": self.task_config_value,
            "baseline_model_id": self.baseline_model_id,
            "cohort": self.cohort.as_dict(),
            "llm_output": self.llm_output.as_dict() if self.llm_output is not None else None,
            "execution": self.execution.as_dict(),
            "models": [model.as_dict() for model in self.models],
        }
