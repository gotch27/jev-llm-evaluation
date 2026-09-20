"""Typed progress notifications emitted by coordinated benchmarks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ModelProgress:
    """Describe durable progress for one model over the shared cohort."""

    total: int
    completed: int
    valid: int
    failed: int
    elapsed_seconds: float

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "ModelProgress":
        """Build display progress from a model-run summary dictionary."""
        return cls(
            total=int(summary["expected"]),
            completed=int(summary["attempted"]),
            valid=int(summary["successful"]),
            failed=int(summary["failed"]),
            elapsed_seconds=float(summary["elapsed_seconds"]),
        )


class BenchmarkObserver(Protocol):
    """Receive benchmark lifecycle events without controlling execution."""

    def benchmark_started(
        self,
        name: str,
        run: Path,
        model_ids: Sequence[str],
        total_examples: int,
    ) -> None: ...

    def model_started(self, model_id: str, progress: ModelProgress) -> None: ...

    def model_progress(self, model_id: str, progress: ModelProgress) -> None: ...

    def prediction_failed(self, model_id: str, example_id: str, error: str) -> None: ...

    def model_completed(self, model_id: str, progress: ModelProgress) -> None: ...

    def model_failed(self, model_id: str, error: BaseException) -> None: ...

    def report_started(self) -> None: ...

    def benchmark_completed(self, run: Path) -> None: ...

    def benchmark_failed(self, error: BaseException) -> None: ...


class NullBenchmarkObserver:
    """Ignore progress notifications for library calls and offline tests."""

    def benchmark_started(
        self,
        name: str,
        run: Path,
        model_ids: Sequence[str],
        total_examples: int,
    ) -> None:
        pass

    def model_started(self, model_id: str, progress: ModelProgress) -> None:
        pass

    def model_progress(self, model_id: str, progress: ModelProgress) -> None:
        pass

    def prediction_failed(self, model_id: str, example_id: str, error: str) -> None:
        pass

    def model_completed(self, model_id: str, progress: ModelProgress) -> None:
        pass

    def model_failed(self, model_id: str, error: BaseException) -> None:
        pass

    def report_started(self) -> None:
        pass

    def benchmark_completed(self, run: Path) -> None:
        pass

    def benchmark_failed(self, error: BaseException) -> None:
        pass
