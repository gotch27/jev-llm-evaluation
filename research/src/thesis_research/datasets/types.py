"""Shared contracts for prepared supervised datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from typesafe_sdk import JSONContent

QuestionType = Literal["choice", "noul", "score"]


@dataclass(frozen=True)
class ChoiceReference:
    """Reference label for a Choice question."""

    label: str


@dataclass(frozen=True)
class NoulReference:
    """Reference truth value for a Noul question."""

    value: bool


@dataclass(frozen=True)
class ScoreReference:
    """Zero-based reference level for a Score question."""

    level: int


Reference: TypeAlias = ChoiceReference | NoulReference | ScoreReference


@dataclass(frozen=True)
class DatasetExample:
    """One stable dataset row before task-specific state mapping."""

    id: str
    state: dict[str, JSONContent]
    reference: Reference


@dataclass(frozen=True)
class ChoiceReferenceSchema:
    """Ordered label inventory expected by a Choice dataset."""

    labels: tuple[str, ...]
    question_type: Literal["choice"] = "choice"


@dataclass(frozen=True)
class NoulReferenceSchema:
    """Boolean reference schema expected by a Noul dataset."""

    question_type: Literal["noul"] = "noul"


@dataclass(frozen=True)
class ScoreReferenceSchema:
    """Number of ordered zero-based levels expected by a Score dataset."""

    level_count: int
    question_type: Literal["score"] = "score"


ReferenceSchema: TypeAlias = ChoiceReferenceSchema | NoulReferenceSchema | ScoreReferenceSchema


@dataclass(frozen=True)
class PreparedDataset:
    """One verified split pinned to an exact source-repository commit.

    The source revision identifies the dataset snapshot, independently of the
    research code revision. Stable example IDs refer to rows in that snapshot.
    """

    dataset_id: str
    revision: str
    split: str
    examples: tuple[DatasetExample, ...]
    reference_schema: ReferenceSchema
    manifest: dict


class DatasetAdapter(Protocol):
    """Prepare and load one named dataset implementation."""

    dataset_id: str
    supported_splits: tuple[str, ...]

    def prepare(self, root: "Path", revision: str) -> "Path":
        """Prepare a pinned dataset revision under the data root."""
        ...

    def load(self, root: "Path", revision: str, split: str) -> PreparedDataset:
        """Load and verify a prepared dataset split."""
        ...


def reference_class(reference: Reference) -> str:
    """Return a stable class key used for stratification and exact metrics."""
    if isinstance(reference, ChoiceReference):
        return reference.label
    if isinstance(reference, NoulReference):
        return "true" if reference.value else "false"
    return str(reference.level)
