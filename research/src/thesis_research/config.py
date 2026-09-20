"""Typed task configuration loaded from versioned TOML files."""

import math
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from typesafe_sdk import JSONContent

QuestionType = Literal["choice"]


@dataclass(frozen=True)
class ChoiceOptionConfig:
    """Define one allowed Choice label and its optional TypeSafe criterion.

    ``criterion`` mirrors TypeSafe's ``JSONContent | None`` value: a string,
    JSON object, JSON array, or ``None`` when the TOML field is omitted.
    """

    label: str
    criterion: JSONContent | None

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe representation of this option."""
        return {"label": self.label, "criterion": deepcopy(self.criterion)}


@dataclass(frozen=True)
class ChoiceQuestionConfig:
    """Define one provider-neutral TypeSafe Choice question."""

    question_type: QuestionType
    state_field: str
    question_id: str
    instructions: JSONContent | None
    options: tuple[ChoiceOptionConfig, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation of this question."""
        return {
            "type": self.question_type,
            "state_field": self.state_field,
            "id": self.question_id,
            "instructions": deepcopy(self.instructions),
            "options": [option.as_dict() for option in self.options],
        }


@dataclass(frozen=True)
class TaskConfig:
    """Represent one split-independent dataset task definition."""

    name: str
    dataset_revision: str
    question: ChoiceQuestionConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation of the task."""
        value: dict[str, Any] = {
            "name": self.name,
            "dataset_revision": self.dataset_revision,
        }
        if self.question is not None:
            value["question"] = self.question.as_dict()
        return value


def load_task_config(path: Path) -> TaskConfig:
    """Load and strictly validate one split-independent task TOML file.

    Criterion and instruction content is passed through as JSON-compatible
    data. TOML has no null literal, so an omitted ``criterion`` represents
    TypeSafe's ``None`` criterion.

    Args:
        path: TOML file containing a dataset revision and optional question.

    Returns:
        A frozen typed task definition.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If fields are missing, unexpected, or not JSON-compatible.
    """
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    required = {"name", "dataset_revision"}
    allowed = required | {"question"}
    _require_fields(value, required, allowed, "Task")

    name = _nonempty_string(value["name"], "Task name")
    revision = _nonempty_string(value["dataset_revision"], "dataset_revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("dataset_revision must be a full lowercase Git commit SHA")
    question = _parse_question(value["question"]) if "question" in value else None
    return TaskConfig(name, revision, question)


def _parse_question(value: object) -> ChoiceQuestionConfig:
    if not isinstance(value, dict):
        raise ValueError("Task question must be a table")
    required = {"type", "state_field", "id", "options"}
    allowed = required | {"instructions"}
    _require_fields(value, required, allowed, "Question")
    question_type = value["type"]
    if question_type != "choice":
        raise ValueError("Question type must be choice")
    options = value["options"]
    if not isinstance(options, list) or not options:
        raise ValueError("Question options must be a nonempty array")
    instructions = (
        _json_content(value["instructions"], "Question instructions")
        if "instructions" in value
        else None
    )
    return ChoiceQuestionConfig(
        question_type=cast(QuestionType, question_type),
        state_field=_nonempty_string(value["state_field"], "Question state_field"),
        question_id=_nonempty_string(value["id"], "Question id"),
        instructions=instructions,
        options=tuple(
            _parse_option(option, index) for index, option in enumerate(options, start=1)
        ),
    )


def _parse_option(value: object, index: int) -> ChoiceOptionConfig:
    if not isinstance(value, dict):
        raise ValueError(f"Question option {index} must be a table")
    _require_fields(value, {"label"}, {"label", "criterion"}, f"Question option {index}")
    criterion = (
        _json_content(value["criterion"], f"Question option {index} criterion")
        if "criterion" in value
        else None
    )
    return ChoiceOptionConfig(
        label=_nonempty_string(value["label"], f"Question option {index} label"),
        criterion=criterion,
    )


def _json_content(value: object, name: str) -> JSONContent:
    if not isinstance(value, str | dict | list):
        raise ValueError(f"{name} must be a string, object, or array")
    _validate_json_value(value, name)
    return cast(JSONContent, value)


def _validate_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} object keys must be strings")
            _validate_json_value(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains a non-JSON value of type {type(value).__name__}")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_fields(
    value: dict[str, Any], required: set[str], allowed: set[str], name: str
) -> None:
    actual = set(value)
    missing = required - actual
    unexpected = actual - allowed
    if missing or unexpected:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
