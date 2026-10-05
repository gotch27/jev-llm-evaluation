"""Typed structured-task configuration loaded from versioned TOML files."""

from __future__ import annotations

import math
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from typesafe_sdk import JSONContent


@dataclass(frozen=True)
class DatasetConfig:
    """Identify one registered dataset and immutable source revision."""

    id: str
    revision: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "revision": self.revision}


@dataclass(frozen=True)
class StateConfig:
    """Map task-state keys to source fields supplied by a dataset adapter."""

    fields: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"fields": dict(self.fields)}


@dataclass(frozen=True)
class ChoiceOptionConfig:
    """Define one allowed Choice label and its optional criterion."""

    label: str
    criterion: JSONContent | None

    def as_dict(self) -> dict[str, object]:
        return {"label": self.label, "criterion": deepcopy(self.criterion)}


@dataclass(frozen=True)
class ChoiceQuestionConfig:
    """Define one provider-neutral TypeSafe Choice question."""

    question_id: str
    instructions: JSONContent | None
    options: tuple[ChoiceOptionConfig, ...]
    question_type: Literal["choice"] = "choice"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.question_type,
            "id": self.question_id,
            "instructions": deepcopy(self.instructions),
            "options": [option.as_dict() for option in self.options],
        }


@dataclass(frozen=True)
class NoulQuestionConfig:
    """Define one provider-neutral TypeSafe yes/no question."""

    question_id: str
    instructions: JSONContent | None
    true_criterion: JSONContent | None
    false_criterion: JSONContent | None
    has_criteria: bool
    question_type: Literal["noul"] = "noul"

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.question_type,
            "id": self.question_id,
            "instructions": deepcopy(self.instructions),
        }
        if self.has_criteria:
            value["criteria"] = {
                "true": deepcopy(self.true_criterion),
                "false": deepcopy(self.false_criterion),
            }
        return value


@dataclass(frozen=True)
class ScoreLevelConfig:
    """Describe one ordered zero-based TypeSafe Score level."""

    criterion: JSONContent

    def as_dict(self) -> dict[str, JSONContent]:
        return {"criterion": deepcopy(self.criterion)}


@dataclass(frozen=True)
class ScoreQuestionConfig:
    """Define one provider-neutral TypeSafe Score question."""

    question_id: str
    instructions: JSONContent | None
    levels: tuple[ScoreLevelConfig, ...]
    question_type: Literal["score"] = "score"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.question_type,
            "id": self.question_id,
            "instructions": deepcopy(self.instructions),
            "levels": [level.as_dict() for level in self.levels],
        }


QuestionConfig: TypeAlias = ChoiceQuestionConfig | NoulQuestionConfig | ScoreQuestionConfig


@dataclass(frozen=True)
class TaskConfig:
    """Represent one split-independent dataset task definition."""

    name: str
    dataset: DatasetConfig
    state: StateConfig
    question: QuestionConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset.as_dict(),
            "state": self.state.as_dict(),
            "question": self.question.as_dict(),
        }


def load_task_config(path: Path) -> TaskConfig:
    """Load a task with explicit dataset, state, and question tables."""
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    return _parse_task(value)


def _parse_task(value: dict[str, Any]) -> TaskConfig:
    required = {"name", "dataset", "state", "question"}
    _require_fields(value, required, required, "Task")
    dataset = value["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a table")
    _require_fields(dataset, {"id", "revision"}, {"id", "revision"}, "dataset")
    return TaskConfig(
        name=_nonempty_string(value["name"], "Task name"),
        dataset=DatasetConfig(
            _nonempty_string(dataset["id"], "dataset id"),
            _nonempty_string(dataset["revision"], "dataset revision"),
        ),
        state=_parse_state(value["state"]),
        question=_parse_question(value["question"]),
    )


def _parse_state(value: object) -> StateConfig:
    if not isinstance(value, dict):
        raise ValueError("state must be a table")
    _require_fields(value, {"fields"}, {"fields"}, "state")
    fields = value["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("state.fields must be a nonempty table")
    parsed = tuple(
        (
            _nonempty_string(output, "state field name"),
            _nonempty_string(source, f"state.fields.{output}"),
        )
        for output, source in fields.items()
    )
    return StateConfig(parsed)


def _parse_question(value: object) -> QuestionConfig:
    if not isinstance(value, dict):
        raise ValueError("Task question must be a table")
    question_type = value.get("type")
    common = {"type", "id", "instructions"}
    if question_type == "choice":
        _require_fields(value, {"type", "id", "options"}, common | {"options"}, "Question")
        return _parse_choice_question(value)
    if question_type == "noul":
        _require_fields(value, {"type", "id"}, common | {"criteria"}, "Question")
        return _parse_noul_question(value)
    if question_type == "score":
        _require_fields(value, {"type", "id", "levels"}, common | {"levels"}, "Question")
        return _parse_score_question(value)
    raise ValueError("Question type must be choice, noul, or score")


def _parse_choice_question(value: dict[str, Any]) -> ChoiceQuestionConfig:
    options = value["options"]
    if not isinstance(options, list) or not options:
        raise ValueError("Question options must be a nonempty array")
    return ChoiceQuestionConfig(
        question_id=_nonempty_string(value["id"], "Question id"),
        instructions=_optional_json_content(value, "instructions", "Question instructions"),
        options=tuple(
            _parse_choice_option(option, index) for index, option in enumerate(options, start=1)
        ),
    )


def _parse_noul_question(value: dict[str, Any]) -> NoulQuestionConfig:
    criteria = value.get("criteria", {})
    if not isinstance(criteria, dict):
        raise ValueError("Noul criteria must be a table")
    _require_fields(criteria, set(), {"true", "false"}, "Noul criteria")
    return NoulQuestionConfig(
        question_id=_nonempty_string(value["id"], "Question id"),
        instructions=_optional_json_content(value, "instructions", "Question instructions"),
        true_criterion=_optional_json_content(criteria, "true", "true criterion"),
        false_criterion=_optional_json_content(criteria, "false", "false criterion"),
        has_criteria="criteria" in value,
    )


def _parse_score_question(value: dict[str, Any]) -> ScoreQuestionConfig:
    levels = value["levels"]
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("Score levels must contain at least two ordered entries")
    return ScoreQuestionConfig(
        question_id=_nonempty_string(value["id"], "Question id"),
        instructions=_optional_json_content(value, "instructions", "Question instructions"),
        levels=tuple(
            _parse_score_level(level, index) for index, level in enumerate(levels, start=1)
        ),
    )


def _parse_choice_option(value: object, index: int) -> ChoiceOptionConfig:
    if not isinstance(value, dict):
        raise ValueError(f"Question option {index} must be a table")
    _require_fields(value, {"label"}, {"label", "criterion"}, f"Question option {index}")
    return ChoiceOptionConfig(
        label=_nonempty_string(value["label"], f"Question option {index} label"),
        criterion=_optional_json_content(value, "criterion", f"Question option {index} criterion"),
    )


def _parse_score_level(value: object, index: int) -> ScoreLevelConfig:
    if not isinstance(value, dict):
        raise ValueError(f"Score level {index} must be a table")
    _require_fields(value, {"criterion"}, {"criterion"}, f"Score level {index}")
    return ScoreLevelConfig(_json_content(value["criterion"], f"Score level {index} criterion"))


def _optional_json_content(value: dict[str, Any], key: str, name: str) -> JSONContent | None:
    return _json_content(value[key], name) if key in value else None


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
    missing = required - set(value)
    unexpected = set(value) - allowed
    if missing or unexpected:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
