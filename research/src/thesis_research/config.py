"""Typed experiment configuration loaded from versioned TOML files."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

Split = Literal["train", "test"]


@dataclass(frozen=True)
class IntentOptionConfig:
    """Describe when one intent label should and should not be selected.

    Attributes:
        label: Exact label spelling used by the dataset.
        use_when: Positive criteria for selecting the label.
        distinguish_from: Boundary separating it from similar labels.
    """

    label: str
    use_when: str
    distinguish_from: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable copy using the TOML field names."""
        return {
            "label": self.label,
            "use_when": self.use_when,
            "distinguish_from": self.distinguish_from,
        }


@dataclass(frozen=True)
class QuestionConfig:
    """Define the state field and closed-set question sent to every model.

    Attributes:
        state_field: Name under which the customer message is placed.
        question_id: Stable identifier used to retrieve the model's answer.
        instructions: Shared classification instructions.
        options: Intent labels and their decision criteria in official order.
    """

    state_field: str
    question_id: str
    instructions: str
    options: tuple[IntentOptionConfig, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy matching the TOML structure."""
        return {
            "state_field": self.state_field,
            "id": self.question_id,
            "instructions": self.instructions,
            "options": [option.as_dict() for option in self.options],
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Represent one validated, immutable experiment configuration.

    Attributes:
        name: Human-readable experiment identifier.
        dataset_revision: Full Git revision pinning the source dataset.
        split: Dataset split selected for the experiment.
        question: Structured question definition, when the experiment needs one.
    """

    name: str
    dataset_revision: str
    split: Split
    question: QuestionConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy suitable for run metadata."""
        value: dict[str, Any] = {
            "name": self.name,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
        }
        if self.question is not None:
            value["question"] = self.question.as_dict()
        return value


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate one versioned experiment TOML file.

    Args:
        path: TOML file containing the dataset and optional question definition.

    Returns:
        An immutable configuration with nested typed values.

    Raises:
        OSError: If the configuration file cannot be read.
        ValueError: If required fields are missing, unexpected, or malformed.
    """
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    required = {"name", "dataset_revision", "split"}
    allowed = required | {"question"}
    _require_fields(value, required, allowed, "Config")

    name = _nonempty_string(value["name"], "Config name")
    revision = _nonempty_string(value["dataset_revision"], "dataset_revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("dataset_revision must be a full lowercase Git commit SHA")
    split = value["split"]
    if split not in ("train", "test"):
        raise ValueError("Config split must be train or test")
    question = _parse_question(value["question"]) if "question" in value else None
    return ExperimentConfig(name, revision, cast(Split, split), question)


def _parse_question(value: object) -> QuestionConfig:
    if not isinstance(value, dict):
        raise ValueError("Config question must be a table")
    fields = {"state_field", "id", "instructions", "options"}
    _require_fields(value, fields, fields, "Question")
    options = value["options"]
    if not isinstance(options, list):
        raise ValueError("Question options must be an array")
    return QuestionConfig(
        state_field=_nonempty_string(value["state_field"], "Question state_field"),
        question_id=_nonempty_string(value["id"], "Question id"),
        instructions=_nonempty_string(value["instructions"], "Question instructions"),
        options=tuple(
            _parse_option(option, index) for index, option in enumerate(options, start=1)
        ),
    )


def _parse_option(value: object, index: int) -> IntentOptionConfig:
    if not isinstance(value, dict):
        raise ValueError(f"Question option {index} must be a table")
    fields = {"label", "use_when", "distinguish_from"}
    _require_fields(value, fields, fields, f"Question option {index}")
    return IntentOptionConfig(
        label=_nonempty_string(value["label"], f"Question option {index} label"),
        use_when=_nonempty_string(value["use_when"], f"Question option {index} use_when"),
        distinguish_from=_nonempty_string(
            value["distinguish_from"], f"Question option {index} distinguish_from"
        ),
    )


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
