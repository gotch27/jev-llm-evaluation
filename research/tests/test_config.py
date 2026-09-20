from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from thesis_research.config import TaskConfig, load_task_config
from thesis_research.datasets.banking77 import REVISION

CONFIG_PATH = (
    Path(__file__).parents[1] / "experiments" / "intent_classification" / "tasks" / "banking77.toml"
)


def test_loads_an_immutable_split_independent_task():
    config = load_task_config(CONFIG_PATH)

    assert isinstance(config, TaskConfig)
    assert config.question is not None
    assert len(config.question.options) == 77
    assert "split" not in config.as_dict()
    with pytest.raises(FrozenInstanceError):
        config.name = "changed"


def test_choice_criteria_accept_string_object_array_and_omission(tmp_path):
    path = tmp_path / "generic-choice.toml"
    path.write_text(
        f'''name = "generic choice"
dataset_revision = "{REVISION}"

[question]
type = "choice"
state_field = "message"
id = "decision"
instructions = ["Select one label", "Use only the state"]

[[question.options]]
label = "string"
criterion = "A text criterion"

[[question.options]]
label = "object"
criterion.use_when = "Use it here"
criterion.nested.priority = 2

[[question.options]]
label = "array"
criterion = ["first", "second"]

[[question.options]]
label = "none"
''',
        encoding="utf-8",
    )

    question = load_task_config(path).question
    assert question is not None
    assert question.instructions == ["Select one label", "Use only the state"]
    assert [option.criterion for option in question.options] == [
        "A text criterion",
        {"use_when": "Use it here", "nested": {"priority": 2}},
        ["first", "second"],
        None,
    ]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('type = "choice"', 'type = "score"', "type must be choice"),
        ('label = "card_arrival"', 'label = " "', "label must be a nonempty"),
        (
            'criterion.distinguish_from = "Use card_delivery_estimate',
            'example = "Where is my card?"\n'
            'criterion.distinguish_from = "Use card_delivery_estimate',
            "unexpected=.*example",
        ),
    ],
)
def test_rejects_invalid_task_fields(tmp_path, old, new, message):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_task_config(path)


def test_rejects_toml_values_that_are_not_json_content(tmp_path):
    path = tmp_path / "non-json.toml"
    path.write_text(
        f'''name = "non-JSON criterion"
dataset_revision = "{REVISION}"

[question]
type = "choice"
state_field = "message"
id = "decision"

[[question.options]]
label = "dated"
criterion = 1979-05-27T07:32:00Z
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="criterion must be a string, object, or array"):
        load_task_config(path)
