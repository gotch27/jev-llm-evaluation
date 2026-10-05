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
[dataset]
id = "banking77"
revision = "{REVISION}"

[state.fields]
message = "text"

[question]
type = "choice"
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
        ('type = "choice"', 'type = "score"', "missing=.*levels"),
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
[dataset]
id = "banking77"
revision = "{REVISION}"

[state.fields]
message = "text"

[question]
type = "choice"
id = "decision"

[[question.options]]
label = "dated"
criterion = 1979-05-27T07:32:00Z
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="criterion must be a string, object, or array"):
        load_task_config(path)


def test_rejects_retired_task_format(tmp_path):
    path = tmp_path / "old-task.toml"
    path.write_text(f'name = "old task"\ndataset_revision = "{REVISION}"\n')
    with pytest.raises(ValueError, match="missing=.*dataset.*question.*state"):
        load_task_config(path)


@pytest.mark.parametrize("table", ["state", "question"])
def test_requires_state_and_question_tables(tmp_path, table):
    import tomllib

    # Use a minimal current task, omitting exactly the table being checked.
    source = """name = "fixture"
[dataset]
id = "boolq"
revision = "fixture"
[state.fields]
passage = "passage"
[question]
type = "noul"
id = "answer"
"""
    sections = {
        "state": '[state.fields]\npassage = "passage"\n',
        "question": '[question]\ntype = "noul"\nid = "answer"\n',
    }
    path = tmp_path / "missing-table.toml"
    path.write_text(source.replace(sections[table], ""))
    assert table not in tomllib.loads(path.read_text())
    with pytest.raises(ValueError, match=f"missing=.*{table}"):
        load_task_config(path)
