from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from thesis_research.config import ExperimentConfig, load_experiment_config

CONFIG_PATH = Path(__file__).parents[1] / "experiments" / "intent_classification" / "banking77.toml"


def test_loads_an_immutable_typed_configuration():
    config = load_experiment_config(CONFIG_PATH)

    assert isinstance(config, ExperimentConfig)
    assert config.question is not None
    assert len(config.question.options) == 77
    with pytest.raises(FrozenInstanceError):
        config.split = "train"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('split = "test"', 'split = "validation"', "train or test"),
        (
            'use_when = "A physical card has already been ordered or dispatched and the customer '
            'is tracking it or reporting that it has not arrived."',
            'use_when = " "',
            "use_when must be a nonempty",
        ),
        (
            'distinguish_from = "Use card_delivery_estimate',
            'example = "Where is my card?"\ndistinguish_from = "Use card_delivery_estimate',
            "unexpected=.*example",
        ),
        (
            'distinguish_from = "Use card_delivery_estimate for a general delivery-time question '
            'before a card is overdue, and order_physical_card for obtaining a card."\n',
            "",
            "missing=.*distinguish_from",
        ),
    ],
)
def test_rejects_invalid_configuration_fields(tmp_path, old, new, message):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)
