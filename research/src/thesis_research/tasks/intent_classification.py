"""Build the shared BANKING77 state and Choice question from experiment config."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from typesafe_sdk import Choice, Questions

from thesis_research.config import ExperimentConfig

STATE_FIELD = "customer_message"
QUESTION_ID = "intent"


@dataclass(frozen=True)
class IntentClassificationTask:
    """Hold the frozen state shape and Choice shared by every model.

    Attributes:
        state_field: Key used to place the original customer message in state.
        question_id: Key used to identify the classification answer.
        choice: Validated TypeSafe Choice containing all official intents.
    """

    state_field: str
    question_id: str
    choice: Choice

    def build_state(self, text: str) -> dict[str, str]:
        """Wrap one untouched customer message in the configured state field.

        Args:
            text: Original nonempty dataset message.

        Returns:
            Single-field state object accepted by every decision client.

        Raises:
            ValueError: If the message is empty or is not a string.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("Customer message must be a nonempty string")
        return {self.state_field: text}

    def questions(self) -> Questions:
        """Return the provider-neutral question mapping expected by clients."""
        return {self.question_id: self.choice}


def build_banking77_task(
    config: ExperimentConfig, expected_labels: Sequence[str]
) -> IntentClassificationTask:
    """Validate the BANKING77 question and construct its TypeSafe Choice.

    The configuration labels must exactly match the dataset inventory in count,
    spelling, uniqueness, and order. No fallback or ``other`` label is added.

    Args:
        config: Validated experiment configuration containing the question.
        expected_labels: Official dataset labels in source order.

    Returns:
        Frozen state builder and provider-neutral Choice question.

    Raises:
        ValueError: If the label inventory, state field, question ID, option
            order, or criteria are incompatible with BANKING77.
    """
    labels = list(expected_labels)
    if len(labels) != 77 or len(set(labels)) != 77 or not all(_nonempty(label) for label in labels):
        raise ValueError("Expected 77 unique nonempty BANKING77 labels")

    question = config.question
    if question is None:
        raise ValueError("Config must contain a question table")

    if question.state_field != STATE_FIELD:
        raise ValueError(f"Question state_field must be {STATE_FIELD!r}")
    if question.question_id != QUESTION_ID:
        raise ValueError(f"Question id must be {QUESTION_ID!r}")
    parsed_options = [
        (option.label, option.use_when, option.distinguish_from) for option in question.options
    ]

    option_labels = [label for label, _, _ in parsed_options]
    duplicates = sorted(label for label, count in Counter(option_labels).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate question option labels: {duplicates}")

    missing = [label for label in labels if label not in option_labels]
    unexpected = [label for label in option_labels if label not in labels]
    if missing or unexpected:
        raise ValueError(
            "Question options do not match dataset labels; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if option_labels != labels:
        raise ValueError("Question options must follow the dataset label order")

    criteria = {
        label: {"use_when": use_when, "distinguish_from": distinguish_from}
        for label, use_when, distinguish_from in parsed_options
    }
    return IntentClassificationTask(
        state_field=STATE_FIELD,
        question_id=QUESTION_ID,
        choice=Choice(instructions=question.instructions, criteria=criteria),
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
