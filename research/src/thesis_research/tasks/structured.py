"""Build provider-neutral structured questions from dataset task configuration."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typesafe_sdk import Choice, JSONContent, Noul, Questions, Score

from thesis_research.config import (
    ChoiceQuestionConfig,
    NoulQuestionConfig,
    ScoreQuestionConfig,
    TaskConfig,
)
from thesis_research.datasets import (
    ChoiceReferenceSchema,
    DatasetExample,
    NoulReferenceSchema,
    ReferenceSchema,
    ScoreReferenceSchema,
)
from thesis_research.datasets.types import QuestionType

StructuredQuestion = Choice | Noul | Score


@dataclass(frozen=True)
class StructuredTask:
    """Hold state construction and one validated structured question."""

    state_fields: tuple[tuple[str, str], ...]
    question_id: str
    question_type: QuestionType
    question: StructuredQuestion

    def validate_state_fields(self, examples: Sequence[DatasetExample]) -> None:
        """Check every mapped source field across the split before model calls.

        Report all invalid mappings together, with counts and sample row IDs,
        so a typo cannot hide behind cohort selection or a later example batch.
        """
        problems = []
        for output_name, source_name in self.state_fields:
            missing = 0
            empty = 0
            ids = []
            for example in examples:
                absent = source_name not in example.state
                value = example.state.get(source_name)
                blank = isinstance(value, str) and not value
                missing += absent
                empty += blank
                if (absent or blank) and len(ids) < 3:
                    ids.append(example.id)
            if missing or empty:
                problems.append(
                    f"{output_name!r} <- {source_name!r}: "
                    f"missing in {missing} rows, empty in {empty} rows; example IDs={ids}"
                )
        if problems:
            raise ValueError("Invalid task state mappings: " + "; ".join(problems))

    def build_state(self, source: Mapping[str, JSONContent]) -> dict[str, JSONContent]:
        """Map model input keys (left) to dataset source fields (right)."""
        if not isinstance(source, Mapping):
            raise ValueError("Dataset state must be a mapping of source fields")
        state: dict[str, JSONContent] = {}
        for output_name, source_name in self.state_fields:
            if source_name not in source:
                raise ValueError(f"Dataset example is missing source field: {source_name}")
            value = source[source_name]
            if isinstance(value, str) and not value:
                raise ValueError(f"Dataset source field must be nonempty: {source_name}")
            state[output_name] = value
        return state

    def questions(self) -> Questions:
        """Return the one-question mapping shared by every model."""
        return {self.question_id: self.question}


def build_structured_task(config: TaskConfig, schema: ReferenceSchema) -> StructuredTask:
    """Validate a task against its dataset reference schema and build it."""
    question_config = config.question
    if question_config.question_type != schema.question_type:
        raise ValueError(
            f"Task question type {question_config.question_type!r} does not match "
            f"dataset reference type {schema.question_type!r}"
        )

    if isinstance(question_config, ChoiceQuestionConfig):
        assert isinstance(schema, ChoiceReferenceSchema)
        option_labels = [option.label for option in question_config.options]
        duplicates = sorted(label for label, count in Counter(option_labels).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate question option labels: {duplicates}")
        expected = list(schema.labels)
        option_set = set(option_labels)
        expected_set = set(expected)
        missing = [label for label in expected if label not in option_set]
        unexpected = [label for label in option_labels if label not in expected_set]
        if missing or unexpected:
            raise ValueError(
                "Question options do not match dataset labels; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if option_labels != expected:
            raise ValueError("Question options must follow the dataset label order")
        question: StructuredQuestion = Choice(
            instructions=question_config.instructions,
            criteria={option.label: option.criterion for option in question_config.options},
        )
    elif isinstance(question_config, NoulQuestionConfig):
        assert isinstance(schema, NoulReferenceSchema)
        criteria = None
        if question_config.has_criteria:
            criteria = {
                "true": question_config.true_criterion,
                "false": question_config.false_criterion,
            }
        question = Noul(instructions=question_config.instructions, criteria=criteria)
    else:
        assert isinstance(question_config, ScoreQuestionConfig)
        assert isinstance(schema, ScoreReferenceSchema)
        if len(question_config.levels) != schema.level_count:
            raise ValueError(
                "Score rubric level count does not match the dataset; "
                f"expected={schema.level_count}, actual={len(question_config.levels)}"
            )
        question = Score(
            instructions=question_config.instructions,
            criteria=[level.criterion for level in question_config.levels],
        )

    return StructuredTask(
        state_fields=config.state.fields,
        question_id=question_config.question_id,
        question_type=question_config.question_type,
        question=question,
    )


def class_labels(schema: ReferenceSchema) -> list[str]:
    """Return ordered discrete class labels for metrics and stratification."""
    if isinstance(schema, ChoiceReferenceSchema):
        return list(schema.labels)
    if isinstance(schema, NoulReferenceSchema):
        return ["false", "true"]
    return [str(level) for level in range(schema.level_count)]
