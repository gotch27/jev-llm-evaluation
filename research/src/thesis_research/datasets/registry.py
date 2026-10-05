"""Registry and provider-neutral loading for supported datasets."""

from pathlib import Path

from thesis_research.datasets.banking77 import Banking77Adapter
from thesis_research.datasets.boolq import BoolQAdapter
from thesis_research.datasets.types import (
    ChoiceReference,
    ChoiceReferenceSchema,
    DatasetAdapter,
    NoulReference,
    NoulReferenceSchema,
    PreparedDataset,
    ScoreReference,
    ScoreReferenceSchema,
)

_ADAPTERS: dict[str, DatasetAdapter] = {
    adapter.dataset_id: adapter for adapter in (Banking77Adapter(), BoolQAdapter())
}


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    """Return the registered adapter for a dataset identifier."""
    try:
        return _ADAPTERS[dataset_id]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_id}") from error


def prepare_dataset(root: Path, dataset_id: str, revision: str) -> Path:
    """Prepare a registered dataset revision."""
    return get_dataset_adapter(dataset_id).prepare(root, revision)


def load_prepared_dataset(
    root: Path,
    dataset_id: str,
    revision: str,
    split: str,
) -> PreparedDataset:
    """Load one verified split through its registered adapter."""
    adapter = get_dataset_adapter(dataset_id)
    if split not in adapter.supported_splits:
        raise ValueError(
            f"Dataset {dataset_id} split must be one of {list(adapter.supported_splits)}"
        )
    prepared = adapter.load(root, revision, split)
    return validate_prepared_dataset(prepared, dataset_id, revision, split)


def validate_prepared_dataset(
    prepared: PreparedDataset,
    dataset_id: str,
    revision: str,
    split: str,
) -> PreparedDataset:
    """Enforce the reference contract shared by all registered adapters."""
    if (
        prepared.dataset_id != dataset_id
        or prepared.revision != revision
        or prepared.split != split
    ):
        raise ValueError("Prepared dataset identity does not match the requested dataset")
    if not prepared.examples:
        raise ValueError("Prepared dataset split must contain at least one example")
    ids = [example.id for example in prepared.examples]
    if any(not isinstance(example_id, str) or not example_id.strip() for example_id in ids) or len(
        ids
    ) != len(set(ids)):
        raise ValueError("Prepared dataset example IDs must be nonempty and unique")

    schema = prepared.reference_schema
    if isinstance(schema, ChoiceReferenceSchema):
        if (
            not schema.labels
            or any(not isinstance(label, str) or not label.strip() for label in schema.labels)
            or len(schema.labels) != len(set(schema.labels))
        ):
            raise ValueError("Choice reference labels must be nonempty and unique")
        allowed = set(schema.labels)
        if any(
            not isinstance(example.reference, ChoiceReference)
            or not isinstance(example.reference.label, str)
            or example.reference.label not in allowed
            for example in prepared.examples
        ):
            raise ValueError("Choice references must belong to the declared label inventory")
    elif isinstance(schema, NoulReferenceSchema):
        if any(
            not isinstance(example.reference, NoulReference)
            or not isinstance(example.reference.value, bool)
            for example in prepared.examples
        ):
            raise ValueError("Noul datasets must contain only boolean references")
    else:
        assert isinstance(schema, ScoreReferenceSchema)
        if type(schema.level_count) is not int or schema.level_count < 2:
            raise ValueError("Score reference schemas require at least two levels")
        if any(
            not isinstance(example.reference, ScoreReference)
            or type(example.reference.level) is not int
            or not 0 <= example.reference.level < schema.level_count
            for example in prepared.examples
        ):
            raise ValueError("Score references must belong to the configured zero-based rubric")
    return prepared
