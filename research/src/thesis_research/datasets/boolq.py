"""BoolQ benchmark adapter over the shared source readers."""

from pathlib import Path

from thesis_datasets.sources.boolq import (
    DATASET_ID,
    EXPECTED,
    FILES,
    LICENSE,
    REVISION,
    SCHEMA,
    SOURCE,
    prepare_boolq,
)
from thesis_datasets.sources.boolq import (
    load_prepared_boolq as load_source_boolq,
)

from thesis_research.datasets.types import (
    DatasetAdapter,
    DatasetExample,
    NoulReference,
    NoulReferenceSchema,
    PreparedDataset,
)

__all__ = [
    "DATASET_ID",
    "EXPECTED",
    "FILES",
    "LICENSE",
    "REVISION",
    "SCHEMA",
    "SOURCE",
    "load_source_boolq",
    "load_prepared_boolq",
    "BoolQAdapter",
    "prepare_boolq",
]


def load_prepared_boolq(root: Path, revision: str, split: str) -> PreparedDataset:
    """Convert verified shared BoolQ records into typed benchmark references."""
    rows, manifest = load_source_boolq(root, revision, split)
    return PreparedDataset(
        dataset_id=DATASET_ID,
        revision=revision,
        split=split,
        examples=tuple(
            DatasetExample(row["id"], row["inputs"], NoulReference(row["target"]["value"]))
            for row in rows
        ),
        reference_schema=NoulReferenceSchema(),
        manifest=manifest,
    )


class BoolQAdapter(DatasetAdapter):
    """Dataset registry adapter for BoolQ."""

    dataset_id = DATASET_ID
    supported_splits = tuple(FILES)

    def prepare(self, root: Path, revision: str) -> Path:
        return prepare_boolq(root, revision)

    def load(self, root: Path, revision: str, split: str) -> PreparedDataset:
        return load_prepared_boolq(root, revision, split)
