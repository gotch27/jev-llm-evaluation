"""BANKING77 benchmark adapter over the shared source readers."""

from pathlib import Path

from thesis_datasets.sources.banking77 import (
    FILES,
    REVISION,
    SOURCE,
    Banking77Example,
    load_banking77,
    load_prepared_banking77,
    prepare_banking77,
    sha256_bytes,
)

from thesis_research.datasets.types import (
    ChoiceReference,
    ChoiceReferenceSchema,
    DatasetAdapter,
    DatasetExample,
    PreparedDataset,
)

__all__ = [
    "FILES",
    "REVISION",
    "SOURCE",
    "Banking77Example",
    "Banking77Adapter",
    "load_banking77",
    "load_prepared_banking77",
    "prepare_banking77",
    "sha256_bytes",
]


class Banking77Adapter(DatasetAdapter):
    """Map shared BANKING77 source rows into the benchmark contract."""

    dataset_id = "banking77"
    supported_splits = ("train", "test")

    def prepare(self, root: Path, revision: str) -> Path:
        return prepare_banking77(root, revision)

    def load(self, root: Path, revision: str, split: str) -> PreparedDataset:
        labels, examples, manifest = load_prepared_banking77(root, revision, split)
        return PreparedDataset(
            dataset_id=self.dataset_id,
            revision=revision,
            split=split,
            examples=tuple(
                DatasetExample(example.id, {"text": example.text}, ChoiceReference(example.label))
                for example in examples
            ),
            reference_schema=ChoiceReferenceSchema(tuple(labels)),
            manifest=manifest,
        )
