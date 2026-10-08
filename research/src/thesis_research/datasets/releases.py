"""Convert shared, verified JSONL releases into benchmark references."""

from pathlib import Path

from thesis_datasets.releases import load_release

from thesis_research.datasets.types import (
    ChoiceReference,
    ChoiceReferenceSchema,
    DatasetExample,
    NoulReference,
    NoulReferenceSchema,
    PreparedDataset,
    ScoreReference,
    ScoreReferenceSchema,
)


def load_dataset_release(
    root: Path, dataset_id: str, revision: str, version: str, split: str
) -> PreparedDataset:
    manifest, splits = load_release(root, dataset_id, version, source_revision=revision)
    if split not in splits:
        raise ValueError(f"Dataset {dataset_id} split must be one of {list(splits)}")
    schema = manifest["reference_schema"]
    kind = schema["type"]
    if kind == "choice":
        reference_schema = ChoiceReferenceSchema(tuple(schema["labels"]))
    elif kind == "noul":
        reference_schema = NoulReferenceSchema()
    else:
        reference_schema = ScoreReferenceSchema(len(schema["levels"]))
    return PreparedDataset(
        dataset_id=dataset_id,
        revision=revision,
        split=split,
        examples=tuple(
            DatasetExample(row["id"], row["inputs"], _reference(row["target"]))
            for row in splits[split]
        ),
        reference_schema=reference_schema,
        manifest=manifest,
    )


def _reference(target: dict):
    if target["type"] == "choice":
        return ChoiceReference(target["label"])
    if target["type"] == "noul":
        return NoulReference(target["value"])
    return ScoreReference(target["level"])
