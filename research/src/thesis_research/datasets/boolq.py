"""Pinned Google BoolQ data loaded from Hugging Face Parquet files."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from thesis_research.datasets.types import (
    DatasetAdapter,
    DatasetExample,
    NoulReference,
    NoulReferenceSchema,
    PreparedDataset,
)

DATASET_ID = "boolq"
REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
SOURCE = "https://huggingface.co/datasets/google/boolq"
LICENSE = "CC BY-SA 3.0"
SCHEMA = {"question": "string", "answer": "bool", "passage": "string"}
FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
}
EXPECTED = {
    "train": {
        "rows": 9427,
        "sha256": "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623",
        "reference_counts": {"false": 3553, "true": 5874},
    },
    "validation": {
        "rows": 3270,
        "sha256": "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
        "reference_counts": {"false": 1237, "true": 2033},
    },
}
_SCHEMA = pa.schema(
    [
        pa.field("question", pa.string(), nullable=True),
        pa.field("answer", pa.bool_(), nullable=True),
        pa.field("passage", pa.string(), nullable=True),
    ]
)


def prepare_boolq(root: Path, revision: str = REVISION) -> Path:
    """Download and verify the two labeled BoolQ splits at one revision."""
    directory = root / DATASET_ID / revision
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        for split in FILES:
            load_prepared_boolq(root, revision, split)
        return directory

    manifest = {
        "dataset_id": DATASET_ID,
        "source": SOURCE,
        "revision": revision,
        "license": LICENSE,
        "schema": SCHEMA,
        "files": {},
    }
    for split, name in FILES.items():
        url = f"{SOURCE}/resolve/{revision}/data/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        checksum = hashlib.sha256(data).hexdigest()
        expected = EXPECTED.get(split) if revision == REVISION else None
        if expected is not None and checksum != expected["sha256"]:
            raise ValueError(f"Downloaded BoolQ checksum mismatch: {name}")
        destination = directory / name
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"Existing download differs: {destination}")
        destination.write_bytes(data)
        manifest["files"][name] = {
            "split": split,
            "url": url,
            "sha256": checksum,
            "rows": expected["rows"] if expected is not None else None,
            "reference_counts": (expected["reference_counts"] if expected is not None else None),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split in FILES:
        load_prepared_boolq(root, revision, split)
    return directory


def load_prepared_boolq(root: Path, revision: str, split: str) -> PreparedDataset:
    """Load a verified labeled BoolQ split from the data root."""
    if split not in FILES:
        raise ValueError(f"BoolQ split must be one of {sorted(FILES)}")
    directory = root / DATASET_ID / revision
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("revision") != revision:
        raise ValueError("BoolQ manifest does not match the requested dataset revision")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("BoolQ manifest does not record the expected source schema")
    if revision == REVISION and (
        manifest.get("source") != SOURCE or manifest.get("license") != LICENSE
    ):
        raise ValueError("BoolQ manifest provenance does not match the pinned source")
    name = FILES[split]
    path = directory / name
    file_record = manifest.get("files", {}).get(name)
    if not isinstance(file_record, dict):
        raise ValueError(f"BoolQ manifest has no record for {name}")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != file_record.get("sha256"):
        raise ValueError(f"Checksum mismatch: {name}")
    canonical = EXPECTED.get(split) if revision == REVISION else None
    if canonical is not None and checksum != canonical["sha256"]:
        raise ValueError(f"Pinned BoolQ checksum mismatch: {name}")

    table = pq.read_table(path)
    if not table.schema.equals(_SCHEMA, check_metadata=False):
        raise ValueError("BoolQ columns must be question:string, answer:bool, passage:string")
    expected_rows = canonical["rows"] if canonical is not None else file_record.get("rows")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(f"BoolQ {split} row count does not match the manifest")

    examples: list[DatasetExample] = []
    for row_number, row in enumerate(table.to_pylist(), start=1):
        question = row.get("question")
        passage = row.get("passage")
        answer = row.get("answer")
        if not isinstance(question, str) or not question:
            raise ValueError(f"Invalid BoolQ question at {split} row {row_number}")
        if not isinstance(passage, str) or not passage:
            raise ValueError(f"Invalid BoolQ passage at {split} row {row_number}")
        if not isinstance(answer, bool):
            raise ValueError(f"Invalid BoolQ answer at {split} row {row_number}")
        examples.append(
            DatasetExample(
                id=f"{split}:{row_number}",
                state={"question": question, "passage": passage},
                reference=NoulReference(answer),
            )
        )
    if {example.reference.value for example in examples} != {False, True}:
        raise ValueError(f"BoolQ {split} must contain both boolean labels")
    if canonical is not None:
        actual_counts = {
            "false": sum(not example.reference.value for example in examples),
            "true": sum(example.reference.value for example in examples),
        }
        if actual_counts != canonical["reference_counts"]:
            raise ValueError(f"BoolQ {split} boolean distribution does not match the source")
    return PreparedDataset(
        dataset_id=DATASET_ID,
        revision=revision,
        split=split,
        examples=tuple(examples),
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
