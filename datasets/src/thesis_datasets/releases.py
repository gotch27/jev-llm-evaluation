"""Content-addressed JSONL releases; no model SDK or training dependencies."""

import errno
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError(f"Invalid dataset or split identifier: {value!r}")
    return value


def release_directory(root: Path, dataset_id: str, version: str) -> Path:
    _component(dataset_id)
    if not isinstance(version, str) or not re.fullmatch(r"[a-f0-9]{64}", version):
        raise ValueError("Dataset version must be a full lowercase SHA-256 digest")
    return root / "releases" / dataset_id / version


def _validate_schema(schema: dict) -> None:
    if not isinstance(schema, dict):
        raise ValueError("Reference schema must be an object")
    kind = schema.get("type")
    if kind == "choice":
        labels = schema.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ValueError("Choice schema needs ordered unique nonempty labels")
    elif kind == "score":
        levels = schema.get("levels")
        if (
            not isinstance(levels, list)
            or len(levels) < 2
            or any(not isinstance(level, str) or not level.strip() for level in levels)
        ):
            raise ValueError("Score schema needs at least two ordered level descriptions")
    elif kind != "noul":
        raise ValueError("Reference schema type must be choice, noul, or score")


def _validate_rows(splits: dict[str, list[dict]], schema: dict) -> None:
    _validate_schema(schema)
    if not isinstance(splits, dict) or not splits:
        raise ValueError("A release needs at least one split")
    ids: set[str] = set()
    for split, rows in splits.items():
        _component(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Split {split} must contain records")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "id",
                "split",
                "inputs",
                "target",
                "metadata",
            }:
                raise ValueError("Records must contain id, split, inputs, target, and metadata")
            example_id = row["id"]
            if not isinstance(example_id, str) or not example_id.strip() or example_id in ids:
                raise ValueError("Example IDs must be nonempty and unique across all splits")
            ids.add(example_id)
            if row["split"] != split:
                raise ValueError(f"Record {example_id} has incorrect split membership")
            if (
                not isinstance(row["inputs"], dict)
                or not row["inputs"]
                or not isinstance(row["metadata"], dict)
                or any(not isinstance(k, str) or not k for k in row["inputs"])
                or any(isinstance(v, str) and not v.strip() for v in row["inputs"].values())
            ):
                raise ValueError(f"Record {example_id} has invalid inputs or metadata")
            target = row["target"]
            kind = schema["type"]
            key = {"choice": "label", "noul": "value", "score": "level"}[kind]
            if (
                not isinstance(target, dict)
                or set(target) != {"type", key}
                or target["type"] != kind
            ):
                raise ValueError(f"Record {example_id} has an incompatible target")
            value = target[key]
            if kind == "choice" and (not isinstance(value, str) or value not in schema["labels"]):
                raise ValueError("Choice target is outside the label inventory")
            if kind == "noul" and type(value) is not bool:
                raise ValueError("Noul target must be a boolean")
            if kind == "score" and (
                type(value) is not int or not 0 <= value < len(schema["levels"])
            ):
                raise ValueError("Score target must be a zero-based level in the rubric")
            encoded(row)  # Reject NaN and non-JSON values before publication.


def publish_release(
    root: Path,
    dataset_id: str,
    source_revision: str,
    splits: dict[str, list[dict]],
    reference_schema: dict,
    provenance: dict,
) -> Path:
    """Publish atomically; identical content reuses a verified immutable release."""
    _component(dataset_id)
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("Source revision must be explicit")
    _validate_rows(splits, reference_schema)
    base = root / "releases" / dataset_id
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publishing-", dir=base) as temporary:
        staged = Path(temporary)
        files = {}
        for split, rows in splits.items():
            path = staged / f"{split}.jsonl"
            with path.open("wb") as stream:
                for row in rows:
                    stream.write(encoded(row) + b"\n")
            files[split] = {"path": path.name, "sha256": checksum(path), "rows": len(rows)}
        identity = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "source_revision": source_revision,
            "reference_schema": reference_schema,
            "files": files,
            "provenance": provenance,
        }
        version = hashlib.sha256(encoded(identity)).hexdigest()
        manifest = {**identity, "version": version}
        (staged / "manifest.json").write_bytes(encoded(manifest) + b"\n")
        destination = release_directory(root, dataset_id, version)
        if destination.exists():
            load_release(root, dataset_id, version, source_revision=source_revision)
        else:
            try:
                staged.rename(destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                load_release(root, dataset_id, version, source_revision=source_revision)
            # TemporaryDirectory tolerates a moved staging directory.
        return destination


def load_release(
    root: Path, dataset_id: str, version: str, *, source_revision: str | None = None
) -> tuple[dict, dict[str, list[dict]]]:
    """Verify identity, every file, targets, and split membership before use."""
    directory = release_directory(root, dataset_id, version)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("version") != version
    ):
        raise ValueError("Dataset release identity or schema version does not match")
    identity = {key: value for key, value in manifest.items() if key != "version"}
    if hashlib.sha256(encoded(identity)).hexdigest() != version:
        raise ValueError("Dataset manifest fingerprint mismatch")
    if source_revision is not None and manifest.get("source_revision") != source_revision:
        raise ValueError("Dataset source revision does not match the configured revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Dataset release has no split files")
    splits = {}
    for split, record in files.items():
        _component(split)
        if not isinstance(record, dict) or record.get("path") != f"{split}.jsonl":
            raise ValueError("Invalid split file path")
        path = directory / record["path"]
        if checksum(path) != record.get("sha256"):
            raise ValueError(f"Dataset checksum mismatch: {split}")
        rows = read_jsonl(path)
        if len(rows) != record.get("rows"):
            raise ValueError(f"Dataset row count mismatch: {split}")
        splits[split] = rows
    _validate_rows(splits, manifest["reference_schema"])
    return manifest, splits
