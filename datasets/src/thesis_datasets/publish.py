"""Publish existing source/translation exports without modifying their records."""

import json
from pathlib import Path

from thesis_datasets.releases import checksum, publish_release, read_jsonl


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _split_records(directory: Path, rows: list[dict], splits: tuple[str, ...]) -> None:
    if any(row.get("split") not in splits for row in rows):
        raise ValueError("Export contains an unexpected split")
    for split in splits:
        actual = read_jsonl(directory / f"{split}.jsonl")
        expected = [row for row in rows if row["split"] == split]
        if actual != expected:
            raise ValueError(f"Export {split} records differ from all.jsonl")


def publish_text_dataset(
    root: Path, dataset_id: str, source: Path, translation: Path | None = None
) -> Path:
    """Freeze an existing complete Yelp or SNIPS export, preserving both languages."""
    if dataset_id not in {"yelp", "snips"}:
        raise ValueError("Text export publisher supports yelp and snips")
    manifest = _json(source / "manifest.json")
    source_rows = read_jsonl(source / "all.jsonl")
    splits = (
        ("train", "validation", "calibration", "test")
        if dataset_id == "yelp"
        else ("train", "heldout")
    )
    source_hash = checksum(source / "all.jsonl")
    expected_hash = manifest.get("source_sha256" if dataset_id == "yelp" else "normalized_sha256")
    if source_hash != expected_hash or len(source_rows) != manifest.get("rows"):
        raise ValueError("Source checksum or row count differs from its manifest")
    _split_records(source, source_rows, splits)
    if dataset_id == "yelp":
        for name, expected in manifest.get("normalized_files", {}).items():
            if name not in {"all.jsonl", *(f"{s}.jsonl" for s in splits)}:
                raise ValueError("Unexpected normalized source file")
            if checksum(source / name) != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
        labels = _json(source / "labels.json")
        if labels != [
            {"label": i, "stars": i + 1, "description": f"{i + 1} stars"} for i in range(5)
        ]:
            raise ValueError("Yelp labels must map 0–4 to 1–5 stars")
        schema = {"type": "score", "levels": [f"{i} stars" for i in range(1, 6)]}
    else:
        labels = _json(source / "labels.json")
        if not isinstance(labels, dict) or set(labels) != {str(i) for i in range(7)}:
            raise ValueError("SNIPS labels must contain the seven ordered intents")
        schema = {"type": "choice", "labels": [labels[str(i)] for i in range(7)]}
    provenance = {
        "source_sha256": source_hash,
        "preparation": {
            key: manifest[key]
            for key in (
                "dataset",
                "repository",
                "license",
                "preparation_version",
                "seed",
                "min_chars",
                "max_chars",
                "normalization",
                "sampling",
                "limits",
                "split_counts",
                "counts",
                "heldout_note",
            )
            if key in manifest
        },
    }
    rows = source_rows
    if translation is not None:
        summary = _json(translation / "summary.json")
        if summary.get("complete") is not True or summary.get("missing_rows") != 0:
            raise ValueError("Only complete translation exports can be published")
        if summary.get("retranslation", {}).get("complete", True) is not True:
            raise ValueError("Selective retranslation is not complete")
        translation_source_hash = (
            summary.get("source_manifest", {}).get("source_sha256")
            if dataset_id == "yelp"
            else summary.get("source_sha256")
        )
        if translation_source_hash != source_hash:
            raise ValueError("Translation was produced from a different source snapshot")
        rows = read_jsonl(translation / "all.jsonl")
        _split_records(translation, rows, splits)
        if (
            len(rows) != len(source_rows)
            or summary.get("source_rows") != len(source_rows)
            or summary.get("translated_rows") != len(rows)
            or [r.get("id") for r in rows] != [r.get("id") for r in source_rows]
        ):
            raise ValueError("Translation IDs, order, or counts differ from the source")
        for original, translated in zip(source_rows, rows, strict=True):
            if any(translated.get(key) != value for key, value in original.items()):
                raise ValueError(f"Translation changed original record: {original['id']}")
            if not isinstance(translated.get("text_mk"), str) or not translated["text_mk"].strip():
                raise ValueError("Translation is missing Macedonian input text")
        provenance["translation"] = {
            key: summary[key]
            for key in ("model", "profile", "prompt", "generation_settings", "base_profile")
            if key in summary
        }
        provenance["translation"]["export_sha256"] = checksum(translation / "all.jsonl")
    canonical: dict[str, list[dict]] = {split: [] for split in splits}
    for row in rows:
        if not isinstance(row.get("text_en"), str) or not row["text_en"].strip():
            raise ValueError("Source record is missing English input text")
        label = row.get("label")
        limit = 5 if dataset_id == "yelp" else 7
        if type(label) is not int or not 0 <= label < limit:
            raise ValueError("Invalid zero-based source label")
        if dataset_id == "yelp":
            if row.get("stars") != label + 1:
                raise ValueError("Yelp stars do not match the source label")
            target = {"type": "score", "level": label}
        else:
            if row.get("intent") != schema["labels"][label]:
                raise ValueError("SNIPS intent differs from the label inventory")
            target = {"type": "choice", "label": row["intent"]}
        canonical[row["split"]].append(
            {
                "id": row["id"],
                "split": row["split"],
                "inputs": {key: row[key] for key in ("text_en", "text_mk") if key in row},
                "target": target,
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {"id", "split", "text_en", "text_mk"}
                },
            }
        )
    return publish_release(root, dataset_id, manifest["revision"], canonical, schema, provenance)


def publish_official_dataset(root: Path, dataset_id: str, revision: str, cache: Path) -> Path:
    """Publish verified BANKING77 or BoolQ source files without downloading."""
    if dataset_id == "banking77":
        from thesis_datasets.sources.banking77 import load_prepared_banking77

        splits = {}
        for split in ("train", "test"):
            labels, examples, manifest = load_prepared_banking77(cache, revision, split)
            splits[split] = [
                {
                    "id": example.id,
                    "split": split,
                    "inputs": {"text": example.text},
                    "target": {"type": "choice", "label": example.label},
                    "metadata": {},
                }
                for example in examples
            ]
        schema = {"type": "choice", "labels": labels}
    elif dataset_id == "boolq":
        from thesis_datasets.sources.boolq import FILES, load_prepared_boolq

        splits = {}
        for split in FILES:
            rows, manifest = load_prepared_boolq(cache, revision, split)
            splits[split] = rows
        schema = {"type": "noul"}
    else:
        raise ValueError("Official source publisher supports banking77 and boolq")
    provenance = {
        "source": manifest["source"],
        "files": manifest["files"],
        "preparation": "Original source order, IDs, input fields, and targets preserved.",
    }
    if "license" in manifest:
        provenance["license"] = manifest["license"]
    return publish_release(root, dataset_id, revision, splits, schema, provenance)
