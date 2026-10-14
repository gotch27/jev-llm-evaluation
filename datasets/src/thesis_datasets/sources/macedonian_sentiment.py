"""Prepare native Macedonian sentiment splits from a verified local MTEB snapshot."""

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from thesis_datasets.releases import checksum, publish_release
from thesis_datasets.sources.paths import source_directory

DATASET_ID = "macedonian-tweet-sentiment"
REVISION = "6ec78a685dad2e2b37af77264676e9dddfa7dc8e"
SOURCE = "https://huggingface.co/datasets/mteb/MacedonianTweetSentimentClassification"
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
FILES = {split: f"{split}-00000-of-00001.parquet" for split in ("train", "test")}
EXPECTED = {
    "train": {
        "rows": 8566,
        "sha256": "b9e4bb35fcb12faf2e04aba1936f4c4f84219433cad0de7f0a9a3e6eb69fe70a",
        "counts": {-1: 4676, 0: 1280, 1: 2610},
    },
    "test": {
        "rows": 1139,
        "sha256": "8f647d4be9bf8b9f71b9b812b641d7a9daa98cbb7628917dbab51435e1357e60",
        "counts": {-1: 508, 0: 200, 1: 431},
    },
}


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _group_id(text: str) -> str:
    return hashlib.sha256(_text_key(text).encode("utf-8")).hexdigest()


def _record(row: dict, source_split: str, row_number: int) -> dict:
    if (
        not isinstance(row.get("text"), str)
        or not row["text"].strip()
        or type(row.get("label")) is not int
        or row["label"] not in LABELS
    ):
        raise ValueError(f"Invalid sentiment source row: {source_split}:{row_number}")
    return {
        "id": f"{DATASET_ID}:{source_split}:{row_number:06d}",
        "split": source_split,
        "inputs": {"text_mk": row["text"]},
        "target": {"type": "choice", "label": LABELS[row["label"]]},
        "metadata": {
            "source_split": source_split,
            "source_row_number": row_number,
            "original_label": row["label"],
            "duplicate_group": _group_id(row["text"]),
        },
    }


def _quotas(size: int) -> dict[str, int]:
    """Round 80/10/10 by largest remainder, with stable tie-breaking."""
    weights = {"train": 8, "validation": 1, "calibration": 1}
    counts = {split: size * weight // 10 for split, weight in weights.items()}
    ranked = sorted(weights, key=lambda split: -(size * weights[split] % 10))
    for split in ranked[: size - sum(counts.values())]:
        counts[split] += 1
    return counts


def prepare_sentiment_rows(
    source_train: list[dict], source_test: list[dict], *, seed: int = 42
) -> tuple[dict[str, list[dict]], dict]:
    """Remove test overlaps; stratify whole duplicate groups; preserve source text and order."""
    if type(seed) is not int or not source_train or not source_test:
        raise ValueError("Preparation requires nonempty train/test splits and an integer seed")
    train = [_record(row, "train", i) for i, row in enumerate(source_train, start=1)]
    test = [_record(row, "test", i) for i, row in enumerate(source_test, start=1)]
    test_groups = {row["metadata"]["duplicate_group"] for row in test}
    excluded = [row for row in train if row["metadata"]["duplicate_group"] in test_groups]
    remaining = [row for row in train if row["metadata"]["duplicate_group"] not in test_groups]
    grouped = defaultdict(list)
    for row in remaining:
        grouped[row["metadata"]["duplicate_group"]].append(row)
    by_label = defaultdict(list)
    for group, rows in grouped.items():
        labels = {row["target"]["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Conflicting training labels within duplicate group: {group}")
        by_label[rows[0]["target"]["label"]].append(group)
    if set(by_label) != set(LABELS.values()):
        raise ValueError("Every sentiment class must remain in the training pool")

    assignment = {}
    for groups in by_label.values():
        groups.sort(key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
        quotas = _quotas(sum(len(grouped[group]) for group in groups))
        counts = dict.fromkeys(quotas, 0)
        for group in groups:
            # Fill the largest outstanding per-class quota; keep duplicates together.
            split = max(quotas, key=lambda name: quotas[name] - counts[name])
            assignment[group] = split
            counts[split] += len(grouped[group])

    splits = {"train": [], "validation": [], "calibration": [], "test": test}
    for row in remaining:
        row["split"] = assignment[row["metadata"]["duplicate_group"]]
        splits[row["split"]].append(row)
    seen_groups = set()
    for split, rows in splits.items():
        if {row["target"]["label"] for row in rows} != set(LABELS.values()):
            raise ValueError(f"Not enough examples to represent every class in {split}")
        groups = {row["metadata"]["duplicate_group"] for row in rows}
        if groups & seen_groups:
            raise ValueError("Duplicate text groups cross prepared split boundaries")
        seen_groups.update(groups)
    audit = {
        "original_split_counts": {"train": len(train), "test": len(test)},
        "excluded_training_rows": [
            {
                "id": row["id"],
                "source_row_number": row["metadata"]["source_row_number"],
                "duplicate_group": row["metadata"]["duplicate_group"],
                "reason": "Text group appears in the original test split",
            }
            for row in excluded
        ],
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "label_counts": {
            split: dict(sorted(Counter(row["target"]["label"] for row in rows).items()))
            for split, rows in splits.items()
        },
        "cross_split_duplicate_groups": 0,
    }
    return splits, audit


def prepare_macedonian_sentiment(root: Path, revision: str = REVISION, *, seed: int = 42) -> Path:
    """Verify downloaded Parquet files and publish a portable, immutable training release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = source_directory(root, DATASET_ID, revision)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("revision") != revision
        or manifest.get("source") != SOURCE
        or manifest.get("schema") != {"text": "string", "label": "int64"}
        or manifest.get("label_mapping") != {str(k): v for k, v in LABELS.items()}
    ):
        raise ValueError("Sentiment source manifest identity, schema, or labels differ")
    rows = {}
    for split, filename in FILES.items():
        path = directory / filename
        recorded = manifest.get("files", {}).get(filename, {})
        digest = checksum(path)
        if digest != recorded.get("sha256"):
            raise ValueError(f"Sentiment source checksum mismatch: {filename}")
        if revision == REVISION and digest != EXPECTED[split]["sha256"]:
            raise ValueError(f"Pinned upstream checksum mismatch: {filename}")
        table = pq.read_table(path)
        schema = pa.schema([pa.field("text", pa.string()), pa.field("label", pa.int64())])
        if not table.schema.equals(schema, check_metadata=False):
            raise ValueError("Sentiment source must have text:string and label:int64 columns")
        if table.num_rows != recorded.get("rows"):
            raise ValueError(f"Sentiment source row count mismatch: {split}")
        rows[split] = table.to_pylist()
        if revision == REVISION and (
            table.num_rows != EXPECTED[split]["rows"]
            or dict(Counter(row["label"] for row in rows[split])) != EXPECTED[split]["counts"]
        ):
            raise ValueError(f"Pinned sentiment source label counts differ: {split}")
    splits, audit = prepare_sentiment_rows(rows["train"], rows["test"], seed=seed)
    provenance = {
        "source": SOURCE,
        "license": manifest.get("license"),
        "source_manifest_sha256": checksum(manifest_path),
        "source_files": manifest["files"],
        "preparation": {
            "version": 1,
            "seed": seed,
            "split_policy": "80/10/10 within each class of the original training pool",
            "assignment": "SHA-256 seed/group order; largest remaining quota; whole text groups",
            "rounding": "Largest remainder; ties: train, validation, calibration",
            "grouping": "NFC Unicode normalization, casefold, collapsed whitespace; SHA-256",
            "text_policy": "Original Macedonian text preserved without preprocessing",
            "test_policy": (
                "Original test text, labels, and order preserved; overlaps excluded from train"
            ),
            "ids": "Dataset ID, original source split, and one-based source row number",
            "output_order": "Original source order within each prepared split",
        },
        "audit": audit,
    }
    return publish_release(
        root,
        DATASET_ID,
        revision,
        splits,
        {"type": "choice", "labels": list(LABELS.values())},
        provenance,
    )
