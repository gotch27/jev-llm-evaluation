"""Training and evaluation must use the same immutable records and split assignments."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier

import pytest

from thesis_datasets.publish import publish_text_dataset
from thesis_datasets.releases import checksum, load_release, publish_release


def record(example_id="train:1", split="train", kind="score", value=0):
    key = {"choice": "label", "noul": "value", "score": "level"}[kind]
    return {
        "id": example_id,
        "split": split,
        "inputs": {"text_en": "Review\u2028with a break"},
        "target": {"type": kind, key: value},
        "metadata": {"stars": 1},
    }


@pytest.mark.parametrize(
    "schema,kind,value",
    [
        ({"type": "choice", "labels": ["a", "b"]}, "choice", "b"),
        ({"type": "noul"}, "noul", False),
        ({"type": "score", "levels": ["1 star", "2 stars"]}, "score", 0),
    ],
)
def test_release_preserves_all_reference_types_and_is_reproducible(tmp_path, schema, kind, value):
    rows = {"train": [record(kind=kind, value=value)]}
    first = publish_release(tmp_path, "example", "source-commit", rows, schema, {"seed": 42})
    before = {path.name: path.read_bytes() for path in first.iterdir()}
    assert (
        publish_release(tmp_path, "example", "source-commit", rows, schema, {"seed": 42}) == first
    )
    manifest, actual = load_release(
        tmp_path, "example", first.name, source_revision="source-commit"
    )
    assert actual == rows
    assert manifest["source_revision"] == "source-commit"
    assert manifest["version"] != "source-commit"
    assert {path.name: path.read_bytes() for path in first.iterdir()} == before
    changed = deepcopy(rows)
    changed["train"][0]["inputs"]["text_en"] += " edited"
    assert (
        publish_release(tmp_path, "example", "source-commit", changed, schema, {"seed": 42})
        != first
    )


def test_release_rejects_tampering_and_never_repairs_in_place(tmp_path):
    schema = {"type": "score", "levels": ["one", "two"]}
    rows = {"train": [record()]}
    path = publish_release(tmp_path, "yelp", "revision", rows, schema, {})
    (path / "train.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        load_release(tmp_path, "yelp", path.name)
    with pytest.raises(ValueError, match="checksum"):
        publish_release(tmp_path, "yelp", "revision", rows, schema, {})
    assert (path / "train.jsonl").read_text() == "changed\n"


def test_concurrent_publication_reuses_the_same_verified_release(tmp_path, monkeypatch):
    rows = {"train": [record()]}
    schema = {"type": "score", "levels": ["one", "two"]}
    barrier = Barrier(2)
    rename = Path.rename

    def concurrent_rename(self, target):
        barrier.wait(timeout=10)
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", concurrent_rename)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_release, tmp_path, "yelp", "source", rows, schema, {})
            for _ in range(2)
        ]
        first, second = [future.result(timeout=15) for future in futures]
    assert first == second
    _, actual = load_release(tmp_path, "yelp", first.name)
    assert actual == rows
    assert list((tmp_path / "releases/yelp").iterdir()) == [first]


def test_manifest_and_source_revision_are_verified(tmp_path):
    path = publish_release(
        tmp_path,
        "yelp",
        "revision",
        {"train": [record()]},
        {"type": "score", "levels": ["one", "two"]},
        {},
    )
    with pytest.raises(ValueError, match="source revision"):
        load_release(tmp_path, "yelp", path.name, source_revision="other")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["provenance"]["seed"] = 99
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        load_release(tmp_path, "yelp", path.name)


@pytest.mark.parametrize("level", [-1, 2, True, "1"])
def test_invalid_score_mapping_is_rejected_before_publication(tmp_path, level):
    with pytest.raises(ValueError, match="zero-based"):
        publish_release(
            tmp_path,
            "yelp",
            "revision",
            {"train": [record(value=level)]},
            {"type": "score", "levels": ["one", "two"]},
            {},
        )
    assert not (tmp_path / "releases").exists()


def test_cross_split_id_reuse_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique across"):
        publish_release(
            tmp_path,
            "yelp",
            "revision",
            {"train": [record()], "test": [record(split="test")]},
            {"type": "score", "levels": ["one", "two"]},
            {},
        )


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture(params=["yelp", "snips"])
def text_export(tmp_path, request):
    dataset_id = request.param
    source = tmp_path / "source"
    translated = tmp_path / "translated"
    source.mkdir()
    translated.mkdir()
    splits = (
        ("train", "validation", "calibration", "test")
        if dataset_id == "yelp"
        else ("train", "heldout")
    )
    limit = 5 if dataset_id == "yelp" else 7
    rows = []
    for split in splits:
        for label in range(limit):
            row = {
                "id": f"{split}:{label}",
                "split": split,
                "label": label,
                "text_en": f"Original {split} {label}",
                "duplicate_group": f"{split}:{label}",
            }
            if dataset_id == "yelp":
                row["stars"] = label + 1
            else:
                row["intent"] = f"intent_{label}"
            rows.append(row)
    write_jsonl(source / "all.jsonl", rows)
    for split in splits:
        write_jsonl(source / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    manifest = {
        "revision": "source-commit",
        "rows": len(rows),
        "seed": 42,
        "source_sha256" if dataset_id == "yelp" else "normalized_sha256": checksum(
            source / "all.jsonl"
        ),
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    labels = (
        [{"label": i, "stars": i + 1, "description": f"{i + 1} stars"} for i in range(5)]
        if (dataset_id == "yelp")
        else {str(i): f"intent_{i}" for i in range(7)}
    )
    (source / "labels.json").write_text(json.dumps(labels))
    translated_rows = [
        {**row, "text_mk": f"Превод {row['id']}", "review_flags": ["check"]} for row in rows
    ]
    write_jsonl(translated / "all.jsonl", translated_rows)
    for split in splits:
        write_jsonl(
            translated / f"{split}.jsonl", [r for r in translated_rows if r["split"] == split]
        )
    summary = {
        "complete": True,
        "missing_rows": 0,
        "source_rows": len(rows),
        "translated_rows": len(rows),
        "source_sha256": checksum(source / "all.jsonl"),
        "source_manifest": manifest,
        "profile": "profile",
        "model": "translation-model",
    }
    (translated / "summary.json").write_text(json.dumps(summary))
    return dataset_id, source, translated, rows


def test_text_snapshot_preserves_ids_languages_targets_and_splits(tmp_path, text_export):
    dataset_id, source, translated, original = text_export
    before = {
        str(path): checksum(path) for folder in (source, translated) for path in folder.iterdir()
    }
    path = publish_text_dataset(tmp_path / "shared", dataset_id, source, translated)
    manifest, splits = load_release(tmp_path / "shared", dataset_id, path.name)
    for source_row in original:
        row = next(r for r in splits[source_row["split"]] if r["id"] == source_row["id"])
        assert row["inputs"] == {"text_en": source_row["text_en"], "text_mk": f"Превод {row['id']}"}
        assert row["metadata"]["label"] == source_row["label"]
        assert "review_flags" not in row["inputs"] and "label" not in row["inputs"]
        if dataset_id == "yelp":
            assert row["target"] == {"type": "score", "level": source_row["stars"] - 1}
        else:
            assert row["target"] == {"type": "choice", "label": source_row["intent"]}
    assert manifest["provenance"]["translation"]["profile"] == "profile"
    assert {
        str(path): checksum(path) for folder in (source, translated) for path in folder.iterdir()
    } == before
    assert publish_text_dataset(tmp_path / "shared", dataset_id, source, translated) == path


@pytest.mark.parametrize(
    "failure", ["incomplete", "wrong_source", "split_changed", "target_changed"]
)
def test_bad_translation_is_rejected(tmp_path, text_export, failure):
    dataset_id, source, translated, _ = text_export
    summary_path = translated / "summary.json"
    summary = json.loads(summary_path.read_text())
    if failure == "incomplete":
        summary["complete"] = False
    elif failure == "wrong_source":
        summary["source_sha256"] = "wrong"
        summary["source_manifest"]["source_sha256"] = "wrong"
    else:
        rows = [json.loads(line) for line in (translated / "all.jsonl").read_text().splitlines()]
        rows[0]["split" if failure == "split_changed" else "label"] = "changed"
        write_jsonl(translated / "all.jsonl", rows)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        publish_text_dataset(tmp_path / "shared", dataset_id, source, translated)
    assert not (tmp_path / "shared/releases").exists()
