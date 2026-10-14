"""Check leakage prevention, source preservation, and reproducible sentiment releases."""

import copy
import json

import pytest

from thesis_datasets.cli import main
from thesis_datasets.releases import checksum, load_release
from thesis_datasets.sources.macedonian_sentiment import (
    DATASET_ID,
    FILES,
    LABELS,
    SOURCE,
    prepare_macedonian_sentiment,
    prepare_sentiment_rows,
)


def _rows(prefix, per_class=40):
    return [
        {"text": f"{prefix} текст {label} број {i}", "label": label}
        for label in LABELS
        for i in range(per_class)
    ]


def _fixture(root):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    revision = "fixture-revision"
    directory = root / DATASET_ID / "source" / revision
    directory.mkdir(parents=True)
    files = {}
    for split in FILES:
        rows = _rows(split, 40 if split == "train" else 1)
        path = directory / FILES[split]
        pq.write_table(pa.Table.from_pylist(rows), path)
        files[path.name] = {"sha256": checksum(path), "rows": len(rows)}
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "source": SOURCE,
        "schema": {"text": "string", "label": "int64"},
        "label_mapping": {str(k): v for k, v in LABELS.items()},
        "license": "cc-by-nc-sa-3.0",
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, revision


def test_stratified_splits_preserve_labels_text_order_and_test():
    train, test = _rows("тренинг"), _rows("тест", 1)
    original_train, original_test = copy.deepcopy(train), copy.deepcopy(test)
    splits, audit = prepare_sentiment_rows(train, test)
    assert train == original_train and test == original_test
    assert audit["split_counts"] == {
        "train": 96,
        "validation": 12,
        "calibration": 12,
        "test": 3,
    }
    assert audit["label_counts"]["validation"] == dict.fromkeys(LABELS.values(), 4)
    assert [row["inputs"]["text_mk"] for row in splits["test"]] == [r["text"] for r in test]
    groups_seen = set()
    ids_seen = set()
    for split, rows in splits.items():
        groups = {row["metadata"]["duplicate_group"] for row in rows}
        ids = {row["id"] for row in rows}
        assert not groups & groups_seen
        assert not ids & ids_seen
        groups_seen.update(groups)
        ids_seen.update(ids)
        source = test if split == "test" else train
        positions = [r["metadata"]["source_row_number"] for r in rows]
        assert positions == sorted(positions)
        for row in rows:
            original = source[row["metadata"]["source_row_number"] - 1]
            assert row["inputs"] == {"text_mk": original["text"]}
            assert row["target"] == {"type": "choice", "label": LABELS[original["label"]]}


def test_test_overlaps_are_excluded_and_training_duplicates_stay_together():
    train = _rows("тренинг") + [
        {"text": "  ДОБРО   утро ", "label": 1},
        {"text": "добро утро", "label": 1},
        {"text": " ист   текст ", "label": 0},
        {"text": "ИСТ текст", "label": 0},
    ]
    test = _rows("тест", 1) + [{"text": "Добро утро", "label": 1}]
    splits, audit = prepare_sentiment_rows(train, test)
    assert len(audit["excluded_training_rows"]) == 2
    assert len(splits["test"]) == len(test)
    retained = [row for split, rows in splits.items() if split != "test" for row in rows]
    assert len(retained) == len(train) - 2
    duplicates = [
        row for row in retained if row["inputs"]["text_mk"] in {" ист   текст ", "ИСТ текст"}
    ]
    assert len(duplicates) == 2
    assert duplicates[0]["split"] == duplicates[1]["split"]
    assert (
        duplicates[0]["metadata"]["duplicate_group"] == duplicates[1]["metadata"]["duplicate_group"]
    )


def test_conflicting_duplicate_labels_fail_instead_of_selecting_an_answer():
    train = _rows("тренинг") + [
        {"text": "добро утро", "label": 1},
        {"text": "ДОБРО УТРО", "label": -1},
    ]
    with pytest.raises(ValueError, match="Conflicting training labels"):
        prepare_sentiment_rows(train, _rows("тест", 1))


@pytest.mark.parametrize("label", [-2, 2, True, "1"])
def test_invalid_labels_fail(label):
    train = _rows("тренинг") + [{"text": "невалидно", "label": label}]
    with pytest.raises(ValueError, match="Invalid sentiment source row"):
        prepare_sentiment_rows(train, _rows("тест", 1))


def test_publication_is_reproducible_and_preserves_source_bytes(tmp_path):
    root = tmp_path / "datasets"
    directory, revision = _fixture(root)
    before = {p.name: checksum(p) for p in directory.iterdir()}
    first = prepare_macedonian_sentiment(root, revision)
    second = prepare_macedonian_sentiment(root, revision)
    assert first == second
    manifest, splits = load_release(root, DATASET_ID, first.name, source_revision=revision)
    assert manifest["reference_schema"] == {"type": "choice", "labels": list(LABELS.values())}
    assert len(splits["train"]) == 96
    assert {p.name: checksum(p) for p in directory.iterdir()} == before
    alternative = prepare_macedonian_sentiment(root, revision, seed=43)
    assert alternative != first
    _, other_splits = load_release(root, DATASET_ID, alternative.name)
    assert other_splits["test"] == splits["test"]
    assert other_splits["validation"] != splits["validation"]


def test_source_checksum_mismatch_rejects_publication(tmp_path):
    root = tmp_path / "datasets"
    directory, revision = _fixture(root)
    with (directory / FILES["train"]).open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        prepare_macedonian_sentiment(root, revision)
    assert not (root / "releases").exists()


def test_prepare_command_publishes_and_reports_splits(tmp_path, monkeypatch, capsys):
    root = tmp_path / "datasets"
    _, revision = _fixture(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "thesis-datasets",
            "prepare",
            "--dataset",
            DATASET_ID,
            "--root",
            str(root),
            "--revision",
            revision,
        ],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["splits"]["train"] == 96
    assert result["excluded_training_rows"] == 0
    assert len(result["version"]) == 64
