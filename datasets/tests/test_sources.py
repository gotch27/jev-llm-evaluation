"""Official source publication preserves original bytes, IDs, order, and labels."""

import csv
import json

from thesis_datasets.publish import publish_official_dataset
from thesis_datasets.releases import checksum, load_release
from thesis_datasets.sources.banking77 import FILES, load_prepared_banking77


def test_banking77_publication_preserves_source_files_and_records(tmp_path):
    root = tmp_path / "datasets"
    path = root / "banking77/source/fixture-revision"
    path.mkdir(parents=True)
    labels = [f"label_{i}" for i in range(77)]
    (path / "categories.json").write_text(json.dumps(labels))
    for split in ("train", "test"):
        with (path / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "category"])
            writer.writerows((f"{split} message {i}", label) for i, label in enumerate(labels))
    manifest = {
        "source": "fixture",
        "revision": "fixture-revision",
        "files": {
            name: {"sha256": checksum(path / name), "url": f"fixture:{name}"} for name in FILES
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    before = {p.name: checksum(p) for p in path.iterdir()}
    release = publish_official_dataset(root, "banking77", "fixture-revision", root)
    manifest, splits = load_release(root, "banking77", release.name)
    for split in ("train", "test"):
        _, expected, _ = load_prepared_banking77(root, "fixture-revision", split)
        assert manifest["reference_schema"]["labels"] == labels
        assert [row["id"] for row in splits[split]] == [e.id for e in expected]
        assert [row["inputs"]["text"] for row in splits[split]] == [e.text for e in expected]
        assert [row["target"]["label"] for row in splits[split]] == [e.label for e in expected]
    assert {p.name: checksum(p) for p in path.iterdir()} == before
