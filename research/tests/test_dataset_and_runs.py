import csv
import json

import pytest

from thesis_research.datasets import (
    load_banking77,
    load_prepared_banking77,
    prepare_banking77,
    sha256_bytes,
)
from thesis_research.datasets.banking77 import FILES, REVISION


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data" / "banking77" / REVISION
    directory.mkdir(parents=True)
    labels = [f"label_{i}" for i in range(77)]
    (directory / "categories.json").write_text(json.dumps(labels))
    for split in ("train", "test"):
        with (directory / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "category"])
            writer.writerows(
                (f"{split} message {i}, with comma", label) for i, label in enumerate(labels)
            )
    manifest = {
        "revision": REVISION,
        "source": "synthetic fixture",
        "files": {
            name: {
                "sha256": sha256_bytes((directory / name).read_bytes()),
                "url": f"fixture:{name}",
            }
            for name in FILES
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_splits_ids_and_labels(dataset):
    labels, train, _ = load_banking77(dataset, "train")
    _, test, _ = load_banking77(dataset, "test")
    assert len(labels) == 77
    assert train[0].id == "train:1" and train[-1].id == "train:77"
    assert test[0].id == "test:1"
    assert train[0].text == "train message 0, with comma"
    assert train[0].label == "label_0"
    assert not {e.id for e in train} & {e.id for e in test}
    assert load_banking77(dataset, "train")[1] == train


def test_cached_prepare_and_tampering(dataset, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Cached prepare must not use network")

    monkeypatch.setattr("thesis_research.datasets.banking77.urlopen", no_network)
    assert prepare_banking77(dataset.parents[1]) == dataset
    (dataset / "train.csv").write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        load_banking77(dataset, "train")


def test_prepared_loader_rejects_wrong_manifest_revision(dataset):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["revision"] = "different-revision"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest revision"):
        load_prepared_banking77(dataset.parents[1], REVISION, "train")


def test_fresh_download_banking77(dataset, tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(
        "thesis_research.datasets.banking77.urlopen",
        lambda url, timeout: io.BytesIO((dataset / url.rsplit("/", 1)[-1]).read_bytes()),
    )
    fresh = prepare_banking77(tmp_path / "fresh")
    assert load_banking77(fresh, "test")[1] == load_banking77(dataset, "test")[1]
    manifest = json.loads((fresh / "manifest.json").read_text())
    assert REVISION in manifest["files"]["test.csv"]["url"]


def test_cli_inspect(dataset, tmp_path):
    import subprocess
    import sys

    config = tmp_path / "experiment.toml"
    config.write_text(f'name = "synthetic CLI verification"\ndataset_revision = "{REVISION}"\n')
    common = ["--config", str(config), "--data-dir", str(tmp_path / "data")]
    command = [sys.executable, "-m", "thesis_research.cli"]
    inspected = subprocess.run(
        [*command, "inspect", *common, "--limit", "1"], capture_output=True, text=True, check=True
    )
    assert '"id": "train:1"' in inspected.stdout
    assert '"id": "test:1"' not in inspected.stdout


def test_reject_wrong_label_inventory(dataset):
    labels_path = dataset / "categories.json"
    labels_path.write_text(json.dumps(["single_label"]))
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["categories.json"]["sha256"] = sha256_bytes(labels_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="77 unique"):
        load_banking77(dataset, "train")
