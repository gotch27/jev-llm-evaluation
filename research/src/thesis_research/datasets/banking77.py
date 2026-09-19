"""Pinned BANKING77 data, with original split and row identities preserved."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets"
FILES = ("categories.json", "train.csv", "test.csv")


@dataclass(frozen=True)
class Banking77Example:
    """Represent one unchanged BANKING77 message and its reference intent.

    Attributes:
        id: Stable identifier formed from the split and original row number.
        text: Original customer message.
        label: Official reference intent label.
    """

    id: str
    text: str
    label: str


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for an in-memory byte sequence."""
    return hashlib.sha256(data).hexdigest()


def prepare_banking77(root: Path, revision: str = REVISION) -> Path:
    """Download and verify the three official BANKING77 source files.

    Existing data is verified and reused. Modified cached files cause an error
    instead of being silently replaced.

    Args:
        root: Data root under which ``banking77/<revision>`` is created.
        revision: Full source-repository revision to download.

    Returns:
        Directory containing the dataset files and provenance manifest.

    Raises:
        OSError: If files cannot be downloaded, read, or written.
        ValueError: If cached or downloaded data fails validation.
    """
    directory = root / "banking77" / revision
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        load_prepared_banking77(root, revision, "train")
        load_prepared_banking77(root, revision, "test")
        return directory
    manifest = {"source": SOURCE, "revision": revision, "files": {}}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        destination = directory / name
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"Existing download differs: {destination}")
        destination.write_bytes(data)
        manifest["files"][name] = {"url": url, "sha256": sha256_bytes(data)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_prepared_banking77(root, revision, "train")
    load_prepared_banking77(root, revision, "test")
    return directory


def load_prepared_banking77(
    root: Path,
    revision: str,
    split: str,
) -> tuple[list[str], list[Banking77Example], dict]:
    """Load a prepared split and confirm its requested source revision.

    Args:
        root: Data root containing revision-specific BANKING77 directories.
        revision: Expected source-repository revision.
        split: Official split to load, either ``"train"`` or ``"test"``.

    Returns:
        The verified label inventory, examples, and provenance manifest.

    Raises:
        OSError: If a required dataset file cannot be read.
        ValueError: If the manifest revision or dataset contents are invalid.
    """
    directory = root / "banking77" / revision
    labels, examples, manifest = load_banking77(directory, split)
    if manifest.get("revision") != revision:
        raise ValueError("Dataset manifest revision does not match requested revision")
    return labels, examples, manifest


def load_banking77(directory: Path, split: str) -> tuple[list[str], list[Banking77Example], dict]:
    """Load and verify one prepared BANKING77 split.

    Args:
        directory: Revision directory created by ``prepare_banking77``.
        split: Official split to load, either ``"train"`` or ``"test"``.

    Returns:
        A tuple containing the official label inventory, examples in original
        row order, and the dataset provenance manifest.

    Raises:
        OSError: If a required dataset file cannot be read.
        ValueError: If checksums, labels, columns, or examples are invalid.
    """
    if split not in ("train", "test"):
        raise ValueError("Split must be train or test")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name in FILES:
        if sha256_bytes((directory / name).read_bytes()) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    labels = json.loads((directory / "categories.json").read_text(encoding="utf-8"))
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) and label for label in labels)
        or len(labels) != 77
        or len(set(labels)) != 77
    ):
        raise ValueError("Expected 77 unique intent labels")
    examples = []
    with (directory / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError("Expected text,category CSV columns")
        for row_number, row in enumerate(reader, start=1):
            if set(row) != {"text", "category"} or not row["text"] or row["category"] not in labels:
                raise ValueError(f"Invalid example at {split} row {row_number}")
            examples.append(Banking77Example(f"{split}:{row_number}", row["text"], row["category"]))
    if {example.label for example in examples} != set(labels):
        raise ValueError(f"Split {split} does not contain all 77 labels")
    return labels, examples, manifest
