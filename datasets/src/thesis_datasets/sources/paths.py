"""Source locations, including explicitly selected historical cache roots."""

import re
from pathlib import Path


def source_directory(root: Path, dataset_id: str, revision: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", revision):
        raise ValueError("Source revision must be a single pinned path component")
    historical = root / dataset_id / revision
    if historical.exists():
        return historical
    return root / dataset_id / "source" / revision
