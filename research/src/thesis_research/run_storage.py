"""Shared filesystem and provenance helpers for durable experiment runs."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def create_run_directory(root: Path) -> tuple[Path, datetime]:
    """Create a unique run directory and return it with its UTC start time.

    The timestamp and random UUID prevent repeated experiments from
    overwriting earlier records.
    """
    timestamp = datetime.now(UTC)
    run = root / f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex}"
    run.mkdir(parents=True, exist_ok=False)
    return run, timestamp


def write_json(path: Path, value: object) -> None:
    """Replace a JSON file atomically to avoid partial progress records."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_checksum(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, str | bool | None]:
    """Return the current commit and whether the working tree has any changes.

    Both fields are ``None`` outside a readable Git repository so record
    creation can continue in exported or copied project directories.
    """

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def artifact_metadata(run: Path, names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Record relative paths and SHA-256 checksums for existing run artifacts."""
    return {
        name: {"path": name, "sha256": file_checksum(run / name)}
        for name in names
        if (run / name).exists()
    }
