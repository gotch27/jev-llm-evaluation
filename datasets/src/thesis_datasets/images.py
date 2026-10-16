"""Verified PNG/JPEG assets for immutable dataset releases."""

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Self


def image_metadata(data: bytes) -> dict:
    """Inspect actual pixels without resizing, re-encoding, or trusting a suffix."""
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Image releases require thesis-datasets[images]") from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Assets must be still PNG or JPEG images")
                width, height = image.size
                media_type = "image/png" if image.format == "PNG" else "image/jpeg"
                image.verify()
            # verify() checks structure; load() also checks that pixels decode.
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("Rotate image pixels upright before publishing")
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as e:
        raise ValueError(f"Invalid image asset: {e}") from e
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "media_type": media_type,
        "width": width,
        "height": height,
    }


@dataclass(frozen=True)
class ImageAsset:
    """An image's portable manifest record and its local, verified file."""

    path: Path
    metadata: dict

    @classmethod
    def from_manifest(cls, directory: Path, record: dict) -> Self:
        """Locate a release asset without reading or decoding its file."""
        metadata = {key: value for key, value in record.items() if key != "path"}
        return cls(directory / record["path"], metadata)

    def read_bytes(self) -> bytes:
        data = self.path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != self.metadata["sha256"]
            or len(data) != self.metadata["bytes"]
        ):
            raise ValueError(f"Image checksum mismatch: {self.metadata['sha256']}")
        return data


def validate_image_references(splits: dict, assets: dict) -> None:
    """Require every declared image input to resolve in the frozen asset manifest."""
    if any(not isinstance(key, str) or not key.strip() for key in assets):
        raise ValueError("Image asset IDs must be nonempty strings")
    referenced = set()
    for rows in splits.values():
        for row in rows:
            for value in row["inputs"].values():
                if isinstance(value, dict) and value.get("type") == "image":
                    if (
                        set(value) != {"type", "asset"}
                        or not isinstance(value["asset"], str)
                        or value["asset"] not in assets
                    ):
                        raise ValueError(f"Unknown or malformed image input in {row['id']}")
                    referenced.add(value["asset"])
    if referenced != set(assets):
        raise ValueError("Every image asset must be referenced by a dataset input")


def load_image_assets(directory: Path, records: dict) -> dict[str, ImageAsset]:
    """Verify asset paths, hashes, dimensions, media types, and decoded pixels."""
    if not isinstance(records, dict) or not records:
        raise ValueError("Image releases require a nonempty assets manifest")
    result = {}
    for asset_id, record in records.items():
        if not isinstance(asset_id, str) or not asset_id.strip() or not isinstance(record, dict):
            raise ValueError("Invalid image asset manifest")
        digest = record.get("sha256")
        suffix = {"image/png": "png", "image/jpeg": "jpg"}.get(record.get("media_type"))
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or suffix is None
            or record.get("path") != f"assets/{digest}.{suffix}"
        ):
            raise ValueError("Invalid image asset path or identity")
        asset = ImageAsset.from_manifest(directory, record)
        path = asset.path
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Image asset escapes its dataset release")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Image checksum mismatch: {asset_id}")
        if image_metadata(data) != asset.metadata:
            raise ValueError(f"Image metadata mismatch: {asset_id}")
        result[asset_id] = asset
    return result
