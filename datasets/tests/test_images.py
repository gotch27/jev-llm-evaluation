"""Image bytes and metadata form part of a release's immutable identity."""

import json

import pytest

from thesis_datasets.images import image_metadata, load_image_assets
from thesis_datasets.releases import load_release, publish_release

Image = pytest.importorskip("PIL.Image")


def publish_image(tmp_path, *, colour="red", kind="noul"):
    source = tmp_path / f"{colour}.png"
    Image.new("RGB", (512, 512), colour).save(source)
    schema = {"type": kind}
    target = {"type": kind, "value": True}
    if kind == "choice":
        schema["labels"] = ["a", "b"]
        target = {"type": kind, "label": "a"}
    elif kind == "score":
        schema["levels"] = ["low", "high"]
        target = {"type": kind, "level": 1}
    rows = {
        "train": [
            {
                "id": "train:1",
                "split": "train",
                "inputs": {"photo": {"type": "image", "asset": "source-photo"}},
                "target": target,
                "metadata": {},
            }
        ]
    }
    release = publish_release(
        tmp_path, "images", "source-v1", rows, schema, {}, image_assets={"source-photo": source}
    )
    return release, rows, schema, source


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_images_preserve_existing_targets_and_publish_reproducibly(tmp_path, kind):
    release, rows, schema, source = publish_image(tmp_path, kind=kind)
    manifest, actual = load_release(tmp_path, "images", release.name)
    assert actual == rows
    assert manifest["schema_version"] == 2
    assets = load_image_assets(release, manifest["assets"])
    assert assets["source-photo"].read_bytes() == source.read_bytes()
    assert assets["source-photo"].metadata["width"] == 512
    before = {
        path.relative_to(release): path.read_bytes()
        for path in release.rglob("*")
        if path.is_file()
    }
    assert (
        publish_release(
            tmp_path, "images", "source-v1", rows, schema, {}, image_assets={"source-photo": source}
        )
        == release
    )
    assert {
        path.relative_to(release): path.read_bytes()
        for path in release.rglob("*")
        if path.is_file()
    } == before
    Image.new("RGB", (512, 512), "blue").save(source)
    assert (
        publish_release(
            tmp_path, "images", "source-v1", rows, schema, {}, image_assets={"source-photo": source}
        )
        != release
    )


def test_modified_image_is_rejected_without_repairing_release(tmp_path):
    release, _, _, _ = publish_image(tmp_path)
    manifest = json.loads((release / "manifest.json").read_text())
    asset = release / manifest["assets"]["source-photo"]["path"]
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Image checksum mismatch"):
        load_release(tmp_path, "images", release.name)
    assert asset.read_bytes() == b"tampered"


def test_missing_assets_are_rejected_before_publication(tmp_path):
    _, rows, schema, _ = publish_image(tmp_path)
    with pytest.raises(ValueError, match="Unknown or malformed image"):
        publish_release(tmp_path, "missing", "source-v1", rows, schema, {})
    assert not (tmp_path / "releases/missing").exists()


def test_manifest_cannot_redirect_assets_outside_release(tmp_path):
    release, _, _, source = publish_image(tmp_path)
    manifest = json.loads((release / "manifest.json").read_text())
    records = manifest["assets"]
    records["source-photo"]["path"] = str(source)
    with pytest.raises(ValueError, match="asset path"):
        load_image_assets(release, records)


def test_release_rejects_files_that_are_not_images(tmp_path):
    _, rows, schema, source = publish_image(tmp_path)
    source.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="Invalid image"):
        publish_release(
            tmp_path,
            "invalid",
            "source-v1",
            rows,
            schema,
            {},
            image_assets={"source-photo": source},
        )


def test_jpeg_metadata_comes_from_decoded_file_content(tmp_path):
    path = tmp_path / "misleading.png"
    Image.new("RGB", (512, 300), "red").save(path, format="JPEG")
    info = image_metadata(path.read_bytes())
    assert info["media_type"] == "image/jpeg"
    assert (info["width"], info["height"]) == (512, 300)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_image_orientation_requires_explicit_preparation(tmp_path, format):
    path = tmp_path / "rotated"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (512, 300), "red").save(path, format=format, exif=exif)
    with pytest.raises(ValueError, match="Rotate image pixels upright"):
        image_metadata(path.read_bytes())


def test_animated_png_is_rejected(tmp_path):
    path = tmp_path / "animated.png"
    Image.new("RGB", (512, 512), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (512, 512), "blue")]
    )
    with pytest.raises(ValueError, match="still PNG or JPEG"):
        image_metadata(path.read_bytes())
