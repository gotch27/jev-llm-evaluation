"""Explicit image-route support and wire encodings for existing decision types."""

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from thesis_datasets.images import ImageAsset, image_metadata

ImageInputs = Mapping[str, ImageAsset]
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
IMAGE_REQUEST_LIMIT_BYTES = 32 * 1024 * 1024
_IMAGE_ROUTES = {
    ("system_one", "typesafe-ai/jev-1.14.0", "typesafe-ai"): "live Noul image check",
    ("system_one", "openai/gpt-6-luna-decisions", "openai"): "live Noul image check",
    ("llm", "openai/gpt-5.6-luna", "openai"): "documented vision and structured output",
    ("llm", "google/gemini-2.5-flash-lite", "google"): "documented vision and structured output",
    ("llm", "alibaba/qwen3.5-flash", "alibaba"): "documented vision and structured output",
}
_ASSET_ROUTE_FAILURES = {
    ("system_one", "liquid/d1", "liquid"): "Gateway TypeSafe image-asset request returned HTTP 422",
    ("system_one", "liquid/d1-ultra", "liquid"): (
        "Gateway TypeSafe image-asset request returned HTTP 422"
    ),
    ("system_one", "perplexity/pplx-decider-v1.1-27b", "perplexity"): (
        "Gateway TypeSafe image-asset request returned HTTP 400"
    ),
}


def load_image(path: Path) -> ImageAsset:
    """Inspect a local PNG/JPEG for a standalone call, without publishing a dataset."""
    path = path.resolve(strict=True)
    asset = ImageAsset(path, image_metadata(path.read_bytes()))
    validate_image_inputs({"image": asset})
    return asset


def image_capability(backend: str, model: str, provider: str, transport: str = "vercel") -> dict:
    """Reject unsupported and unverified routes rather than silently dropping images."""
    route = (backend, model, provider)
    evidence = _IMAGE_ROUTES.get(route) if transport == "vercel" else None
    if evidence is None:
        reason = _ASSET_ROUTE_FAILURES.get(route) if transport == "vercel" else None
        reason = reason or "this exact model/provider/transport is unverified"
        raise ValueError(
            f"Image input is not enabled for {model} via {provider} ({transport}): {reason}"
        )
    return {
        "checked_at": "2026-10-15",
        "protocol": "typesafe_assets" if backend == "system_one" else "chat_image_url",
        "evidence": evidence,
        "headers": dict(IMAGE_HEADERS) if backend == "system_one" else {},
        "detail": None if backend == "system_one" else "auto",
    }


def validate_image_inputs(images: ImageInputs) -> None:
    """Enforce the initial shared PNG/JPEG contract before sending any request."""
    expanded_bytes = 0
    for name, asset in images.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(asset, ImageAsset):
            raise ValueError("Image inputs must map nonempty state names to ImageAsset objects")
        info = asset.metadata
        width, height = info["width"], info["height"]
        if (
            info["media_type"] not in {"image/png", "image/jpeg"}
            or not 256 <= width <= 1568
            or not 256 <= height <= 1568
            or not 78_400 <= width * height <= 2_458_624
            or info["bytes"] > 16 * 1024 * 1024
        ):
            raise ValueError(f"Image {name!r} exceeds the shared image format/size limits")
        asset.read_bytes()
        expanded_bytes += info["bytes"]
    if expanded_bytes > IMAGE_REQUEST_LIMIT_BYTES:
        raise ValueError("Expanded image inputs exceed 32 MiB")


def validate_image_request(state: Any, questions: Mapping, images: ImageInputs) -> None:
    """Bound the base64-expanded request before any participant starts."""
    validate_image_inputs(images)
    if not isinstance(state, dict) or not set(images) <= set(state):
        raise ValueError("Each image must identify an existing top-level state field")
    text = {
        "state": state,
        "questions": {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in questions.items()
        },
    }
    # Reserve framing space and count repeated image uses, including base64 expansion.
    estimated = len(json.dumps(text).encode("utf-8")) + 512 * (len(images) + 1)
    estimated += sum(4 * ((asset.metadata["bytes"] + 2) // 3) for asset in images.values())
    if estimated > IMAGE_REQUEST_LIMIT_BYTES:
        raise ValueError("Encoded image request exceeds the shared 32 MiB budget")


def typesafe_image_body(state: Any, images: ImageInputs) -> tuple[dict, dict]:
    """Replace declared state image fields with opaque references and upload assets."""
    validate_image_inputs(images)
    if not isinstance(state, dict) or not set(images) <= set(state):
        raise ValueError("Each image must identify an existing top-level state field")
    wire_state = dict(state)
    assets = {}
    for name, asset in images.items():
        reference = "image-" + asset.metadata["sha256"]
        wire_state[name] = reference
        assets[reference] = {
            "type": "image",
            "format": "png" if asset.metadata["media_type"] == "image/png" else "jpg",
            "data_base64": base64.b64encode(asset.read_bytes()).decode("ascii"),
        }
    return wire_state, assets


def chat_image_parts(images: ImageInputs) -> list[dict]:
    """Label each image with its task-state field and send actual image message parts."""
    validate_image_inputs(images)
    parts = []
    for name, asset in images.items():
        data = base64.b64encode(asset.read_bytes()).decode("ascii")
        parts.extend(
            [
                {"type": "text", "text": f"Image for state field `{name}`:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{asset.metadata['media_type']};base64,{data}",
                        "detail": "auto",
                    },
                },
            ]
        )
    return parts
