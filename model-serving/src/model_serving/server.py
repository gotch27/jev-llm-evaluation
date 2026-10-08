"""Serve one explicit Laya checkpoint through the TypeSafe HTTP contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

CHECKPOINT_PATTERNS = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


@dataclass(frozen=True)
class ServingSettings:
    checkpoint: str
    model_id: str
    revision: str | None = None
    device: str = "mps"
    max_len: int | None = None
    head_max_len: int | None = None
    models_dir: Path = Path("models")

    def __post_init__(self) -> None:
        if not self.checkpoint.strip() or not self.model_id.strip():
            raise ValueError("checkpoint and model_id must be nonempty")
        if self.device not in ("mps", "cpu"):
            raise ValueError("device must be mps or cpu")
        for name in ("max_len", "head_max_len"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_len and self.head_max_len and self.head_max_len > self.max_len:
            raise ValueError("head_max_len must not exceed max_len")


def checkpoint_fingerprint(directory: Path) -> dict[str, Any]:
    """Hash the files Laya loads, excluding Hub cache bookkeeping."""
    paths = [directory / "rl_agent_config.json", directory / "model.safetensors"]
    for folder in ("encoder", "tokenizer"):
        paths.extend(path for path in (directory / folder).rglob("*") if path.is_file())
    files = {}
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"Incomplete checkpoint: missing {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(partial(handle.read, 1024 * 1024), b""):
                digest.update(block)
        files[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def load_checkpoint(settings: ServingSettings) -> tuple[Any, dict[str, Any]]:
    """Resolve a pinned Hub export or a local directory, then preload eager inference."""
    import laya
    from huggingface_hub import snapshot_download

    directory = Path(settings.checkpoint).expanduser()
    if directory.is_dir():
        if settings.revision is not None:
            raise ValueError("A local export uses file fingerprints; omit --revision")
        source = {"kind": "local", "path": str(directory.resolve()), "revision": None}
    else:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", settings.checkpoint):
            raise ValueError("checkpoint must be an existing directory or an explicit Hub repo ID")
        if not settings.revision or not re.fullmatch(r"[0-9a-f]{40}", settings.revision):
            raise ValueError("Hub checkpoints require a pinned 40-character commit --revision")
        destination = (
            settings.models_dir / settings.checkpoint.replace("/", "--") / settings.revision
        )
        directory = Path(
            snapshot_download(
                repo_id=settings.checkpoint,
                revision=settings.revision,
                local_dir=destination,
                allow_patterns=CHECKPOINT_PATTERNS,
            )
        )
        source = {"kind": "hub", "repo_id": settings.checkpoint, "revision": settings.revision}
    # Pass a concrete directory so Laya's alias/router resolution cannot select another model.
    agent = laya.load(str(directory.resolve()), device=settings.device, backend="eager")
    cfg = getattr(agent, "cfg", {})
    max_len = settings.max_len or cfg.get("max_len", 512)
    head_max_len = settings.head_max_len or cfg.get("head_max_len", 192)
    if head_max_len > max_len:
        raise ValueError("Effective head_max_len must not exceed max_len")
    encoder_config = getattr(
        getattr(getattr(agent, "model", None), "encoder", None), "config", None
    )
    capacity = getattr(encoder_config, "max_position_embeddings", None)
    if capacity is not None and max_len > capacity:
        raise ValueError(f"max_len={max_len} exceeds checkpoint encoder capacity {capacity}")
    # Laya may repair tokenizer configuration during load; fingerprint the actual loaded export.
    provenance = {
        "checkpoint": source,
        "fingerprint": checkpoint_fingerprint(directory),
        "requested_device": settings.device,
        "backend": "eager",
        "max_len": max_len,
        "head_max_len": head_max_len,
        "versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("thesis-model-serving", "laya", "torch", "transformers", "fastapi")
            },
        },
    }
    return agent, provenance


class SystemOneRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1)


def validate_questions(questions: dict[str, dict[str, Any]]) -> None:
    for question_id, question in questions.items():
        if not question_id:
            raise ValueError("Every question needs a nonempty ID")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"{question_id}: Choice requires all named options in criteria")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(f"{question_id}: Score requires ordered criteria")
        elif kind != "noul":
            raise ValueError(f"{question_id}: unsupported question type {kind!r}")


def input_loss(response: dict[str, Any], questions: dict[str, dict[str, Any]]) -> str | None:
    usage = response.get("usage", {})
    if usage.get("truncated") or usage.get("state_tokens_dropped", 0) > 0:
        return "State was truncated by the token budget"
    for question_id, stats in usage.get("options", {}).items():
        if stats.get("distinct", 0) < stats.get("total", 0):
            return f"{question_id}: options collapsed into indistinguishable token spans"
    for question_id, question in questions.items():
        answer = response.get("answers", {}).get(question_id, {})
        if question["type"] == "choice":
            expected = set(question["criteria"])
        elif question["type"] == "score":
            expected = {str(i) for i in range(len(question["criteria"]))}
        else:
            continue
        probabilities = {str(key) for key in answer.get("probabilities", {})}
        if probabilities != expected:
            return f"{question_id}: returned probabilities lost or changed options"
    return None


def runtime_metadata(agent: Any, question_count: int = 1) -> dict[str, Any]:
    model = getattr(agent, "model", None)
    parameter = next(model.parameters(), None) if model is not None else None
    dtype_for = getattr(agent, "dtype_for", None)
    return {
        "actual_device": str(getattr(agent, "device", "unknown")),
        "parameter_dtype": str(parameter.dtype) if parameter is not None else "unknown",
        "inference_dtype": str(dtype_for(question_count)) if dtype_for else "unknown",
        "cpu_fallback_count": getattr(agent, "cpu_fallback_count", 0),
        "last_fallback_reason": getattr(agent, "last_fallback_reason", None),
    }


def create_app(settings: ServingSettings, agent: Any, provenance: dict[str, Any]) -> FastAPI:
    """Inject one preloaded agent. A single worker also serializes cancelled requests."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-inference")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="Local Laya TypeSafe service", lifespan=lifespan)

    def infer(payload: SystemOneRequest) -> JSONResponse:
        started = time.perf_counter()
        before = getattr(agent, "cpu_fallback_count", 0)
        raw: dict[str, Any] = {}
        status = 200
        try:
            raw = agent.predict(
                payload.state,
                payload.questions,
                max_len=provenance["max_len"],
                head_max_len=provenance["head_max_len"],
            )
            loss = input_loss(raw, payload.questions)
            if loss:
                status = 422
                raw = {**raw, "error": {"category": "ModelInputLimit", "message": loss}}
        except (ValueError, TypeError) as error:
            status = 422
            raw = {"error": {"category": type(error).__name__, "message": str(error)}}
        except Exception as error:
            status = 500
            raw = {"error": {"category": type(error).__name__, "message": str(error)}}
        metadata = {
            **provenance,
            **runtime_metadata(agent, len(payload.questions)),
            "inference_seconds": time.perf_counter() - started,
            "cpu_fallbacks_this_request": getattr(agent, "cpu_fallback_count", 0) - before,
        }
        # Errors retain native answers/usage; HTTP status prevents accepting them as predictions.
        return JSONResponse(
            {**raw, "model": settings.model_id, "provider": "local", "server_metadata": metadata},
            status_code=status,
        )

    @app.post("/v1/systemone")
    async def system_one(payload: SystemOneRequest):
        try:
            if payload.model != settings.model_id:
                raise ValueError(f"This process serves only model {settings.model_id!r}")
            if payload.state is None:
                raise ValueError("state must not be null")
            validate_questions(payload.questions)
        except ValueError as error:
            return JSONResponse(
                {"error": {"category": "InvalidRequest", "message": str(error)}}, status_code=422
            )
        return await asyncio.get_running_loop().run_in_executor(executor, infer, payload)

    @app.get("/health")
    async def health():
        return {"ready": True, "model": settings.model_id, "server_metadata": provenance}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local export directory or Hub repo ID")
    parser.add_argument("--model-id", required=True, help="Versioned alias accepted in requests")
    parser.add_argument("--revision", help="Required full commit SHA for a Hub checkpoint")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = ServingSettings(
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        models_dir=args.models_dir,
    )
    agent, provenance = load_checkpoint(settings)
    print(
        json.dumps({"model": settings.model_id, **provenance, **runtime_metadata(agent)}, indent=2)
    )
    import uvicorn

    uvicorn.run(create_app(settings, agent, provenance), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
