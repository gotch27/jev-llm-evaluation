import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from model_serving.server import (
    ServingSettings,
    checkpoint_fingerprint,
    create_app,
    load_checkpoint,
)

QUESTIONS = {
    "intent": {"type": "choice", "criteria": {f"intent-{i}": None for i in range(77)}},
    "yes": {"type": "noul", "instructions": {"question": "Is it true?"}},
    "score": {"type": "score", "criteria": ["bad", "okay", "good"]},
}
PAYLOAD = {"model": "laya-test-v1", "state": {"message": "frozen"}, "questions": QUESTIONS}


class FakeAgent:
    device = "cpu"
    cpu_fallback_count = 0

    def __init__(self, *, error=None, usage=None, delay=0):
        self.error, self.usage, self.delay = error, usage or {}, delay
        self.calls = []
        self.active = self.max_active = 0
        self.started = threading.Event()

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        self.started.set()
        try:
            time.sleep(self.delay)
            if self.error:
                raise self.error
            answers = {
                "intent": {
                    "type": "choice",
                    "choice": "intent-0",
                    "confidence": 1.0,
                    "probabilities": {f"intent-{i}": float(i == 0) for i in range(77)},
                },
                "yes": {"type": "noul", "noul": 0.8},
                "score": {
                    "type": "score",
                    "score": 1.5,
                    "confidence": 0.5,
                    "legend": {0: "bad", 1: "okay", 2: "good"},
                    "probabilities": {0: 0.0, 1: 0.5, 2: 0.5},
                },
            }
            return {
                "model": "laya-rl-agent",
                "answers": answers,
                "usage": {"input_tokens": 400, "output_tokens": 0, **self.usage},
            }
        finally:
            self.active -= 1


def app_for(agent):
    return create_app(
        ServingSettings(checkpoint="unused", model_id="laya-test-v1"),
        agent,
        {"max_len": 8192, "head_max_len": 8192, "fingerprint": {"sha256": "fixture"}},
    )


def test_native_answers_payload_usage_and_provenance():
    agent = FakeAgent()
    with TestClient(app_for(agent)) as client:
        result = client.post("/v1/systemone", json=PAYLOAD)
        assert client.get("/health").json()["ready"]
    assert result.status_code == 200
    body = result.json()
    assert body["model"] == "laya-test-v1" and body["provider"] == "local"
    assert body["answers"]["score"]["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    assert len(body["answers"]["intent"]["probabilities"]) == 77
    assert "cost" not in body["usage"]
    assert body["server_metadata"]["actual_device"] == "cpu"
    assert body["server_metadata"]["fingerprint"]["sha256"] == "fixture"
    assert agent.calls == [(PAYLOAD["state"], QUESTIONS, {"max_len": 8192, "head_max_len": 8192})]


@pytest.mark.parametrize(
    "change",
    [
        {"model": "auto"},
        {"state": None},
        {"questions": {}},
        {"questions": {"q": {"type": "unsupported"}}},
    ],
)
def test_invalid_requests_do_not_run_inference(change):
    agent = FakeAgent()
    with TestClient(app_for(agent)) as client:
        assert client.post("/v1/systemone", json={**PAYLOAD, **change}).status_code == 422
    assert not agent.calls


@pytest.mark.parametrize(
    "usage",
    [
        {"truncated": True, "state_tokens_dropped": 100},
        {"options": {"intent": {"total": 77, "distinct": 30}}},
    ],
)
def test_input_loss_is_an_error_with_native_usage_and_answers(usage):
    with TestClient(app_for(FakeAgent(usage=usage))) as client:
        result = client.post("/v1/systemone", json=PAYLOAD)
    assert result.status_code == 422
    body = result.json()
    assert body["error"]["category"] == "ModelInputLimit"
    assert body["usage"]["input_tokens"] == 400
    assert body["answers"]["yes"]["noul"] == 0.8


def test_missing_options_are_not_accepted():
    class DroppingAgent(FakeAgent):
        def predict(self, *args, **kwargs):
            raw = super().predict(*args, **kwargs)
            raw["answers"]["intent"]["probabilities"].pop("intent-76")
            return raw

    with TestClient(app_for(DroppingAgent())) as client:
        result = client.post("/v1/systemone", json=PAYLOAD)
    assert result.status_code == 422
    assert "options" in result.json()["error"]["message"]


@pytest.mark.parametrize(
    "error,status",
    [(ValueError("only 20 of 77 markers fit"), 422), (RuntimeError("out of memory"), 500)],
)
def test_agent_errors_keep_diagnostics_and_metadata(error, status):
    with TestClient(app_for(FakeAgent(error=error))) as client:
        result = client.post("/v1/systemone", json=PAYLOAD)
    assert result.status_code == status
    assert result.json()["error"]["message"] == str(error)
    assert result.json()["server_metadata"]["max_len"] == 8192


def test_concurrent_and_cancelled_requests_stay_serialized():
    async def run():
        agent = FakeAgent(delay=0.04)
        app = app_for(agent)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://local"
            ) as client:
                first = asyncio.create_task(client.post("/v1/systemone", json=PAYLOAD))
                await asyncio.to_thread(agent.started.wait, 1)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                results = await asyncio.gather(
                    *(client.post("/v1/systemone", json=PAYLOAD) for _ in range(3))
                )
                assert (await client.get("/health")).status_code == 200
                assert all(result.status_code == 200 for result in results)
        assert len(agent.calls) == 4 and agent.max_active == 1

    asyncio.run(run())


def export_fixture(tmp_path):
    (tmp_path / "rl_agent_config.json").write_text(
        json.dumps({"max_len": 512, "head_max_len": 192})
    )
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    for folder in ("encoder", "tokenizer"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "config.json").write_text("{}")
    return tmp_path


def test_checkpoint_fingerprint_tracks_weights_and_ignores_cache(tmp_path):
    export_fixture(tmp_path)
    original = checkpoint_fingerprint(tmp_path)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "download").write_text("unrelated")
    assert checkpoint_fingerprint(tmp_path) == original
    (tmp_path / "model.safetensors").write_bytes(b"new weights")
    assert checkpoint_fingerprint(tmp_path)["sha256"] != original["sha256"]


@pytest.mark.parametrize("hub", [False, True])
def test_load_one_explicit_checkpoint_with_pinned_revision(monkeypatch, tmp_path, hub):
    import huggingface_hub
    import laya

    directory = export_fixture(tmp_path)
    calls, downloads = [], []
    agent = SimpleNamespace(cfg={"max_len": 512, "head_max_len": 192})
    monkeypatch.setattr(laya, "load", lambda *args, **kwargs: calls.append((args, kwargs)) or agent)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs) or str(directory),
    )
    settings = ServingSettings(
        checkpoint="owner/checkpoint" if hub else str(directory),
        model_id="version-v1",
        revision="a" * 40 if hub else None,
        max_len=8192,
        head_max_len=8192,
    )
    loaded, provenance = load_checkpoint(settings)
    assert loaded is agent
    assert calls == [((str(directory.resolve()),), {"device": "mps", "backend": "eager"})]
    assert provenance["max_len"] == 8192
    assert provenance["fingerprint"] == checkpoint_fingerprint(directory)
    if hub:
        assert downloads[0]["revision"] == "a" * 40
        assert downloads[0]["allow_patterns"] == [
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ]
    else:
        assert not downloads


@pytest.mark.parametrize(
    "checkpoint,revision", [("owner/model", None), ("owner/model", "main"), ("laya", "a" * 40)]
)
def test_unpinned_hub_and_automatic_aliases_are_rejected(checkpoint, revision):
    with pytest.raises(ValueError):
        load_checkpoint(ServingSettings(checkpoint=checkpoint, model_id="v1", revision=revision))


def test_local_export_rejects_hub_revision(tmp_path):
    export_fixture(tmp_path)
    with pytest.raises(ValueError, match="omit --revision"):
        load_checkpoint(ServingSettings(checkpoint=str(tmp_path), model_id="v1", revision="a" * 40))
