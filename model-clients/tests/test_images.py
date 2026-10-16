"""Concurrent image transports and fail-fast model/request compatibility."""

import asyncio
import base64
import json
import random
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from PIL import Image
from thesis_datasets.images import ImageAsset, image_metadata
from typesafe_sdk import Choice, Noul, Score

from thesis_model_clients import (
    TypeSafeHTTPDecisionClient,
    VercelLLMDecisionClient,
    VercelSystemOneDecisionClient,
    load_image,
)
from thesis_model_clients.images import image_capability, validate_image_request
from thesis_model_clients.vercel_provider import VercelGatewayProvider


@pytest.fixture
def images(tmp_path):
    result = {}
    for colour in ("red", "blue"):
        path = tmp_path / f"{colour}.png"
        Image.new("RGB", (512, 512), colour).save(path)
        result[colour] = load_image(path)
    return result


@pytest.mark.parametrize(
    "model,provider",
    [("typesafe-ai/jev-1.14.0", "typesafe-ai"), ("openai/gpt-6-luna-decisions", "openai")],
)
def test_system_one_images_send_pixels_with_all_existing_question_types(
    images, monkeypatch, model, provider
):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "unused-test-key")
    requests = []

    async def reply(request):
        body = json.loads(request.content)
        requests.append((request, body))
        await asyncio.sleep(0)
        return httpx2.Response(
            200,
            json={
                "model": model,
                "answers": {
                    "visible": {"type": "noul", "noul": 0.9},
                    "colour": {
                        "type": "choice",
                        "choice": "red",
                        "confidence": 1,
                        "probabilities": {"red": 1, "blue": 0},
                    },
                    "rating": {
                        "type": "score",
                        "score": 1,
                        "confidence": 1,
                        "legend": {"0": "low", "1": "high"},
                        "probabilities": {"0": 0, "1": 1},
                    },
                },
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "provider_metadata": {
                    "gateway": {"cost": "0.0001", "routing": {"resolvedProvider": provider}}
                },
            },
        )

    async def run():
        async with VercelSystemOneDecisionClient(
            model, provider, _transport=httpx2.MockTransport(reply)
        ) as client:
            return await asyncio.gather(
                *(
                    client.evaluate(
                        {"photo": {"type": "image", **asset.metadata}},
                        {
                            "visible": Noul(instructions="Is there a square?"),
                            "colour": Choice(criteria={"red": None, "blue": None}),
                            "rating": Score(criteria=["low", "high"]),
                        },
                        images={"photo": asset},
                    )
                    for asset in images.values()
                )
            )

    results = asyncio.run(run())
    assert len(requests) == 2
    for (request, body), asset in zip(requests, images.values(), strict=True):
        assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
        assert body["providerOptions"] == {"gateway": {"only": [provider]}}
        reference = body["state"]["photo"]
        assert base64.b64decode(body["assets"][reference]["data_base64"]) == asset.read_bytes()
        assert body["assets"][reference]["format"] == "png"
    assert all(set(result.answers) == {"visible", "colour", "rating"} for result in results)
    assert all(result.usage.cost_usd == 0.0001 for result in results)


@pytest.mark.parametrize(
    "model,provider_name",
    [
        ("openai/gpt-5.6-luna", "openai"),
        ("google/gemini-2.5-flash-lite", "google"),
        ("alibaba/qwen3.5-flash", "alibaba"),
    ],
)
def test_llm_image_context_is_isolated_between_concurrent_examples_and_text_calls(
    images, model, provider_name
):
    requests = []

    async def create(**request):
        requests.append(request)
        await asyncio.sleep(0)
        raw = {"model": model, "usage": {"cost": 0.001}}
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answers":{"q":true}}'))],
            model_dump=lambda **kwargs: raw,
        )

    async def close():
        pass

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close
    )
    provider = VercelGatewayProvider(model, provider_name, "unused-test-key", _client=fake)

    async def run():
        async with VercelLLMDecisionClient(
            model, provider_name, output_mode="discrete", _provider=provider
        ) as client:
            await asyncio.gather(
                *(
                    client.evaluate({"photo": "image"}, {"q": Noul()}, images={"photo": asset})
                    for asset in images.values()
                )
            )
            await client.evaluate("text only", {"q": Noul()})

    asyncio.run(run())
    for request, asset in zip(requests[:2], images.values(), strict=True):
        parts = request["messages"][-1]["content"]
        assert parts[-2]["text"] == "Image for state field `photo`:"
        data_url = parts[-1]["image_url"]["url"]
        assert base64.b64decode(data_url.split(",", 1)[1]) == asset.read_bytes()
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["extra_body"]["providerOptions"]["gateway"]["only"] == [provider_name]
    assert isinstance(requests[-1]["messages"][-1]["content"], str)
    assert provider.image_parts.get() is None


def test_request_budget_counts_base64_expansion_and_repeated_image_fields(tmp_path):
    path = tmp_path / "noise.png"
    size = (1568, 1568)
    pixels = random.Random(0).randbytes(size[0] * size[1] * 3)
    Image.frombytes("RGB", size, pixels).save(path)
    asset = ImageAsset(path, image_metadata(path.read_bytes()))
    assert 4 * asset.metadata["bytes"] < 32 * 1024 * 1024
    state = {name: {"type": "image", **asset.metadata} for name in ("a", "b", "c", "d")}
    with pytest.raises(ValueError, match="Encoded image request exceeds"):
        validate_image_request(state, {"q": Noul()}, dict.fromkeys(state, asset))


def test_self_hosted_client_rejects_images_even_with_an_enabled_hosted_model_id(images):
    requests = []

    async def reply(request):
        requests.append(request)
        return httpx2.Response(500)

    async def run():
        async with TypeSafeHTTPDecisionClient(
            "typesafe-ai/jev-1.14.0",
            "typesafe-ai",
            base_url="http://localhost:8000",
            _transport=httpx2.MockTransport(reply),
        ) as client:
            await client.evaluate(
                {"photo": "image"}, {"q": Noul()}, images={"photo": images["red"]}
            )

    with pytest.raises(ValueError, match="self_hosted.*unverified"):
        asyncio.run(run())
    assert requests == []


@pytest.mark.parametrize("provider", ["other", "digitalocean"])
def test_image_capability_requires_exact_provider(provider):
    with pytest.raises(ValueError, match="not enabled"):
        image_capability("system_one", "typesafe-ai/jev-1.14.0", provider)


def test_standalone_image_loader_rejects_undersized_files(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (64, 64), "red").save(path)
    with pytest.raises(ValueError, match="shared image format/size limits"):
        load_image(path)


def test_loaded_image_remains_attached_to_its_file_after_changing_directory(tmp_path, monkeypatch):
    path = tmp_path / "photo.png"
    Image.new("RGB", (512, 512), "red").save(path)
    original = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    asset = load_image(Path("photo.png"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    Image.new("RGB", (512, 512), "blue").save(elsewhere / "photo.png")
    monkeypatch.chdir(elsewhere)
    assert asset.read_bytes() == original
    wire = {"photo": {"type": "image", **asset.metadata}}
    validate_image_request(wire, {"q": Noul()}, {"photo": asset})
