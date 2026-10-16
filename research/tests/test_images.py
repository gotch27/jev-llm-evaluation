"""Image transport, capability checks, frozen cohorts, and resume integrity."""

import asyncio
import base64
import json
import random
from types import SimpleNamespace

import httpx2
import pytest
from PIL import Image
from thesis_datasets.images import ImageAsset, image_metadata
from thesis_datasets.releases import publish_release
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.benchmark.validation import validate_benchmark
from thesis_research.clients import (
    TypeSafeHTTPDecisionClient,
    VercelLLMDecisionClient,
    VercelSystemOneDecisionClient,
)
from thesis_research.clients.contracts import CallRecord, DecisionResult, UsageTotals
from thesis_research.clients.vercel_provider import VercelGatewayProvider
from thesis_research.config import load_task_config
from thesis_research.images import image_capability, validate_image_request


@pytest.fixture
def images(tmp_path):
    result = {}
    for colour in ("red", "blue"):
        path = tmp_path / f"{colour}.png"
        Image.new("RGB", (512, 512), colour).save(path)
        result[colour] = ImageAsset(path, image_metadata(path.read_bytes()))
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


def write_plan(tmp_path, images, kind="noul", *, models=None):
    key, value, schema = "value", True, {"type": "noul"}
    question = 'type = "noul"\nid = "q"\ninstructions = "Is the image red?"\n'
    if kind == "choice":
        key, value, schema = "label", "red", {"type": "choice", "labels": ["red", "blue"]}
        question = (
            'type = "choice"\nid = "q"\n'
            '[[question.options]]\nlabel = "red"\n'
            '[[question.options]]\nlabel = "blue"\n'
        )
    elif kind == "score":
        key, value, schema = "level", 1, {"type": "score", "levels": ["low", "high"]}
        question = (
            'type = "score"\nid = "q"\n'
            '[[question.levels]]\ncriterion = "low"\n'
            '[[question.levels]]\ncriterion = "high"\n'
        )
    rows = {
        "train": [
            {
                "id": f"train:{i}",
                "split": "train",
                "inputs": {"photo": {"type": "image", "asset": f"secret-label-{colour}"}},
                "target": {"type": kind, key: value},
                "metadata": {"hidden": "reference"},
            }
            for i, colour in enumerate(images, 1)
        ]
    }
    release = publish_release(
        tmp_path,
        "image-fixture",
        "source-v1",
        rows,
        schema,
        {},
        image_assets={f"secret-label-{colour}": asset.path for colour, asset in images.items()},
    )
    task = tmp_path / "task.toml"
    task.write_text(
        f'name = "Image fixture"\n[dataset]\nid = "image-fixture"\n'
        f'revision = "source-v1"\nversion = "{release.name}"\n'
        '[state.images]\nphoto = "photo"\n[question]\n' + question
    )
    plan = tmp_path / "plan.toml"
    models = models or [("jev", "typesafe-ai/jev-1.14.0", "typesafe-ai")]
    plan.write_text(
        'name = "Image fixture"\ntask_config = "task.toml"\nbaseline_model_id = "jev"\n'
        '[cohort]\nsplit = "train"\nstrategy = "all"\n'
        "[execution]\nexample_concurrency = 1\nmodel_concurrency = 2\n"
        + "".join(
            f'[[models]]\nid = "{alias}"\nbackend = "system_one"\n'
            f'model = "{model}"\nprovider = "{provider}"\n'
            for alias, model, provider in models
        )
    )
    return plan, release


class ImageFixtureClient:
    def __init__(self, kind="noul", fail_after=None):
        self.kind, self.fail_after, self.calls = kind, fail_after, []

    async def evaluate(self, state, questions, *, images):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("fixture interruption")
        assert images["photo"].metadata["sha256"] == state["photo"]["sha256"]
        assert images["photo"].read_bytes()
        self.calls.append(state)
        answer = NoulAnswer(noul=0.9)
        if self.kind == "choice":
            answer = ChoiceAnswer(choice="red", confidence=1, probabilities={"red": 1, "blue": 0})
        elif self.kind == "score":
            answer = ScoreAnswer(
                score=1, confidence=1, probabilities={0: 0, 1: 1}, legend={0: "low", 1: "high"}
            )
        usage = UsageTotals(10, 0, 0.001, 0, 0)
        model = "typesafe-ai/jev-1.14.0"
        call = CallRecord(
            tuple(questions),
            model,
            model,
            "typesafe-ai",
            "typesafe-ai",
            0.001,
            usage,
            None,
            {"fixture": True},
        )
        return DecisionResult(
            {"q": answer},
            {},
            model,
            "typesafe-ai",
            (model,),
            ("typesafe-ai",),
            usage,
            0.001,
            (call,),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_image_benchmark_keeps_frozen_pixels_and_existing_metrics(tmp_path, images, kind):
    plan, _ = write_plan(tmp_path, images, kind)
    validation = validate_benchmark(plan, tmp_path)
    assert validation["cohort_examples"] == 2
    assert validation["image_fields"] == ["photo"]
    client = ImageFixtureClient(kind)
    run = asyncio.run(run_benchmark(plan, tmp_path, tmp_path / "outputs", _clients={"jev": client}))
    assert len(client.calls) == 2
    cohort = (run / "cohort.json").read_text()
    assert "secret-label" not in cohort and "hidden" not in cohort
    assert str(tmp_path) not in cohort
    assert json.loads(cohort)["examples"][0]["state"]["photo"]["sha256"]
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["image_capabilities"]["jev"]["protocol"] == "typesafe_assets"
    summary = json.loads((run / "report/summary.json").read_text())
    assert summary["models"]["jev"]["accuracy"] == 1
    assert (run / "report/index.html").exists()


def test_all_incompatible_routes_fail_before_any_model_calls(tmp_path, images):
    plan, _ = write_plan(
        tmp_path,
        images,
        models=[
            ("jev", "typesafe-ai/jev-1.14.0", "typesafe-ai"),
            ("laya", "convaiinnovations/laya", "boundless"),
            ("liquid", "liquid/d1", "liquid"),
        ],
    )
    client = ImageFixtureClient()
    with pytest.raises(ValueError, match="no model calls made") as error:
        asyncio.run(run_benchmark(plan, tmp_path, tmp_path / "outputs", _clients={"jev": client}))
    assert "laya:" in str(error.value) and "liquid:" in str(error.value)
    assert client.calls == []
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("size", [(64, 64), (256, 256), (1600, 512)])
def test_invalid_images_outside_selected_cohort_fail_before_model_calls(tmp_path, images, size):
    path = images["blue"].path
    Image.new("RGB", size, "blue").save(path)
    images["blue"] = ImageAsset(path, image_metadata(path.read_bytes()))
    plan, _ = write_plan(tmp_path, images)
    # Seed 1 selects the valid red image; the blue image still needs full-split checks.
    plan.write_text(
        plan.read_text().replace('strategy = "all"', 'strategy = "random"\nsize = 1\nseed = 1')
    )
    client = ImageFixtureClient()
    with pytest.raises(ValueError, match="shared image format/size limits"):
        asyncio.run(run_benchmark(plan, tmp_path, tmp_path / "outputs", _clients={"jev": client}))
    assert client.calls == []


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


def test_validate_cli_needs_no_gateway_credential(tmp_path, images, monkeypatch, capsys):
    import sys

    from thesis_research.cli import main

    plan, _ = write_plan(tmp_path, images)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["thesis-research", "validate", "--plan", str(plan), "--data-dir", str(tmp_path)],
    )
    main()
    assert json.loads(capsys.readouterr().out)["image_fields"] == ["photo"]


def test_resume_rechecks_image_bytes_and_preserves_completed_examples(tmp_path, images):
    plan, release = write_plan(tmp_path, images)
    with pytest.raises(ValueError, match="preserved record"):
        asyncio.run(
            run_benchmark(
                plan,
                tmp_path,
                tmp_path / "outputs",
                _clients={"jev": ImageFixtureClient(fail_after=1)},
            )
        )
    run = next((tmp_path / "outputs").iterdir())
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    manifest = json.loads((release / "manifest.json").read_text())
    path = release / next(iter(manifest["assets"].values()))["path"]
    original = path.read_bytes()
    path.write_bytes(b"modified")
    client = ImageFixtureClient()
    with pytest.raises(ValueError, match="Image checksum"):
        asyncio.run(resume_benchmark(run, tmp_path, _clients={"jev": client}))
    assert client.calls == []
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    path.write_bytes(original)
    asyncio.run(resume_benchmark(run, tmp_path, _clients={"jev": client}))
    assert len(client.calls) == 1


@pytest.mark.parametrize("completed", [False, True])
def test_capability_changes_block_only_unfinished_image_runs(
    tmp_path, images, monkeypatch, completed
):
    from thesis_research.benchmark import runner

    plan, _ = write_plan(tmp_path, images)
    client = ImageFixtureClient(fail_after=None if completed else 1)
    if completed:
        run = asyncio.run(
            run_benchmark(plan, tmp_path, tmp_path / "outputs", _clients={"jev": client})
        )
    else:
        with pytest.raises(ValueError, match="fixture interruption"):
            asyncio.run(
                run_benchmark(plan, tmp_path, tmp_path / "outputs", _clients={"jev": client})
            )
        run = next((tmp_path / "outputs").iterdir())
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        runner, "validate_image_models", lambda *args: {"jev": {"protocol": "changed"}}
    )
    resumed = ImageFixtureClient()
    if completed:
        assert asyncio.run(resume_benchmark(run, tmp_path, _clients={"jev": resumed})) == run
    else:
        with pytest.raises(ValueError, match="Saved image capabilities differ"):
            asyncio.run(resume_benchmark(run, tmp_path, _clients={"jev": resumed}))
    assert resumed.calls == []
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("provider", ["other", "digitalocean"])
def test_image_capability_requires_exact_provider(provider):
    with pytest.raises(ValueError, match="not enabled"):
        image_capability("system_one", "typesafe-ai/jev-1.14.0", provider)


def test_image_configuration_requires_frozen_release_and_disallows_box(tmp_path, images):
    plan, _ = write_plan(tmp_path, images)
    task = plan.parent / "task.toml"
    text = task.read_text()
    task.write_text("\n".join(line for line in text.splitlines() if not line.startswith("version")))
    with pytest.raises(ValueError, match="published dataset version"):
        load_task_config(task)
    task.write_text(text.replace('type = "noul"', 'type = "box"'))
    with pytest.raises(ValueError, match="choice, noul, or score"):
        load_task_config(task)
