"""Image transport, capability checks, frozen cohorts, and resume integrity."""

import asyncio
import json

import pytest
from PIL import Image
from thesis_datasets.images import ImageAsset, image_metadata
from thesis_datasets.releases import publish_release
from thesis_model_clients import CallRecord, DecisionResult, UsageTotals
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.benchmark.validation import validate_benchmark
from thesis_research.config import load_task_config


@pytest.fixture
def images(tmp_path):
    result = {}
    for colour in ("red", "blue"):
        path = tmp_path / f"{colour}.png"
        Image.new("RGB", (512, 512), colour).save(path)
        result[colour] = ImageAsset(path, image_metadata(path.read_bytes()))
    return result


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
