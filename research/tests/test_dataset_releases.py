"""Shared release loading, hidden targets, reporting, and resume."""

import asyncio
import json
from pathlib import Path

import pytest
from thesis_datasets.releases import publish_release
from typesafe_sdk import ScoreAnswer

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.clients import DecisionResult, UsageTotals
from thesis_research.config import DatasetConfig, load_task_config
from thesis_research.datasets import (
    ChoiceReference,
    NoulReference,
    ScoreReference,
    get_dataset_adapter,
    load_prepared_dataset,
    prepare_dataset,
    registry,
)
from thesis_research.tasks import build_structured_task


@pytest.mark.parametrize(
    "dataset_id,schema,target,reference",
    [
        (
            "snips",
            {"type": "choice", "labels": ["a", "b"]},
            {"type": "choice", "label": "b"},
            ChoiceReference("b"),
        ),
        ("boolq", {"type": "noul"}, {"type": "noul", "value": False}, NoulReference(False)),
        (
            "yelp",
            {"type": "score", "levels": ["one", "two"]},
            {"type": "score", "level": 0},
            ScoreReference(0),
        ),
    ],
)
def test_shared_release_maps_targets_and_keeps_metadata_hidden(
    tmp_path, dataset_id, schema, target, reference
):
    row = {
        "id": "train:1",
        "split": "train",
        "inputs": {"text_mk": "Текст"},
        "target": target,
        "metadata": {"stars": 5, "review_flags": ["check"]},
    }
    release = publish_release(tmp_path, dataset_id, "source", {"train": [row]}, schema, {})
    prepared = load_prepared_dataset(tmp_path, dataset_id, "source", "train", version=release.name)
    assert prepared.examples[0].state == {"text_mk": "Текст"}
    assert prepared.examples[0].reference == reference
    assert prepared.manifest["version"] == release.name
    assert prepare_dataset(tmp_path, dataset_id, "source", version=release.name) == release
    with pytest.raises(ValueError, match="source revision"):
        load_prepared_dataset(tmp_path, dataset_id, "wrong", "train", version=release.name)
    with pytest.raises(ValueError, match="split must"):
        load_prepared_dataset(tmp_path, dataset_id, "source", "test", version=release.name)


@pytest.mark.parametrize("dataset_id", ["yelp", "snips"])
def test_derived_datasets_require_frozen_version(tmp_path, dataset_id):
    with pytest.raises(ValueError, match="explicit published"):
        load_prepared_dataset(tmp_path, dataset_id, "source", "train")
    with pytest.raises(ValueError, match="explicit published"):
        prepare_dataset(tmp_path, dataset_id, "source")


def test_historical_dataset_config_serialization_does_not_gain_a_version():
    assert DatasetConfig("boolq", "source").as_dict() == {"id": "boolq", "revision": "source"}


def test_custom_dataset_preparation_retains_its_adapter_contract(tmp_path, monkeypatch):
    class Adapter:
        def prepare(self, root, revision):
            assert root == tmp_path and revision == "source"
            return root / "custom-prepared"

    monkeypatch.setitem(registry._ADAPTERS, "custom-fixture", Adapter())
    assert get_dataset_adapter("custom-fixture").prepare(tmp_path, "source") == (
        tmp_path / "custom-prepared"
    )
    assert prepare_dataset(tmp_path, "custom-fixture", "source") == tmp_path / "custom-prepared"


@pytest.mark.parametrize("version", ["latest", "a" * 63, "A" * 64, 42])
def test_task_rejects_unpinned_dataset_version(tmp_path, version):
    task = tmp_path / "task.toml"
    task.write_text(f"""name = "fixture"
[dataset]
id = "yelp"
revision = "source"
version = {json.dumps(version)}
[state.fields]
review = "text_mk"
[question]
type = "score"
id = "rating"
[[question.levels]]
criterion = "one"
[[question.levels]]
criterion = "two"
""")
    with pytest.raises(ValueError, match="dataset version"):
        load_task_config(task)


@pytest.mark.parametrize(
    "dataset_id,experiment,size",
    [("yelp", "review_rating", 5), ("snips", "intent_classification", 7)],
)
def test_bilingual_plans_pin_same_snapshot_and_only_use_training(dataset_id, experiment, size):
    root = Path(__file__).parents[1] / "experiments" / experiment
    tasks = [
        load_task_config(root / "tasks" / f"{dataset_id}-{language}.toml")
        for language in ("en", "mk")
    ]
    assert tasks[0].dataset == tasks[1].dataset
    for language, task in zip(("en", "mk"), tasks, strict=True):
        assert task.dataset.version is not None
        assert [field for _, field in task.state.fields] == [f"text_{language}"]
        plan = load_benchmark_plan(
            root / "benchmarks" / f"{dataset_id}-{language}-local-smoke.toml"
        )
        assert plan.cohort.split == "train" and plan.cohort.size == size
        assert plan.models[0].transport == "self_hosted"


def test_yelp_release_benchmark_report_and_resume_skip_completed_rows(tmp_path):
    rows = [
        {
            "id": f"train:{i}",
            "split": "train",
            "inputs": {"text_mk": f"Оцена {i}"},
            "target": {"type": "score", "level": i},
            "metadata": {"stars": i + 1},
        }
        for i in range(2)
    ]
    release = publish_release(
        tmp_path / "data",
        "yelp",
        "source",
        {"train": rows},
        {"type": "score", "levels": ["one", "two"]},
        {"seed": 42},
    )
    task_path = tmp_path / "task.toml"
    task_path.write_text(f'''name = "Yelp fixture"
[dataset]
id = "yelp"
revision = "source"
version = "{release.name}"
[state.fields]
review = "text_mk"
[question]
type = "score"
id = "rating"
[[question.levels]]
criterion = "one"
[[question.levels]]
criterion = "two"
''')
    task = build_structured_task(
        load_task_config(task_path),
        load_prepared_dataset(
            tmp_path / "data", "yelp", "source", "train", version=release.name
        ).reference_schema,
    )
    assert task.build_state(rows[0]["inputs"]) == {"review": "Оцена 0"}
    plan = tmp_path / "plan.toml"
    plan.write_text("""name = "Yelp local fixture"
baseline_model_id = "local"
task_config = "task.toml"
[cohort]
split = "train"
strategy = "random"
size = 2
seed = 42
[execution]
example_concurrency = 1
model_concurrency = 1
[[models]]
id = "local"
backend = "system_one"
transport = "self_hosted"
model = "fixture"
provider = "local"
base_url = "http://127.0.0.1:8000"
""")

    class Client:
        def __init__(self, fail=False):
            self.states = []
            self.fail = fail

        async def evaluate(self, state, questions):
            if self.fail and len(self.states) == 1:
                raise RuntimeError("interrupted")
            self.states.append(state)
            level = int(state["review"].split()[-1])
            answer = ScoreAnswer(
                score=float(level),
                confidence=1.0,
                probabilities={0: float(level == 0), 1: float(level == 1)},
                legend={0: "one", 1: "two"},
            )
            return DecisionResult(
                answers={"rating": answer},
                errors={},
                requested_model="fixture",
                requested_provider="local",
                resolved_models=("fixture",),
                resolved_providers=("local",),
                usage=UsageTotals(10, 1, None, 0, 0),
                latency_seconds=0.01,
                calls=(),
            )

        async def aclose(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.aclose()

    interrupted = Client(fail=True)
    with pytest.raises(ValueError, match="preserved record"):
        asyncio.run(
            run_benchmark(
                plan, tmp_path / "data", tmp_path / "outputs", _clients={"local": interrupted}
            )
        )
    run = next((tmp_path / "outputs").iterdir())
    resumed = Client()
    asyncio.run(resume_benchmark(run, tmp_path / "data", _clients={"local": resumed}))
    assert len(resumed.states) == 1 and resumed.states[0] != interrupted.states[0]
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["dataset"]["version"] == release.name
    assert metadata["status"] == "completed"
    assert (run / "report/index.html").exists()
    report = json.loads((run / "report/summary.json").read_text())
    assert report["models"]["local"]["accuracy"] == 1
