"""Coordinated benchmark execution, recovery, and statistical reports."""

import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typesafe_sdk import ChoiceAnswer

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.benchmark.cohort import select_banking77_cohort
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.report import exact_mcnemar_p_value
from thesis_research.benchmark.types import CohortSpec
from thesis_research.clients import CallRecord, DecisionError, DecisionResult, UsageTotals
from thesis_research.config import load_task_config
from thesis_research.datasets import Banking77Example, sha256_bytes
from thesis_research.datasets.banking77 import FILES, REVISION

TASK_CONFIG = (
    Path(__file__).parents[1] / "experiments" / "intent_classification" / "tasks" / "banking77.toml"
)
BENCHMARKS_DIRECTORY = TASK_CONFIG.parent.parent / "benchmarks"


def test_without_criteria_smoke_plan_only_changes_the_task_variant():
    standard = load_benchmark_plan(BENCHMARKS_DIRECTORY / "banking77-smoke.toml")
    without_criteria = load_benchmark_plan(
        BENCHMARKS_DIRECTORY / "banking77-without-criteria-smoke.toml"
    )

    assert standard.task_config.name == "banking77.toml"
    assert without_criteria.task_config.name == "banking77-without-criteria.toml"
    assert standard.cohort == without_criteria.cohort
    assert standard.llm_output == without_criteria.llm_output
    assert standard.execution == without_criteria.execution
    assert standard.models == without_criteria.models


@pytest.fixture
def benchmark_dataset(tmp_path):
    config = load_task_config(TASK_CONFIG)
    labels = [option.label for option in config.question.options]
    directory = tmp_path / "data" / "banking77" / REVISION
    directory.mkdir(parents=True)
    (directory / "categories.json").write_text(json.dumps(labels))
    for split in ("train", "test"):
        with (directory / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "category"])
            writer.writerows(
                (f"{split} message {index}", label) for index, label in enumerate(labels)
            )
    manifest = {
        "revision": REVISION,
        "source": "synthetic fixture",
        "files": {
            name: {
                "sha256": sha256_bytes((directory / name).read_bytes()),
                "url": f"fixture:{name}",
            }
            for name in FILES
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "data", labels


@dataclass
class Activity:
    active_models: int = 0
    max_active_models: int = 0


class FakeDecisionClient:
    def __init__(
        self,
        labels,
        *,
        model_id,
        wrong_indexes=(),
        failed_indexes=(),
        raise_at=None,
        activity=None,
    ):
        self.labels = labels
        self.model_id = model_id
        self.wrong_indexes = set(wrong_indexes)
        self.failed_indexes = set(failed_indexes)
        self.raise_at = raise_at
        self.activity = activity
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def evaluate(self, state, questions):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            self.calls.append((state, questions))
            index = int(state["customer_message"].rsplit(" ", 1)[-1])
            if index == self.raise_at:
                raise RuntimeError("simulated process interruption")
            if index in self.failed_indexes:
                error = DecisionError("ProviderError", "simulated outage")
                usage = UsageTotals(None, None, None, None, None)
                call = CallRecord(
                    question_ids=("intent",),
                    requested_model=self.model_id,
                    resolved_model=None,
                    requested_provider="fake",
                    resolved_provider=None,
                    latency_seconds=0.001,
                    usage=usage,
                    error=error,
                    raw={"error": "simulated outage"},
                )
                return DecisionResult(
                    answers={},
                    errors={"intent": error},
                    requested_model=self.model_id,
                    requested_provider="fake",
                    resolved_models=(),
                    resolved_providers=(),
                    usage=usage,
                    latency_seconds=0.001,
                    calls=(call,),
                )
            label = (
                self.labels[(index + 1) % len(self.labels)]
                if index in self.wrong_indexes
                else self.labels[index]
            )
            answer = ChoiceAnswer(choice=label, confidence=1.0, probabilities={label: 1.0})
            usage = UsageTotals(10, 2, 0.001, 0, 0)
            call = CallRecord(
                question_ids=("intent",),
                requested_model=self.model_id,
                resolved_model=self.model_id,
                requested_provider="fake",
                resolved_provider="fake",
                latency_seconds=0.001,
                usage=usage,
                error=None,
                raw={"response": {"label": label}},
            )
            return DecisionResult(
                answers={"intent": answer},
                errors={},
                requested_model=self.model_id,
                requested_provider="fake",
                resolved_models=(self.model_id,),
                resolved_providers=("fake",),
                usage=usage,
                latency_seconds=0.001,
                calls=(call,),
            )
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True

    async def __aenter__(self):
        if self.activity is not None:
            self.activity.active_models += 1
            self.activity.max_active_models = max(
                self.activity.max_active_models,
                self.activity.active_models,
            )
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.activity is not None:
            self.activity.active_models -= 1
        await self.aclose()


def write_plan(
    tmp_path,
    *,
    model_concurrency=2,
    example_concurrency=3,
    llm_output_mode="label",
):
    plan = tmp_path / "benchmark.toml"
    plan.write_text(
        f"""name = "synthetic benchmark"
task_config = {json.dumps(str(TASK_CONFIG))}

[cohort]
split = "train"
strategy = "stratified_random"
size = 77
seed = 42

[llm_output]
mode = {json.dumps(llm_output_mode)}

[execution]
example_concurrency = {example_concurrency}
model_concurrency = {model_concurrency}

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "llm-a"
backend = "llm"
model = "creator/llm-a"
provider = "creator"

[[models]]
id = "llm-b"
backend = "llm"
model = "creator/llm-b"
provider = "creator"
"""
    )
    return plan


def test_plan_can_select_full_llm_probabilities(tmp_path):
    plan = load_benchmark_plan(write_plan(tmp_path, llm_output_mode="probabilities"))

    assert plan.llm_output.mode == "probabilities"


def test_plan_rejects_invalid_split_and_llm_output_mode(tmp_path):
    invalid_mode = write_plan(tmp_path, llm_output_mode="confidence")
    with pytest.raises(ValueError, match="llm_output mode must be label or probabilities"):
        load_benchmark_plan(invalid_mode)

    invalid_split = write_plan(tmp_path)
    invalid_split.write_text(invalid_split.read_text().replace('split = "train"', 'split = "dev"'))
    with pytest.raises(ValueError, match="cohort split must be train or test"):
        load_benchmark_plan(invalid_split)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_command_runs_all_models_and_writes_analysis_ready_report(
    benchmark_dataset,
    tmp_path,
):
    data, labels = benchmark_dataset
    plan = write_plan(tmp_path)
    activity = Activity()
    clients = {
        "jev": FakeDecisionClient(labels, model_id="jev", activity=activity),
        "llm-a": FakeDecisionClient(
            labels,
            model_id="llm-a",
            wrong_indexes={0},
            activity=activity,
        ),
        "llm-b": FakeDecisionClient(
            labels,
            model_id="llm-b",
            failed_indexes={1},
            activity=activity,
        ),
    }
    run = asyncio.run(run_benchmark(plan, data, tmp_path / "outputs", _clients=clients))

    assert activity.max_active_models == 2
    assert all(client.closed and len(client.calls) == 77 for client in clients.values())
    assert all(client.max_active <= 3 for client in clients.values())
    assert (run / "plan.toml").read_bytes() == plan.read_bytes()
    assert (run / "task.toml").read_bytes() == TASK_CONFIG.read_bytes()
    assert len(json.loads((run / "cohort.json").read_text())["examples"]) == 77

    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert set(metadata["models"]) == {"jev", "llm-a", "llm-b"}
    assert all(value["status"] == "completed" for value in metadata["models"].values())

    model_metrics = read_jsonl(run / "report" / "model_metrics.jsonl")
    assert [row["model_id"] for row in model_metrics] == ["jev", "llm-a", "llm-b"]
    assert [row["correct"] for row in model_metrics] == [77, 76, 76]
    assert len(read_jsonl(run / "report" / "per_label_metrics.jsonl")) == 3 * 77
    assert len(read_jsonl(run / "report" / "model_outcomes.jsonl")) == 3 * 77
    comparisons = read_jsonl(run / "report" / "pairwise_statistics.jsonl")
    assert [row["comparison_model_id"] for row in comparisons] == ["llm-a", "llm-b"]
    assert comparisons[0]["baseline_only_correct"] == 1
    assert comparisons[0]["comparison_only_correct"] == 0
    assert comparisons[0]["accuracy_difference"] == pytest.approx(-1 / 77)
    assert comparisons[1]["both_valid"] == 76
    assert not list(run.rglob("*.png"))

    llm_summary = json.loads((run / "models" / "llm-a" / "summary.json").read_text())
    assert llm_summary["usage"]["input_tokens"] == 770
    llm_prediction = read_jsonl(run / "models" / "llm-a" / "predictions.jsonl")[0]
    assert set(llm_prediction) == {"id", "label"}
    assert set(read_jsonl(run / "models" / "jev" / "predictions.jsonl")[0]) == {
        "id",
        "label",
        "confidence",
        "probabilities",
    }
    assert (
        json.loads((run / "models" / "llm-a" / "metadata.json").read_text())["output_mode"]
        == "label"
    )
    llm_decision = read_jsonl(run / "models" / "llm-a" / "decisions.jsonl")[0]["decision"]
    assert llm_decision["choice_output_mode"] == "label"
    assert llm_decision["answers"]["intent"] == {"type": "choice", "choice": labels[1]}
    assert llm_decision["calls"][0]["raw"] == {"response": {"label": labels[1]}}


def test_failed_benchmark_resumes_only_missing_examples(benchmark_dataset, tmp_path):
    data, labels = benchmark_dataset
    plan = write_plan(tmp_path, model_concurrency=1, example_concurrency=1)
    initial_clients = {
        "jev": FakeDecisionClient(labels, model_id="jev"),
        "llm-a": FakeDecisionClient(labels, model_id="llm-a", raise_at=2),
        "llm-b": FakeDecisionClient(labels, model_id="llm-b"),
    }
    outputs = tmp_path / "outputs"
    with pytest.raises(ValueError, match="preserved record"):
        asyncio.run(run_benchmark(plan, data, outputs, _clients=initial_clients))
    run = next(outputs.iterdir())
    assert len(read_jsonl(run / "models" / "jev" / "predictions.jsonl")) == 77
    assert len(read_jsonl(run / "models" / "llm-a" / "predictions.jsonl")) == 2
    assert len(read_jsonl(run / "models" / "llm-b" / "predictions.jsonl")) == 77

    resumed_llm = FakeDecisionClient(labels, model_id="llm-a")
    assert asyncio.run(resume_benchmark(run, data, _clients={"llm-a": resumed_llm})) == run
    assert len(resumed_llm.calls) == 75
    assert json.loads((run / "metadata.json").read_text())["status"] == "completed"
    assert len(read_jsonl(run / "models" / "llm-a" / "predictions.jsonl")) == 77
    assert (run / "report" / "summary.json").exists()

    # A completed run is idempotent and does not need clients or API credentials.
    assert asyncio.run(resume_benchmark(run, data)) == run


def test_resume_rejects_changed_frozen_inputs(benchmark_dataset, tmp_path):
    data, labels = benchmark_dataset
    plan = write_plan(tmp_path)
    clients = {
        model_id: FakeDecisionClient(labels, model_id=model_id)
        for model_id in ("jev", "llm-a", "llm-b")
    }
    run = asyncio.run(run_benchmark(plan, data, tmp_path / "outputs", _clients=clients))
    (run / "cohort.json").write_text("{}")
    with pytest.raises(ValueError, match="Saved benchmark input changed"):
        asyncio.run(resume_benchmark(run, data))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("model_concurrency = 0", "positive integer"),
        ("size = 1", "at least 77"),
    ],
)
def test_plan_or_cohort_validation(benchmark_dataset, tmp_path, replacement, message):
    data, labels = benchmark_dataset
    plan = write_plan(tmp_path)
    if replacement.startswith("model"):
        plan.write_text(plan.read_text().replace("model_concurrency = 2", replacement))
        with pytest.raises(ValueError, match=message):
            load_benchmark_plan(plan)
    else:
        plan.write_text(plan.read_text().replace("size = 77", replacement))
        clients = {
            model_id: FakeDecisionClient(labels, model_id=model_id)
            for model_id in ("jev", "llm-a", "llm-b")
        }
        with pytest.raises(ValueError, match=message):
            asyncio.run(run_benchmark(plan, data, tmp_path / "outputs", _clients=clients))


def test_exact_mcnemar_p_value():
    assert exact_mcnemar_p_value(0, 0) == 1
    assert exact_mcnemar_p_value(1, 9) == pytest.approx(0.021484375)
    assert exact_mcnemar_p_value(9, 1) == pytest.approx(0.021484375)
    assert 0 <= exact_mcnemar_p_value(0, 3080) <= 1


def test_random_smoke_cohort_is_small_and_reproducible():
    examples = [Banking77Example(f"train:{index}", str(index), "intent") for index in range(10)]
    spec = CohortSpec("train", "random", 1, 20260920)
    first = select_banking77_cohort(examples, ["intent"], spec)
    second = select_banking77_cohort(examples, ["intent"], spec)
    assert first == second
    assert len(first) == 1
