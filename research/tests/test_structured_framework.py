import asyncio
import hashlib
import io
import json
import math
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

from thesis_research.benchmark.cohort import (
    cohort_record,
    examples_from_cohort,
    select_cohort,
)
from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.report import generate_report
from thesis_research.benchmark.runner import resume_benchmark, run_benchmark
from thesis_research.benchmark.types import CohortSpec
from thesis_research.clients import CallRecord, DecisionResult, UsageTotals
from thesis_research.config import load_task_config
from thesis_research.datasets import (
    ChoiceReference,
    ChoiceReferenceSchema,
    DatasetAdapter,
    DatasetExample,
    NoulReference,
    NoulReferenceSchema,
    PreparedDataset,
    ScoreReference,
    ScoreReferenceSchema,
    load_prepared_boolq,
    prepare_boolq,
    validate_prepared_dataset,
)
from thesis_research.datasets import registry as dataset_registry
from thesis_research.evaluation import evaluate_structured, read_prediction_jsonl
from thesis_research.prediction import predict_examples
from thesis_research.tasks import StructuredTask, build_structured_task

BOOLQ_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"


def write_noul_task(path: Path, revision: str = BOOLQ_REVISION) -> Path:
    path.write_text(
        f'''name = "synthetic Noul"

[dataset]
id = "boolq"
revision = "{revision}"

[state.fields]
passage = "passage"
question = "question"

[question]
type = "noul"
id = "answer"
instructions = "Answer from the passage."

[question.criteria]
true = "The answer is yes."
false = "The answer is no."
'''
    )
    return path


def write_score_task(path: Path) -> Path:
    path.write_text(
        """name = "synthetic Score"

[dataset]
id = "synthetic-score"
revision = "revision-1"

[state.fields]
response = "text"

[question]
type = "score"
id = "quality"
instructions = { goal = "Rate quality" }

[[question.levels]]
criterion = "Poor"

[[question.levels]]
criterion = { description = "Acceptable" }

[[question.levels]]
criterion = ["Excellent", "No defects"]
"""
    )
    return path


def test_builds_noul_and_score_tasks(tmp_path):
    noul_config = load_task_config(write_noul_task(tmp_path / "noul.toml"))
    noul = build_structured_task(noul_config, NoulReferenceSchema())
    assert isinstance(noul.question, Noul)
    assert noul.build_state({"passage": "Evidence", "question": "Is it true?"}) == {
        "passage": "Evidence",
        "question": "Is it true?",
    }
    assert noul.question.criteria == {
        "true": "The answer is yes.",
        "false": "The answer is no.",
    }

    score_config = load_task_config(write_score_task(tmp_path / "score.toml"))
    score = build_structured_task(score_config, ScoreReferenceSchema(3))
    assert isinstance(score.question, Score)
    assert score.question.criteria == [
        "Poor",
        {"description": "Acceptable"},
        ["Excellent", "No defects"],
    ]
    with pytest.raises(ValueError, match="level count"):
        build_structured_task(score_config, ScoreReferenceSchema(4))
    with pytest.raises(ValueError, match="does not match"):
        build_structured_task(noul_config, ScoreReferenceSchema(2))


def test_state_validation_reports_all_invalid_mappings_across_rows(tmp_path):
    task = build_structured_task(
        load_task_config(write_noul_task(tmp_path / "task.toml")), NoulReferenceSchema()
    )
    examples = [
        DatasetExample("train:1", {"passage": "P", "question": "Q"}, NoulReference(True)),
        DatasetExample("train:2", {"passage": ""}, NoulReference(False)),
        DatasetExample("train:3", {"question": ""}, NoulReference(True)),
    ]
    with pytest.raises(ValueError) as error:
        task.validate_state_fields(examples)
    assert "'passage' <- 'passage': missing in 1 rows, empty in 1 rows" in str(error.value)
    assert "'question' <- 'question': missing in 1 rows, empty in 1 rows" in str(error.value)
    assert "train:2" in str(error.value) and "train:3" in str(error.value)


def test_boolq_output_names_can_change_independently_of_source_names(tmp_path):
    config = load_task_config(write_noul_task(tmp_path / "task.toml"))
    config = replace(
        config, state=replace(config.state, fields=(("context", "passage"), ("query", "question")))
    )
    task = build_structured_task(config, NoulReferenceSchema())
    example = DatasetExample("train:1", {"passage": "P", "question": "Q"}, NoulReference(True))
    task.validate_state_fields([example])
    assert task.build_state(example.state) == {"context": "P", "query": "Q"}


def test_prediction_validates_later_batches_before_any_model_call(tmp_path):
    task = build_structured_task(
        load_task_config(write_noul_task(tmp_path / "task.toml")), NoulReferenceSchema()
    )
    examples = [
        DatasetExample("train:1", {"passage": "P", "question": "Q"}, NoulReference(True)),
        DatasetExample("train:2", {"passage": "P"}, NoulReference(False)),
    ]
    client = FakeStructuredClient([NoulAnswer(noul=0.9)])
    predictions = io.StringIO()
    decisions = io.StringIO()
    with pytest.raises(ValueError, match="Invalid task state mappings"):
        asyncio.run(
            predict_examples(
                client,
                examples,
                task,
                NoulReferenceSchema(),
                predictions,
                decisions,
                max_concurrency=1,
            )
        )
    assert client.calls == 0
    assert predictions.getvalue() == decisions.getvalue() == ""


def test_prepared_dataset_rejects_reference_outside_score_rubric():
    prepared = PreparedDataset(
        dataset_id="score-fixture",
        revision="revision-1",
        split="train",
        examples=(DatasetExample("train:1", {"text": "A"}, ScoreReference(3)),),
        reference_schema=ScoreReferenceSchema(3),
        manifest={},
    )
    with pytest.raises(ValueError, match="zero-based rubric"):
        validate_prepared_dataset(prepared, "score-fixture", "revision-1", "train")


@pytest.mark.parametrize(
    ("schema", "reference"),
    [
        (NoulReferenceSchema(), NoulReference(1)),
        (NoulReferenceSchema(), NoulReference("false")),
        (ScoreReferenceSchema(3), ScoreReference(True)),
        (ScoreReferenceSchema(3), ScoreReference(1.5)),
        (ScoreReferenceSchema(3), ScoreReference("1")),
        (ScoreReferenceSchema(3.0), ScoreReference(1)),
        (ScoreReferenceSchema(True), ScoreReference(0)),
        (ChoiceReferenceSchema(("",)), ChoiceReference("")),
        (ChoiceReferenceSchema((" ",)), ChoiceReference(" ")),
        (ChoiceReferenceSchema(([],)), ChoiceReference("a")),
        (ChoiceReferenceSchema(("a",)), ChoiceReference([])),
    ],
)
def test_prepared_dataset_rejects_malformed_reference_values(schema, reference):
    prepared = PreparedDataset(
        dataset_id="fixture",
        revision="revision-1",
        split="train",
        examples=(DatasetExample("train:1", {"text": "A"}, reference),),
        reference_schema=schema,
        manifest={},
    )
    with pytest.raises(ValueError):
        validate_prepared_dataset(prepared, "fixture", "revision-1", "train")


@pytest.mark.parametrize("example_id", ["", " ", 1, []])
def test_prepared_dataset_rejects_malformed_ids(example_id):
    prepared = PreparedDataset(
        dataset_id="fixture",
        revision="revision-1",
        split="train",
        examples=(DatasetExample(example_id, {"text": "A"}, NoulReference(True)),),
        reference_schema=NoulReferenceSchema(),
        manifest={},
    )
    with pytest.raises(ValueError, match="IDs must be nonempty and unique"):
        validate_prepared_dataset(prepared, "fixture", "revision-1", "train")


class FakeStructuredClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.closed = False
        self.calls = 0

    async def evaluate(self, state, questions):
        self.calls += 1
        answer = next(self.answers)
        question_id = next(iter(questions))
        usage = UsageTotals(1, 1, 0.0, 0, 0)
        call = CallRecord(
            question_ids=(question_id,),
            requested_model="fake/model",
            resolved_model="fake/model",
            requested_provider="fake",
            resolved_provider="fake",
            latency_seconds=0.01,
            usage=usage,
            error=None,
            raw={"state": state},
        )
        return DecisionResult(
            answers={question_id: answer},
            errors={},
            requested_model="fake/model",
            requested_provider="fake",
            resolved_models=("fake/model",),
            resolved_providers=("fake",),
            usage=usage,
            latency_seconds=0.01,
            calls=(call,),
        )

    async def aclose(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()


@pytest.mark.parametrize(
    ("question", "schema", "reference", "answer"),
    [
        (
            Choice(criteria={"a": None}),
            ChoiceReferenceSchema(("a",)),
            ChoiceReference("a"),
            ChoiceAnswer(choice="a", confidence=float("nan"), probabilities={"a": 1.0}),
        ),
        (
            Choice(criteria={"a": None}),
            ChoiceReferenceSchema(("a",)),
            ChoiceReference("a"),
            ChoiceAnswer(choice="a", confidence=1.0, probabilities={"unknown": 1.0}),
        ),
        (
            Noul(),
            NoulReferenceSchema(),
            NoulReference(True),
            NoulAnswer.model_construct(noul=True),
        ),
        (
            Score(criteria=["Low", "High"]),
            ScoreReferenceSchema(2),
            ScoreReference(1),
            ScoreAnswer(
                score=1.0,
                confidence=float("nan"),
                probabilities={0: 0.0, 1: 1.0},
                legend={0: "Low", 1: "High"},
            ),
        ),
        (
            Score(criteria=["Low", "High"]),
            ScoreReferenceSchema(2),
            ScoreReference(1),
            ScoreAnswer.model_construct(
                score=1.0,
                confidence=1.0,
                probabilities={False: 0.0, True: 1.0},
                legend={0: "Low", 1: "High"},
            ),
        ),
    ],
)
def test_malformed_answers_fail_consistently_in_progress_and_evaluation(
    question, schema, reference, answer
):
    task = StructuredTask((("text", "text"),), "answer", schema.question_type, question)
    examples = [DatasetExample("train:1", {"text": "A"}, reference)]
    predictions = io.StringIO()
    decisions = io.StringIO()
    progress = []
    summary = asyncio.run(
        predict_examples(
            FakeStructuredClient([answer]),
            examples,
            task,
            schema,
            predictions,
            decisions,
            max_concurrency=1,
            on_progress=progress.append,
        )
    )
    assert summary["successful"] == progress[-1]["successful"] == 0
    assert summary["failed"] == progress[-1]["failed"] == 1
    prediction = json.loads(predictions.getvalue())
    assert "error" in prediction
    assert json.loads(decisions.getvalue())["prediction"] == prediction
    metrics, _ = evaluate_structured(
        examples, schema, read_prediction_jsonl(predictions.getvalue().encode())
    )
    assert metrics["prediction_counts"]["failed"] == 1


def test_serializes_noul_and_score_predictions(tmp_path):
    noul_task = build_structured_task(
        load_task_config(write_noul_task(tmp_path / "noul.toml")),
        NoulReferenceSchema(),
    )
    noul_examples = [
        DatasetExample(
            "train:1",
            {"passage": "Evidence", "question": "Question"},
            NoulReference(True),
        )
    ]
    predictions = io.StringIO()
    decisions = io.StringIO()
    asyncio.run(
        predict_examples(
            FakeStructuredClient([NoulAnswer(noul=0.5)]),
            noul_examples,
            noul_task,
            NoulReferenceSchema(),
            predictions,
            decisions,
            max_concurrency=1,
        )
    )
    assert json.loads(predictions.getvalue()) == {
        "id": "train:1",
        "value": True,
        "probability": 0.5,
    }
    discrete_predictions = io.StringIO()
    asyncio.run(
        predict_examples(
            FakeStructuredClient([NoulAnswer(noul=0.5)]),
            noul_examples,
            noul_task,
            NoulReferenceSchema(),
            discrete_predictions,
            io.StringIO(),
            max_concurrency=1,
            record_answer_details=False,
        )
    )
    assert json.loads(discrete_predictions.getvalue()) == {
        "id": "train:1",
        "value": True,
    }

    score_task = build_structured_task(
        load_task_config(write_score_task(tmp_path / "score.toml")),
        ScoreReferenceSchema(3),
    )
    score_examples = [DatasetExample("train:1", {"text": "Response"}, ScoreReference(1))]
    predictions = io.StringIO()
    decisions = io.StringIO()
    asyncio.run(
        predict_examples(
            FakeStructuredClient(
                [
                    ScoreAnswer(
                        score=1.5,
                        confidence=0.4,
                        legend={0: "Poor", 1: "Okay", 2: "Great"},
                        probabilities={0: 0.1, 1: 0.45, 2: 0.45},
                    )
                ]
            ),
            score_examples,
            score_task,
            ScoreReferenceSchema(3),
            predictions,
            decisions,
            max_concurrency=1,
        )
    )
    score_prediction = json.loads(predictions.getvalue())
    assert score_prediction["level"] == 1
    assert score_prediction["score"] == 1.5
    assert score_prediction["probabilities"] == {"0": 0.1, "1": 0.45, "2": 0.45}

    discrete_predictions = io.StringIO()
    asyncio.run(
        predict_examples(
            FakeStructuredClient(
                [
                    ScoreAnswer(
                        score=1.5,
                        confidence=0.4,
                        legend={0: "Poor", 1: "Okay", 2: "Great"},
                        probabilities={0: 0.1, 1: 0.45, 2: 0.45},
                    )
                ]
            ),
            score_examples,
            score_task,
            ScoreReferenceSchema(3),
            discrete_predictions,
            io.StringIO(),
            max_concurrency=1,
            record_answer_details=False,
        )
    )
    assert json.loads(discrete_predictions.getvalue()) == {
        "id": "train:1",
        "level": 1,
        "score": 1.5,
    }


def test_rejects_inconsistent_noul_and_score_probability_records():
    noul = [DatasetExample("1", {"text": "A"}, NoulReference(True))]
    noul_summary, _ = evaluate_structured(
        noul,
        NoulReferenceSchema(),
        _predictions([{"id": "1", "value": False, "probability": 0.8}]),
    )
    assert noul_summary["prediction_counts"]["invalid"] == 1

    score = [DatasetExample("1", {"text": "A"}, ScoreReference(1))]
    score_summary, _ = evaluate_structured(
        score,
        ScoreReferenceSchema(3),
        _predictions(
            [
                {
                    "id": "1",
                    "level": 0,
                    "score": 1.2,
                    "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
                }
            ]
        ),
    )
    assert score_summary["prediction_counts"]["invalid"] == 1


def _predictions(records):
    return read_prediction_jsonl("\n".join(json.dumps(record) for record in records).encode())


def test_noul_metrics_include_failures_and_probability_quality():
    examples = [
        DatasetExample(str(index), {"text": str(index)}, NoulReference(value))
        for index, value in enumerate([True, False, True, False])
    ]
    predictions = _predictions(
        [
            {"id": "0", "value": True, "probability": 0.9},
            {"id": "1", "value": True, "probability": 0.6},
            {"id": "2", "error": "timeout"},
        ]
    )
    summary, outcomes = evaluate_structured(examples, NoulReferenceSchema(), predictions)
    assert summary["accuracy"] == 0.25
    assert summary["prediction_counts"] == {
        "valid": 2,
        "missing": 1,
        "invalid": 0,
        "failed": 1,
    }
    assert summary["probability_count"] == 2
    assert summary["probability_coverage"] == 0.5
    assert summary["brier_score"] == pytest.approx((0.1**2 + 0.6**2) / 2)
    assert summary["log_loss"] == pytest.approx(-(math.log(0.9) + math.log(0.4)) / 2)
    assert [outcome["status"] for outcome in outcomes] == [
        "valid",
        "valid",
        "failed",
        "missing",
    ]


def test_score_metrics_use_expected_score_and_discrete_level():
    examples = [
        DatasetExample(str(index), {"text": str(index)}, ScoreReference(level))
        for index, level in enumerate([0, 1, 2])
    ]
    predictions = _predictions(
        [
            {"id": "0", "level": 0, "score": 0.2},
            {"id": "1", "level": 2, "score": 1.8},
            {"id": "2", "error": "timeout"},
        ]
    )
    summary, _ = evaluate_structured(examples, ScoreReferenceSchema(3), predictions)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["numeric_count"] == 2
    assert summary["numeric_coverage"] == pytest.approx(2 / 3)
    assert summary["mean_absolute_error"] == pytest.approx(0.5)
    assert summary["root_mean_squared_error"] == pytest.approx(math.sqrt(0.34))
    assert -1 <= summary["quadratic_weighted_kappa"] <= 1


def _write_boolq_fixture(root: Path, revision: str) -> Path:
    directory = root / "boolq" / revision
    directory.mkdir(parents=True)
    rows = {
        "train": [
            {"question": "First?", "answer": True, "passage": "First passage."},
            {"question": "Second?", "answer": False, "passage": "Second passage."},
        ],
        "validation": [
            {"question": "Third?", "answer": False, "passage": "Third passage."},
            {"question": "Fourth?", "answer": True, "passage": "Fourth passage."},
        ],
    }
    files = {}
    for split, values in rows.items():
        name = f"{split}-00000-of-00001.parquet"
        path = directory / name
        pq.write_table(
            pa.table(
                {
                    "question": [row["question"] for row in values],
                    "answer": [row["answer"] for row in values],
                    "passage": [row["passage"] for row in values],
                }
            ),
            path,
        )
        files[name] = {
            "split": split,
            "url": f"fixture:{name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(values),
        }
    manifest = {
        "dataset_id": "boolq",
        "source": "fixture",
        "revision": revision,
        "license": "CC BY-SA 3.0",
        "schema": {"question": "string", "answer": "bool", "passage": "string"},
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_boolq_loader_and_cached_tamper_detection(tmp_path):
    revision = "fixture-revision"
    directory = _write_boolq_fixture(tmp_path / "data", revision)
    prepared = load_prepared_boolq(tmp_path / "data", revision, "train")
    assert [example.id for example in prepared.examples] == ["train:1", "train:2"]
    assert prepared.examples[0].state == {
        "question": "First?",
        "passage": "First passage.",
    }
    assert prepared.examples[0].reference == NoulReference(True)
    (directory / "train-00000-of-00001.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        load_prepared_boolq(tmp_path / "data", revision, "train")


def test_boolq_prepare_downloads_both_splits(tmp_path, monkeypatch):
    source = _write_boolq_fixture(tmp_path / "source", "fixture-revision")

    def fake_urlopen(url, timeout):
        name = url.rsplit("/", 1)[-1]
        return io.BytesIO((source / name).read_bytes())

    monkeypatch.setattr("thesis_research.datasets.boolq.urlopen", fake_urlopen)
    prepared = prepare_boolq(tmp_path / "download", "fixture-revision")
    assert prepared == tmp_path / "download" / "boolq" / "fixture-revision"
    assert (
        len(load_prepared_boolq(tmp_path / "download", "fixture-revision", "validation").examples)
        == 2
    )

    def no_network(*args, **kwargs):
        pytest.fail("A verified BoolQ cache must not use the network")

    monkeypatch.setattr("thesis_research.datasets.boolq.urlopen", no_network)
    assert prepare_boolq(tmp_path / "download", "fixture-revision") == prepared


def test_boolq_rejects_corrupted_pinned_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "thesis_research.datasets.boolq.urlopen",
        lambda *args, **kwargs: io.BytesIO(b"not the pinned parquet file"),
    )
    with pytest.raises(ValueError, match="Downloaded BoolQ checksum mismatch"):
        prepare_boolq(tmp_path / "data", BOOLQ_REVISION)


def test_generic_cohort_restore(tmp_path):
    examples = [
        DatasetExample("train:1", {"passage": "A", "question": "Q1"}, NoulReference(True)),
        DatasetExample("train:2", {"passage": "B", "question": "Q2"}, NoulReference(False)),
        DatasetExample("train:3", {"passage": "C", "question": "Q3"}, NoulReference(True)),
    ]
    task = build_structured_task(
        load_task_config(write_noul_task(tmp_path / "noul.toml")),
        NoulReferenceSchema(),
    )
    selected = select_cohort(
        examples,
        NoulReferenceSchema(),
        CohortSpec("train", "stratified_random", 2, 7),
    )
    saved = cohort_record(
        selected,
        dataset_id="boolq",
        dataset_revision=BOOLQ_REVISION,
        split="train",
        schema=NoulReferenceSchema(),
        spec=CohortSpec("train", "stratified_random", 2, 7),
        task=task,
    )
    assert (
        examples_from_cohort(
            saved,
            examples,
            dataset_id="boolq",
            dataset_revision=BOOLQ_REVISION,
            split="train",
            task=task,
        )
        == selected
    )

    for changed in (
        {**saved, "dataset_id": None},
        {**saved, "question_type": None},
        {
            **saved,
            "examples": [
                {"id": row["id"], "text": "old", "reference_label": "old"}
                for row in saved["examples"]
            ],
        },
    ):
        with pytest.raises(ValueError, match="dataset ID|question type|state differs"):
            examples_from_cohort(
                changed,
                examples,
                dataset_id="boolq",
                dataset_revision=BOOLQ_REVISION,
                split="train",
                task=task,
            )


@pytest.mark.parametrize("tampering", [None, "missing_decision", "mismatched_prediction"])
def test_boolq_benchmark_generates_noul_reports(tmp_path, tampering):
    revision = "fixture-revision"
    _write_boolq_fixture(tmp_path / "data", revision)
    task = write_noul_task(tmp_path / "task.toml", revision)
    plan = tmp_path / "plan.toml"
    plan.write_text(
        f'''name = "BoolQ fixture"
task_config = "{task.name}"

[cohort]
split = "train"
strategy = "all"

[llm_output]
mode = "probabilities"

[execution]
example_concurrency = 1
model_concurrency = 2

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "llm"
backend = "llm"
model = "fake/model"
provider = "fake"
'''
    )
    clients = {
        name: FakeStructuredClient([NoulAnswer(noul=0.9), NoulAnswer(noul=0.1)])
        for name in ("jev", "llm")
    }
    run = asyncio.run(
        run_benchmark(plan, tmp_path / "data", tmp_path / "outputs", _clients=clients)
    )
    report = json.loads((run / "report" / "summary.json").read_text())
    assert report["question_type"] == "noul"
    assert report["models"]["jev"]["accuracy"] == 1
    metrics = [
        json.loads(line)
        for line in (run / "report" / "model_metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["brier_score"] == pytest.approx(0.01) for row in metrics)
    outcomes = [
        json.loads(line)
        for line in (run / "report" / "model_outcomes.jsonl").read_text().splitlines()
    ]
    assert all(row["question_type"] == "noul" for row in outcomes)
    assert {row["predicted_value"] for row in outcomes} == {False, True}
    comparisons = [
        json.loads(line)
        for line in (run / "report" / "pairwise_statistics.jsonl").read_text().splitlines()
    ]
    assert comparisons[0]["question_type"] == "noul"

    if tampering is not None:
        decisions_path = run / "models" / "llm" / "decisions.jsonl"
        decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
        if tampering == "missing_decision":
            decisions.pop()
        else:
            decisions[0]["prediction"]["value"] = False
        decisions_path.write_text("\n".join(json.dumps(record) for record in decisions))
        prepared = load_prepared_boolq(tmp_path / "data", revision, "train")
        with pytest.raises(ValueError, match="different IDs|differs from decision"):
            generate_report(
                run,
                load_benchmark_plan(plan).models,
                list(prepared.examples),
                prepared.reference_schema,
            )


@pytest.mark.parametrize("output_mode", ["discrete", "probabilities"])
def test_score_benchmark_runs_and_resumes_through_dataset_registry(
    tmp_path, monkeypatch, output_mode
):
    schema = ScoreReferenceSchema(3)
    examples = tuple(
        DatasetExample(
            f"train:{index + 1}", {"text": f"Response {index}"}, ScoreReference(index % 3)
        )
        for index in range(6)
    )
    prepared = PreparedDataset(
        "synthetic-score", "revision-1", "train", examples, schema, {"source": "synthetic fixture"}
    )

    class ScoreAdapter(DatasetAdapter):
        dataset_id = "synthetic-score"
        supported_splits = ("train",)

        def prepare(self, root, revision):
            return root

        def load(self, root, revision, split):
            assert revision == "revision-1" and split == "train"
            return prepared

    monkeypatch.setitem(dataset_registry._ADAPTERS, "synthetic-score", ScoreAdapter())
    task = write_score_task(tmp_path / "task.toml")
    plan = tmp_path / "plan.toml"
    plan.write_text(
        f'''name = "Score fixture"
task_config = "{task.name}"

[cohort]
split = "train"
strategy = "stratified_random"
size = 3
seed = 7

[llm_output]
mode = "{output_mode}"

[execution]
example_concurrency = 1
model_concurrency = 2

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "llm"
backend = "llm"
model = "fake/model"
provider = "fake"
'''
    )
    selected = select_cohort(examples, schema, load_benchmark_plan(plan).cohort)

    def answers_for(rows, probabilities):
        answers = []
        for row in rows:
            level = row.reference.level
            distribution = {
                index: (0.8 if index == level else 0.1) if probabilities else float(index == level)
                for index in range(3)
            }
            answers.append(
                ScoreAnswer(
                    score=sum(index * probability for index, probability in distribution.items()),
                    confidence=distribution[level],
                    probabilities=distribution,
                    legend={0: "Poor", 1: "Acceptable", 2: "Excellent"},
                )
            )
        return answers

    class InterruptedScoreClient(FakeStructuredClient):
        async def evaluate(self, state, questions):
            if self.calls == 1:
                raise RuntimeError("simulated interruption after one saved Score result")
            return await super().evaluate(state, questions)

    jev = FakeStructuredClient(answers_for(selected, probabilities=True))
    llm = InterruptedScoreClient(
        answers_for(selected, probabilities=output_mode == "probabilities")
    )
    with pytest.raises(ValueError, match="preserved record"):
        asyncio.run(
            run_benchmark(
                plan, tmp_path / "data", tmp_path / "outputs", _clients={"jev": jev, "llm": llm}
            )
        )
    run = next((tmp_path / "outputs").iterdir())
    predictions_path = run / "models" / "llm" / "predictions.jsonl"
    first_prediction = json.loads(predictions_path.read_text())
    assert jev.calls == 3 and llm.calls == 1
    assert jev.closed and llm.closed
    resumed = FakeStructuredClient(
        answers_for(selected[1:], probabilities=output_mode == "probabilities")
    )
    asyncio.run(resume_benchmark(run, tmp_path / "data", _clients={"llm": resumed}))
    assert resumed.calls == 2 and resumed.closed
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert predictions[0] == first_prediction
    assert {row["level"] for row in predictions} == {0, 1, 2}
    assert all(("probabilities" in row) == (output_mode == "probabilities") for row in predictions)
    cohort = json.loads((run / "cohort.json").read_text())
    assert cohort["question_type"] == "score"
    assert all(set(row["state"]) == {"response"} for row in cohort["examples"])
    report = json.loads((run / "report" / "summary.json").read_text())
    assert report["question_type"] == "score"
    assert all(metrics["accuracy"] == 1 for metrics in report["models"].values())
    metrics = report["models"]["llm"]
    assert metrics["numeric_coverage"] == 1
    assert metrics["quadratic_weighted_kappa"] == 1
    assert metrics["mean_absolute_error"] == pytest.approx(
        0.2 if output_mode == "probabilities" else 0.0
    )
    comparisons = [
        json.loads(line)
        for line in (run / "report" / "pairwise_statistics.jsonl").read_text().splitlines()
    ]
    assert comparisons[0]["question_type"] == "score"
    assert comparisons[0]["both_correct"] == 3
    assert json.loads((run / "metadata.json").read_text())["status"] == "completed"
