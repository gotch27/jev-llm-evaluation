import asyncio
import csv
import json
from pathlib import Path

import pytest
from typesafe_sdk import ChoiceAnswer

from thesis_research.clients import (
    CallRecord,
    DecisionError,
    DecisionResult,
    UsageTotals,
)
from thesis_research.config import load_experiment_config
from thesis_research.datasets import sha256_bytes
from thesis_research.datasets.banking77 import FILES, REVISION
from thesis_research.evaluation import run_classification_evaluation
from thesis_research.prediction_run import run_prediction_experiment

CONFIG_PATH = Path(__file__).parents[1] / "experiments" / "intent_classification" / "banking77.toml"


@pytest.fixture
def prediction_dataset(tmp_path):
    config = load_experiment_config(CONFIG_PATH)
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


class FakeDecisionClient:
    def __init__(self, labels, *, failed_indexes=(), unknown_indexes=()):
        self.labels = labels
        self.failed_indexes = set(failed_indexes)
        self.unknown_indexes = set(unknown_indexes)
        self.active = 0
        self.max_active = 0
        self.closed = False
        self.calls = []

    async def evaluate(self, state, questions):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            self.calls.append((state, questions))
            index = int(state["customer_message"].rsplit(" ", 1)[-1])
            if index in self.failed_indexes:
                error = DecisionError("ProviderError", "temporary outage", {"attempt": 1})
                usage = UsageTotals(None, None, None, None, None)
                call = CallRecord(
                    question_ids=("intent",),
                    requested_model="fake-model",
                    resolved_model=None,
                    requested_provider="fake-provider",
                    resolved_provider=None,
                    latency_seconds=0.01,
                    usage=usage,
                    error=error,
                    raw={"error": "temporary outage"},
                )
                return DecisionResult(
                    answers={},
                    errors={"intent": error},
                    requested_model="fake-model",
                    requested_provider="fake-provider",
                    resolved_models=(),
                    resolved_providers=(),
                    usage=usage,
                    latency_seconds=0.01,
                    calls=(call,),
                )

            label = "unknown_intent" if index in self.unknown_indexes else self.labels[index]
            answer = ChoiceAnswer(choice=label, confidence=1.0, probabilities={label: 1.0})
            usage = UsageTotals(10, 2, 0.001, 0, 0)
            call = CallRecord(
                question_ids=("intent",),
                requested_model="fake-model",
                resolved_model="resolved-model",
                requested_provider="fake-provider",
                resolved_provider="resolved-provider",
                latency_seconds=0.01,
                usage=usage,
                error=None,
                raw={"request": {"state": state}, "response": {"label": label}},
            )
            return DecisionResult(
                answers={"intent": answer},
                errors={},
                requested_model="fake-model",
                requested_provider="fake-provider",
                resolved_models=("resolved-model",),
                resolved_providers=("resolved-provider",),
                usage=usage,
                latency_seconds=0.01,
                calls=(call,),
            )
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_prediction_run_is_bounded_ordered_recorded_and_evaluable(prediction_dataset, tmp_path):
    data, labels = prediction_dataset
    fake = FakeDecisionClient(labels, failed_indexes={1}, unknown_indexes={2})
    run = asyncio.run(
        run_prediction_experiment(
            CONFIG_PATH,
            data,
            tmp_path / "prediction-outputs",
            backend="jev",
            model="fake-model",
            provider=None,
            max_concurrency=2,
            limit=4,
            _client=fake,
        )
    )

    predictions = read_jsonl(run / "predictions.jsonl")
    assert [record["id"] for record in predictions] == ["test:1", "test:2", "test:3", "test:4"]
    assert predictions[0]["label"] == labels[0]
    assert predictions[1]["error"] == "ProviderError: temporary outage"
    assert "unknown_intent" in predictions[2]["error"]
    assert predictions[3]["label"] == labels[3]
    assert all("reference_label" not in prediction for prediction in predictions)
    assert fake.max_active == 2
    assert fake.closed

    decisions = read_jsonl(run / "decisions.jsonl")
    assert decisions[0]["state"] == {"customer_message": "test message 0"}
    assert decisions[0]["decision"]["calls"][0]["raw"]["response"] == {"label": labels[0]}
    assert decisions[1]["decision"]["errors"]["intent"]["diagnostics"] == {"attempt": 1}

    summary = json.loads((run / "summary.json").read_text())
    assert summary["expected"] == summary["attempted"] == 4
    assert summary["successful"] == 2
    assert summary["failed"] == 2
    assert summary["usage"]["input_tokens"] is None
    assert summary["resolved_models"] == ["resolved-model"]

    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["split"] == "test"
    assert metadata["selection"] == {
        "limit": 4,
        "split_examples": 77,
        "selected_examples": 4,
    }
    assert set(metadata["artifacts"]) == {
        "config.toml",
        "predictions.jsonl",
        "decisions.jsonl",
        "summary.json",
    }

    evaluation = run_classification_evaluation(
        CONFIG_PATH,
        run / "predictions.jsonl",
        data,
        tmp_path / "evaluation-outputs",
    )
    evaluation_summary = json.loads((evaluation / "summary.json").read_text())
    assert evaluation_summary["total"] == 77
    assert evaluation_summary["prediction_counts"] == {
        "valid": 2,
        "missing": 73,
        "invalid": 0,
        "failed": 2,
    }


def test_prediction_setup_failure_is_preserved(prediction_dataset, tmp_path, monkeypatch):
    data, _ = prediction_dataset
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    outputs = tmp_path / "prediction-outputs"

    with pytest.raises(ValueError, match="preserved record"):
        asyncio.run(
            run_prediction_experiment(
                CONFIG_PATH,
                data,
                outputs,
                backend="jev",
                model="fake-model",
                provider=None,
                limit=1,
            )
        )

    run = next(outputs.iterdir())
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "AI_GATEWAY_API_KEY" in metadata["error"]
    assert set(metadata["artifacts"]) == {"config.toml"}


def test_prediction_arguments_are_validated_before_creating_output(tmp_path):
    outputs = tmp_path / "outputs"
    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(
            run_prediction_experiment(
                CONFIG_PATH,
                tmp_path / "data",
                outputs,
                backend="jev",
                model="fake-model",
                provider=None,
                max_concurrency=0,
            )
        )
    assert not outputs.exists()
