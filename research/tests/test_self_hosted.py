"""Benchmark plans for independently served models."""

from pathlib import Path

import pytest

from thesis_research.benchmark.plan import load_benchmark_plan
from thesis_research.benchmark.types import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/boolean_question_answering/benchmarks/boolq-local-smoke.toml"


def test_old_model_serialization_is_unchanged_and_local_settings_roundtrip():
    assert ModelSpec("jev", "system_one", "typesafe-ai/jev", "typesafe-ai").as_dict() == {
        "id": "jev",
        "backend": "system_one",
        "model": "typesafe-ai/jev",
        "provider": "typesafe-ai",
    }
    spec = load_benchmark_plan(PLAN).models[0]
    assert spec.transport == "self_hosted"
    assert spec.as_dict()["base_url"] == "http://127.0.0.1:8000"
    assert "api_key_env" not in spec.as_dict() and "timeout_seconds" not in spec.as_dict()


@pytest.mark.parametrize(
    "replacement",
    [
        ('transport = "self_hosted"', 'transport = "unknown"'),
        ('transport = "self_hosted"', 'transport = "vercel"'),
        ('backend = "system_one"', 'backend = "llm"'),
        ('base_url = "http://127.0.0.1:8000"', 'base_url = "file:///tmp/model"'),
        ('base_url = "http://127.0.0.1:8000"', "timeout_seconds = -1"),
    ],
)
def test_invalid_local_plan_settings_are_rejected(tmp_path, replacement):
    path = tmp_path / "plan.toml"
    path.write_text(PLAN.read_text().replace(*replacement))
    with pytest.raises(ValueError):
        load_benchmark_plan(path)


def test_three_local_plans_preserve_original_cohorts_and_task_settings():
    for hosted_path in (ROOT / "experiments").rglob("*-system-one-smoke.toml"):
        hosted = load_benchmark_plan(hosted_path)
        local = load_benchmark_plan(
            hosted_path.with_name(hosted_path.name.replace("-system-one-smoke", "-local-smoke"))
        )
        assert local.cohort == hosted.cohort
        assert local.task_config == hosted.task_config
        assert local.execution.example_concurrency == local.execution.model_concurrency == 1
        assert local.baseline_model_id == "laya-local"
        assert [model.id for model in local.models] == ["laya-local"]
        assert local.models[0].transport == "self_hosted"
