"""Reusable decision records retain answers, partial failures, and unknown cost."""

import json

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from thesis_model_clients import (
    CallRecord,
    DecisionError,
    DecisionResult,
    UsageTotals,
    decision_result_to_dict,
)


@pytest.fixture
def decision():
    usage = UsageTotals(12, 3, None, 1, 0)
    error = DecisionError("ModelRefusal", "Declined this question", {"question": "blocked"})
    return DecisionResult(
        answers={
            "intent": ChoiceAnswer(choice="card", confidence=1, probabilities={"card": 1}),
            "visible": NoulAnswer(noul=0.9),
            "rating": ScoreAnswer(
                score=1, confidence=1, probabilities={0: 0, 1: 1}, legend={0: "low", 1: "high"}
            ),
        },
        errors={"blocked": error},
        requested_model="model-v1",
        requested_provider="requested-provider",
        resolved_models=("resolved-v1",),
        resolved_providers=("resolved-provider",),
        usage=usage,
        latency_seconds=2,
        calls=(
            CallRecord(
                ("intent", "visible", "rating", "blocked"),
                "model-v1",
                "resolved-v1",
                "requested-provider",
                "resolved-provider",
                1,
                usage,
                None,
                {"provider_response": {"refusal": "blocked"}},
            ),
        ),
    )


@pytest.mark.parametrize("mode", ["discrete", "probabilities", "native_probabilities"])
def test_decision_records_are_json_safe_and_keep_partial_failures(decision, mode):
    record = json.loads(json.dumps(decision_result_to_dict(decision, answer_output_mode=mode)))
    assert record["requested_model"] == "model-v1"
    assert record["resolved_models"] == ["resolved-v1"]
    assert record["resolved_providers"] == ["resolved-provider"]
    assert record["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cost_usd": None,
        "retries": 1,
        "malformed_response_retries": 0,
    }
    assert record["latency_seconds"] == 2
    assert record["calls"][0]["latency_seconds"] == 1
    assert record["calls"][0]["raw"] == {"provider_response": {"refusal": "blocked"}}
    assert record["errors"]["blocked"] == {
        "error_type": "ModelRefusal",
        "message": "Declined this question",
        "diagnostics": {"question": "blocked"},
    }
    if mode == "discrete":
        assert record["answers"] == {
            "intent": {"type": "choice", "choice": "card"},
            "visible": {"type": "noul", "value": True},
            "rating": {"type": "score", "score": 1},
        }
    else:
        assert record["answers"]["intent"]["probabilities"] == {"card": 1}
        assert record["answers"]["visible"]["noul"] == 0.9
        assert record["answers"]["rating"]["probabilities"] == {"0": 0, "1": 1}


def test_unknown_serialization_mode_is_rejected(decision):
    with pytest.raises(ValueError, match="answer_output_mode must be"):
        decision_result_to_dict(decision, answer_output_mode="unknown")
