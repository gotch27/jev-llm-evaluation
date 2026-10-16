"""Provider-neutral decision contracts and concrete API clients."""

from thesis_model_clients.contracts import (
    CallRecord,
    DecisionClient,
    DecisionError,
    DecisionResult,
    UsageTotals,
)
from thesis_model_clients.factory import create_decision_client
from thesis_model_clients.images import ImageAsset, ImageInputs, image_capability, load_image
from thesis_model_clients.serialization import AnswerOutputMode, decision_result_to_dict
from thesis_model_clients.system_one import (
    TypeSafeHTTPDecisionClient,
    VercelSystemOneDecisionClient,
)
from thesis_model_clients.vercel_llm import VercelLLMDecisionClient

__all__ = [
    "AnswerOutputMode",
    "CallRecord",
    "DecisionClient",
    "DecisionError",
    "DecisionResult",
    "ImageAsset",
    "ImageInputs",
    "TypeSafeHTTPDecisionClient",
    "VercelSystemOneDecisionClient",
    "VercelLLMDecisionClient",
    "UsageTotals",
    "create_decision_client",
    "decision_result_to_dict",
    "image_capability",
    "load_image",
]
