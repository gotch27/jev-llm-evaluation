"""Provider-neutral decision contracts and concrete API clients."""

from thesis_research.clients.contracts import (
    CallRecord,
    DecisionClient,
    DecisionError,
    DecisionResult,
    UsageTotals,
)
from thesis_research.clients.factory import create_decision_client
from thesis_research.clients.system_one import (
    TypeSafeHTTPDecisionClient,
    VercelSystemOneDecisionClient,
)
from thesis_research.clients.vercel_llm import VercelLLMDecisionClient

__all__ = [
    "CallRecord",
    "DecisionClient",
    "DecisionError",
    "DecisionResult",
    "TypeSafeHTTPDecisionClient",
    "VercelSystemOneDecisionClient",
    "VercelLLMDecisionClient",
    "UsageTotals",
    "create_decision_client",
]
