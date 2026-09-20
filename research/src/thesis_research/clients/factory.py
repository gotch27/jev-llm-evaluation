"""Construction of configured decision clients."""

from thesis_research.clients.contracts import DecisionClient
from thesis_research.clients.jev import JevDecisionClient
from thesis_research.clients.vercel_llm import VercelLLMDecisionClient


def create_decision_client(
    backend: str,
    model: str,
    provider: str | None,
    *,
    max_concurrency: int,
    llm_output_mode: str = "probabilities",
) -> DecisionClient:
    """Create a decision client for one configured model backend.

    Args:
        backend: Backend name, either ``"jev"`` or ``"llm"``.
        model: Provider model identifier.
        provider: Pinned Vercel AI Gateway provider, or ``None`` for Jev.
        max_concurrency: Shared upper bound on active provider requests.
        llm_output_mode: LLM answer format; ignored by the Jev backend.

    Returns:
        An unopened asynchronous client ready for context-manager use.

    Raises:
        ValueError: If the backend or its provider setting is invalid, or a
            required API key is absent.
    """
    if backend == "jev":
        if provider is not None:
            raise ValueError("--provider is only valid with the llm backend")
        return JevDecisionClient(model, max_concurrency=max_concurrency)
    if backend == "llm":
        if provider is None or not provider.strip():
            raise ValueError("--provider is required with the llm backend")
        return VercelLLMDecisionClient(
            model,
            provider,
            output_mode=llm_output_mode,
            max_concurrency=max_concurrency,
        )
    raise ValueError("backend must be jev or llm")
