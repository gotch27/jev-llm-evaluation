"""Construction of configured decision clients."""

from thesis_research.clients.contracts import DecisionClient
from thesis_research.clients.system_one import VercelSystemOneDecisionClient
from thesis_research.clients.vercel_llm import VercelLLMDecisionClient


def create_decision_client(
    backend: str,
    model: str,
    provider: str,
    *,
    max_concurrency: int,
    llm_output_mode: str | None = None,
    reasoning_effort: str | None = None,
) -> DecisionClient:
    """Create a decision client for one configured model backend.

    Args:
        backend: Backend name, either ``"system_one"`` or ``"llm"``.
        model: Provider model identifier.
        provider: Pinned Vercel AI Gateway provider.
        max_concurrency: Shared upper bound on active provider requests.
        llm_output_mode: Explicit LLM answer format; ignored by System One.
        reasoning_effort: Explicit Vercel reasoning effort for an LLM, or
            ``None`` to leave the setting unspecified.

    Returns:
        An unopened asynchronous client ready for context-manager use.

    Raises:
        ValueError: If the backend or its provider setting is invalid, or a
            required API key is absent.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a nonempty pinned provider")
    if backend == "system_one":
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is only valid with the llm backend")
        return VercelSystemOneDecisionClient(model, provider, max_concurrency=max_concurrency)
    if backend == "llm":
        if llm_output_mode not in ("discrete", "probabilities"):
            raise ValueError("llm_output_mode must be discrete or probabilities for LLMs")
        return VercelLLMDecisionClient(
            model,
            provider,
            output_mode=llm_output_mode,
            reasoning_effort=reasoning_effort,
            max_concurrency=max_concurrency,
        )
    raise ValueError("backend must be system_one or llm")
