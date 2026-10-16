"""Construction of configured decision clients."""

from thesis_model_clients.contracts import DecisionClient
from thesis_model_clients.system_one import (
    TypeSafeHTTPDecisionClient,
    VercelSystemOneDecisionClient,
)
from thesis_model_clients.vercel_llm import VercelLLMDecisionClient


def create_decision_client(
    backend: str,
    model: str,
    provider: str,
    *,
    max_concurrency: int,
    llm_output_mode: str | None = None,
    reasoning_effort: str | None = None,
    transport: str = "vercel",
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: float | None = None,
) -> DecisionClient:
    """Create a decision client for one configured model backend.

    Args:
        backend: Backend name, either ``"system_one"`` or ``"llm"``.
        model: Provider model identifier.
        provider: Pinned gateway provider or self-hosted provenance label.
        max_concurrency: Shared upper bound on active provider requests.
        llm_output_mode: Explicit LLM answer format; ignored by System One.
        reasoning_effort: Explicit Vercel reasoning effort for an LLM, or
            ``None`` to leave the setting unspecified.
        transport: Vercel (default) or a self-hosted TypeSafe HTTP service.
        base_url: API root required for self-hosted services.
        api_key_env: Optional self-hosted authentication environment variable.
        timeout_seconds: Self-hosted request timeout; defaults to 120 seconds.

    Returns:
        An unopened asynchronous client ready for context-manager use.

    Raises:
        ValueError: If the backend or its provider setting is invalid, or a
            required API key is absent.
    """
    if transport not in ("vercel", "self_hosted"):
        raise ValueError("transport must be vercel or self_hosted")
    if transport == "vercel" and any(
        value is not None for value in (base_url, api_key_env, timeout_seconds)
    ):
        raise ValueError("base_url, api_key_env, and timeout_seconds require self_hosted transport")
    if transport == "self_hosted" and backend != "system_one":
        raise ValueError("self_hosted transport requires the system_one backend")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a nonempty provider label")
    if backend == "system_one":
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is only valid with the llm backend")
        if transport == "self_hosted":
            if base_url is None:
                raise ValueError("self_hosted transport requires base_url")
            return TypeSafeHTTPDecisionClient(
                model,
                provider,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_seconds=120.0 if timeout_seconds is None else timeout_seconds,
                max_concurrency=max_concurrency,
            )
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
