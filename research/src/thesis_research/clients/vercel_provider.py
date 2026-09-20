"""Strict Vercel AI Gateway transport used by the System One adapter."""

from dataclasses import asdict, dataclass
from typing import Any

import openai
from system_one_adapter.providers import Message, ProviderResult, translating
from system_one_adapter.providers.openai import AsyncOpenAIProvider
from typesafe_sdk import TypeSafeError

from thesis_research.clients.contracts import DEFAULT_TIMEOUT_SECONDS

VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


@dataclass(frozen=True)
class VercelGatewayProviderResult(ProviderResult):
    """Extend an adapter response with reproducibility metadata.

    Attributes:
        request: Complete request without authentication credentials.
        raw_response: JSON-safe provider response.
        resolved_model: Model reported by Vercel AI Gateway.
        resolved_provider: Upstream provider reported by Vercel AI Gateway.
        cost_usd: Request cost reported by Vercel AI Gateway, when available.
    """

    request: dict[str, Any]
    raw_response: dict[str, Any]
    resolved_model: str
    resolved_provider: str
    cost_usd: float | None


class VercelGatewayProvider(AsyncOpenAIProvider):
    """Send strict OpenAI-compatible requests to a pinned Vercel AI Gateway route.

    Requests require native JSON Schema output and one exact upstream provider.
    Model fallbacks are disabled by omitting Vercel's optional ``models`` list.

    Args:
        model_name: Canonical Vercel model identifier.
        provider_name: Only upstream provider allowed to serve the request.
        api_key: Vercel AI Gateway credential passed directly to the HTTP client.
        reasoning_effort: Explicit provider reasoning effort, or ``None`` to
            leave the setting unspecified.
        timeout_seconds: HTTP timeout applied to each request.
        _client: Optional OpenAI-compatible client injected by offline tests.
    """

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        api_key: str,
        *,
        reasoning_effort: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        _client: Any | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
        self.model_name = model_name
        self.provider_name = provider_name
        self.reasoning_effort = reasoning_effort
        self.api = "chat_completions"
        self._client = _client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=VERCEL_AI_GATEWAY_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Send one strict structured-output chat-completions request.

        Args:
            messages: Adapter-generated conversation sent to the model.
            schema: JSON Schema generated for the isolated TypeSafe question.
            structured: Whether native structured output was requested.

        Returns:
            Adapter text and usage enriched with routing and raw diagnostics.

        Raises:
            ValueError: If the adapter attempts an unstructured request.
            TypeSafeError: If Vercel AI Gateway omits required response information.
        """
        if not structured:
            raise ValueError("Vercel AI Gateway evaluations require native structured output")
        extra_body: dict[str, Any] = {
            "providerOptions": {
                "gateway": {
                    "only": [self.provider_name],
                }
            }
        }
        if self.reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [asdict(message) for message in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "evaluation", "schema": schema, "strict": True},
            },
            "extra_body": extra_body,
        }
        with translating(self.translate_error):
            response = await self._client.chat.completions.create(**request)
        raw = response.model_dump(mode="json")
        if response.usage is None:
            raise TypeSafeError("Vercel AI Gateway response did not include token usage")
        if not response.choices:
            raise TypeSafeError("Vercel AI Gateway response did not include a choice")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeSafeError("Vercel AI Gateway response did not include text content")
        return VercelGatewayProviderResult(
            text=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            request=request,
            raw_response=raw,
            resolved_model=str(raw.get("model") or self.model_name),
            resolved_provider=_resolved_provider(raw, self.provider_name),
            cost_usd=_cost(raw),
        )


def _resolved_provider(raw: dict[str, Any], requested_provider: str) -> str:
    direct = raw.get("provider")
    if isinstance(direct, str) and direct:
        return direct
    gateway = _gateway_metadata(raw)
    routing = gateway.get("routing")
    if isinstance(routing, dict):
        for key in ("resolvedProvider", "resolved_provider"):
            value = routing.get(key)
            if isinstance(value, str) and value:
                return value
    return requested_provider


def _cost(raw: dict[str, Any]) -> float | None:
    usage = raw.get("usage")
    if isinstance(usage, dict):
        parsed = _numeric_value(usage.get("cost"))
        if parsed is not None:
            return parsed
    parsed = _numeric_value(_gateway_metadata(raw).get("cost"))
    if parsed is not None:
        return parsed
    return None


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _gateway_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("provider_metadata", raw.get("providerMetadata"))
    if not isinstance(metadata, dict):
        return {}
    gateway = metadata.get("gateway")
    return gateway if isinstance(gateway, dict) else {}
