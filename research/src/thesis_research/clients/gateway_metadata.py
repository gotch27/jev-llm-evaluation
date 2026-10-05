"""Read reported Vercel routing and cost from complete response bodies."""

import math
from typing import Any


def gateway_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Return gateway metadata from either Vercel response naming convention."""
    metadata = raw.get("provider_metadata", raw.get("providerMetadata"))
    if not isinstance(metadata, dict):
        return {}
    gateway = metadata.get("gateway")
    return gateway if isinstance(gateway, dict) else {}


def resolved_provider(raw: dict[str, Any]) -> str | None:
    """Return the serving provider only when the response reports it."""
    direct = raw.get("provider")
    if isinstance(direct, str) and direct:
        return direct
    routing = gateway_metadata(raw).get("routing")
    if isinstance(routing, dict):
        for key in ("resolvedProvider", "resolved_provider", "finalProvider"):
            value = routing.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def reported_cost(raw: dict[str, Any]) -> float | None:
    """Read finite nonnegative billed cost; absent cost remains unknown."""
    usage = raw.get("usage")
    if isinstance(usage, dict):
        parsed = _cost_value(usage.get("cost"))
        if parsed is not None:
            return parsed
    return _cost_value(gateway_metadata(raw).get("cost"))


def _cost_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        cost = float(value)
    except (ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None
