"""Timing statistics and runner provenance for benchmark records."""

from __future__ import annotations

import math
import platform
import statistics
from datetime import datetime
from typing import Any, Sequence


def runner_environment(location: str | None) -> dict[str, str]:
    """Describe the machine executing one benchmark attempt.

    The location is a user-supplied label because a local process cannot
    reliably infer its cloud region or physical location. Hostnames and IP
    addresses are deliberately omitted.
    """
    normalized = "unspecified" if location is None else location.strip()
    if not normalized:
        raise ValueError("runner_location must be nonempty when supplied")
    local_time = datetime.now().astimezone()
    offset = local_time.strftime("%z")
    formatted_offset = f"{offset[:3]}:{offset[3:]}" if offset else "unknown"
    return {
        "location": normalized,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "timezone": str(local_time.tzinfo),
        "utc_offset": formatted_offset,
    }


def latency_statistics(values: Sequence[float]) -> dict[str, int | float | None]:
    """Summarize nonnegative durations using linear-interpolated percentiles.

    p50 is the median; p90 and p95 describe the slower tail of observed calls.
    Interpolation can produce a duration not present in the measurements.
    Small cohorts provide only a coarse description of tail latency.
    """
    durations = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in durations):
        raise ValueError("Latency values must be finite and nonnegative")
    if not durations:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "maximum": None,
        }
    ordered = sorted(durations)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "standard_deviation": statistics.pstdev(ordered),
        "minimum": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def latency_fields(prefix: str, statistics_value: dict[str, Any]) -> dict[str, Any]:
    """Flatten one latency summary for a tidy report row."""
    return {
        f"{prefix}_{name}_seconds" if name != "count" else f"{prefix}_count": value
        for name, value in statistics_value.items()
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
