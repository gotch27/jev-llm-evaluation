"""Deterministic latency summaries and non-sensitive runner provenance."""

import pytest

from thesis_research.benchmark.timing import latency_statistics, runner_environment


def test_latency_statistics_use_linear_interpolation():
    statistics = latency_statistics([1.0, 2.0, 3.0, 4.0, 5.0])

    assert statistics == {
        "count": 5,
        "mean": 3.0,
        "standard_deviation": pytest.approx(2**0.5),
        "minimum": 1.0,
        "p50": 3.0,
        "p90": pytest.approx(4.6),
        "p95": pytest.approx(4.8),
        "maximum": 5.0,
    }
    assert latency_statistics([])["p95"] is None
    assert latency_statistics([2.5])["p95"] == 2.5


def test_runner_environment_records_label_without_hostname():
    environment = runner_environment(" local-mac-oslo ")

    assert environment["location"] == "local-mac-oslo"
    assert environment["python_version"]
    assert environment["utc_offset"]
    assert "hostname" not in environment
    with pytest.raises(ValueError, match="nonempty"):
        runner_environment("   ")
