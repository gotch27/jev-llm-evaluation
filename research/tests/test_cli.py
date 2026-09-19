"""Command-line application behavior."""

import os

import pytest

from thesis_research import cli


@pytest.mark.parametrize(
    ("existing_value", "expected_value"),
    [(None, "file-key"), ("external-key", "external-key")],
)
def test_cli_loads_dotenv_without_overriding_environment(
    tmp_path,
    monkeypatch,
    existing_value,
    expected_value,
):
    """Load the local .env before parsing while respecting an external value."""
    (tmp_path / ".env").write_text("AI_GATEWAY_API_KEY=file-key\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    if existing_value is not None:
        monkeypatch.setenv("AI_GATEWAY_API_KEY", existing_value)

    class EnvironmentLoaded(Exception):
        pass

    def verify_environment():
        assert os.environ["AI_GATEWAY_API_KEY"] == expected_value
        raise EnvironmentLoaded

    monkeypatch.setattr(cli, "_build_parser", verify_environment)
    with pytest.raises(EnvironmentLoaded):
        cli.main()
