# Local model serving

This is an independent Python 3.12 uv project. It serves checkpoints over HTTP;
it does not import the research benchmark package or run training.

- Read README.md before changing the server or setup.
- Keep reusable code in src/model_serving/ and tests in tests/.
- Keep pyproject.toml and uv.lock aligned. Run uv commands from this directory.
- Preserve TypeSafe-compatible requests and responses, explicit checkpoint selection,
  all question options, raw usage, and runtime provenance. Report truncation and failures.
- Keep checkpoints in ignored models/ and generated records in ignored outputs/.
- Use loopback by default. Never read research credentials or route through a cloud provider.
- Run pytest, Ruff lint/format checks, and package builds for implementation changes.
- No fine-tuning, commits, or pushes without explicit user authorization.
