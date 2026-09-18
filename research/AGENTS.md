# Research code

Python code for reproducible Jev and LLM experiments. Read `README.md` here for setup and layout.

- Use Python 3.12 and uv. Run commands from `research/`; keep `pyproject.toml` and `uv.lock` aligned.
- Keep reusable code in `src/thesis_research/`, experiment configurations in `experiments/`, and tests in `tests/`.
- Keep setup and layout documentation in the main `README.md`; do not add subfolder READMEs or empty placeholder modules.
- Build only what the current experiment needs. Prefer small functions and explicit configuration; avoid premature frameworks.
- Requested coding work authorizes the necessary code, configuration, test, and technical README changes. Vault updates require separate authorization.
- Keep model settings, prompts, retries, and output handling explicit. Record dataset revisions, splits, model versions, configuration, timing, usage, errors, and code revision for each run.
- Preserve failed runs. Keep test data separate from prompt development and tuning.
- Store credentials outside versioned files. Downloaded data belongs in `data/` and generated runs in `outputs/`, both ignored.
- Check changes with `uv run ruff check .` and `uv run ruff format --check .`. Run relevant pytest tests when implementation exists; test meaningful behavior such as metrics, splits, and error handling.
- Report what changed, what was checked, and what remains unfinished. Never present simulated or unrun experiments as measured results.
