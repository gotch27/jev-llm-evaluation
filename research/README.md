# Research code

Python experiment code for evaluating Jev and LLMs. This is an initial scaffold; model clients, dataset loaders, the runner, and metrics are still to be implemented.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```sh
uv sync --locked
uv run python -c "import thesis_research"
uv run ruff check .
uv run ruff format --check .
```

Python 3.12 is selected in `.python-version`; dependency versions are recorded in `uv.lock`. pytest is available for tests as implementation is added. There are no tests yet.

## Layout

Keep reusable implementation in `src/` and experiment settings in `experiments/`:

```text
research/
├── pyproject.toml
├── uv.lock
├── .python-version
├── AGENTS.md
├── README.md
├── src/thesis_research/
│   ├── models/          # Jev and LLM adapters
│   ├── datasets/        # Loading and selecting examples
│   ├── evaluation/      # Metrics and result summaries
│   └── runner.py       # Execute experiments and record outputs
├── experiments/
│   └── intent_classification/  # TOML run configurations
├── tests/
├── data/               # Ignored dataset downloads
└── outputs/            # Ignored results, grouped by run
```

The directories are tracked with `.gitkeep` files. Create modules and configurations as they are implemented; `runner.py` is planned and does not exist yet. Downloaded data and generated outputs remain ignored. Keep setup and layout documentation in this README rather than adding one per subfolder.

Record dataset sources, revisions, checksums, and splits. Preserve raw responses, predictions, errors, configuration, and provenance for each run, including failed runs.

Provider SDKs will be added when the corresponding clients are implemented. Keep credentials in environment variables or ignored `.env` files; loading a `.env` file is not implemented yet.

Study decisions and interpretation belong in [the vault](../vault/Thesis%20Home.md). Run records should identify the code commit, dataset revision, model configuration, command, and output location.
