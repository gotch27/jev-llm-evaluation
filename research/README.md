# Research code

Python components for the thesis experiments. The project prepares and scores BANKING77
predictions and provides a shared structured interface for calling Jev and LLMs through
Vercel AI Gateway. The command-line application connects those pieces; selecting models and
executing live experiments remain research decisions.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run all commands below from `research/`:

```sh
uv sync --locked
cp .env.example .env
uv run thesis-research --help
```

Open `.env` and set `AI_GATEWAY_API_KEY` to your Vercel AI Gateway key. The CLI loads this file
automatically and does not replace a value already supplied by your shell or PyCharm. `.env` is
ignored by Git; `.env.example` is versioned because it contains only the variable name.

In PyCharm, open the repository root, select `research/.venv/bin/python` as the interpreter,
and set the run configuration's working directory to `research/`. To run the CLI from a Python
run configuration, select **Module name** and enter `thesis_research.cli`; use the command and
arguments shown below as its parameters. You do not need to duplicate the API key in PyCharm's
environment-variable settings when the working directory is `research/`. `uv.lock` records
dependency versions.

## Structured model clients

Both clients accept the same state and TypeSafe `Choice`, `Score`, and `Noul` questions and
return a `DecisionResult`. Jev receives the complete question set in one request. An LLM receives
one isolated request per question; those requests run concurrently with at most five active by
default. For BANKING77 there will be one Choice question with 77 options, so this distinction does
not create extra requests.

Both paths use `AI_GATEWAY_API_KEY`. Jev uses Vercel's TypeSafe-compatible endpoint while keeping
the TypeSafe Python SDK and its native state-and-questions interface:

```python
import asyncio

from dotenv import load_dotenv
from thesis_research.clients import JevDecisionClient
from typesafe_sdk import Choice


async def main() -> None:
    questions = {
        "intent": Choice(
            instructions="Which intent best matches this message?",
            criteria={"card_arrival": None, "change_pin": None},
        )
    }
    async with JevDecisionClient("typesafe-ai/jev") as client:
        result = await client.evaluate("When will my card arrive?", questions)
        print(result.answers["intent"])


load_dotenv()
asyncio.run(main())
```

For an LLM, supply a canonical Vercel model ID and pin one upstream provider. Virtual models and
routing aliases are rejected. Prefer a versioned model ID when Vercel offers one. The selected
model/provider must support native JSON-schema output.

```python
import asyncio

from dotenv import load_dotenv
from thesis_research.clients import VercelLLMDecisionClient
from typesafe_sdk import Choice


async def main() -> None:
    questions = {
        "intent": Choice(
            instructions="Which intent best matches this message?",
            criteria={"card_arrival": None, "change_pin": None},
        )
    }
    async with VercelLLMDecisionClient(
        model="creator/model-id",
        provider="upstream-provider-slug",
    ) as client:
        result = await client.evaluate("When will my card arrive?", questions)
        print(result.answers["intent"])


load_dotenv()
asyncio.run(main())
```

The examples call `load_dotenv()` because they are library-level Python scripts. The command-line
application does this automatically. Do not put credentials in source files, TOML configurations,
or command-line arguments.

LLMs always use native structured output and return full probability distributions. Invalid sums
are normalized, while the original values and normalization error remain in each call's raw
diagnostics. LLM requests pin the upstream provider with Vercel's `only` option and omit the
optional model-fallback list. They preserve the request and raw response, including routing and
cost metadata returned by the gateway, without authentication headers.

`DecisionResult` provides typed `answers`, per-question `errors`, requested and resolved model and
provider identifiers, aggregate token/cost/retry usage, total wall-clock latency, and individual
call records. If any call has unknown usage or cost, that aggregate is `None` rather than an
incomplete total. Vercel's TypeSafe-compatible response currently does not report per-request
cost or the number of SDK retries, so those successful Jev fields are `None`.

Both clients use the same explicit transient retry policy: two retries with bounded backoff,
retry-after support, and a 30-second retry budget. HTTP calls have a 60-second timeout. Malformed
LLM output is not retried. Model calling is asynchronous; always use the clients as async context
managers or call `aclose()`.

## Prepare and inspect BANKING77

```sh
uv run thesis-research prepare
uv run thesis-research inspect --limit 5
```

The default configuration is `experiments/intent_classification/banking77.toml`. It selects
all 77 labels and the official test split for scoring. `inspect` always displays training
examples, regardless of the configured evaluation split. Use training data for development;
reserve test data for evaluation.

Data comes from [PolyAI's official dataset repository](https://github.com/PolyAI-LDN/task-specific-datasets#banking),
pinned to commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. The source provides 10,003 training
examples and 3,080 test examples. Files are stored under `data/banking77/<revision>/`, alongside
a manifest containing the source, revision, URLs, and SHA-256 checksums. Subsequent preparation
uses the cached files and verifies checksums. Downloads use a 60-second timeout with no automatic
retries. If preparation fails, its files remain for inspection; rerunning can complete an
interrupted download. A checksum mismatch stops processing rather than replacing modified data.

The dataset is licensed under [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE).
When using it in the thesis, cite Casanueva et al. (2020),
[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).

The loader returns examples with `id`, `text`, and `label`. IDs such as `train:1` and `test:1`
use the split and one-based original CSV data-row number (excluding the header). They are
stable within the pinned revision. Text, label spelling, and split membership are preserved.

## BANKING77 Choice definition

The experiment configuration contains the complete provider-neutral classification task. Each
of the 77 official labels has a concise `use_when` definition and a `distinguish_from` boundary.
The criteria were derived only from training examples and contain no example utterances. The
question uses the original label order and spelling, including `Refund_not_showing_up` and
`reverted_card_payment?`.

The task builder validates the configuration against labels loaded from the pinned dataset. It
then produces the same state and TypeSafe `Choice` for Jev or an LLM:

```python
from pathlib import Path

from thesis_research.config import load_experiment_config
from thesis_research.datasets import load_banking77
from thesis_research.tasks import build_banking77_task

config_path = Path("experiments/intent_classification/banking77.toml")
config = load_experiment_config(config_path)
dataset_dir = Path("data/banking77") / config.dataset_revision
labels, training_examples, _ = load_banking77(dataset_dir, "train")
task = build_banking77_task(config, labels)

state = task.build_state(training_examples[0].text)
questions = task.questions()
```

The state is `{"customer_message": <original text>}` and the sole question ID is `intent`.
There is no `other` option because BANKING77 is a closed-set task. The configuration records a
dataset-specific naming quirk: `get_physical_card` means finding or receiving the physical card's
PIN in the training data, while `order_physical_card` means obtaining the card itself.

This defines the task but does not run it. Dataset prediction orchestration and model selection
are described below.

## Run predictions

The `predict` command loads the configured split, builds the frozen BANKING77 task, calls one
backend for every selected example, and prints the unique prediction-run directory it created.
Set `AI_GATEWAY_API_KEY` in `research/.env` first.

For Jev, use Vercel's published evaluation-model ID:

```sh
uv run thesis-research predict \
  --backend jev \
  --model typesafe-ai/jev
```

For an LLM, supply a canonical model ID and the pinned upstream provider:

```sh
uv run thesis-research predict \
  --backend llm \
  --model creator/model-id \
  --provider upstream-provider-slug
```

The default configuration predicts the complete official test split. Use `--limit N` for a
bounded software or connectivity run and `--max-concurrency N` to change the default limit of five
active requests. A limited prediction file is intentionally incomplete and the evaluator will
count the omitted examples as missing. For prompt or model development, create a separate
versioned configuration with `split = "train"`; keep the test split reserved for the final runs.

Every attempt gets a new directory under `outputs/predictions/` containing:

- `config.toml`: the complete dataset and Choice configuration used by the run.
- `predictions.jsonl`: ordered labels or explicit errors, ready for the evaluator.
- `decisions.jsonl`: each state, probability distribution, timing, usage, provider call record,
  raw response, and diagnostic error.
- `summary.json`: prediction counts, aggregate usage, elapsed time, and resolved routing.
- `metadata.json`: dataset provenance, model settings, command, code revision and dirty status,
  progress, status, and artifact checksums.

Files are flushed after each bounded batch and failed or interrupted attempts remain available for
inspection. No API keys are written to the run directory. Individual provider failures are saved
as failed predictions while the remaining examples continue.

Vercel currently exposes Jev as the unversioned `typesafe-ai/jev` identifier. Run records preserve
that identifier, the response-reported model, the provider, timestamp, raw response, and code
revision. Confirm the resolved model metadata with a small connectivity run before using it for a
thesis-quality benchmark.

## Evaluate saved predictions

Supply UTF-8 JSONL: one JSON object per line, containing `id` and exactly one of `label` or
`error`. For example, the following illustrates the format only, not actual predictions:

```jsonl
{"id": "test:1", "label": "card_arrival"}
{"id": "test:2", "error": "Request timed out"}
```

Use exact dataset labels. Error messages must be nonempty strings. Blank lines are ignored;
malformed records stop evaluation. Unknown labels or non-string label values count as invalid
predictions. Predictions are matched by ID, so file order does not matter. Duplicate IDs and
IDs outside the configured split are rejected. Omitted IDs count as missing predictions.

Place your prediction file in an ignored directory, then run:

```sh
uv run thesis-research evaluate --predictions outputs/predictions.jsonl
```

For a generated run, replace `RUN_DIRECTORY` with the directory printed by `predict`:

```sh
uv run thesis-research evaluate \
  --predictions RUN_DIRECTORY/predictions.jsonl
```

All commands accept `--config PATH` and `--data-dir PATH`; evaluation also accepts
`--output-dir PATH`. Paths are relative to your working directory. To score development
predictions, copy the experiment configuration and change `split` to `"train"`.

Each evaluation attempt creates a unique directory under `outputs/` containing:

- `config.toml` and `predictions.jsonl`: exact copies of the supplied inputs, when readable.
- `metadata.json`: UTC timestamp, configuration, dataset provenance, prediction checksum,
  code commit and dirty status, command, and completion status or error.
- `summary.json`: metrics and counts, when evaluation succeeds.
- `outcomes.jsonl`: each example's reference label, supplied prediction, status, and correctness.

Failed attempts remain recorded. Existing run directories are never overwritten. Git fields
are `null` if Git information cannot be read. A commit and dirty flag do not capture uncommitted
code: use a clean committed version for reproducible thesis runs. Prediction files should retain
extra provider information if available; this scorer preserves those fields but does not validate
model provenance. Prediction runs generated by this project record model settings, the frozen
prompt, usage, latency, cost when available, and raw provider diagnostics separately.

### Metric definitions

- **Accuracy:** correct predictions divided by all expected examples, including missing,
  invalid, and failed predictions. Values range from 0 to 1; multiply by 100 for percentages.
- **Per-label precision:** correct predictions of that label divided by all valid predictions
  of it. **Recall:** correct predictions of that label divided by its reference support.
- **F1:** harmonic mean of precision and recall. Undefined ratios are reported as zero.
- **Macro-F1:** unweighted mean F1 over all 77 configured labels.
- **Confusion matrix:** rows are reference labels; columns are predicted labels plus a final
  `<unsuccessful>` column covering missing, invalid, and failed predictions. These errors
  reduce recall for the reference label. Separate status counts distinguish their causes.

Preparing data, mocked provider tests, and checking the software do not constitute measured model
results.

## Code layout and checks

- `src/thesis_research/datasets/`: pinned data preparation and loading.
- `src/thesis_research/config.py`: typed, validated experiment configuration.
- `src/thesis_research/tasks/`: validated state and question construction.
- `src/thesis_research/clients/`: shared result contracts plus Jev and Vercel AI Gateway clients.
- `src/thesis_research/prediction.py`: bounded prediction and result serialization.
- `src/thesis_research/prediction_run.py`: durable prediction-run orchestration.
- `src/thesis_research/evaluation/`: prediction parsing, metrics, and evaluation-run records.
- `src/thesis_research/run_storage.py`: shared durable-record and Git metadata helpers.
- `src/thesis_research/cli.py`: argument parsing and command dispatch only.
- `experiments/intent_classification/`: versioned TOML configurations.
- `tests/`: offline tests with synthetic fixtures, not model experiments.
- `data/` and `outputs/`: ignored dataset downloads and generated records.

```sh
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
```

Study decisions and interpretation belong in [the vault](../vault/Thesis%20Home.md).
