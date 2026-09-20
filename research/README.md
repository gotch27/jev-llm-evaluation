# Research code

Python components for the thesis experiments. The project prepares BANKING77, provides a shared
structured interface for calling Jev and LLMs through Vercel AI Gateway, and runs coordinated
benchmarks in which every configured model receives the same frozen cohort. Benchmark reports
are saved as machine-readable tables for later statistical analysis and visualization.

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
        output_mode="label",
    ) as client:
        result = await client.evaluate("When will my card arrive?", questions)
        print(result.answers["intent"])


load_dotenv()
asyncio.run(main())
```

The examples call `load_dotenv()` because they are library-level Python scripts. The command-line
application does this automatically. Do not put credentials in source files, TOML configurations,
or command-line arguments.

LLMs always use native structured output. `output_mode="label"` requires one exact Choice label;
it does not ask for or save a self-reported confidence. `output_mode="probabilities"` requires a
probability for every Choice label. Invalid probability sums are normalized, while the original
values and normalization error remain in each call's raw diagnostics. Benchmark plans set this
once for all configured LLMs. Jev continues to return its native probability distribution in
either mode.

LLM requests pin the upstream provider with Vercel's `only` option and omit the optional
model-fallback list. They preserve the request and raw response, including routing and cost
metadata returned by the gateway, without authentication headers.

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

The default task configuration is
`experiments/intent_classification/tasks/banking77.toml`. It defines all 77 labels and their
criteria. Dataset splits belong to benchmark plans: use `split = "train"` for development and
smoke checks, and reserve `split = "test"` for final evaluation. `inspect` always displays
training examples.

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

The task directory contains two provider-neutral variants shared by Jev and every LLM:

- `banking77.toml` gives every label a concise `use_when` definition and a `distinguish_from`
  boundary. These criteria were derived only from training examples and contain no example
  utterances.
- `banking77-without-criteria.toml` keeps the identical dataset revision, instruction, label order, and
  spelling, but omits every criterion. TypeSafe therefore receives `null` for every label.

Both use all 77 official labels, including the original spellings `Refund_not_showing_up` and
`reverted_card_payment?`. A benchmark selects a variant through its `task_config` path, so the
choice is preserved in the saved run record.

Every option has a `label` and an optional singular `criterion`. The criterion is deliberately
generic: it can be a string, object, array, or omitted. Omission maps to JSON `null`. For example,
all of these task fragments are valid and reach TypeSafe without changing their JSON shape:

```toml
[[question.options]]
label = "card_arrival"
criterion = "Use when a dispatched card has not arrived."

[[question.options]]
label = "card_arrival"
criterion.use_when = "A dispatched card has not arrived."
criterion.distinguish_from = "Use card_delivery_estimate for a general delivery-time question."

[[question.options]]
label = "card_arrival"
criterion = ["card ordered", "card dispatched", "card missing"]

[[question.options]]
label = "card_arrival" # no criterion; TypeSafe receives null
```

The versioned BANKING77 task uses the object form because its two named fields make the intent
boundaries easier to review. `criterion` is the value for one option; TypeSafe's outer `criteria`
mapping is built as `{label: criterion}`.

The task builder validates the configuration against labels loaded from the pinned dataset. It
then produces the same state and TypeSafe `Choice` for Jev or an LLM:

```python
from pathlib import Path

from thesis_research.config import load_task_config
from thesis_research.datasets import load_banking77
from thesis_research.tasks import build_banking77_task

config_path = Path("experiments/intent_classification/tasks/banking77.toml")
config = load_task_config(config_path)
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

This defines the task but does not make model calls. A benchmark plan connects that task to a
cohort and a list of models.

## Run a coordinated benchmark

`experiments/intent_classification/benchmarks/banking77-smoke.toml` is a one-example connectivity
check. It references the shared task and calls Jev plus five LLMs:

```toml
name = "BANKING77 smoke benchmark"
task_config = "../tasks/banking77.toml"

[cohort]
split = "train"
strategy = "random"
size = 1
seed = 20260920

[llm_output]
mode = "label"

[execution]
example_concurrency = 1
model_concurrency = 6

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "gpt-5-4-mini"
backend = "llm"
model = "openai/gpt-5.4-mini"
provider = "openai"

# The file also contains Claude Haiku 4.5, Gemini 3 Flash,
# Grok 4.1 Fast Non-Reasoning, and DeepSeek V3.2.
```

`banking77-without-criteria-smoke.toml` uses the task without criteria. It deliberately keeps the
same training split, random seed, cohort size, models, and LLM output mode, so both smoke plans
select the same example and differ only in the task definition.

The plan requires exactly one Jev entry and at least one LLM. Model IDs are stable local names used
in file paths and reports. LLM providers are pinned; Jev is pinned to TypeSafe's Vercel provider by
the client. `cohort.split` must be `train` or `test`. `random` selects any requested number of
examples and is suitable for smoke checks. `stratified_random` requires at least 77 examples and
selects every intent before adding a second example from any intent. A seed makes either selection
repeatable. Use `strategy = "all"` without `size` or `seed` when the complete configured split
should be evaluated.

`llm_output.mode = "label"` makes every LLM return only the selected label. This is the default
choice for the accuracy and F1 benchmark because no probability is needed for those metrics. Set
it to `"probabilities"` when an experiment needs the complete LLM distribution. The setting does
not change Jev's native output, and label mode does not produce a confidence value.

`example_concurrency` bounds active examples inside each model run. `model_concurrency` bounds how
many models run at the same time. The smoke plan uses `6`, so Jev and all five LLMs are called
together. For quality runs, use `1` to avoid cross-model load affecting latency. The chosen
execution settings are saved with the benchmark.

After preparing the data and setting `AI_GATEWAY_API_KEY`, run:

```sh
uv run thesis-research benchmark \
  --plan experiments/intent_classification/benchmarks/banking77-smoke.toml
```

The command prints a unique directory under `outputs/benchmarks/`. Every model receives the exact
same saved cohort. A benchmark cannot accidentally compare 300 Jev examples with 600 LLM examples.
Individual provider failures become explicit unsuccessful predictions and remain in the accuracy
denominator. If a process or setup failure stops a model, completed models and completed example IDs
are preserved. Resume that same benchmark with:

```sh
uv run thesis-research benchmark --resume RUN_DIRECTORY
```

In an interactive terminal, the CLI keeps one progress row per model. Each row shows whether the
model is waiting, running, done, or failed; completed examples; structurally valid and failed
predictions; processing rate; elapsed time; and approximate ETA. Prediction failures and lifecycle
events are printed above the live rows so they remain in terminal scrollback. Error messages are
shortened there, while `decisions.jsonl` retains the complete diagnostics.

Progress advances after each flushed example batch. With `example_concurrency = 1`, it advances
one example at a time; with a larger value, it advances by that batch size. Models still progress
independently and do not wait for one another between batches. If `model_concurrency` is smaller
than the number of configured models, queued models remain visibly marked as waiting.

The displayed rate measures completed examples divided by that model's elapsed run time. `valid`
means the provider returned a structurally valid allowed label; it does not mean the prediction was
correct. `elapsed` is the model run's wall-clock time, including client setup, response validation,
record serialization, and cleanup. The narrower provider-call latency is recorded separately in
`decisions.jsonl`. Accuracy and F1 are calculated only in the final report. When output is
redirected or the terminal does not support live rendering, progress bars are disabled
automatically and concise lifecycle and failure logs remain on standard error. Standard output
still ends with only the run directory, making it safe to capture in a shell script. Use PyCharm's
Terminal tool window when you want the full live display.

Resume verifies that `plan.toml`, `task.toml`, and `cohort.json` have not changed. It calls only the
models and example IDs still missing. An explicit error prediction is a completed provider attempt;
it is evaluated as incorrect rather than silently retried during resume.

In PyCharm, use module `thesis_research.cli`, working directory `research/`, and parameters such as:

```text
benchmark --plan experiments/intent_classification/benchmarks/banking77-smoke.toml
```

Each parent benchmark contains:

```text
plan.toml
task.toml
cohort.json
metadata.json
models/<model-id>/metadata.json
models/<model-id>/predictions.jsonl
models/<model-id>/decisions.jsonl
models/<model-id>/summary.json
report/summary.json
report/model_metrics.jsonl
report/per_label_metrics.jsonl
report/pairwise_statistics.jsonl
report/model_outcomes.jsonl
```

`predictions.jsonl` contains one stable example ID and either a label or an explicit error. In LLM
label mode it contains no confidence or probabilities. `decisions.jsonl` preserves the state,
selected label, timing, usage, routing, raw response, and diagnostics for each call. Probability
mode additionally preserves the distribution, normalization details, and confidence. Jev records
its native distribution in both modes. Raw diagnostics can contain adapter internals, but they are
not treated as a model-reported confidence in label mode. No API keys are saved. Model summaries
record progress and aggregate usage; incomplete provider usage becomes `null` instead of an
understated total. Model metadata records its output mode. Parent metadata records the dataset
provenance, exact inputs and checksums, code revision, dirty-tree flag, attempts, and status of
every model.

The report is generated automatically after every model has completed. It contains no graphs. Its
JSON and JSONL files preserve the values needed to create graphs later:

- `model_metrics.jsonl`: accuracy, macro-F1, correct and status counts, elapsed time, token usage,
  retry counts, and reported cost per model.
- `per_label_metrics.jsonl`: precision, recall, F1, and support for every model-label pair.
- `model_outcomes.jsonl`: one correctness/status row per model and example.
- `pairwise_statistics.jsonl`: Jev versus each LLM, including accuracy difference, the paired
  correctness contingency table, valid-label agreement, and the exact two-sided McNemar p-value.
- `summary.json`: the complete metrics above plus every model's confusion matrix.

Accuracy is correct predictions divided by every cohort example, including invalid and failed
predictions. Macro-F1 is the unweighted mean of all 77 intent F1 scores. Confusion-matrix rows are
reference labels; columns are predicted labels plus `<unsuccessful>`. The McNemar result tests the
paired difference in correctness because the models see the same examples. These saved tables can
later be loaded into pandas, R, or plotting software without repeating paid calls.

Vercel currently exposes Jev as the unversioned `typesafe-ai/jev` identifier. Run records preserve
the requested identifier, response-reported model and provider, timestamp, raw response, and code
revision. Confirm resolved routing with a small development benchmark before a thesis-quality run.
Preparing data, mocked provider tests, and checking the software do not constitute measured model
results.

## Code layout and checks

- `src/thesis_research/datasets/`: pinned data preparation and loading.
- `src/thesis_research/config.py`: typed, validated task configuration.
- `src/thesis_research/tasks/`: validated state and question construction.
- `src/thesis_research/clients/`: shared result contracts plus Jev and Vercel AI Gateway clients.
- `src/thesis_research/prediction.py`: bounded prediction and result serialization.
- `src/thesis_research/benchmark/`: plans, shared cohorts, resumable model runs, and reports.
- `src/thesis_research/benchmark/progress.py`: typed benchmark progress notifications.
- `src/thesis_research/terminal.py`: Rich progress rows and human-readable event logs.
- `src/thesis_research/evaluation/`: provider-independent prediction parsing and metrics.
- `src/thesis_research/run_storage.py`: shared durable-record and Git metadata helpers.
- `src/thesis_research/cli.py`: argument parsing and command dispatch only.
- `experiments/intent_classification/tasks/`: versioned prompts, labels, and criteria.
- `experiments/intent_classification/benchmarks/`: dataset split, LLM output, model, cohort, and
  execution plans.
- `tests/`: offline tests with synthetic fixtures, not model experiments.
- `data/` and `outputs/`: ignored dataset downloads and generated records.

```sh
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
```

Study decisions and interpretation belong in [the vault](../vault/Thesis%20Home.md).
