# Research code

Python components for the thesis experiments. The project prepares pinned labeled datasets,
builds one shared TypeSafe `Choice`, `Noul`, or `Score` question per task, calls Jev and LLMs
through Vercel AI Gateway, and runs coordinated benchmarks in which every model receives the same
frozen cohort. Reports are saved as machine-readable tables for later statistical analysis and
visualization. BANKING77 is the Choice task and BoolQ is the first Noul task.

See [Experiment configuration reference](CONFIGURATION.md) for complete task and benchmark TOML
examples, supported fields, allowed values, and validation rules.

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
default. BANKING77 has one Choice question with 77 options, so this distinction does
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
        output_mode="discrete",
    ) as client:
        result = await client.evaluate("When will my card arrive?", questions)
        print(result.answers["intent"])


load_dotenv()
asyncio.run(main())
```

The examples call `load_dotenv()` because they are library-level Python scripts. The command-line
application does this automatically. Do not put credentials in source files, TOML configurations,
or command-line arguments.

LLMs always use native structured output. `output_mode="discrete"` requires only the structured
decision needed by the task and does not invent confidence data. `output_mode="probabilities"`
also requests the question-specific distribution. Invalid probability sums are normalized while
the original values and normalization error remain in raw diagnostics. These are the only two
output modes. Benchmark plans require an explicit mode shared by all configured LLMs. Jev retains
its native answer details.

LLM requests pin the upstream provider with Vercel's `only` option and omit the optional
model-fallback list. They preserve the request and raw response, including routing and cost
metadata returned by the gateway, without authentication headers.

`DecisionResult` provides typed `answers`, per-question `errors`, requested and resolved model and
provider identifiers, aggregate token/cost/retry usage, total wall-clock latency, and individual
call records. If any call has unknown usage or cost, that aggregate is `None` rather than an
incomplete total. Jev requests are billed through AI Gateway. The client preserves the full HTTP
response body because the TypeSafe SDK's typed response drops gateway extensions. Reported cost
comes from `provider_metadata.gateway.cost`; routing and generation IDs remain in the raw record.
Missing cost stays `None`, while an explicitly reported zero remains zero. See
[Vercel's TypeSafe API response](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe).
Jev HTTP attempts are counted separately for each evaluation, including exhausted retries and
connection failures. A retry count of zero means the initial attempt was sufficient; `None` means
the count could not be observed, such as a failure from an externally injected client.

Both clients use the same explicit transient retry policy: two retries with bounded backoff,
retry-after support, and a 30-second retry budget. HTTP calls have a 60-second timeout. Malformed
LLM output is not retried. Model calling is asynchronous; always use the clients as async context
managers or call `aclose()`.

`DecisionClient` and `DatasetAdapter` are Python Protocols. The concrete clients and dataset
adapters explicitly inherit their contracts to make those relationships visible. Each client's
semaphore bounds its active requests across examples; it is separate from model concurrency.

## Prepare and inspect datasets

```sh
uv run thesis-research prepare \
  --config experiments/intent_classification/tasks/banking77.toml
uv run thesis-research inspect \
  --config experiments/intent_classification/tasks/banking77.toml \
  --split train --limit 5

uv run thesis-research prepare \
  --config experiments/boolean_question_answering/tasks/boolq.toml
uv run thesis-research inspect \
  --config experiments/boolean_question_answering/tasks/boolq.toml \
  --split validation --limit 2
```

`prepare` and `inspect` read the dataset ID and revision from the selected task, then dispatch to
its registered adapter. Both commands require `--config`; omission fails before any data work.
`inspect` reads
the split named by `--split` and prints its reference distribution plus JSON examples. Dataset
adapters reject unsupported split names.

### BANKING77 provenance

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

### BoolQ provenance

BoolQ is downloaded directly from [Google’s Hugging Face repository](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) at commit
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The pinned Parquet files contain 9,427 training rows
and 3,270 validation rows with the schema `question: string`, `answer: bool`, and
`passage: string`. Their verified SHA-256 checksums are:

- Train: `4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623`
- Validation: `52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8`

The training references contain 5,874 `true` and 3,553 `false` values. Validation contains
2,033 `true` and 1,237 `false` values. The pinned adapter verifies these distributions as well as
the files themselves.

Files and a provenance manifest are stored under `data/boolq/<revision>/`. Preparation validates
the checksum, Parquet schema, row counts, nonempty source fields, and the presence of both boolean
classes. Stable IDs use the original one-based row position, such as `train:1` and
`validation:1`. Cached files are verified before reuse; modified files stop loading.

The source metadata lists [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The labeled validation split is reserved for held-out evaluation because the published test labels
are unavailable. Training data is used for task development and the two-row smoke cohort.

## Generic task definition

A dataset example's `state` holds its source inputs. A task selects and renames those fields to
build model state and defines exactly one structured question. The BoolQ
state is `{"passage": ..., "question": ...}`; its hidden boolean answer becomes a typed Noul
reference and is never sent to a model. Its frozen instruction is stored in
`experiments/boolean_question_answering/tasks/boolq.toml`.

In `[state.fields]`, the left side names the model input and the right side names an adapter's
source field. For example, `customer_message = "text"` selects the adapter's `text` value and
sends it under `customer_message`. Renaming the model key is supported; update instructions that
mention it. Renaming a source field requires the adapter to supply that name. Before model calls,
the task validates every mapping across the full prepared split and reports missing or empty
fields together with counts and example IDs. Frozen cohort states must still match when resuming.

A dataset revision identifies the exact source-repository commit, independently of the research
code revision. SHA-256 fingerprints verify that cached data and frozen experiment inputs retain
their exact bytes.

The builder validates Choice labels and order against the dataset inventory, Noul tasks against a
boolean reference schema, and Score rubric length against zero-based reference levels. See
[Experiment configuration reference](CONFIGURATION.md) for complete Choice, Noul, and Score TOML
forms, state mappings, criteria, prediction records, and metrics.

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
from thesis_research.datasets import load_prepared_dataset
from thesis_research.tasks import build_structured_task

config_path = Path("experiments/intent_classification/tasks/banking77.toml")
config = load_task_config(config_path)
prepared = load_prepared_dataset(Path("data"), config.dataset.id, config.dataset.revision, "train")
task = build_structured_task(config, prepared.reference_schema)

state = task.build_state(prepared.examples[0].state)
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
check. It references the shared task and calls Jev plus the three candidate LLMs:

```toml
name = "BANKING77 smoke benchmark"
task_config = "../tasks/banking77.toml"

[cohort]
split = "train"
strategy = "random"
size = 1
seed = 20260920

[llm_output]
mode = "discrete"

[execution]
example_concurrency = 1
model_concurrency = 4

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "gpt-5-6-luna"
backend = "llm"
model = "openai/gpt-5.6-luna"
provider = "openai"
reasoning_effort = "low"

# The file also contains Gemini 2.5 Flash Lite and Qwen 3.5 Flash,
# both with reasoning_effort = "none".
```

`banking77-without-criteria-smoke.toml` uses the task without criteria. It deliberately keeps the
same training split, random seed, cohort size, models, and LLM output mode, so both smoke plans
select the same example and differ only in the task definition.

`banking77-timing-pilot.toml` is a separate training-only timing-method check using 10 random
examples. It runs models and examples sequentially and measures every selected example, including
the first request. It makes no additional warm-up calls and is not a thesis experiment.
Ten examples check the timing machinery; they do not support dependable tail-latency estimates.

`banking77-model-selection.toml` compares the same candidates on a deterministic, stratified
training cohort of 770 messages: 10 examples from each of the 77 intents. This is development data
for choosing the final comparison models. It is not the final held-out result, and it does not use
the reserved test split. Run the smoke plan and inspect its recorded costs and errors before
starting this larger plan.

The plan requires exactly one Jev entry and at least one LLM. Model IDs are stable local names used
in file paths and reports. LLM providers are pinned; Jev is pinned to TypeSafe's Vercel provider by
the client. An LLM can set `reasoning_effort` to `none`, `minimal`, `low`, `medium`, `high`, or
`xhigh`; omission leaves the provider default unchanged. The current candidate plans set
Luna to `low` and disable reasoning for Gemini Flash Lite and Qwen Flash. Jev does not accept this
field. `cohort.split` is validated by the selected dataset: BANKING77 supports `train` and `test`,
while BoolQ supports `train` and `validation`. `random` selects any requested number of examples
and is suitable for smoke checks. `stratified_random` selects across Choice labels, boolean
values, or Score levels before adding another example from a class. It therefore requires at
least 77 examples for BANKING77 and at least two for BoolQ. A seed makes either selection
repeatable. Use `strategy = "all"` without `size` or `seed` for the complete configured split.

`llm_output.mode = "discrete"` makes every LLM return only the structured decision needed for the
task. Set it to `"probabilities"` when an experiment needs Choice distributions, BoolQ
`P(true)`, or Score level probabilities. The field is required; the software does not choose a
mode implicitly. The setting does not change Jev's native answer details, and discrete mode does not
produce artificial confidence values.

`example_concurrency` bounds active examples inside each model run. `model_concurrency` bounds how
many models run at the same time. The smoke plan uses `4`, so Jev and all three LLMs are called
together. The model-selection plan uses `1` for both settings so latency measurements are not
affected by local competition between candidates. The chosen execution settings are saved with
the benchmark.

After preparing the data and setting `AI_GATEWAY_API_KEY`, run:

```sh
uv run thesis-research benchmark \
  --plan experiments/intent_classification/benchmarks/banking77-smoke.toml \
  --runner-location local-mac-oslo
```

The BoolQ smoke plan selects one `true` and one `false` training example and uses probability
mode. It is versioned but is not executed by setup or tests:

```sh
uv run thesis-research benchmark \
  --plan experiments/boolean_question_answering/benchmarks/boolq-smoke.toml \
  --runner-location local-mac-oslo
```

If all four smoke rows complete with valid predictions, inspect the generated model summaries and
then run the training-only candidate comparison:

```sh
uv run thesis-research benchmark \
  --plan experiments/intent_classification/benchmarks/banking77-model-selection.toml \
  --runner-location local-mac-oslo
```

`--runner-location` is an optional descriptive label, not an inferred location. Use it for measured
runs. Each attempt records the label, operating system, architecture, Python version, timezone, and
UTC offset without saving a hostname or IP address.

The command prints a unique directory under `outputs/benchmarks/`. Every model receives the exact
same saved cohort. A benchmark cannot accidentally compare 300 Jev examples with 600 LLM examples.
Individual provider failures become explicit unsuccessful predictions and remain in the accuracy
denominator. If a process or setup failure stops a model, completed models and completed example IDs
are preserved. Resume that same benchmark with:

```sh
uv run thesis-research benchmark --resume RUN_DIRECTORY
```

If transient gateway failures such as exhausted 429 or 504 retries were saved as explicit errors,
archive those failed attempts and retry their IDs during resume:

```sh
uv run thesis-research benchmark --resume RUN_DIRECTORY --retry-errors
```

Successful predictions are retained. The removed error prediction and its complete decision
diagnostics are appended to `models/<model-id>/retry_history.jsonl`, so recovery does not erase the
original provider failure. Use this option only after the original process has stopped.

Resume accepts the current task format, `discrete` or `probabilities`, and cohorts containing
`state` and typed `reference` records. Earlier BANKING77 task formats, the retired `label` mode,
and `{id,text,reference_label}` cohorts are unsupported. Historical outputs remain archival data;
the software does not rewrite them or infer missing historical costs. Saved inputs and every
model's settings, paired records, usage, and timing are validated before error records are
archived or unfinished IDs are called. Invalid records leave the saved run unchanged.

In an interactive terminal, the CLI keeps one progress row per model. Each row shows whether the
model is waiting, running, done, or failed; completed examples; structurally valid and failed
predictions; processing rate; elapsed time; and approximate ETA. Prediction failures and lifecycle
events are printed above the live rows so they remain in terminal scrollback. Error messages are
shortened there, while `decisions.jsonl` retains the complete diagnostics.

Progress advances after each flushed example batch. With `example_concurrency = 1`, it advances
one example at a time; with a larger value, it advances by that batch size. Models still progress
independently and do not wait for one another between batches. If `model_concurrency` is smaller
than the number of configured models, queued models remain visibly marked as waiting.

The displayed rate measures completed examples divided by measured prediction time. `valid` means
the provider returned a structurally valid allowed label; it does not mean the prediction was
correct. Model summaries separately retain total attempt time, measured prediction time, decision
latency, and provider-call latency. Accuracy and F1 are calculated only in the final report. When output is
redirected or the terminal does not support live rendering, progress bars are disabled
automatically and concise lifecycle and failure logs remain on standard error. Standard output
still ends with only the run directory, making it safe to capture in a shell script. Use PyCharm's
Terminal tool window when you want the full live display.

Resume verifies that `plan.toml`, `task.toml`, and `cohort.json` have not changed. It calls only the
models and example IDs still missing. By default, an explicit error prediction is a completed
provider attempt and is evaluated as incorrect. `--retry-errors` is the explicit recovery path for
transient infrastructure failures.

In PyCharm, use module `thesis_research.cli`, working directory `research/`, and parameters such as:

```text
benchmark --plan experiments/intent_classification/benchmarks/banking77-smoke.toml --runner-location local-mac-oslo
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
models/<model-id>/retry_history.jsonl  # only when saved errors were retried
models/<model-id>/summary.json
report/summary.json
report/model_metrics.jsonl
report/per_label_metrics.jsonl
report/pairwise_statistics.jsonl
report/model_outcomes.jsonl
```

`predictions.jsonl` contains one stable example ID and either a typed result or an explicit error.
Choice rows use `label`; Noul rows use `value` and optionally `probability`; Score rows use
`level`, expected `score`, and optional probability details. In LLM discrete mode, confidence and
probability fields are omitted. `decisions.jsonl` preserves state, typed answers, timing, usage,
routing, raw responses, and diagnostics. Probability mode additionally preserves distributions
and normalization details. Jev records its native answer details. Raw diagnostics can contain
adapter internals, but the evaluator uses only the documented prediction fields. No API keys are
saved. Model summaries record progress, measured and total elapsed time, latency distributions,
and measured usage. Vercel's reported cost is read from `usage.cost` or gateway metadata when
available, which lets a smoke run inform
the budget for a larger cohort. Incomplete provider usage becomes `null` instead of an understated
total. Model metadata records its output mode and any explicit reasoning effort. Parent metadata
records the dataset
provenance, exact inputs and checksums, code revision, dirty-tree flag, runner environment,
attempts, and status of every model.

Decision latency includes waiting for a client semaphore. Individual provider-call latency starts
after acquiring a slot and includes SDK retries and backoff. Both clients use these definitions.
These durations also include network and SDK work. In each latency summary, `p50` is the median;
`p90` and `p95` describe approximately the 90th and 95th percentiles. Values are linearly
interpolated between measurements and can be durations no individual call took. Small cohorts
provide only a coarse view of the slower tail.

The report is generated automatically after every model has completed. It contains no graphs. Its
JSON and JSONL files preserve the values needed to create graphs later:

- `model_metrics.jsonl`: question type, accuracy, macro-F1, correct and status counts, Noul or
  Score metrics where applicable, measured and total elapsed time, latency statistics, token
  usage, retry counts, and reported cost.
- `per_label_metrics.jsonl`: precision, recall, F1, and support for every model-label pair.
- `model_outcomes.jsonl`: one correctness/status row per model and example, including question
  type, reference and predicted type-specific fields, end-to-end latency, and usage.
- `pairwise_statistics.jsonl`: Jev versus each LLM, including accuracy difference, the paired
  correctness contingency table, valid-label agreement, and the exact two-sided McNemar p-value.
- `summary.json`: the complete metrics above plus every model's confusion matrix.

Accuracy is correct predictions divided by every cohort example, including missing, invalid, and
failed predictions. Macro-F1 is the unweighted mean over the dataset’s discrete classes.
Confusion-matrix rows are reference classes; columns are predicted classes plus `<unsuccessful>`.
Noul adds Brier score, clipped log loss, and probability coverage. Score adds MAE, RMSE,
quadratic weighted kappa, and numeric coverage. The McNemar result tests the paired difference in
correctness because the models see the same examples. These tables can later be loaded into
pandas, R, or plotting software without repeating paid calls.

Latency summaries include successful decisions, all decisions, and individual provider calls.
They report count, mean, population standard deviation, minimum, p50, p90, p95, and maximum using
linear-interpolated percentiles. Failed calls remain visible in the all-decision and provider-call
summaries rather than silently disappearing.

Vercel currently exposes Jev as the unversioned `typesafe-ai/jev` identifier. Run records preserve
the requested identifier, response-reported model and provider, timestamp, raw response, and code
revision. Confirm resolved routing with a small development benchmark before a thesis-quality run.
Preparing data, mocked provider tests, and checking the software do not constitute measured model
results.

## Analyze completed benchmarks

Version analysis notebooks in `notebooks/`, reusable loading and plotting code in
`src/thesis_research/analysis.py`, and generated tables, figures, and executed notebook copies
in `outputs/analysis/<run-id>/`. The saved benchmark is the source of truth. Analysis reads its
frozen reports without making model calls or modifying the original run.

Install the optional analysis dependencies and open Jupyter from `research/`:

```sh
uv sync --locked --group analysis
uv run --group analysis jupyter lab notebooks/banking77_model_selection.ipynb
```

Use the project's `.venv` kernel if opening the notebook in PyCharm or another editor.
The first notebook defaults to the completed 770-example BANKING77 model-selection run.
Change `RUN_ID` in its first code cell to analyze a different completed BANKING77 run, then run
the cells from top to bottom. It includes run provenance, cohort checks, model metrics and
recorded costs, intent-level errors, latency distributions, and Jev–LLM paired correctness.
Keep the versioned notebook cleared of execution outputs; save executed copies under `outputs/`.

`notebooks/banking77_smoke_10.ipynb` defaults to the completed 10-example BANKING77
training smoke run. Run its cells to save tables, figures, and a provenance manifest in
`outputs/analysis/<run-id>/`. This small cohort checks the pipeline and is not sufficient
for model selection or final performance claims.

The same figures and tables can be generated without opening Jupyter:

```sh
uv run --group analysis python -m thesis_research.analysis \
  outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be
```

The exporter writes four CSV tables (`model_comparison`, `per_label_metrics`, `paired_comparison`,
and `confusion_pairs`), five figures as vector PDF and 300-dpi PNG (`classification_quality`,
`latency_ecdf`, `accuracy_latency`, `intent_f1`, and `paired_correctness`), and a manifest recording
source checksums, analysis code checksum, and plotting-library versions. PDFs are suitable for
including in the thesis; PNGs are useful for previews. Output defaults to
`outputs/analysis/<run-id>/`; `--output-dir` overrides it. Re-running replaces the derived exports.
The first workflow supports completed Choice benchmarks; BoolQ and Score analysis can be added
when those experiments need it.

Training results are labeled as development/model selection in the figures. Every figure shows
its dataset, split, and sample size. Unavailable costs remain missing. Accuracy retains failures
in its denominator, while latency plots include all measured decisions. Per-intent scores from
the 770-example cohort have only 10 references each and should be interpreted accordingly.
The oracle accuracy in the paired table uses reference labels and is only an upper bound on
combined performance, not a measured combined system.

These figures are descriptive point estimates. For final thesis comparisons, freeze the task,
models, split, and planned comparisons first, then add an appropriate uncertainty method and
multiple-comparison policy. Saved McNemar p-values are currently exploratory and unadjusted.
Use the same plotting functions on the final held-out run to keep the presentation consistent.

Check analysis code alongside the existing research checks:

```sh
uv run --group analysis python -m pytest
uv run ruff check .
uv run ruff format --check .
```

## Code layout and checks

- `src/thesis_research/datasets/`: shared typed examples, dataset registry, and pinned adapters.
- `src/thesis_research/config.py`: typed, validated task configuration.
- `src/thesis_research/tasks/`: validated state and question construction.
- `src/thesis_research/clients/`: shared result contracts plus Jev and Vercel AI Gateway clients.
- `src/thesis_research/prediction.py`: bounded prediction and result serialization.
- `src/thesis_research/benchmark/`: plans, shared cohorts, resumable model runs, and reports.
- `src/thesis_research/benchmark/model_records.py`: model-record validation and metric summaries.
- `src/thesis_research/benchmark/timing.py`: runner provenance and latency statistics.
- `src/thesis_research/benchmark/progress.py`: typed benchmark progress notifications.
- `src/thesis_research/terminal.py`: Rich progress rows and human-readable event logs.
- `src/thesis_research/evaluation/structured.py`: provider-independent parsing and metrics for all
  three question types.
- `src/thesis_research/run_storage.py`: shared durable-record and Git metadata helpers.
- `src/thesis_research/cli.py`: argument parsing and command dispatch only.
- `src/thesis_research/analysis.py`: completed Choice report validation, tables, and figure exports.
- `notebooks/`: versioned experiment analysis notebooks with cleared outputs.
- `experiments/*/tasks/`: versioned dataset, state mapping, question, labels/rubrics, and criteria.
- `experiments/*/benchmarks/`: dataset split, LLM output, model, cohort, and execution plans.
- `tests/`: offline tests with synthetic fixtures, not model experiments.
- `data/` and `outputs/`: ignored dataset downloads and generated records.

```sh
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
```

Study decisions and interpretation belong in [the vault](../vault/Thesis%20Home.md).
