# Experiment configuration reference

The research project separates **what is asked** from **how a comparison is run**:

- A task TOML defines the dataset revision, state field, structured question, labels, and optional
  criteria.
- A benchmark TOML selects a task, dataset cohort, LLM output format, concurrency, and models.

Both files are copied into every benchmark output directory. Paths in `task_config` are resolved
relative to the benchmark TOML that contains them. API keys never belong in either file; the CLI
loads `AI_GATEWAY_API_KEY` from `research/.env`.

The current runner supports BANKING77 intent classification with one TypeSafe `Choice` question.
The configuration structure keeps the task definition separate from the benchmark, but adding a
different dataset or question type still requires corresponding Python dataset and task code.

## Task TOML

The complete runnable task files are:

- `experiments/intent_classification/tasks/banking77.toml`
- `experiments/intent_classification/tasks/banking77-without-criteria.toml`

A shortened example shows the structure:

```toml
name = "BANKING77 intent classification"
dataset_revision = "57ec275d8078af65b7731c2a98be812d844a6d6b"

[question]
type = "choice"
state_field = "customer_message"
id = "intent"
instructions = "Select the single intent that best matches the message."

[[question.options]]
label = "card_arrival"
criterion.use_when = "A dispatched card has not arrived."
criterion.distinguish_from = "Use card_delivery_estimate for a general timing question."

[[question.options]]
label = "card_linking"
criterion = "The customer wants to link an existing card to the app."

[[question.options]]
label = "exchange_rate"
# Omitting criterion sends null for this label.

# A complete BANKING77 task continues with all 77 official labels in source order.
```

### Top-level fields

| Field | Required | Accepted value | Meaning |
| --- | --- | --- | --- |
| `name` | yes | nonempty string | Human-readable task name saved with the run. |
| `dataset_revision` | yes | 40-character lowercase Git SHA | Exact dataset source revision. |
| `question` | yes for the current benchmark | table | Structured question definition. |

### Question fields

| Field | Required | Accepted value | Meaning |
| --- | --- | --- | --- |
| `type` | yes | `"choice"` | The only task question type currently supported by the TOML loader. |
| `state_field` | yes | nonempty string | State key that receives the source text. BANKING77 requires `customer_message`. |
| `id` | yes | nonempty string | Answer key. BANKING77 requires `intent`. |
| `instructions` | no | string, object, or array | Shared instruction passed to Jev and every LLM. |
| `options` | yes | one or more `[[question.options]]` tables | Allowed Choice labels and their criteria. |

Each option requires a nonempty `label`. Its optional singular `criterion` accepts a string,
object, or array. Omitting it represents JSON `null`, because TOML has no null literal. Nested
criterion content may contain ordinary JSON-compatible strings, numbers, booleans, arrays, and
objects. TypeSafe receives the options as an outer mapping shaped like
`{"criteria": {"label": criterion}}`.

For BANKING77, the option labels must match all 77 official dataset labels exactly, uniquely, and
in source order. There is no automatically added `other` label. The task with criteria uses an
object containing `use_when` and `distinguish_from`; these names are a task-authoring convention,
not fixed configuration fields. The alternative task omits every criterion.

## Benchmark TOML

The runnable smoke and timing-pilot plans are under
`experiments/intent_classification/benchmarks/`. A minimal plan that compares Jev with one LLM
has this form:

```toml
name = "BANKING77 development comparison"
task_config = "../tasks/banking77.toml"

[cohort]
split = "train"
strategy = "random"
size = 10
seed = 20260920

[llm_output]
mode = "label"

[execution]
example_concurrency = 1
model_concurrency = 2

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "example-llm"
backend = "llm"
model = "creator/model-id"
provider = "provider-slug"
```

### Top-level fields

Every field below is required. Unknown fields are rejected.

| Field | Accepted value | Meaning |
| --- | --- | --- |
| `name` | nonempty string | Human-readable benchmark name. |
| `task_config` | nonempty path string | Task TOML, resolved relative to this benchmark file. |
| `cohort` | table | Shared examples sent to every model. |
| `llm_output` | table | Structured output required from every configured LLM. |
| `execution` | table | Concurrency limits. |
| `models` | two or more `[[models]]` tables | Exactly one Jev model and at least one LLM. |

### Cohort options

`split` accepts `"train"` or `"test"`. Use the training split for task development and software
checks; reserve the test split for the final frozen evaluation.

`strategy` accepts:

| Strategy | `size` | `seed` | Behavior |
| --- | --- | --- | --- |
| `"all"` | forbidden | forbidden | Use the entire selected split. |
| `"random"` | required positive integer | required integer | Select a deterministic simple random sample. |
| `"stratified_random"` | required positive integer, at least 77 for BANKING77 | required integer | Select across every label before taking additional examples round-robin. |

A requested size cannot exceed the selected split. The chosen cohort is saved before model calls
and reused unchanged for every model and any resumed attempt.

### LLM output options

`llm_output.mode` accepts:

- `"label"`: require one exact Choice label. No self-reported confidence is requested or saved.
- `"probabilities"`: require a probability for every Choice label. Normalization and the original
  distribution are preserved in diagnostics.

The setting applies to every LLM in the benchmark. It does not change Jev's native answer format;
Jev retains its probability distribution in either mode.

### Execution options

Both values are required positive integers:

- `example_concurrency` bounds simultaneously processed examples within each active model run.
- `model_concurrency` bounds how many configured models run at the same time.

Concurrent models progress independently. For comparative latency measurements, prefer
`model_concurrency = 1` so providers do not compete for local or network resources. A connectivity
smoke check may run all configured models together.

`banking77-timing-pilot.toml` is a training-only development plan with sequential model and example
execution. It measures every selected example, including the first request, and makes no additional
warm-up calls. It is a timing-method check, not a thesis experiment. The connectivity smoke plans
run their models together.

### Model options

Each model table supports:

| Field | Required | Accepted value | Meaning |
| --- | --- | --- | --- |
| `id` | yes | unique lowercase letters, digits, `_`, or `-`, starting with a letter or digit | Stable local identifier used in paths and reports. |
| `backend` | yes | `"jev"` or `"llm"` | Selects the client implementation. |
| `model` | yes | nonempty canonical Vercel model ID | Requested model. Prefer a versioned ID when available. |
| `provider` | only for LLMs | nonempty provider slug | Pins the upstream LLM provider. It must be omitted for Jev. |

Virtual model aliases and provider fallback lists are not supported. The selected LLM model and
provider must support native JSON-schema output.

## Commands

Run commands from `research/`:

```sh
# Prepare the dataset revision referenced by a task.
uv run thesis-research prepare \
  --config experiments/intent_classification/tasks/banking77.toml

# Inspect training examples without accessing the test split.
uv run thesis-research inspect \
  --config experiments/intent_classification/tasks/banking77.toml \
  --limit 5

# Run a new coordinated benchmark.
uv run thesis-research benchmark \
  --plan experiments/intent_classification/benchmarks/banking77-smoke.toml \
  --runner-location local-mac-oslo

# Resume only missing models or example IDs in an existing run.
uv run thesis-research benchmark --resume outputs/benchmarks/RUN_DIRECTORY
```

`--runner-location` is an optional descriptive label. Use it for measured runs, for example
`local-mac-oslo`, `aws-eu-north-1`, or `aws-us-east-1`. It is a command option rather than a TOML
field because the same frozen plan may be executed or resumed from different environments. Each
attempt records its own label, operating system, machine architecture, Python version, timezone,
and UTC offset without saving the hostname or IP address.

The CLI rejects missing, unexpected, or inconsistent configuration fields before paid model work
begins. A new run copies `plan.toml`, `task.toml`, and `cohort.json` into its output directory.
Resume verifies those frozen inputs before continuing.

Measured decision latencies are summarized with count, mean, population standard deviation,
minimum, p50, p90, p95, and maximum. Percentiles use linear interpolation. The report keeps
separate summaries for successful decisions, all decisions, and underlying provider calls, while
`model_outcomes.jsonl` contains each measured example's end-to-end latency and usage.
