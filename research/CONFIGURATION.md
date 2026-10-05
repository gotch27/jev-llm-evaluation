# Experiment configuration reference

The project separates the supervised task from the benchmark execution:

- A **task TOML** pins a registered dataset revision, maps dataset fields into model state, and
  defines one TypeSafe `Choice`, `Noul`, or `Score` question.
- A **benchmark TOML** selects a dataset split and cohort, the LLM answer mode, concurrency, and
  the Jev and LLM models that receive that same cohort.

Both files and the selected cohort are copied into every benchmark run. API keys belong only in
`research/.env`; the CLI reads `AI_GATEWAY_API_KEY` from there.

## Task TOML

Every new task has four parts:

```toml
name = "Human-readable task name"

[dataset]
id = "registered-dataset-id"
revision = "immutable-source-revision"

[state.fields]
model_state_key = "dataset_source_field"

[question]
type = "choice" # or "noul" or "score"
id = "stable_answer_key"
instructions = "Instructions shared by Jev and every LLM."
```

`dataset.id` selects a Python adapter in `src/thesis_research/datasets/registry.py`.
`dataset.revision` is passed to that adapter and preserved in the run. The registered datasets are:

| Dataset | ID | Supported splits | Reference type |
| --- | --- | --- | --- |
| BANKING77 | `banking77` | `train`, `test` | Choice label |
| BoolQ | `boolq` | `train`, `validation` | Boolean Noul value |

`[state.fields]` maps the keys sent to the model to source fields supplied by the dataset adapter.
For example, `customer_message = "text"` sends `{"customer_message": row["text"]}`. BoolQ maps
both `passage` and `question`; the hidden `answer` field is never included in the state.
Changing a left-hand key changes the model input name; update instructions that mention it.
Changing a right-hand source name requires the adapter to provide that field. Before calling any
model, the benchmark checks every mapping across the full split, including rows outside a selected
cohort. Missing or empty source fields are reported together with row counts and sample IDs.
The revision pins the dataset's source-repository commit, separately from the research code commit.

Instructions and criteria accept a TOML string, object, or array. Nested content may use any
JSON-compatible values that TOML can express. Omitting an optional criterion maps it to JSON
`null` when the TypeSafe question is built.

### Choice

```toml
[question]
type = "choice"
id = "intent"
instructions = "Select exactly one intent."

[[question.options]]
label = "card_arrival"
criterion = "Use when an ordered card has not arrived."

[[question.options]]
label = "card_delivery_estimate"
criterion.use_when = "The customer asks how long card delivery normally takes."
criterion.distinguish_from = "Use card_arrival after an expected delivery is overdue."

[[question.options]]
label = "exchange_rate" # no criterion; TypeSafe receives null for this option
```

Each option requires a unique, nonempty `label`. Options must reproduce the dataset adapter’s
complete label inventory in its exact order. `criterion` is the content for one option; the task
builder forms TypeSafe’s outer `criteria` mapping as `{label: criterion}`. Names such as
`use_when` and `distinguish_from` are authoring conventions inside an object, not reserved fields.

The complete task files are:

- `experiments/intent_classification/tasks/banking77.toml`
- `experiments/intent_classification/tasks/banking77-without-criteria.toml`

### Noul

`Noul` represents a yes/no proposition. BoolQ asks whether the answer to the supplied question is
yes, based only on the supplied passage:

```toml
[question]
type = "noul"
id = "answer"
instructions = "Based only on the passage, determine whether the answer is yes."

[question.criteria]
true = "The passage supports a yes answer."
false = "The passage supports a no answer."
```

The whole `[question.criteria]` table is optional. If present, `true` and `false` are individually
optional and accept the same generic criterion content as Choice. BoolQ’s full task is
`experiments/boolean_question_answering/tasks/boolq.toml`.

### Score

`Score` uses an ordered zero-based rubric. Array order defines level `0`, level `1`, and so on:

```toml
[question]
type = "score"
id = "quality"
instructions = "Rate the response quality against the rubric."

[[question.levels]]
criterion = "Incorrect or unusable."

[[question.levels]]
criterion = { description = "Partly correct", defects = ["minor omission"] }

[[question.levels]]
criterion = "Fully correct."
```

At least two levels are required. Every level requires a `criterion` value. The number of
configured levels must equal the dataset adapter’s Score reference schema, and every reference
level must be within that range. Score is implemented and covered by synthetic tests; there is no
versioned real Score dataset task yet.

## Benchmark TOML

```toml
name = "BoolQ Noul smoke benchmark"
task_config = "../tasks/boolq.toml"

[cohort]
split = "train"
strategy = "stratified_random"
size = 2
seed = 20260920

[llm_output]
mode = "probabilities"

[execution]
example_concurrency = 1
model_concurrency = 4

[[models]]
id = "jev"
backend = "jev"
model = "typesafe-ai/jev"

[[models]]
id = "example-llm"
backend = "llm"
model = "creator/model-id"
provider = "provider-slug"
reasoning_effort = "low"
```

Every field is required, and unknown fields are rejected. `task_config` is resolved relative to
the benchmark file. A plan requires exactly one Jev model and at least one LLM.

### Cohort

`split` is any nonempty name, then the selected dataset adapter validates it. BANKING77 supports
`train` and `test`; BoolQ supports `train` and `validation`.

| Strategy | `size` | `seed` | Behavior |
| --- | --- | --- | --- |
| `all` | omitted | omitted | Uses the complete selected split. |
| `random` | positive integer | integer | Selects a deterministic simple random sample. |
| `stratified_random` | positive integer | integer | Selects across each reference class before taking additional rows round-robin. |

Strata are Choice labels, boolean values, or Score levels. A stratified size must be at least the
number of classes. The saved cohort contains stable IDs, exact model states, and typed references;
it is reused unchanged by all models and by resumed attempts.

### LLM output

`llm_output.mode` accepts:

- `discrete`: save only the structured decision needed for exact evaluation. Choice saves a
  label, Noul saves a boolean, and Score saves its derived level and expected score. It does not
  invent confidence or probability values.
- `probabilities`: also save the probability information returned through the adapter. Choice
  saves its distribution and confidence, Noul saves `P(true)`, and Score saves its level
  distribution and confidence.

This setting controls LLM serialization. Jev keeps its native structured answer details. For
Noul, the saved boolean is true when `P(true) >= 0.5`. For Score, the saved discrete level is the
largest-probability level, with the lowest level chosen on a tie; the expected score supplied by
TypeSafe is saved separately.

### Execution and models

`example_concurrency` bounds active examples within each model. `model_concurrency` bounds active
models. Models progress independently; lowering model concurrency can make timing comparisons
easier to interpret.

Each model table accepts:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique lowercase local identifier used in paths and reports. |
| `backend` | yes | `jev` or `llm`. |
| `model` | yes | Canonical Vercel model ID. |
| `provider` | LLM only | Pinned upstream provider slug. |
| `reasoning_effort` | optional, LLM only | `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |

Jev must omit `provider` and `reasoning_effort`. LLMs require a provider and native structured
output support. Virtual aliases and provider fallback lists are rejected.

## Prediction records and metrics

Each `predictions.jsonl` row contains a stable ID and exactly one typed result or error:

```json
{"id":"train:1","label":"card_arrival"}
{"id":"train:2","label":"card_arrival","confidence":0.8,"probabilities":{"card_arrival":0.8,"card_delivery_estimate":0.2}}
{"id":"train:1","value":true}
{"id":"train:2","value":false,"probability":0.18}
{"id":"train:1","level":2,"score":1.7}
{"id":"train:2","level":1,"score":1.4,"confidence":0.6,"probabilities":{"0":0.1,"1":0.6,"2":0.3}}
{"id":"train:3","error":"ProviderError: request timed out"}
```

Duplicate and unexpected IDs are rejected. Missing, malformed, and explicit-error predictions
remain distinct failure counts, and all count as incorrect. Every question type reports exact
accuracy, macro-F1, per-class precision/recall/F1/support, a confusion matrix, and paired McNemar
comparisons. Noul additionally reports Brier score, clipped log loss, and probability coverage.
Score additionally reports MAE, RMSE, quadratic weighted kappa, and numeric coverage.

The JSONL report tables include `question_type` and retain the existing Choice label fields while
adding boolean, probability, level, and score fields where applicable. They remain suitable for
pandas, R, or later visualization without repeating model calls.

## Commands

Run these from `research/`. `prepare` and `inspect` require an explicit `--config`:

```sh
# BANKING77
uv run thesis-research prepare \
  --config experiments/intent_classification/tasks/banking77.toml
uv run thesis-research inspect \
  --config experiments/intent_classification/tasks/banking77.toml \
  --split train --limit 5

# BoolQ: download both pinned labeled splits, then inspect either split
uv run thesis-research prepare \
  --config experiments/boolean_question_answering/tasks/boolq.toml
uv run thesis-research inspect \
  --config experiments/boolean_question_answering/tasks/boolq.toml \
  --split train --limit 2
uv run thesis-research inspect \
  --config experiments/boolean_question_answering/tasks/boolq.toml \
  --split validation --limit 2

# Defined only; do not run until model access and cost are intended
uv run thesis-research benchmark \
  --plan experiments/boolean_question_answering/benchmarks/boolq-smoke.toml \
  --runner-location local-mac-oslo

# Resume a saved run, with an optional explicit retry of archived error rows
uv run thesis-research benchmark --resume outputs/benchmarks/RUN_DIRECTORY
uv run thesis-research benchmark \
  --resume outputs/benchmarks/RUN_DIRECTORY --retry-errors
```

`--runner-location` is descriptive provenance. New runs copy the plan, task, and cohort before
calling models. Resume validates those frozen inputs and every model's saved settings and paired
records before archiving errors or calling unfinished IDs. Invalid records leave the saved run
unchanged. The current format requires `[dataset]`, `[state.fields]`, and `[question]`, an output mode of
`discrete` or `probabilities`, and cohorts with model `state` and typed `reference` records.
Earlier BANKING77 task formats, `mode = "label"`, and `{id,text,reference_label}` cohorts are
unsupported. Existing historical runs are preserved as archival records.
