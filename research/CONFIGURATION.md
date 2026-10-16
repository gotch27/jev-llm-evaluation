# Experiment configuration reference

The project separates the supervised task from the benchmark execution:

- A **task TOML** pins a dataset source revision and optional prepared version, maps dataset fields into model state, and
  defines one TypeSafe `Choice`, `Noul`, or `Score` question.
- A **benchmark TOML** selects a dataset split and cohort, the LLM answer mode, concurrency, and
  the System One models and LLMs that receive that same cohort.

Both files and the selected cohort are copied into every benchmark run. API keys belong in
environment variables or ignored `research/.env`, never in TOML. Hosted calls use
`AI_GATEWAY_API_KEY`; self-hosted authentication is optional and selects a separate variable.

## Task TOML

Every new task has four parts:

```toml
name = "Human-readable task name"

[dataset]
id = "registered-dataset-id"
revision = "immutable-source-revision"
# For a shared release, also set version to its full 64-character SHA-256 ID.

[state.fields]
model_state_key = "dataset_source_field"

[question]
type = "choice" # or "noul" or "score"
id = "stable_answer_key"
instructions = "Instructions shared by every configured model."
```

`dataset.revision` identifies the upstream source commit. `dataset.version`, when present,
selects exactly `../datasets/releases/<id>/<version>/`; it is a full lowercase SHA-256 digest
covering inputs, targets, split membership, schema, and preparation provenance. Every split is
verified before loading. No latest-version pointer is followed. Yelp and SNIPS require a version.
Without it, BANKING77 and BoolQ retain their source-pinned adapter behavior. Supported datasets are:

| Dataset | ID | Supported splits | Reference type |
| --- | --- | --- | --- |
| BANKING77 | `banking77` | `train`, `test` | Choice label |
| BoolQ | `boolq` | `train`, `validation` | Boolean Noul value |
| Yelp | `yelp` | `train`, `validation`, `calibration`, `test` | Score level 0–4 = stars 1–5 |
| SNIPS | `snips` | `train`, `heldout` | Seven Choice intents |

Yelp/SNIPS bilingual releases expose `text_en` and `text_mk`. Select one explicitly, such as
`review = "text_mk"`. Labels, ratings, and translation review flags are kept outside the inputs.
The checked-in Yelp/SNIPS tasks pin the published snapshot used during this migration. Publish a
different prepared export as a new version and update a new task file; do not edit a frozen run.

### Image state

Image tasks must pin `dataset.version` to a published image release. Any dataset ID with a valid
shared release can be loaded without adding a source downloader. Map top-level image inputs with
`[state.images]`; the publisher stores their bytes and provenance as described in
[Image releases](../datasets/README.md#image-releases). For an image-only Noul task, use this
fragment after the pinned `[dataset]` table:

```toml
[state.images]
photo = "photo"

[question]
type = "noul"
id = "visible_damage"
instructions = "Does the product in `photo` show visible damage?"
```

`[state.fields]` can accompany images for text such as a question or product description. At
least one mapping table is required; each supplied table must be nonempty, and output names
must be distinct. Mapping an image reference through `state.fields` is an error. The current
image support covers state inputs, with the existing Choice/Noul/Score definitions and metrics.
Box and images inside question definitions are unsupported.

All participants receive the same image bytes. The initial shared contract accepts still PNG
and JPEG images: 256–1568 pixels per side, 78,400–2,458,624 total pixels, at most 16 MiB per file,
and a conservative 32 MiB encoded-request budget. Preparation must produce compliant images;
benchmark execution performs no resizing or other image transformations.

Enabled image routes are Jev 1.14.0 / `typesafe-ai` and GPT-6 Luna Decisions / `openai` using the
TypeSafe API, plus the existing GPT-5.6 Luna / `openai`, Gemini 2.5 Flash Lite / `google`, and
Qwen 3.5 Flash / `alibaba` chat routes. The exact model/provider combination is required.
Jev and GPT-6 Luna Decisions had live Noul connectivity checks; the LLM chat image paths have
offline transport tests and catalog evidence. The unversioned Jev route, local services, unknown
routes, Laya, Microsoft, and ScaleDown are blocked for image tasks. Liquid d1, d1 Ultra, and
Perplexity advertise vision, but their TypeSafe asset requests failed in live checks and remain
blocked for this encoding. Their text configurations continue to work.

Every model's capability is checked before creating a benchmark or constructing clients. All
incompatibilities are reported together, and the plan is rejected as a whole. Every prepared
image input is checked before model execution. New image runs record the capability snapshot;
unfinished resumes require that snapshot and the frozen image checksums to still match.

Validate a selected plan and its data without credentials, model calls, or output creation:

```sh
uv run thesis-research validate --plan path/to/image-benchmark.toml
```

The regular `benchmark` command performs the same image checks automatically.

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
baseline_model_id = "jev"
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
backend = "system_one"
model = "typesafe-ai/jev"
provider = "typesafe-ai"

[[models]]
id = "example-llm"
backend = "llm"
model = "creator/model-id"
provider = "provider-slug"
reasoning_effort = "low"
```

Unknown fields are rejected. `task_config` is resolved relative to the benchmark file.
A plan requires one or more models and a `baseline_model_id` naming one of them. There is no
required backend category. Single-model runs use that sole model as baseline and have no pairwise
comparisons. `[llm_output]` is optional for System One-only plans and required if any LLM is included.

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

This setting controls LLM serialization. System One always records `native_probabilities` and
keeps its native structured answer details independently of `[llm_output]`. For
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
| `backend` | yes | `system_one` or `llm`. |
| `model` | yes | Canonical Vercel model ID. |
| `provider` | yes | Pinned upstream provider slug. |
| `reasoning_effort` | optional, LLM only | `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |

System One models must omit `reasoning_effort`. The hosted routes below were checked against
[Gateway's TypeSafe model list](https://ai-gateway.vercel.sh/typesafe/v1/models) and public
per-model provider endpoints on 2026-10-15:

| Local `id` | Gateway `model` | Pinned `provider` |
| --- | --- | --- |
| `jev` | `typesafe-ai/jev` | `typesafe-ai` |
| `laya` | `convaiinnovations/laya` | `boundless` |
| `liquid-d1` | `liquid/d1` | `liquid` |
| `gpt-6-luna-decisions` | `openai/gpt-6-luna-decisions` | `openai` |
| `microsoft-decision-1` | `microsoft/microsoft-decision-1` | `azure` |
| `scaledown-classify-1` | `scaledown/scaledown-classify-1` | `scaledown` |
| `pplx-decider-v1-1-27b` | `perplexity/pplx-decider-v1.1-27b` | `perplexity` |
| `liquid-d1-ultra` | `liquid/d1-ultra` | `liquid` |
| `jev-1-14-0` | `typesafe-ai/jev-1.14.0` | `typesafe-ai` |

Use Laya's standard ID. All models pin their provider with `gateway.only`. Local IDs use hyphens
in place of version dots; the exact Gateway IDs retain their dots. The unversioned Jev alias may
resolve to the same release as the pinned Jev entry. Freeze a versioned route for thesis runs.
LLMs require native structured output support; virtual aliases and provider fallback lists are
rejected. Historical frozen `jev` plans require their original code revision for resume.

The three `*-system-one-smoke.toml` plans include all nine routes and retain Jev as baseline,
with their existing training cohorts, question definitions, and concurrency. All additions use
the existing TypeSafe endpoint, text state, Choice/Noul/Score types, and Vercel credential:

```toml
[[models]]
id = "microsoft-decision-1"
backend = "system_one"
model = "microsoft/microsoft-decision-1"
provider = "azure"
```

The five added routes have offline fixture coverage but have not had live inference validation.
Run the training smoke plans before drawing conclusions about account access or provider limits.

System One responses can refuse individual questions. These become `ModelRefusal` errors with
the raw refusal in their diagnostics; successful answers and billed usage remain recorded.
An HTTP-successful refusal is not retried.

## Self-hosted System One

Omitting `transport` uses the existing Vercel transport. A self-hosted System One model instead
specifies an API root; the SDK appends `/v1/systemone`:

```toml
[[models]]
id = "laya-local"
backend = "system_one"
transport = "self_hosted"
model = "laya-base"
provider = "local"
base_url = "http://127.0.0.1:8000"
# api_key_env = "LOCAL_DECISION_API_KEY" # optional Bearer authentication
# timeout_seconds = 120.0              # optional; positive, finite seconds
```

`provider` is a provenance label for self-hosted models. No `gateway.only` routing options are
sent. The endpoint must return TypeSafe-compatible Choice/Noul/Score answers, model identity,
and usage. Health endpoints and checkpoint metadata are optional. Missing reported cost remains
unknown, and reported zero remains zero. Raw response extensions are retained for auditing.

Without `api_key_env`, requests contain no Authorization header, even when hosted credentials
exist in the environment. When set, the named variable must contain the server's key;
`AI_GATEWAY_API_KEY` cannot be selected. URLs cannot contain credentials, query parameters, or
fragments. `base_url`, `api_key_env`, and `timeout_seconds` require `self_hosted`; this transport
requires the `system_one` backend. Reasoning effort remains exclusive to hosted LLMs.

For a mixed run, add a hosted model to the same plan and select either configured ID as baseline:

```toml
[[models]]
id = "jev"
backend = "system_one"
model = "typesafe-ai/jev"
provider = "typesafe-ai"
```

With the local stanza above and `baseline_model_id = "laya-local"`, this compares Jev against
local Laya on identical frozen states and questions. The Jev calls use the Vercel key and incur
provider charges. Add `[llm_output]` only when including LLMs.

The independent [model-serving project](../model-serving/README.md) supplies a Laya server and
separate commands. Exported checkpoints can be served directly. Change the model alias when
weights change. Resume requires restarting the original checkpoint, alias, token budgets, and
URL; the benchmark freezes settings but does not enforce a checkpoint-manifest protocol.
Existing Vercel model serialization is unchanged so historical completed runs and resume settings
remain compatible.

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
pandas, R, or later visualization without repeating model calls. New completed runs also print
terminal statistics and generate `report/index.html` automatically. Add `--open-report` to the
benchmark command to open that dashboard. Choice, Noul and Score dashboards use the saved
metrics; no extra model calls or notebook dependencies are needed. A visualization failure does
not change completed benchmark status.

Use `uv run thesis-research report RUN_DIRECTORY --open-report` to export a dashboard and show
statistics for an existing completed run. The default export is `outputs/reports/<run-id>/`;
`--output-dir` can select another directory outside the source run. Historical records stay
unchanged. Rendering provenance and any rendering error are saved in `visualization.json`.

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
