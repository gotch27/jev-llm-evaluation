# SNIPS → Macedonian, through Vercel AI Gateway

Downloads the original English SNIPS intent dataset and translates every utterance into
Macedonian. This prepares a classification dataset; it does not fine-tune any model.
Runs locally on your Mac using **Python 3.10+ and its standard library only**. Translation
is performed remotely by Vercel AI Gateway. No GPU, Node.js, pip packages or hosting needed.

## Run

```sh
cd /Users/gorazdfilipovski/Documents/graduate-thesis/datasets/snips
python3 pipeline.py configure
python3 pipeline.py translate
```

`configure` asks for your **Vercel AI Gateway key**, with hidden input, and writes a private
`.env` file. Run it only once; it refuses to overwrite an existing file. Alternatively, set
`AI_GATEWAY_API_KEY` in your environment. Environment values take precedence over `.env`.
Keys can be created in the [Vercel AI Gateway dashboard](https://vercel.com/docs/ai-gateway/authentication-and-byok/api-keys).
Your Gateway account needs access/credits for the selected model.

The default is **`anthropic/claude-haiku-5.5`**, with 20 utterances per request, 3 concurrent
requests and reasoning disabled. The command translates **the whole dataset**, automatically
downloads it if missing, and exports the results. Rerunning the same command resumes from
the SQLite cache; successfully saved translations are not requested again.

For a small pilot before the full run:

```sh
python3 pipeline.py translate --per-intent 10
```

This translates 70 unique utterances, sampled deterministically across all 7 intents.
Pilot results are reused by the subsequent full run with the same model and prompt.
A successful pilot exits with code 0, but its export is deliberately marked **partial**.

Compare all three inexpensive candidates on the same 35 utterances:

```sh
python3 pipeline.py compare
```

Read `comparison/review.csv`: English source, intent and each model's Macedonian translation
side by side. `comparison/usage.json` reports cumulative token usage and estimated costs.
Check meaning, negation, natural Macedonian, correct numbers and entity preservation. Choose
the cheapest model that meets your quality standard; there is **no automatic quality winner**.
This pilot is a qualitative screen, not a rigorous statistical translation benchmark.

To use Luna instead:

```sh
python3 pipeline.py translate --model openai/gpt-6-luna
```

To compare just Haiku and Luna, or increase the pilot:

```sh
python3 pipeline.py compare --models anthropic/claude-haiku-5.5 openai/gpt-6-luna --per-intent 10
```

Each model/prompt/settings combination has its own cache and export directory. Switching
models does not mix their translations. Comparison translations are reused by a full run
with the same model. Dataset labels are never sent to the translation model.

## Model selection and cost

Official Vercel catalog checked **2026-10-08** (USD, base standard rates for short requests):

| Gateway ID | Input / 1M tokens | Output / 1M tokens | Role |
|---|---:|---:|---|
| `anthropic/claude-haiku-5.5` | $0.10 | $0.50 | Default; matches your requested model |
| `openai/gpt-6-luna` | $0.10 | $0.50 | Equal base price; compare Macedonian quality |
| `google/gemini-2.5-flash-lite` | $0.10 | $0.40 | Cheapest output among these three |

**Recommendation:** start with the Haiku pilot. No live quality comparison has been performed
in this setup because no Gateway key was available. Price and provider descriptions alone
do not prove which model translates Macedonian best. Gemini has a slightly lower output
rate, but the saving may be small for this dataset; inspect quality before choosing it.

```sh
python3 pipeline.py models
```

This reads the current public Gateway catalog and saves `model-catalog.json`, including
prices and supported parameters. Translation refreshes the catalog before running, checks
the exact requested model exists, uses JSON Schema outputs and disables reasoning when
supported. It sends temperature=0 only to models advertising support; Haiku 5.5 does not.
There is no automatic switch to another model if a request fails.

Token usage depends on Macedonian tokenization, JSON overhead and retries. For illustration,
500,000 input + 1,000,000 output tokens cost $0.55 at Haiku/Luna base rates. That is **not a
measured total-dataset estimate**. The pipeline reports a base list-price estimate from
actual returned usage, including rejected responses. Unknown-usage requests, provider price
differences, cache discounts/writes, regional surcharges and your account's credits may alter
actual billing. Use the Gateway dashboard for the invoice and budget settings.

Primary references:

- [Haiku 5.5 on Vercel](https://vercel.com/ai-gateway/models/claude-haiku-5.5)
- [GPT-6 Luna on Vercel](https://vercel.com/ai-gateway/models/gpt-6-luna)
- [Gemini 2.5 Flash Lite on Vercel](https://vercel.com/ai-gateway/models/gemini-2.5-flash-lite)
- [Public model catalog](https://ai-gateway.vercel.sh/v1/models)
- [Official OpenAI Luna documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Gateway Chat Completions](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions)
- [Gateway structured outputs](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/structured-outputs)
- [Gateway reasoning settings](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/reasoning)

## Dataset provenance and thesis considerations

Source: [sonos/nlu-benchmark](https://github.com/sonos/nlu-benchmark), pinned to
`b86ac7f1577868c42158d0dec77db50956046696`, under **CC0-1.0**.
The raw JSON files and license are kept unchanged in `raw/`. Every file has its source URL
and SHA-256 checksum in `source/manifest.json`.

| Intent | Full train | Held out |
|---|---:|---:|
| AddToPlaylist | 1,942 | 100 |
| BookRestaurant | 1,973 | 100 |
| GetWeather | 2,000 | 100 |
| PlayMusic | 2,000 | 100 |
| RateBook | 1,956 | 100 |
| SearchCreativeWork | 1,954 | 100 |
| SearchScreeningEvent | 1,959 | 100 |
| **Total** | **13,784** | **700** |

This is the original **full-train + validate** release, **14,484 rows**. We call the original
`validate_*` files `heldout`, rather than implying they are a new calibration set or a new
official test split. Common processed versions instead carve 700 examples out of full train,
giving 13,084 train / 700 validation / 700 test. That carving has not been done here.

Original texts are reconstructed by concatenating the annotated text segments in order,
without altering whitespace/case. The original PlayMusic full-train file has legacy CESU-8
surrogate pairs for some emoji. Normalization converts these pairs to the corresponding
Unicode characters, preserving the emoji rather than replacing it or decoding it as Latin-1;
the manifest records source encoding and the original bytes remain unchanged.
Exact duplicate texts share a SHA-256 `duplicate_group`:
**14,318 unique texts** across the original rows. The audit records all exact duplicate
groups, cross-split overlaps and any conflicting labels. Translation is paid for once per
unique text and copied back to every original row, preserving their order, labels and IDs.

For the later training/validation/calibration/test experiment, group duplicates together
before splitting, and resolve original cross-split overlaps explicitly. After translation,
also check whether different English messages collapsed into identical Macedonian messages.
No experimental splits or labels are silently changed by this pipeline.

Translation instructions preserve foreign proper names, titles and service names in their
original spelling; ordinary language becomes Macedonian Cyrillic. This is a deliberate
controlled translation policy, not proof that these requests resemble naturally collected
Macedonian speech. Keep that distinction in the thesis, and review a sample as a Macedonian
speaker before fine-tuning.

**Scope:** translated utterance-level **intent classification**. Original entity/slot
annotations remain in the raw source. Their character offsets/segments are not aligned to
the Macedonian text; this output is not a translated slot-filling benchmark.

Dataset citation: Coucke et al. (2018), [Snips Voice Platform: an embedded Spoken Language
Understanding system for private-by-design voice interfaces](https://arxiv.org/abs/1805.10190).

## Files and outputs

Downloaded raw data, normalized source exports, catalog snapshots, caches, logs, and
translation outputs stay local and are ignored by Git. The pinned source revision and
preparation commands above reproduce the source files after cloning the repository.

```text
raw/                        unchanged original JSON, README and license
source/all.jsonl            all original English rows + IDs, labels, provenance
source/train.jsonl          original full training set
source/heldout.jsonl        original validate set
source/labels.json          fixed mapping of integer label → intent
source/manifest.json        revision, counts and checksums
source/audit.json           exact duplicates and overlap report
model-catalog.json          Gateway capabilities/prices, checked timestamp
cache/translations.sqlite3  durable translations and per-attempt usage/errors
logs/pipeline.log           append-only operational log
translated/<model>/<profile>/
  all.jsonl                 complete bilingual dataset, only when all rows are done
  train.jsonl               complete translated train
  heldout.jsonl             complete translated heldout
  *.partial.jsonl           available rows while the dataset is incomplete
  review.jsonl              rows flagged by simple review heuristics
  summary.json              completion state, prompt/settings, provenance, usage
comparison/review.csv       side-by-side pilot, when compare is run
comparison/usage.json       cumulative usage by model/profile
```

Each translated record retains source fields and adds `text_mk`, `language: "mk"`,
`requested_model`, `returned_model`, `response_id`, `translated_at`, `translation_profile`
and `review_flags`. Use **`text_mk`** as the classifier input; the integer `label` and string
`intent` remain unchanged. There is no gold intent embedded in the translated input.

`summary.json.complete` must be true before treating an export as the complete dataset.
Partial files contain only completed rows; missing examples are never filled with English
or empty translations. Output files are written atomically. Cache updates commit entire
valid batches. A valid pilot can exit successfully with a partial dataset export.

Review flags check for missing Cyrillic, some unusual Cyrillic letters, changed digit runs,
unchanged text and suspicious length. These are **heuristics**, not Macedonian language
identification or semantic quality validation. Latin proper names are allowed. Digit flags
can be legitimate, e.g. spelling "2" as a word. Read flagged rows rather than assuming errors.

## Reliability and troubleshooting

- Default: up to **5 attempts per batch**, exponential backoff with jitter, honoring
  `Retry-After` on rate limits; retry HTTP 408/409/425/429/5xx and network failures.
- Retry malformed/truncated responses, missing/duplicate IDs and empty translations;
  nothing from an invalid batch is committed as a completed translation.
- Other HTTP errors, including 400/401/402/403/404, stop the job for you to fix configuration,
  model access or credits. Logs omit response bodies and authentication headers.
- Ctrl-C/SIGTERM stop scheduling work, let in-flight requests finish, and export progress.
  You may wait up to the request timeout, default 90 seconds. Abrupt shutdown can require
  paying again for a request that returned but was not committed locally; no remote API can
  guarantee exactly-once billing across a lost response. SQLite retains committed batches.
- A process lock prevents concurrent jobs from modifying the same cache/export. Retryable
  exhausted batches are left pending; the rerun retries those messages.
- Model aliases may evolve at the provider. Requested/returned model IDs and timestamps are
  stored for provenance; this does not guarantee immutable backend model weights.

Useful commands:

```sh
python3 pipeline.py status
python3 pipeline.py export
python3 pipeline.py download
python3 pipeline.py translate --workers 1 --batch-size 10 --retries 7
python3 -m unittest -v
```

`status` and `export` use the saved catalog offline when present. `download` verifies raw
checksums and rebuilds normalized source files. `--retries` is the total attempt count,
including the first request. Smaller batches can help if a model repeatedly truncates JSON;
you can also increase `--max-tokens`. Exit codes: 0 success, 1 failure/incomplete explicit
export, 130 interruption. Unit/integration tests use a local fake HTTP server, with no real
translation calls or charges.
