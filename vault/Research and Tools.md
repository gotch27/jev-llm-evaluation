# Research and Tools

[← Thesis home](Thesis%20Home.md)

## Sources and reading notes

Starting references carried over from the proposal discussion:

- [Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe AI documentation](https://docs.typesafe.ai/introduction)

For each source, record its citation or URL, date accessed, useful points, and relevance to the thesis. Distinguish the author's claims from our own findings.

### Jev and System One: naming and conceptual background

Recorded and sources accessed on 2026-09-18.

- **Jev — Jevons paradox.** In the FAQ of its [announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (Diogo Almeida, 2026-09-15), TypeSafe states that Jev is named after William Stanley Jevons. The [Jevons paradox](https://en.wikipedia.org/wiki/Jevons_paradox) describes how greater efficiency in using a resource can lower its effective cost and stimulate enough demand to increase total consumption. TypeSafe draws an analogy with machine intelligence: it expects lower costs to enable more uses and greater demand. This is the company's motivation and expectation, not a measured finding of this thesis.
- **System One — Kahneman's System 1 / System 2 distinction.** The same announcement identifies Daniel Kahneman's *Thinking, Fast and Slow* (2011) as the inspiration for the model-class name. System 1 refers to fast, automatic, intuitive thinking; System 2 to slower, effortful, deliberate reasoning. Kahneman also explains the distinction in his [Nobel Prize autobiographical account](https://www.nobelprize.org/prizes/economic-sciences/2002/kahneman/biographical/). TypeSafe uses the name to frame its focus on fast, structured decisions.

**Relevance to the thesis:** These connections provide background for explaining Jev's name and intended role. The efficiency-and-demand connection offers context for the interest in execution cost, while the cognitive analogy helps explain TypeSafe's terminology. The analogy alone does not establish that Jev reproduces human cognition, that LLMs correspond to System 2, or that either approach is more reliable; performance conclusions require experimental evidence.

## Models, frameworks, and datasets

### BANKING77

The first implemented use case uses [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking), pinned to source revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. The source contains 10,003 training examples, 3,080 test examples, and 77 intent labels. The loader preserves the original split, label spelling, and row order, assigns stable IDs such as `train:1`, and verifies the three source files using recorded SHA-256 checksums.

The dataset is licensed under [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE). Dataset paper: Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807). The source was first accessed on 2026-09-18; the pinned revision and license were recorded during implementation on 2026-09-19.

### BoolQ

The framework's first implemented Noul task uses [Google's BoolQ repository on Hugging Face](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5), pinned to revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`. Its labeled Parquet splits contain 9,427 training rows and 3,270 validation rows. Each row supplies a passage and question as model inputs, with a hidden boolean reference answer. The adapter verifies pinned checksums, schema, row counts, and boolean distributions, and assigns stable split-and-row IDs.

The [dataset card](https://huggingface.co/datasets/google/boolq/blob/main/README.md), consulted on 2026-10-06, lists CC BY-SA 3.0 and the paper Clark et al. (2019), *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*. Cached-file provenance and checksums are documented in the [research setup](../research/README.md#boolq-provenance).

The [task](../research/experiments/boolean_question_answering/tasks/boolq.toml) asks for a yes/no decision based only on the supplied passage. Its [smoke plan](../research/experiments/boolean_question_answering/benchmarks/boolq-smoke.toml) selects one true and one false training example and exercises LLM probability output. No live BoolQ run is stored in the current benchmark outputs. The labeled validation split is reserved for held-out evaluation. This task's implementation does not settle which use case follows intent classification in the thesis.

### Model access and structured decisions

Jev and LLMs share an asynchronous `state + questions` interface using TypeSafe `Choice`, `Score`, and `Noul` question types. Jev is called through Vercel's TypeSafe-compatible endpoint with the TypeSafe SDK and receives all questions in one request. LLM questions use TypeSafe's System One Adapter and Vercel AI Gateway; each question is an isolated structured-output request, with independent failures and bounded concurrency.

Vercel model and provider identifiers are pinned in benchmark TOMLs. Models may also pin the gateway's provider-agnostic reasoning effort. The training-only candidate plan uses Jev, GPT-5.6 Luna through OpenAI at low reasoning, Gemini 2.5 Flash Lite through Google with reasoning disabled, and Qwen 3.5 Flash through Alibaba with reasoning disabled. These are development candidates; the final thesis comparison models have not been selected. Credentials are loaded from the ignored `research/.env` file through `AI_GATEWAY_API_KEY` and are never stored in task files, benchmark plans, outputs, or the vault.

The two supported LLM output modes are `discrete` and `probabilities`. Discrete output provides a Choice label, Noul boolean, or Score level; probability output preserves the corresponding distribution or Noul `P(true)`. Jev retains its native answer details. The old `label` mode alias has been removed.

The implementation records requested routing separately from response-reported routing, plus timing, usage, raw diagnostics, and errors. Per-request cost is read from `usage.cost` or gateway metadata. Jev's full HTTP response must be retained because the typed TypeSafe SDK response drops gateway extensions, including `provider_metadata.gateway.cost`. Vercel documents the general gateway cost field in its [cost guide](https://vercel.com/academy/ai-gateway/ai-gateway-pricing), consulted on 2026-10-06. The [2026-10-05 BANKING77 smoke run](Use%20Cases/Intent%20Classification.md#2026-10-05-ten-example-smoke-check) contains nonzero Jev cost in those extensions. Missing cost remains unknown; an explicitly reported zero remains zero. Missing historical Jev cost cannot be reconstructed from the saved typed dumps.

Jev HTTP attempts are counted per evaluation, including exhausted retries and connection failures. The retry policy is distinct from the observed retry count: zero means no additional attempt, while unknown counts remain `None`. Client semaphores bound active provider requests across examples; benchmark settings separately bound active examples and models.

### Python tools

The implementation uses Python 3.12 and uv. Versions pinned in the framework commit's `uv.lock` include:

- TypeSafe SDK 0.7.0;
- System One Adapter 0.2.0;
- OpenAI Python library 3.16.2 for the adapter's OpenAI-compatible provider interface;
- httpx2 2.13.0 for Jev HTTP transport and per-evaluation attempt tracking;
- PyArrow 25.0.1 for pinned BoolQ Parquet files;
- python-dotenv 1.2.3 for local environment loading;
- Rich 14.3.4 for terminal progress;
- pytest 9.1.1 and Ruff 0.16.8 for development checks.

These are implementation dependencies, not the final set of comparison models.

## Setup and reproducibility

The project uses one local Git repository to version the Markdown documentation and research code together. Create commits or push changes only with Gorazd's explicit authorization; permission to edit notes does not include either action. macOS metadata and `.idea/` editor settings are ignored. Read the Markdown notes in `vault/` using the IDE; keep Python research code in the sibling `research/` directory.

See [research setup](../research/README.md) for environment details and commands. Record code revisions and output locations for each run. Keep credentials out of these notes.

### Configuration files

Task and benchmark settings are kept separate so one task definition can be compared across different cohorts and model sets:

- A **task TOML** contains `[dataset]` with its ID and source revision, `[state.fields]` mappings, and one `[question]` of type Choice, Noul, or Score with its instructions and options, boolean criteria, or ordered rubric.
- A **benchmark TOML** references one task and selects the dataset split, cohort strategy, LLM output mode, concurrency, and Jev/LLM configurations, including optional LLM reasoning effort.

In `[state.fields]`, a left-hand key names the model input and its right-hand value names the dataset source field. Output names can change independently; changing a source name requires the adapter to supply it. The benchmark validates all mappings across the full selected split before model calls. A dataset revision identifies its source snapshot independently of the research code revision.

The detailed [experiment configuration reference](../research/CONFIGURATION.md) documents supported fields, validation, TOML examples, and commands. Active task and benchmark files live under `research/experiments/intent_classification/` and `research/experiments/boolean_question_answering/`. Both `prepare` and `inspect` require an explicit `--config`; no default task is selected.

### Python project implementation

The repository was reorganized on 2026-09-18 after choosing one repository for notes and code and Python for implementation. The reusable framework is complete through commit `c93f2c4` on 2026-10-06. Registered adapters prepare BANKING77 and BoolQ through a shared dataset contract; one generic task builder and runner support Choice, Noul, and Score. Concrete dataset adapters and decision clients explicitly inherit their Protocol contracts.

The committed framework snapshot passed 141 offline tests, Ruff lint and formatting, dependency-lock consistency, and staged diff checks. Tests cover malformed inputs and answers, failures, shared cohorts, interruption, safe resume, timing, metrics, and reports; Score also passes full benchmark and resume tests in both output modes. All supported BANKING77 and BoolQ splits and current task mappings were validated against cached files. Score currently has synthetic coverage and no registered real dataset. These checks establish software readiness, not measured model quality.

Each benchmark freezes the task, plan, and cohort before model calls. It preserves predictions and explicit failures by stable example ID, model diagnostics, dataset provenance, file checksums, code revision, dirty-tree status, commands, timing, usage, and reports. Choice reports exact classification metrics; Noul adds Brier score, log loss, and probability coverage; Score adds MAE, RMSE, quadratic weighted kappa, and numeric coverage.

Timing includes every selected example, including the first request, without separate warm-up calls. Decision latency includes client semaphore waiting; provider-call latency starts after acquiring a slot and includes SDK retries and backoff. Reports retain measured and total elapsed time, these latency distributions, and runner-location and operating-system context. The p50, p90, and p95 values are linearly interpolated percentiles; small cohorts provide only a coarse picture of tail latency.

Normal resume calls only missing IDs. Before any new request or error archival, it validates frozen inputs, dataset mappings, and every model's saved settings and paired records. For a transient infrastructure failure, `--retry-errors` preserves complete failed attempts in `retry_history.jsonl`, then retries failed and unfinished IDs while retaining successes. Earlier BANKING77 task formats, `mode = "label"`, and cohorts containing `{id,text,reference_label}` are unsupported; their outputs remain archival records. Outputs stay under ignored `research/outputs/`; experiment notes link exact run directories, including failures.

The training-only `banking77-timing-pilot.toml` runs ten examples with one active model and one active example at a time to check timing machinery. The separate `banking77-model-selection.toml` selects 10 training examples per intent and also uses sequential execution. A completed candidate run and a later concurrent ten-example smoke check are recorded in [Intent Classification — Experiments](Use%20Cases/Intent%20Classification.md#experiments). Their outputs are development records, not held-out thesis results.

An optional Choice analysis module and BANKING77 notebooks also exist locally, together with a ten-example smoke plan. Their code, notebook, dependency, and documentation changes remain uncommitted and are separate from framework commit `c93f2c4`.

No frozen held-out thesis evaluation has been recorded. The comparison models, final output mode, test cohort, execution host, repetition count, and handling of missing provider cost data remain to be fixed before that evaluation.
