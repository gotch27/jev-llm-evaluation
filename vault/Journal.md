# Journal

[← Thesis home](Thesis%20Home.md)

Use authorized dated entries to retain decisions and their reasons, mentor feedback, and meaningful changes. Keep current open questions in [Plan — Open questions](Plan.md#open-questions). Link related research or experiments instead of duplicating them.

## Dated entries

Add entries newest first when Gorazd authorizes saving them. Include the date, the agreed decision or change, its reason, and any approved follow-up actions.

### 2026-10-06 — Reusable structured benchmark framework complete

Committed the framework as `c93f2c4` after code review and readiness checks. Shared dataset, task, cohort, prediction, and evaluation contracts now support Choice, Noul, and Score, with pinned BANKING77 and BoolQ adapters. Removed retired task parsing, the `label` output alias, compatibility wrappers, and unused helpers; task configuration is explicit, and state mappings are validated across the full split before calls. Resume now checks every model's saved settings and records before rewriting error records or making requests. These changes make the experiment inputs and failure behavior explicit while preserving historical outputs.

Corrected Jev recording to retain gateway cost and routing extensions from the full HTTP response and count actual HTTP retries, including exhausted attempts. Unknown cost remains unknown and is not evidence of free inference. Client timing now consistently separates semaphore waiting from provider-call duration. The committed framework snapshot passed 141 offline tests, Ruff lint and formatting, lock consistency, and staged diff checks; cached dataset splits and task mappings were also verified. Score has full synthetic benchmark and resume coverage, while no live BoolQ run is stored yet. Software completion is separate from thesis evaluation.

Reviewed saved development artifacts and updated their status in [Intent Classification](Use%20Cases/Intent%20Classification.md). The September 20 interrupted run remains archival because its task, output mode, and cohort use retired formats. A separate 770-example comparison is complete, and the later smoke run records Jev cost. Final model selection and the held-out study configuration remain open. Local analysis code, notebooks, and the ten-example smoke plan remain uncommitted separately from the framework.

### 2026-10-05 — BANKING77 smoke check records Jev cost

The [ten-example training smoke run](Use%20Cases/Intent%20Classification.md#2026-10-05-ten-example-smoke-check) completed with valid predictions from all four candidates. It used discrete LLM output and four concurrent models on `local-mac-skopje`. Its saved Jev responses include nonzero gateway cost, serving-provider metadata, and observed retry counts. This provides successful Choice and cost-recording evidence; its small concurrent cohort does not establish comparative latency or model selection. The record predates `c93f2c4` and identifies `c55ac82` with a dirty working tree. Artifacts were reviewed on 2026-10-06.

### 2026-10-01 — Training-only candidate comparison completed

The [saved 770-example comparison](Use%20Cases/Intent%20Classification.md#2026-10-01-completed-candidate-comparison) completed for Jev, GPT-5.6 Luna, Gemini 2.5 Flash Lite, and Qwen 3.5 Flash, with 770 valid predictions per model. It began on September 30 in UTC, or October 1 in Europe/Skopje, and used the criteria task, discrete LLM output, 10 training examples per intent, and sequential execution. The saved record identifies `local-mac` and commit `c55ac82` with a dirty working tree. Historical Jev cost and retries are unavailable in this run. It is development evidence for reviewing candidates; no final model selection or held-out thesis evaluation is recorded. Frozen inputs, reports, and available model-artifact checksums were verified on 2026-10-06.

### 2026-09-20 — Candidate selection and recoverable provider failures

Added the training-only BANKING77 candidate plan in code commit `ba6cf69`. It uses the criteria task, exact-label LLM output, a seeded stratified cohort of 770 training messages with 10 examples per intent, and sequential model and example execution. The development candidates are Jev, GPT-5.6 Luna through OpenAI at low reasoning, Gemini 2.5 Flash Lite through Google with reasoning disabled, and Qwen 3.5 Flash through Alibaba with reasoning disabled. These configurations support model selection and are not yet the frozen thesis comparison.

The first development attempt was interrupted during Jev after 315 saved calls: 298 valid responses, 12 gateway timeouts, and 5 provider-capacity errors. No LLM candidate had started. The [preserved run](../research/outputs/benchmarks/20260920T173035363058Z-ec3686cd1df144adaebd77e45f6ddf64/) records `local-mac-oslo` and commit `3656c99` with uncommitted configuration changes. Treat this as an operational failure record, not model-quality evidence. Commit `193316d` added opt-in `--retry-errors` recovery: it archives full failed attempts, retains successful predictions, and retries failed plus unfinished IDs. Next: resume after provider capacity settles, complete candidate selection, and only then freeze the held-out thesis evaluation.

### 2026-09-20 — Reproducible latency records without warm-up calls

Extended the coordinated BANKING77 benchmark in code commit `12d2f0c` to preserve per-example end-to-end latency, provider-call latency, measured and total execution time, p50/p90/p95 and other distribution statistics, and runner-environment provenance. Added a ten-example training-only timing pilot that runs models and examples sequentially to reduce local contention. Decided to measure every selected example, including the first request, without additional warm-up calls: warm-ups were unnecessary for quality evaluation and could add cost, provider-specific caching effects, and methodological complexity. No thesis experiment has been run. Next: choose the final execution host and repetition count together with the remaining benchmark decisions.

### 2026-09-20 — Coordinated standalone benchmark foundation

Implemented the first reusable comparison pipeline for [intent classification](Use%20Cases/Intent%20Classification.md). A benchmark freezes one BANKING77 cohort and sends it to exactly one Jev configuration and one or more LLM configurations, preventing comparisons over different samples. It supports bounded model and example concurrency, interrupted-run resumption, explicit failed predictions, terminal progress, and paired machine-readable reports for later statistics and visualization. LLM output is configurable as an exact label or complete probability distribution; Jev retains its native distribution. No thesis experiment has been run. Next: choose the final models, task variant, output mode, test cohort, latency procedure, and cost source before creating the first evaluation plan.

### 2026-09-19 — BANKING77 task and structured model clients

Confirmed BANKING77 for the first independent use case and pinned the source revision, official training/test splits, 77-label inventory, and file checksums. Implemented a shared TypeSafe `state + questions` interface for Jev and LLMs through Vercel AI Gateway, together with task variants containing per-label criteria or null criteria. Criteria were derived from training data without copying example messages; the official test split remains reserved. Kept task TOMLs separate from benchmark TOMLs so model sets, cohorts, output modes, and execution settings can change without duplicating the task definition.

### 2026-09-18 — One repository and Python research project

Replaced the earlier expectation of a separate future code repository with one Git repository containing the Markdown vault and sibling `research/` Python project. Chose Python 3.12 with uv, pytest, and Ruff, with downloaded data and generated outputs ignored. This keeps documentation and code revisions aligned while preserving separate instructions for the vault and practical work.

### 2026-09-18 — First independent use case: intent classification

Chose [intent classification](Use%20Cases/Intent%20Classification.md) to compare Jev and LLMs on a single defined decision before adding workflow complexity. Approved recording BANKING77 as a dataset candidate and an easy-to-hard progression as a provisional design. Standalone results come first; combined use will be investigated afterward. Next: inspect BANKING77's categories and training examples before confirming the dataset.

### 2026-09-18 — TypeSafe evaluations as background

Added TypeSafe's four published workflow evaluation cases, linked from [Use Cases — TypeSafe's published evaluations](Use%20Cases.md#typesafes-published-evaluations), as background to discuss before our independent experiments. Organized them as separate notes under `Use Cases/` and made [Use Cases](Use%20Cases.md) the overview. Clarified that these notes document the provider's work, so removed placeholders for our experiments and results. Complete reproduction materials have not been found in the sources reviewed; this does not establish that reproduction is impossible. Our own experiments will use separate notes, even for similar tasks. Follow-up: choose independently evaluable tasks and datasets rather than assume direct reproduction is available.

### 2026-09-17 — Version control for the vault

Agreed to start local Git history before detailed planning so the development of ideas and decisions is recoverable. Keep the vault separate from the future research/code repository, using simple commits after meaningful changes. A private remote can be added later; none is planned yet. See [Research and Tools — Setup and reproducibility](Research%20and%20Tools.md#setup-and-reproducibility).
