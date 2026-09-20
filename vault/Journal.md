# Journal

[← Thesis home](Thesis%20Home.md)

Use authorized dated entries to retain decisions and their reasons, mentor feedback, and meaningful changes. Keep current open questions in [Plan — Open questions](Plan.md#open-questions). Link related research or experiments instead of duplicating them.

## Dated entries

Add entries newest first when Gorazd authorizes saving them. Include the date, the agreed decision or change, its reason, and any approved follow-up actions.

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
