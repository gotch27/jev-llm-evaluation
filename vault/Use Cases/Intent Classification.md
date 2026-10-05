# Intent Classification

[← Use cases](../Use%20Cases.md)

Agreed on 2026-09-18 as our first independent use case. Evaluate Jev and LLMs separately on assigning a user message to one predefined intent. Status updated on 2026-10-06: the BANKING77 training-only candidate comparison is complete, and a later ten-example smoke check is saved. The final thesis models and held-out evaluation configuration remain undecided.

## Purpose

Compare classification quality, response time, and execution cost, starting with a simple task structure and later investigating harder distinctions between intents.

## Dataset

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking) contains banking-related customer messages assigned to 77 intent labels. The implementation pins PolyAI's dataset repository to commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, containing 10,003 training examples and 3,080 test examples. It preserves the published splits, original label spelling, and stable row-based identifiers. The files are verified by SHA-256 checksums and distributed under CC BY 4.0.

Use the training split to develop and inspect the task definition. Reserve official test examples for model evaluation after the final task, models, and benchmark configuration are frozen. Cite Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), when using the dataset in the thesis.

## Task definition

The implemented task is closed-set classification across all 77 official labels. Each model receives the same state field, `customer_message`, and the same TypeSafe `Choice` question, `intent`. There is no `other` or uncertainty label.

Two versioned task variants currently exist:

- [BANKING77 with criteria](../../research/experiments/intent_classification/tasks/banking77.toml) gives each label a `use_when` definition and a `distinguish_from` boundary. The criteria were derived from training data and contain no example messages.
- [BANKING77 without criteria](../../research/experiments/intent_classification/tasks/banking77-without-criteria.toml) preserves the instruction and label inventory but sends a null criterion for every label.

The singular `criterion` value is configurable as a string, object, array, or omission. The task builder converts the options into TypeSafe's outer `{label: criterion}` criteria mapping and verifies that the 77 labels match the pinned dataset exactly and in source order.

Task TOMLs contain `[dataset]`, `[state.fields]`, and `[question]` tables. The mapping `customer_message = "text"` names the model input on the left and selects the adapter's source field on the right. These names are configurable; all source mappings are checked across the full split before model calls. Benchmark TOMLs separately select the split and cohort, models, LLM output mode, per-model reasoning effort, and concurrency. See the [research configuration reference](../../research/CONFIGURATION.md) for fields and examples.

## Comparison approach

Give Jev and every configured LLM the same frozen cohort, message state, instruction, and available intent labels. Compare predictions against the dataset references by stable example ID. A benchmark requires exactly one Jev configuration and at least one LLM configuration, and prevents comparisons over mismatched example sets.

The LLM answer format is an explicit benchmark setting:

- `discrete` requires one exact intent label and does not request self-reported confidence.
- `probabilities` requires a probability for every intent and preserves the original and normalized distributions.

This setting applies to the configured LLMs. Jev retains its native probability distribution in either mode. The old `label` mode alias is unsupported. The final mode for the thesis experiment has not been chosen.

Cohorts may use the complete split, a seeded random sample, or a seeded stratified sample. The stratified strategy covers every intent before adding further examples. The same saved cohort is reused by all models and by resumed attempts.

Follow [Plan — Evaluation approach](../Plan.md#evaluation-approach).

## Measures

The implemented evaluator records:

- accuracy over every expected example, treating missing, invalid, and failed predictions as incorrect;
- macro-F1 across all 77 labels;
- per-label precision, recall, F1, and support;
- a confusion matrix with a separate unsuccessful column;
- per-example end-to-end latency and provider-call latency, including failed decisions;
- measured and total execution time plus mean, standard deviation, minimum, p50, p90, p95, and maximum latency;
- runner-environment provenance, token usage, reported cost, retries, and failures when available;
- paired Jev-versus-LLM correctness, accuracy difference, valid-label agreement, and the exact two-sided McNemar p-value.

Reports are stored as JSON and tidy JSONL tables suitable for statistical analysis and visualization. Optional local analysis code and BANKING77 notebooks exist but remain uncommitted separately from the framework.

## Implementation status

The reusable framework is implemented in [`research/`](../../research/README.md), through code commit `c93f2c4` on 2026-10-06:

- pinned dataset preparation and verification;
- generic dataset and task contracts supporting Choice, Noul, and Score, with BANKING77 and BoolQ adapters;
- validated task and benchmark TOMLs, required CLI task configuration, and full-split state-field checks;
- shared asynchronous Jev and LLM decision clients through Vercel AI Gateway;
- coordinated multi-model cohorts with bounded concurrency and resumable execution;
- durable prediction, diagnostic, provenance, and metric records;
- per-example and provider-call timing with analysis-ready latency distributions;
- a sequential, training-only timing pilot with explicit runner-location metadata and no warm-up calls;
- terminal progress rows and persistent failure logs;
- explicit recovery that validates every model's saved settings and records before archiving failed attempts or calling unfinished IDs;
- full Jev HTTP-response preservation for reported cost and routing, measured HTTP retry counts, and consistent client timing definitions;
- 141 passing framework tests plus lint, formatting, lock consistency, and staged diff checks.

Implementation readiness does not constitute a frozen thesis experiment. Earlier BANKING77 task formats, `label` mode, and legacy cohort rows are unsupported by the current runner; historical outputs are preserved.

## Later combined experiment

After establishing standalone results, investigate whether sending uncertain Jev predictions to an LLM improves the quality, time, or cost trade-off. The routing method remains undecided.

## Experiments

No frozen held-out thesis experiment has been recorded. The saved runs below use training data. The sequential ten-example timing pilot remains a configured development check; the October smoke run used a different plan with concurrent models.

The [candidate model-selection plan](../../research/experiments/intent_classification/benchmarks/banking77-model-selection.toml) uses the criteria task, discrete LLM output, and a seeded stratified training cohort of 770 messages: 10 per intent. It runs models and examples sequentially. The candidates are Jev, GPT-5.6 Luna through OpenAI with low reasoning, Gemini 2.5 Flash Lite through Google with reasoning disabled, and Qwen 3.5 Flash through Alibaba with reasoning disabled. These are development configurations; final thesis models have not been selected.

### 2026-09-20: Interrupted candidate run

The first attempt was interrupted during Jev after 315 saved attempts: 298 valid responses and 17 infrastructure failures, comprising 12 gateway timeouts and 5 provider-capacity errors. No LLM candidate had started. The preserved [run directory](../../research/outputs/benchmarks/20260920T173035363058Z-ec3686cd1df144adaebd77e45f6ddf64/) records commit `3656c99` with a dirty working tree and the `local-mac-oslo` runner. Its frozen task, `label` mode, and cohort use retired formats, so it is now archival and cannot resume through the current runner. The completed comparison below is a separate saved run.

### 2026-10-01: Completed candidate comparison

**Question:** How do the four configured candidates perform on the same development cohort?

**Setup:** The frozen criteria task uses BANKING77 revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, 770 training examples, 10 per intent, and seed `20260920`. LLM output is `discrete`; example and model concurrency are both one. Quality, latency, failures, usage, and paired correctness are recorded using the measures above.

**Run:** [Saved comparison](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/), started at `2026-09-30T22:26:53Z` and completed at `2026-09-30T23:20:32Z`, or 2026-10-01 00:26:53–01:20:32 in Europe/Skopje. The [metadata](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/metadata.json) records `local-mac`, macOS arm64, Python 3.12.11, and commit `c55ac82` with a dirty working tree. Equivalent command from `research/`:

```sh
uv run thesis-research benchmark \
  --plan experiments/intent_classification/benchmarks/banking77-model-selection.toml \
  --runner-location local-mac
```

Use the saved [plan](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/plan.toml), [task](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/task.toml), and [cohort](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/cohort.json) when inspecting this run; active source configurations can change.

**Result:** All four models completed 770 valid predictions, with no missing, malformed, or explicit-error predictions. Quality and latency values are retained in the report; the historical Jev cost and retry counts are unknown. See [Results](#results) and [Interpretation](#interpretation). No final candidate selection is recorded.

### 2026-10-05: Ten-example smoke check

The [saved smoke run](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/) checks successful Choice requests, recording, and report generation with the same four candidates. It selects ten random training examples with seed `20260920`, uses discrete LLM output and the criteria task, and runs one example per model with four models concurrently. It is separate from the sequential timing pilot and the stratified candidate comparison.

The run lasted from 22:55:51 to 22:56:11 in Europe/Skopje on 2026-10-05. Its [metadata](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/metadata.json) records `local-mac-skopje`, macOS arm64, Python 3.12.11, commit `c55ac82`, and a dirty working tree. The command used `experiments/intent_classification/benchmarks/banking77-smoke-10.toml` with `--runner-location local-mac-skopje`; this additional source plan remains uncommitted. The frozen [plan](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/plan.toml), [task](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/task.toml), and [cohort](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/cohort.json) preserve its inputs.

All 40 predictions were structurally valid. Jev's ten calls retained gateway cost and routing metadata, with a reported total cost of **USD 0.002645412** and zero retries. See its [summary](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/models/jev/summary.json) and [decisions](../../research/outputs/benchmarks/20261005T205551771992Z-606b44f66f534bd3bd66a3be190a9fc4/models/jev/decisions.jsonl). LLM serving-provider fields are unreported in this run; the requested provider pins remain in the metadata. Ten examples and concurrent models do not establish comparative performance or dependable tail latency.

Record the eventual frozen configuration and thesis run using [Use Cases — Experiment records](../Use%20Cases.md#experiment-records).

## Results

No frozen held-out thesis results are recorded. The following development values come from the [770-example report](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/report/summary.json) and saved model summaries. Accuracy uses all 770 examples; mean decision latency includes all recorded decisions. Cost is the reported total for that run, not an extrapolated price.

| Model | Correct / 770 | Accuracy | Macro-F1 | Mean decision latency (s) | Reported cost (USD) |
| --- | --- | --- | --- | --- | --- |
| Jev | 641 | 83.25% | 0.8301 | 0.4275 | Unknown |
| GPT-5.6 Luna | 671 | 87.14% | 0.8660 | 1.8903 | 0.12754221 |
| Gemini 2.5 Flash Lite | 630 | 81.82% | 0.8161 | 0.7035 | 0.00801447 |
| Qwen 3.5 Flash | 625 | 81.17% | 0.8095 | 1.1464 | 0.42887910 |

The [per-example outcomes](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/report/model_outcomes.jsonl), [per-label metrics](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/report/per_label_metrics.jsonl), and [paired statistics](../../research/outputs/benchmarks/20260930T222653327742Z-d103c737bc91438482be7af965fb77be/report/pairwise_statistics.jsonl) remain the detailed record. Frozen input and report checksums and available model-artifact checksums were verified on 2026-10-06 without rerunning models or rewriting outputs.

## Interpretation

The completed comparison provides development evidence for discussing candidate selection. It uses training data also used to develop criteria, has only ten reference examples per intent, and records one pass on one host. Its point estimates and exploratory paired statistics do not establish held-out performance or a final model choice. Historical missing Jev cost prevents a complete four-model cost comparison for that run; the later smoke cost cannot replace the missing total.

Both completed runs predate framework commit `c93f2c4` and record a dirty working tree, so their recorded commit alone does not identify all executing source changes. Their frozen inputs and outputs remain inspectable. The October smoke supports successful Choice calling and Jev cost recording in that working tree; live BoolQ evaluation remains outstanding, and Score has synthetic coverage only.

## Still to decide

- Which development candidates advance to the final Jev-versus-LLM comparison.
- Whether the criteria and without-criteria tasks are both thesis experiments or one is only developmental.
- LLM output mode for the final comparison.
- Full test split or a predefined test sample.
- Execution host and repetition count for a defensible latency comparison; model and example execution are sequential for comparative timing.
- How to handle a final run if a provider omits direct cost data.
- Whether confidence calibration or uncertainty routing will be studied later.
