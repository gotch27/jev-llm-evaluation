# Intent Classification

[← Use cases](../Use%20Cases.md)

Agreed on 2026-09-18 as our first independent use case. Evaluate Jev and LLMs separately on assigning a user message to one predefined intent. BANKING77 is confirmed for the initial implementation; the final comparison models and thesis-run configuration remain undecided.

## Purpose

Compare classification quality, response time, and execution cost, starting with a simple task structure and later investigating harder distinctions between intents.

## Dataset

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking) contains banking-related customer messages assigned to 77 intent labels. The implementation pins PolyAI's dataset repository to commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, containing 10,003 training examples and 3,080 test examples. It preserves the published splits, original label spelling, and stable row-based identifiers. The files are verified by SHA-256 checksums and distributed under CC BY 4.0.

Use the training split to develop and inspect the task definition. Keep the official test split untouched until the final task, models, and benchmark configuration are frozen. Cite Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), when using the dataset in the thesis.

## Task definition

The implemented task is closed-set classification across all 77 official labels. Each model receives the same state field, `customer_message`, and the same TypeSafe `Choice` question, `intent`. There is no `other` or uncertainty label.

Two versioned task variants currently exist:

- [BANKING77 with criteria](../../research/experiments/intent_classification/tasks/banking77.toml) gives each label a `use_when` definition and a `distinguish_from` boundary. The criteria were derived from training data and contain no example messages.
- [BANKING77 without criteria](../../research/experiments/intent_classification/tasks/banking77-without-criteria.toml) preserves the instruction and label inventory but sends a null criterion for every label.

The singular `criterion` value is configurable as a string, object, array, or omission. The task builder converts the options into TypeSafe's outer `{label: criterion}` criteria mapping and verifies that the 77 labels match the pinned dataset exactly and in source order.

Task TOMLs contain the dataset revision, state field, question, instruction, labels, and optional criteria. Benchmark TOMLs separately select the split and cohort, models, LLM output format, and concurrency. See the [research configuration reference](../../research/CONFIGURATION.md) for the complete fields and examples.

## Comparison approach

Give Jev and every configured LLM the same frozen cohort, message state, instruction, and available intent labels. Compare predictions against the dataset references by stable example ID. A benchmark requires exactly one Jev configuration and at least one LLM configuration, and prevents comparisons over mismatched example sets.

The LLM answer format is an explicit benchmark setting:

- `label` requires one exact intent label and does not request self-reported confidence.
- `probabilities` requires a probability for every intent and preserves the original and normalized distributions.

This setting applies to the configured LLMs. Jev retains its native probability distribution in either mode. The final mode for the thesis experiment has not been chosen.

Cohorts may use the complete split, a seeded random sample, or a seeded stratified sample. The stratified strategy covers every intent before adding further examples. The same saved cohort is reused by all models and by resumed attempts.

Follow [Plan — Evaluation approach](../Plan.md#evaluation-approach).

## Measures

The implemented evaluator records:

- accuracy over every expected example, treating missing, invalid, and failed predictions as incorrect;
- macro-F1 across all 77 labels;
- per-label precision, recall, F1, and support;
- a confusion matrix with a separate unsuccessful column;
- elapsed time, provider-call timing, token usage, reported cost, retries, and failures when available;
- paired Jev-versus-LLM correctness, accuracy difference, valid-label agreement, and the exact two-sided McNemar p-value.

Reports are stored as JSON and tidy JSONL tables suitable for later statistical analysis and visualization. Graph generation is intentionally deferred.

## Implementation status

The practical foundation is implemented in [`research/`](../../research/README.md), through code commit `b4d8d16`:

- pinned dataset preparation and verification;
- validated task and benchmark TOMLs;
- shared asynchronous Jev and LLM decision clients through Vercel AI Gateway;
- coordinated multi-model cohorts with bounded concurrency and resumable execution;
- durable prediction, diagnostic, provenance, and metric records;
- terminal progress rows and persistent failure logs;
- offline tests for dataset handling, configuration, model clients, prediction failures, evaluation, resumption, and reporting.

Implementation readiness does not constitute a completed experiment.

## Later combined experiment

After establishing standalone results, investigate whether sending uncertain Jev predictions to an LLM improves the quality, time, or cost trade-off. The routing method remains undecided.

## Experiments

No thesis experiment has been run yet. The existing smoke plans are development and connectivity configurations, not recorded thesis experiments. Record the eventual frozen configuration and run using [Use Cases — Experiment records](../Use%20Cases.md#experiment-records).

## Results

No measured thesis results yet.

## Interpretation

Add interpretation and limitations after the frozen evaluation has been run.

## Still to decide

- Final Jev and LLM model identifiers and pinned providers.
- Whether the criteria and without-criteria tasks are both thesis experiments or one is only developmental.
- LLM output mode for the final comparison.
- Full test split or a predefined test sample.
- Concurrency and repetition procedure for defensible latency comparison.
- Reliable cost measurement when a provider does not report cost directly.
- Whether confidence calibration or uncertainty routing will be studied later.
