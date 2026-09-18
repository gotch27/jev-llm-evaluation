# Intent Classification

[← Use cases](../Use%20Cases.md)

Agreed on 2026-09-18 as our first independent use case. Evaluate Jev and LLMs separately on assigning a user message to one predefined intent. The dataset and detailed experiment design remain provisional.

## Purpose

Compare classification quality, response time, and execution cost, starting with a simple task structure and later investigating harder distinctions between intents.

## Dataset candidate

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking) contains banking-related messages with intent labels and published training and test splits. Source accessed 2026-09-18. Inspect its categories and training examples before confirming it for this use case.

## Proposed experiment progression

- Start with a predefined subset of clearly distinguishable intents.
- Increase difficulty by introducing closely related intents and expanding the available categories.
- Define the subsets before evaluation and report them separately from the full dataset.

This progression is a proposal, not yet a finalized experiment design.

## Comparison approach

Give each model the same messages and available intent categories. Compare predictions against dataset labels. Use training data for prompt development and keep test data for evaluation. Follow [Plan — Evaluation approach](../Plan.md#evaluation-approach).

## Later combined experiment

After establishing standalone results, investigate whether sending uncertain Jev predictions to an LLM improves the quality, time, or cost trade-off. The routing method remains undecided.

## Experiments

No runs yet. Record configurations and runs using [Use Cases — Experiment records](../Use%20Cases.md#experiment-records).

## Results

No measured results yet.

## Interpretation

Add interpretation and limitations once results exist.

## Still to decide

Dataset confirmation, intent subsets, comparison models, prompts, metrics, and sample sizes.
