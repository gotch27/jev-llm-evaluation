# Use Cases

[[Thesis Home|← Thesis home]]

This note indexes use cases and provides the shared experiment-record format. Notes under `Use Cases/` distinguish TypeSafe’s published evaluations from our independent work. Published-evaluation notes contain task descriptions, methods, sources, and limitations. Our experiments, results, and interpretation belong in separate notes, even when the tasks are similar. Code, datasets, and raw outputs stay in the research repository or their actual storage locations.

## Use cases

Direction clarified on 2026-09-18: discuss TypeSafe's published evaluations as background before presenting our independent experiments. The following notes document TypeSafe's work; our own tasks and configurations remain to be chosen.

- [[Use Cases/Security Incidents|Security Incidents]]
- [[Use Cases/Agent Trace Observability|Agent Trace Observability]]
- [[Use Cases/Invoice Processing|Invoice Processing]]
- [[Use Cases/Customer Service|Customer Service]]

## TypeSafe evaluation methodology

TypeSafe compares Jev and LLM configurations on four decision workflows using accuracy, response time, and cost. Their structured workflows combine narrow model judgments with rules in code; they also compare standalone prompts. Reference labels come from averaged responses of GPT-6 Astra and Claude Fable 5.1 at high thinking settings. Reported accuracy therefore measures agreement with that model-generated reference, not independently established ground truth.

These are comparisons of models within workflows, not evidence of benefits from combining Jev and an LLM. Our standalone-first approach remains in [[Plan#Evaluation approach]]. The precise task boundaries for our comparisons are still to be defined.

The published pages show workflow descriptions, charts, and selected examples. Complete reproduction materials, including the full datasets and runnable evaluation code, have not been found in the sources reviewed. This is not a claim that reproduction is impossible.

Source: [TypeSafe workflow evaluations](https://evals.typesafe.ai/), accessed 2026-09-18. Treat the published evaluations as provider evidence, distinct from our own results.

## Experiment records

Record each of our experiments in a separate note from TypeSafe’s published evaluations, with a consistent name or identifier linking its setup, results, and interpretation. Keep TypeSafe’s published methods and results clearly attributed and separate from our own experiments and findings.

For each experiment, record:

- **Question:** what we want to learn.
- **Setup:** task, dataset and split, models and versions, prompts or configuration, and comparison conditions.
- **Measures:** metrics and how they are calculated.
- **Run:** date, code commit, commands, and links to raw outputs.
- **Results:** observations, including failed or inconclusive runs.
- **Interpretation:** what the evidence supports, limitations, and next steps.

## Findings across experiments

Once results exist, link cross-case comparisons and potential thesis figures or tables here. Keep detailed results and interpretation in the individual use-case notes.
