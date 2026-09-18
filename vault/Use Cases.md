# Use Cases

[← Thesis home](Thesis%20Home.md)

This note indexes use cases and provides the shared experiment-record format. Notes under `Use Cases/` distinguish TypeSafe’s published evaluations from our independent work. Published-evaluation notes contain task descriptions, methods, sources, and limitations. Our experiments, results, and interpretation belong in separate notes, even when the tasks are similar. Code lives in `research/` in the same repository. Datasets and raw outputs stay in ignored data/output directories or their actual storage locations.

## Use cases

Direction clarified on 2026-09-18: discuss TypeSafe's published evaluations as background before presenting our independent experiments.

### TypeSafe's published evaluations

- [Security Incidents](Use%20Cases/Security%20Incidents.md)
- [Agent Trace Observability](Use%20Cases/Agent%20Trace%20Observability.md)
- [Invoice Processing](Use%20Cases/Invoice%20Processing.md)
- [Customer Service](Use%20Cases/Customer%20Service.md)

### Our use cases

- [Intent Classification](Use%20Cases/Intent%20Classification.md) — first independent use case, agreed on 2026-09-18. BANKING77 and the detailed experiment design remain provisional.

## TypeSafe evaluation methodology

TypeSafe compares Jev and LLM configurations on four decision workflows using accuracy, response time, and cost. Their structured workflows combine narrow model judgments with rules in code; they also compare standalone prompts. Reference labels come from averaged responses of GPT-6 Astra and Claude Fable 5.1 at high thinking settings. Reported accuracy therefore measures agreement with that model-generated reference, not independently established ground truth.

These are comparisons of models within workflows, not evidence of benefits from combining Jev and an LLM. Our standalone-first approach remains in [Plan — Evaluation approach](Plan.md#evaluation-approach). The precise task boundaries for our comparisons are still to be defined.

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
