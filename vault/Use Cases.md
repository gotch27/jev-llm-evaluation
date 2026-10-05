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

- [Intent Classification](Use%20Cases/Intent%20Classification.md) — first independent use case, agreed on 2026-09-18. The BANKING77 training-only candidate comparison is complete; the final model set and thesis benchmark remain open.

The framework also includes a [BoolQ Noul task](Research%20and%20Tools.md#boolq) and synthetic Score coverage. Their implementation does not settle the open choice of the next thesis use case.

## TypeSafe evaluation methodology

TypeSafe compares Jev and LLM configurations on four decision workflows using accuracy, response time, and cost. Their structured workflows combine narrow model judgments with rules in code; they also compare standalone prompts. Reference labels come from averaged responses of GPT-6 Astra and Claude Fable 5.1 at high thinking settings. Reported accuracy therefore measures agreement with that model-generated reference, not independently established ground truth.

These are comparisons of models within workflows, not evidence of benefits from combining Jev and an LLM. Our standalone-first approach remains in [Plan — Evaluation approach](Plan.md#evaluation-approach). The BANKING77 task is defined; its final thesis comparison settings remain open.

The published pages show workflow descriptions, charts, and selected examples. Complete reproduction materials, including the full datasets and runnable evaluation code, have not been found in the sources reviewed. This is not a claim that reproduction is impossible.

Source: [TypeSafe workflow evaluations](https://evals.typesafe.ai/), accessed 2026-09-18. Treat the published evaluations as provider evidence, distinct from our own results.

## Experiment records

Record each of our experiments in a separate note from TypeSafe’s published evaluations, with a consistent name or identifier linking its setup, results, and interpretation. Keep TypeSafe’s published methods and results clearly attributed and separate from our own experiments and findings.

Distinguish software tests and connectivity smoke checks from development experiments and frozen thesis evaluations. A working API call or smoke plan verifies infrastructure; it is not a model-quality result. Record actual development runs separately from held-out thesis results, and use “no runs yet” only when no run of that kind has been recorded.

For each experiment, record:

- **Question:** what we want to learn.
- **Setup:** task, dataset and split, models and versions, prompts or configuration, and comparison conditions.
- **Measures:** metrics and how they are calculated.
- **Run:** date, code commit, commands, runner environment, and links to raw outputs.
- **Results:** observations, including failed or inconclusive runs.
- **Interpretation:** what the evidence supports, limitations, and next steps.

## Findings across experiments

No measured thesis results exist yet. Once results exist, link cross-case comparisons and potential thesis figures or tables here. Keep detailed results and interpretation in the individual use-case notes.
