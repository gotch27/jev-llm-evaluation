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

### Model access and structured decisions

Jev and LLMs share an asynchronous `state + questions` interface using TypeSafe `Choice`, `Score`, and `Noul` question types. Jev is called through Vercel's TypeSafe-compatible endpoint with the TypeSafe SDK and receives all questions in one request. LLM questions use TypeSafe's System One Adapter and Vercel AI Gateway; each question is an isolated structured-output request, with independent failures and bounded concurrency.

Vercel model and provider identifiers are pinned in benchmark TOMLs. The final thesis comparison models have not been selected. Credentials are loaded from the ignored `research/.env` file through `AI_GATEWAY_API_KEY` and are never stored in task files, benchmark plans, outputs, or the vault.

The LLM output mode is configured per benchmark as either an exact label or a complete probability distribution. Jev retains its native probability distribution in either case. The implementation records requested and resolved routing, timing, usage, raw diagnostics, and errors where the provider makes them available. Missing usage or cost is stored as unknown rather than silently treated as zero.

### Python tools

The initial implementation uses Python 3.12 and uv. Versions currently pinned in `uv.lock` include:

- TypeSafe SDK 0.7.0;
- System One Adapter 0.2.0;
- OpenAI Python library 3.16.2 for the adapter's OpenAI-compatible provider interface;
- python-dotenv 1.2.3 for local environment loading;
- Rich 14.3.4 for terminal progress;
- pytest 9.1.1 and Ruff 0.16.8 for development checks.

These are implementation dependencies, not the final set of comparison models.

## Setup and reproducibility

The project uses one local Git repository to version the Markdown documentation and research code together. Create commits or push changes only with Gorazd's explicit authorization; permission to edit notes does not include either action. macOS metadata and `.idea/` editor settings are ignored. Read the Markdown notes in `vault/` using the IDE; keep Python research code in the sibling `research/` directory.

See [research setup](../research/README.md) for environment details and commands. Record code revisions and output locations for each run. Keep credentials out of these notes.

### Configuration files

Task and benchmark settings are kept separate so one task definition can be compared across different cohorts and model sets:

- A **task TOML** contains the dataset revision, state field, question, instruction, labels, and optional criteria.
- A **benchmark TOML** references one task and selects the dataset split, cohort strategy, LLM output mode, concurrency, and Jev/LLM configurations.

The detailed [experiment configuration reference](../research/CONFIGURATION.md) documents every currently supported field, allowed value, validation rule, TOML example, and command. The active files live under `research/experiments/intent_classification/tasks/` and `research/experiments/intent_classification/benchmarks/`.

### Python project implementation

The repository was reorganized on 2026-09-18 after choosing one repository for notes and code and Python for implementation. The practical foundation through code commit `12d2f0c` now includes dataset preparation, task construction, Jev and LLM clients, coordinated and resumable benchmark execution, evaluation, machine-readable reporting, terminal progress, and reproducible latency recording. The test suite uses synthetic fixtures and mocked providers; passing tests are software validation, not model-quality evidence.

Each benchmark freezes the task, plan, and cohort before model calls. It preserves predictions and explicit failures by stable example ID, model diagnostics, dataset provenance, file checksums, code revision, dirty-tree status, commands, timing, usage, and generated reports. Timing records include every selected example, including the first request, without separate warm-up calls. Reports retain measured and total execution time, end-to-end decision latency, provider-call latency, p50/p90/p95 and other distribution statistics, and the user-supplied runner-location label with operating-system context. Outputs remain under ignored `research/outputs/`; thesis experiment notes should link the exact run directory after an authorized run is completed.

The training-only `banking77-timing-pilot.toml` runs ten examples with one active model and one active example at a time. It exists to validate the timing and reporting method before the final benchmark; its outputs are development records, not thesis results.

No thesis experiment has been run yet. The comparison models, final output mode, test cohort, execution host, repetition count, and cost source remain to be fixed before evaluation.
