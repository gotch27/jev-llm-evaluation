# Plan

[← Thesis home](Thesis%20Home.md)

Agreed direction recorded on 2026-09-18 and implementation status updated on 2026-09-20. Develop further decisions in chat and save them here only when Gorazd authorizes changes.

## Thesis title

Евалуација на моделот Jev: споредба и комбинирана примена со големи јазични модели

## Short description

Дипломската работа ќе се фокусира на истражување и евалуација на моделот Jev од TypeSafe, споредба со големи јазични модели (LLMs) и испитување на можностите за нивна комбинирана примена. Целта е да се утврдат предностите и ограничувањата на моделот преку експерименти со задачи од различни области. Во теоретскиот дел, врз основа на јавно достапните информации, ќе бидат разгледани дизајнот на моделите, методите и целите на нивното тренирање, како и нивната поврзаност со начинот на работа и намената.

Во практичниот дел ќе бидат спроведени повторливи експерименти за споредба на Jev и големите јазични модели според квалитетот на резултатите, времето на одговор и трошоците за извршување. Ќе биде испитано и во кои задачи нивната комбинирана примена нуди предности во однос на самостојната употреба. Резултатите ќе послужат за формулирање препораки за избор на соодветниот пристап според барањата на задачата.

## Research questions and scope

- Understand Jev's capabilities, limitations, and practical use cases.
- Explain documented differences between Jev and LLMs in model design, training methods, training objectives, and intended use.
- Evaluate several use cases from different areas, chosen together with Gorazd. Include tasks expected to favor LLMs and tasks expected to favor Jev; test those expectations rather than assume them as findings.
- Investigate when combining Jev and LLMs improves on using either approach alone.
- Build reusable code that supports the experiments and their evaluation.

The central questions are where each approach works well, where it struggles, and when a combined workflow offers a useful trade-off between result quality, response time, and cost. Conclusions will be limited to the tasks, models, and configurations evaluated.

## Main components

### Technical foundations

Read and study the publicly available material needed to explain how the models are built and trained. Distinguish disclosed technical details from provider claims and information that is not publicly available. Keep sources and reading notes in [Research and Tools](Research%20and%20Tools.md).

### Comparative and combined-model experiments

First choose tasks for standalone evaluation of Jev against LLMs, preferably tasks that can later support combined experiments using the same datasets. Establish individual model performance before testing whether combining them improves the quality, time, or cost of completing the task. Combined experiments evaluate the whole workflow and do not replace standalone model evaluation.

Select specific tasks and combinations together rather than committing to one application domain. Record methods, configurations, results, limitations, and interpretation in the individual use-case notes linked from [Use Cases](Use%20Cases.md).

### Reusable software modules

Develop shared modules for calling the Jev and LLM APIs, handling task inputs and datasets, running experiments, and logging and evaluating results. Reuse these modules across experiments and combined workflows. Keep implementation in `research/` within this repository, with raw outputs in ignored output directories or external storage, and link them from the vault.

The initial standalone foundation is implemented for BANKING77: pinned data preparation, shared structured Jev/LLM clients, task and benchmark configuration, frozen cohorts, coordinated and resumable execution, evaluation reports, and terminal progress. This establishes reusable components but does not itself constitute a thesis experiment or result.

### Optional demonstration

A small application demonstrating a selected use case is an optional extension. It can reuse the experimental modules; it is not a required thesis deliverable.

## Evaluation approach

- Compare result quality, response time, and execution cost on the selected tasks.
- For standalone comparisons, give Jev and the LLMs the same task, inputs, and output requirements, and assess correctness against reference labels.
- For intent classification, calculate accuracy, macro-F1, per-label metrics, confusion matrices, unsuccessful prediction counts, and paired Jev-versus-LLM comparisons over the same cohort. Preserve machine-readable per-example outcomes for later analysis and visualization.
- Then evaluate combined workflows against the standalone baselines, preferably on the same tasks and datasets, accounting for the quality, time, and cost of the entire workflow. Assess trade-offs rather than assume the combination is better.
- Make experiments repeatable by recording datasets and splits, model versions, prompts, configuration, code revisions, commands, and output locations.
- Preserve failed and inconclusive results, and use the evidence to explain strengths and limitations and formulate practical recommendations.

## Milestones and next actions

TypeSafe's four published workflow evaluations, indexed in [Use Cases — Use cases](Use%20Cases.md#use-cases), remain background before our independent experiments. Intent classification with BANKING77 is selected as the first standalone use case, and its practical foundation is implemented. No thesis evaluation has been run.

The next milestone is to freeze the first thesis benchmark: select the comparison models and providers, decide the task variant and LLM output mode, choose the test cohort, define the latency and cost procedure, and create a dedicated evaluation plan. Development remains on the training split until those decisions are complete. Run the standalone evaluation before selecting or implementing a combined Jev–LLM workflow.

## Open questions

- Which Jev and LLM model/provider versions will form the first BANKING77 comparison?
- Will the final comparison use the criteria task, the without-criteria task, or treat their difference as a separate experiment?
- Will LLMs return labels or complete probability distributions in the final comparison?
- Will the evaluation use the complete BANKING77 test split or a frozen predefined sample?
- How will latency be measured and repeated, and how will cost be obtained when gateway responses omit it?
- Which standalone use case follows intent classification, and which completed standalone task is most suitable for a later combined workflow?
