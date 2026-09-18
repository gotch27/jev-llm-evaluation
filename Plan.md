# Plan

[[Thesis Home|← Thesis home]]

Agreed direction recorded on 2026-09-18. Develop further decisions in chat and save them here only when Gorazd authorizes changes.

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

Read and study the publicly available material needed to explain how the models are built and trained. Distinguish disclosed technical details from provider claims and information that is not publicly available. Keep sources and reading notes in [[Research and Tools]].

### Comparative and combined-model experiments

First choose tasks for standalone evaluation of Jev against LLMs, preferably tasks that can later support combined experiments using the same datasets. Establish individual model performance before testing whether combining them improves the quality, time, or cost of completing the task. Combined experiments evaluate the whole workflow and do not replace standalone model evaluation.

Select specific tasks and combinations together rather than committing to one application domain. Record methods, configurations, results, and limitations in [[Experiments]].

### Reusable software modules

Develop shared modules for calling the Jev and LLM APIs, handling task inputs and datasets, running experiments, and logging and evaluating results. Reuse these modules across experiments and combined workflows. Keep implementation and raw outputs in the separate research repository and link them from the vault.

### Optional demonstration

A small application demonstrating a selected use case is an optional extension. It can reuse the experimental modules; it is not a required thesis deliverable.

## Evaluation approach

- Compare result quality, response time, and execution cost on the selected tasks.
- For standalone comparisons, give Jev and the LLMs the same task, inputs, and output requirements, and assess correctness against reference labels.
- Then evaluate combined workflows against the standalone baselines, preferably on the same tasks and datasets, accounting for the quality, time, and cost of the entire workflow. Assess trade-offs rather than assume the combination is better.
- Make experiments repeatable by recording datasets and splits, model versions, prompts, configuration, code revisions, commands, and output locations.
- Preserve failed and inconclusive results, and use the evidence to explain strengths and limitations and formulate practical recommendations.

## Milestones and next actions

The next discussion is to choose the initial standalone evaluation tasks together, followed by datasets and experimental configurations. Establish standalone results before selecting and evaluating combined approaches. Reading the technical foundations and developing reusable modules will support the experiments. The resulting evidence will support the analysis and recommendations; the optional demo follows selection of a suitable use case.

## Open questions

- Which standalone tasks will represent use cases expected to favor Jev or LLMs, and which can later support combined experiments?
- Which datasets, comparison models, task-specific metrics, and experimental configurations will we use for those cases?
