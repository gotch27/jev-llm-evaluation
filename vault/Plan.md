# Plan

[← Thesis home](Thesis%20Home.md)

Agreed direction recorded on 2026-09-18; implementation and saved-run status updated on 2026-10-06. Develop further decisions in chat and save them here only when Gorazd authorizes changes.

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

The reusable structured benchmark framework is implemented and committed as `c93f2c4`. It supports provider-neutral Choice, Noul, and Score tasks, registered BANKING77 and BoolQ adapters, explicit state-field mappings, frozen cohorts, coordinated and resumable execution, preserved provider failures, typed evaluation reports, terminal progress, timing, provenance, and gateway-reported cost capture. The committed framework snapshot passed 141 offline tests plus lint, formatting, and dependency-lock checks. BANKING77 has saved live development runs; BoolQ has a configured training-only smoke plan but no saved live run, and Score is validated with synthetic datasets. See [Research and Tools — Python project implementation](Research%20and%20Tools.md#python-project-implementation).

### Optional demonstration

A small application demonstrating a selected use case is an optional extension. It can reuse the experimental modules; it is not a required thesis deliverable.

## Evaluation approach

- Compare result quality, response time, and execution cost on the selected tasks.
- For standalone comparisons, give Jev and the LLMs the same task, inputs, and output requirements, and assess correctness against reference labels.
- For intent classification, calculate accuracy, macro-F1, per-label metrics, confusion matrices, unsuccessful prediction counts, and paired Jev-versus-LLM comparisons over the same cohort. Preserve machine-readable per-example outcomes for later analysis and visualization.
- Record every selected example's latency, including the first request, without additional warm-up calls. Preserve measured and total execution time, provider-call latency, runner environment, and distribution statistics such as p50, p90, and p95. Use sequential execution when comparing latency so local contention does not distort the models differently.
- Then evaluate combined workflows against the standalone baselines, preferably on the same tasks and datasets, accounting for the quality, time, and cost of the entire workflow. Assess trade-offs rather than assume the combination is better.
- Make experiments repeatable by recording datasets and splits, model versions, prompts, configuration, code revisions, commands, and output locations.
- Preserve failed and inconclusive results, and use the evidence to explain strengths and limitations and formulate practical recommendations.

## Milestones and next actions

TypeSafe's four published workflow evaluations, indexed in [Use Cases — Use cases](Use%20Cases.md#use-cases), remain background for our independent experiments. Intent classification with BANKING77 is the first standalone use case. The training-only comparison of Jev, GPT-5.6 Luna, Gemini 2.5 Flash Lite, and Qwen 3.5 Flash completed on 770 messages, with 10 examples per intent, on 2026-10-01 in Europe/Skopje time. A ten-example smoke run completed on 2026-10-05 and captured nonzero Jev cost. The earlier interrupted run remains preserved, but its retired input format is unsupported by the current runner. These are development records; no frozen held-out thesis evaluation has been recorded. Details and limitations are in [Intent Classification — Experiments](Use%20Cases/Intent%20Classification.md#experiments).

The next milestone is to review the completed candidate comparison and choose which models advance. Then freeze the first thesis benchmark by deciding the task variant and LLM output mode, test cohort, execution host, latency repetitions, and treatment of missing cost data. Missing historical Jev cost cannot be reconstructed from the saved typed responses. Development model calls remain on the training split until those decisions are complete. The live BoolQ smoke check remains outstanding for the framework's Noul path. Run the standalone evaluation before selecting or implementing a combined Jev–LLM workflow.

## Open questions

- Which of the development candidates will form the final BANKING77 comparison?
- Will the final comparison use the criteria task, the without-criteria task, or treat their difference as a separate experiment?
- Will LLMs use `discrete` or `probabilities` output in the final comparison?
- Will the evaluation use the complete BANKING77 test split or a frozen predefined sample?
- Which execution host and number of repetitions will be used for the final latency comparison?
- How will a thesis comparison handle missing gateway-reported cost, including the unavailable historical Jev total?
- Which standalone use case follows intent classification, and which completed standalone task is most suitable for a later combined workflow?
