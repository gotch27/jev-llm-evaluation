# Macedonian sentiment fine-tuning: experiment summary

Prepared on 15 October 2026 from the frozen local test reports and recorded remote setup.
This summarizes a single-seed experiment, not the final replicated thesis result.
Its findings and limitations concern this Macedonian tweet sentiment dataset and protocol;
they are not general findings about all Laya fine-tuning tasks.

## Research question

Does task-specific supervised fine-tuning improve English and multilingual Laya on native
Macedonian tweet sentiment classification, and how does freezing the encoder affect the result?

The task is three-way Choice classification: negative, neutral, positive. It evaluates this
specific task in Macedonian; it does not establish general Macedonian language competence.

## Data and separation of roles

- Source: [mteb/MacedonianTweetSentimentClassification](https://huggingface.co/datasets/mteb/MacedonianTweetSentimentClassification),
  native Macedonian tweets; no translation in this experiment.
- Pinned source revision: `6ec78a685dad2e2b37af77264676e9dddfa7dc8e`.
- Prepared release: `e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc`.
- Five training texts overlapping the original test set were excluded. No exact normalized
  text groups cross the prepared split boundaries; fuzzy similarity and author/thread grouping
  are not covered by that check.
- Remaining original training pool: train 6,849, validation 856, calibration 856.
- Original test split retained: 1,139 examples (508 negative, 200 neutral, 431 positive).
- Validation selected checkpoints; calibration fitted temperatures; test measured the fixed
  selections. Test predictions were not used to update weights or fit temperatures.

## Models and training

- English base: `convaiinnovations/laya`, 421,293,827 parameters; revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- Multilingual base: `convaiinnovations/laya-multilingual`, 321,908,995 parameters; revision
  `1720e3e3357cfe1e281542e223f8273b0890ca34`.
- Four runs: English/full, English/frozen-encoder, multilingual/full, multilingual/frozen-encoder.
  All started from their original base, not earlier Yelp or trial checkpoints.
- Eight epochs, seed 42, FP32, supervised `soft-ce` loss. This is an explicit supervised
  fine-tuning objective, not a reproduction of the authors' default RLCD recipe.
- Encoder learning rate `2.5e-5` when unfrozen; decision-head learning rate `1e-4`.
- Micro-batch 4, gradient accumulation 4: effective batch up to 16.
- Same Macedonian question, fixed option order, maximum 512 tokens, head budget 192.
  The setup audit reported no truncation in any prepared split for either tokenizer.
- Epoch selected by validation macro-F1, breaking ties using lower cross-entropy. Selected
  epochs: English/full 5, English/frozen 7, multilingual/full 8, multilingual/frozen 8.
- Encoder weight changes/freeze behavior and checkpoint reloads were checked. Exports use
  Laya's FP16 serialization; final evaluation reloads them for FP32 inference.
- Hardware: remote NVIDIA GeForce RTX 5080, approximately 15.47 GiB VRAM. Remote runtime:
  Python 3.12, Laya 0.3.29, PyTorch 2.13.0+cu130, Transformers 5.19.0.

## Final test classification results

All six rows use temperature 1.0. Fitted scalar temperatures preserve these class predictions.

| Model | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| English base | 26.6% | 0.2445 |
| English, full fine-tuning | 54.9% | 0.5392 |
| English, frozen encoder | 46.4% | 0.3158 |
| Multilingual base | 41.9% | 0.3936 |
| Multilingual, full fine-tuning | 64.8% | 0.6317 |
| Multilingual, frozen encoder | 62.9% | 0.5276 |
| Always predict negative (training-majority reference) | 44.6% | 0.2056 |

Full fine-tuning improved English accuracy by 28.3 percentage points and multilingual accuracy
by 22.9 points relative to their bases. Macro-F1 improved by 0.2947 and 0.2382 respectively.

A paired tweet-level bootstrap (5,000 resamples, fixed analysis seed 20261015) gave approximate
95% percentile intervals for the accuracy improvements: English +24.9 to +31.6 points;
multilingual +19.4 to +26.4 points. These are conditional on the fixed models and original
labels, assume independent tweet sampling, and exclude training-seed variability, annotation
errors, and distribution shift. The smaller multilingual full-versus-frozen accuracy difference
of 1.84 points had an interval spanning zero; it should not be described as a demonstrated
accuracy advantage. Its macro-F1 difference was 0.1041, interval approximately 0.0644–0.1432.

## Confidence calibration

Separate Laya temperature maps were fitted on the 856 calibration examples for each fine-tune.
No model weights were changed. The base models remained uncalibrated in the main comparison.
An optional English-base map exists remotely but was excluded from the final main evaluation.

| Fine-tuned model | Temperature | Test NLL before → after | Test ECE before → after | Test Brier before → after |
| --- | ---: | ---: | ---: | ---: |
| English full | 1.4944 | 3.6428 → 2.4547 | 0.4457 → 0.4326 | 0.8956 → 0.8811 |
| English frozen | 1.0310 | 1.1772 → 1.1660 | 0.2021 → 0.1945 | 0.7044 → 0.6994 |
| Multilingual full | 2.1084 | 3.1520 → 1.5345 | 0.3423 → 0.3141 | 0.6881 → 0.6566 |
| Multilingual frozen | 1.1520 | 0.9593 → 0.9202 | 0.1041 → 0.0758 | 0.5222 → 0.5136 |

Lower is better for these three probability metrics. They all improved on test after calibration,
but the fully tuned models remained overconfident. Multilingual/full had the highest observed
classification scores; multilingual/frozen had better probability metrics among the fine-tunes.

## Generalization and qualitative limitations

- Recorded best validation accuracy was 97.7% for English/full and 97.9% for multilingual/full,
  substantially above test accuracy. Validation/calibration were subdivided from the original
  training pool, while test remained the distinct original split. Vocabulary and annotation
  differences and overfitting are plausible contributors; the complete cause is unresolved.
  These validation figures come from remote logs supplied in the discussion; original epoch
  history reports have not yet been copied into the local result archive.
- Multilingual/full called 218 of 431 positive test tweets neutral. Multilingual/frozen
  correctly recognized only 31 of 200 neutral test tweets. Their weaknesses differ materially.
- After calibration, 330 of multilingual/full's 401 mistakes still had at least 95% top-class
  probability. A low fitting-cohort calibration error did not establish reliable test confidence.
- One reviewer assessed 43 selected multilingual/full errors: 14 reference labels plausible,
  5 unclear, 24 questionable; 32 references positive and 11 negative, none neutral.
  This was post-test, model-visible, nonrandom error analysis. It is not a dataset-wide error
  estimate or an independently adjudicated replacement gold standard. Original labels remain intact.
- Humour, sarcasm, context, mixed sentiment, and individual-word associations recur in the notes.
  They motivate hypotheses, not causal attribution of model decisions.
- Only one training seed and one common epoch/learning-rate budget were tested. Frozen models
  were not separately optimized to convergence. Generalization to other tasks/domains is unknown.

## Conclusion and proposed next work

Observed supervised fine-tuning gains on the fixed reference test set are substantial. The
English and multilingual checkpoints both improved; full tuning gave higher macro-F1 than
head-only tuning under this budget. Confidence calibration helped probability metrics but
did not resolve generalization failures or annotation ambiguity.

Preserve this first completed experiment. Proposed next work is to repeat the four conditions
with two additional predeclared seeds and report variation, keeping the current protocol fixed.
A three-level Score formulation is an optional separate ablation, not an automatic fix for
uncertain or disputed labels. Model/task changes motivated by this test review need a separately
defined follow-up evaluation; the current test must not become a tuning set.

Training variability motivation: [Dodge et al. (2020)](https://arxiv.org/abs/2002.06305).
Dataset origin: [Jovanoski et al. (2015)](https://aclanthology.org/R15-1034/).

## Artifact locations

- Local dataset and complete test results: see [the folder guide](README.md).
- Local combined results, relative to the repository root:
  `model-training/outputs/macedonian-sentiment/final-test/summary.json` and `summary.csv`.
- Local user review and bootstrap analysis: `model-training/outputs/macedonian-sentiment/error-review/`.
- Selected fine-tuned weights and original training/calibration artifacts remain remote, as
  recorded in the copied reports. They are not included merely by copying prediction results.
- English base inference was verified locally on the Mac's MPS GPU. The remaining five
  selected exports need transfer/local verification; this does not require retraining.

This document does not modify research benchmarks, thesis notes, dataset labels, or model weights.
