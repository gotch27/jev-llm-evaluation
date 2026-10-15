# Metrics for the Macedonian tweet sentiment experiment

Read after [this experiment's folder guide](README.md). The metric definitions are general,
but all worked examples, counts, comparisons, and conclusions here concern this specific
Macedonian sentiment experiment. Source: the frozen repository-root path
`model-training/outputs/macedonian-sentiment/final-test/summary.json` and per-tweet predictions.
The numerical example throughout is `multilingual-full`, selected epoch 8.

## 1. What counts as a correct prediction?

The model returns three probabilities, one per class. We take the class with the largest
probability and compare it with the dataset's reference label. The metrics below measure
agreement with those original references. Disputed references are not silently corrected.

Every model sees the same 1,139 test tweets:

- Negative reference: 508.
- Neutral reference: 200.
- Positive reference: 431.

## 2. Start with the confusion matrix

A confusion matrix counts predictions by reference class. Here rows are reference labels,
and columns are predicted labels. The diagonal entries are agreements with the reference.

| Reference ↓ / prediction → | Negative | Neutral | Positive | Row total |
| --- | ---: | ---: | ---: | ---: |
| Negative | **368** | 123 | 17 | 508 |
| Neutral | 12 | **183** | 5 | 200 |
| Positive | 26 | 218 | **187** | 431 |
| Column total | 406 | 524 | 209 | 1,139 |

Examples of how to read it:

- 368 negative-reference tweets were predicted negative.
- 218 positive-reference tweets were predicted neutral.
- The model predicted neutral 524 times, although only 200 tweets have a neutral reference.

## 3. Accuracy: how often does the chosen label agree?

```text
accuracy = correct predictions / all predictions
         = (368 + 183 + 187) / 1,139
         = 738 / 1,139
         = 64.8%
```

It made 401 disagreements with the references. Accuracy gives every tweet equal weight;
classes with more tweets therefore influence the total more.

Always predicting negative, the training cohort's largest class, gets 508/1,139 = 44.6%
accuracy on this test. This is why an accuracy above one third is not by itself a useful
success criterion for this imbalanced three-class dataset.

## 4. Precision: when it predicts a class, how often is that right?

For positive predictions:

```text
positive precision = correctly predicted positive / all predicted positive
                   = 187 / 209
                   = 89.5%
```

Among the model's 209 positive calls, 187 matched a positive reference and 22 did not.
Its positive predictions are usually right when it makes them.

Neutral precision is much lower: 183/524 = 34.9%. Approximately two thirds of its neutral
calls disagree with the reference. Many are actually labelled positive or negative.

## 5. Recall: how much of a reference class does it find?

```text
positive recall = correctly predicted positive / all reference-positive tweets
                = 187 / 431
                = 43.4%
```

It finds fewer than half of the reference-positive tweets. This can coexist with 89.5%
precision: the model makes relatively few positive calls, which tend to be correct, but
misses many other positives.

Neutral recall is 183/200 = 91.5%. It finds most reference-neutral tweets, but by calling
many non-neutral tweets neutral too. High recall alone does not make that class reliable.

## 6. F1: combine precision and recall for one class

F1 is their harmonic mean: it rewards a class only when both quantities are reasonably high.

```text
F1 = 2 × precision × recall / (precision + recall)

Equivalently:
F1 = 2 × true positives / (2 × true positives + false positives + false negatives)
```

For positive: true positives = 187, false positives = 22, false negatives = 244.

```text
positive F1 = 374 / (374 + 22 + 244) = 0.5844
```

Here "positive" in "true positive" is a technical counting term: the class currently
being evaluated. A correctly recognized neutral tweet is a true positive for the neutral
class. It does not mean that all these terms refer only to positive sentiment.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 90.6% | 72.4% | 0.8053 |
| Neutral | 34.9% | 91.5% | 0.5055 |
| Positive | 89.5% | 43.4% | 0.5844 |

## 7. Macro-F1: give every class equal importance

"Macro" means calculate each class's F1 separately, then average the three values:

```text
macro-F1 ≈ (0.8053 + 0.5055 + 0.5844) / 3 = 0.6317
```

Each class contributes one third, even though the test set contains 508 negative, 200 neutral,
and 431 positive tweets. Macro-F1 is not accuracy and does not mean "63.17% of tweets correct."
It also does not equal the harmonic mean of averaged precision and averaged recall; we average
the class F1 values themselves.

The always-negative baseline has F1 zero for neutral and positive, yielding macro-F1 0.2056
despite 44.6% accuracy. Macro-F1 makes neglect of entire classes visible.

## 8. Compare multilingual full and frozen

| Metric | Full tuning | Frozen encoder |
| --- | ---: | ---: |
| Correct predictions | 738 | 717 |
| Accuracy | 64.8% | 62.9% |
| Negative F1 | 0.8053 | 0.7099 |
| Neutral F1 | 0.5055 | 0.2238 |
| Positive F1 | 0.5844 | 0.6491 |
| Macro-F1 | 0.6317 | 0.5276 |

The full model correctly recognizes 152 more neutral tweets (183 versus 31), but 59 fewer
negative tweets and 72 fewer positive tweets. The net difference is just 21 additional
correct labels. It does not dominate the frozen model on every class.

The full model's higher macro-F1 reflects better balance across classes, especially the
substantial improvement in neutral F1. The frozen model has better positive F1 and better
negative/positive recall, but misses most neutral-reference tweets.

These are observed single-run differences. Our paired-bootstrap accuracy-difference interval
for these two fixed models spans zero; a broad accuracy-superiority claim is unwarranted.
The macro-F1-difference interval is positive, conditional on the fixed models/references.

## 9. Probability quality is a separate question

Classification metrics use only which option has the largest probability. A prediction can
be 51% or 99% confident and still make the same class decision. Confidence metrics evaluate
the probabilities themselves.

Here confidence means the largest of the three class probabilities, `max(p)`. In Laya's
inference API that is `answer_confidence`. Its field named `confidence` for Choice/Score is
an entropy-based concentration measure; it is not the quantity used for these ECE calculations.

After calibration, multilingual/full's average confidence was 95.98%, while its accuracy
was 64.79%. Many mistakes therefore still carry very high confidence.

### Cross-entropy / negative log-likelihood (NLL)

The `loss` column in this evaluation is mean negative log-probability assigned to the
reference class, using natural logarithms:

```text
NLL = average[-ln(probability assigned to the reference)]
```

For an illustrative single tweet: assigning the reference probability 0.8 costs about 0.223;
assigning it 0.01 costs about 4.605. Confident mistakes can dominate this average.
Lower is better; zero is ideal; there is no fixed upper limit. It is not an error percentage.

The full model improved from 3.1520 to 1.5345 after calibration. The frozen model improved
from 0.9593 to 0.9202. Therefore the frozen model has better NLL despite lower accuracy.

For orientation, assigning exactly 1/3 to every class gives NLL ln(3) ≈ 1.0986. The calibrated
full model still has worse NLL than this uniform-probability reference. That does not negate
its classification gain; it demonstrates how costly its confident errors remain.

### Expected calibration error (ECE)

Laya groups top-class confidences into 15 equally spaced bins between zero and one. Within
each occupied bin it compares average confidence with empirical accuracy. ECE is the
absolute gap in each bin, weighted by the fraction of predictions in that bin.

Illustrative example: predictions averaging 80% confidence but only 60% correct have a
20-percentage-point gap within that bin. This is not a calculation on our actual bin data.

Our calibrated ECE values are 0.3141 for full and 0.0758 for frozen. These correspond to
weighted bin gaps of approximately 31.4 and 7.6 percentage points. ECE is not the error
rate, depends on binning/sample size, and does not guarantee reliability for every class
or subgroup.

### Multiclass Brier score

The `brier` field in these reports measures squared error over all three probabilities:

```text
Brier = average[sum over classes (predicted_probability - reference_target)^2]
```

The reference target is one for the reference class and zero for the others. We sum over
classes without dividing by three. With one-hot targets this convention ranges from 0 to 2.
Lower is better. Uniform probabilities give 2/3 ≈ 0.6667.

After calibration: full = 0.6566; frozen = 0.5136. Again, frozen has better probability quality.

`brier_top1`, also stored in the JSON, is different: it evaluates only the largest probability
as a prediction of whether the chosen label is correct. It is not the multiclass `brier`
used in our main tables. Do not mix differently defined Brier scores across tools.

## 10. What calibration changed

| Multilingual model | NLL raw → calibrated | ECE raw → calibrated | Brier raw → calibrated |
| --- | ---: | ---: | ---: |
| Full | 3.1520 → 1.5345 | 0.3423 → 0.3141 | 0.6881 → 0.6566 |
| Frozen | 0.9593 → 0.9202 | 0.1041 → 0.0758 | 0.5222 → 0.5136 |

One positive temperature divides all three logits before softmax. It preserves their order,
so it cannot change predicted labels, accuracy, confusion matrices, or class/macro-F1.
It can improve or worsen confidence metrics. Here the fitted temperatures improved all
three probability metrics on test, but did not make the full model reliably calibrated.

## 11. How to describe these results

- Accuracy: fraction of tweets whose chosen label matches the reference.
- Per-class precision: reliability of predictions of that class.
- Per-class recall: fraction of that reference class found.
- Per-class F1: combined precision and recall for that class.
- Macro-F1: equally weighted mean of the three class F1 scores.
- NLL/Brier: quality of probability predictions, with confident mistakes penalized.
- ECE: binned agreement between top-class confidence and observed correctness.

Use macro-F1 alongside accuracy as the principal classification comparison in this
imbalanced three-class task, with class-wise results to explain trade-offs. Report probability
metrics separately: the best observed classifier need not have the best probability quality.

All these measurements use the original reference labels. The selected human review is a
qualitative limitation analysis; it does not replace the reference set or establish a
dataset-wide label-error rate. The validation/test gap remains a major limitation.

The next experimental priority is a fixed replication plan across additional training seeds,
after preserving the selected checkpoints and training/calibration artifacts. A Score task
is a separate potential ablation, not a reinterpretation of the completed Choice experiment.
