# Macedonian sentiment experiment: files, models, and results

This guide describes only the **Macedonian tweet sentiment experiment with Laya**. Its
scripts, question, settings, dataset versions, and results are specific to this task.
Read [SUMMARY.md](SUMMARY.md) for this experiment's research summary and
[METRICS.md](METRICS.md) for calculations using its actual test results.
The [workspace index](../../README.md) explains the broader folder organization.

Inventory checked on 15 October 2026. Local paths below are relative to the thesis root:
`/Users/gorazdfilipovski/Documents/graduate-thesis/`.

## 1. Where this fits in the thesis

```text
graduate-thesis/
├── datasets/                 Shared source data, translations, and frozen releases
├── model-training/           Parent workspace for individually named experiments
│   ├── README.md             General workspace index; no dataset-specific results
│   ├── experiments/
│   │   └── macedonian-sentiment/
│   │       ├── README.md     This experiment's files, models, and execution guide
│   │       ├── SUMMARY.md
│   │       ├── METRICS.md
│   │       ├── questions.json
│   │       ├── train_sentiment.py
│   │       ├── calibrate_sentiment.py
│   │       ├── evaluate_sentiment_test.py
│   │       ├── review_sentiment_results.py
│   │       └── tests/        Evaluation, calibration, and training-flow regression tests
│   └── outputs/macedonian-sentiment/     Ignored local artifacts; not model weights
│       ├── final-test/
│       ├── error-review/
│       └── local-smoke/
├── model-serving/            Existing Python environment and HTTP server for inference
│   └── models/               Actual downloaded checkpoints; ignored by Git
├── research/                 Existing benchmark code; unchanged in this experiment phase
└── vault/                    Thesis notes; unchanged in this experiment phase
```

The parent folder added in this fine-tuning phase is `model-training/`; this experiment's
source and documentation are under `model-training/experiments/macedonian-sentiment/`. We reused
`datasets/` for the frozen native Macedonian release and `model-serving/` for its existing
Laya runtime. Earlier English/Macedonian Yelp and SNIPS data remain separate datasets.

The current dataset releases are BANKING77, BoolQ, bilingual Yelp, bilingual SNIPS, and
Macedonian tweet sentiment. Only the last was used in this completed sentiment experiment.
See [the dataset guide](../../../datasets/README.md) for all releases and their version IDs.

## 2. What each script does

The filenames in this table are relative to this experiment folder, unless prefixed with
a repository-root path.

| File | Purpose | Where it runs |
| --- | --- | --- |
| `train_sentiment.py` | Fine-tune all weights or freeze the encoder; validate each epoch and save improving checkpoints | Remote CUDA machine; copied there as `train_full.py` |
| `calibrate_sentiment.py` | Fit a separate temperature on the fixed calibration cohort | Remote CUDA machine |
| `evaluate_sentiment_test.py` | Evaluate the six selected checkpoints and four fixed calibration maps | Remote CUDA machine |
| `review_sentiment_results.py` | Verify copied predictions and build the offline comparison/review page | Mac; no model inference |
| `tests/` | Check metrics, calibration, checkpoint selection, frozen-encoder handling, evaluation, and review generation with mocked GPU/model loading | Mac, using the existing Laya environment |
| `questions.json` | Exact Macedonian question and three option descriptions used in the experiment | Both machines |

The remote preparation, baseline, and trial scripts (`audit_tokens.py`,
`evaluate_sentiment.py`, `train_trial.py`) were created during setup on the remote machine.
They are not currently archived in this local folder. The final workflow scripts above
are local source files; do not confuse remote setup trials with the selected full runs.

The question file was reconstructed from the recorded instructions and verified against
the question SHA-256 in the final results:
`c5edfc2b7ddf9e3124e520f6eace2c9cedb3a1fe608942c5fa6adf07da61bafa`.

## 3. Dataset: source files versus the experiment's frozen release

Original upstream files:

```text
datasets/macedonian-tweet-sentiment/source/
  6ec78a685dad2e2b37af77264676e9dddfa7dc8e/
```

The exact prepared data used for this experiment:

```text
datasets/releases/macedonian-tweet-sentiment/
  e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc/
    train.jsonl          6,849 examples: update model weights
    validation.jsonl       856 examples: select an epoch
    calibration.jsonl      856 examples: fit confidence temperatures
    test.jsonl           1,139 examples: final evaluation
    manifest.json        Identity, checksums, counts, and preparation provenance
```

These are native Macedonian tweets, not translations. The original test split was retained;
five overlapping training texts were excluded before subdividing the training pool.
The long directory name identifies the exact prepared data, not a model.

## 4. The six selected models

There are two base architectures and four fine-tuned exports. English and multilingual
describe the base checkpoints, not which language was used for fine-tuning: all four
fine-tunes used the same Macedonian tweets and question.

| Alias | Starting checkpoint | What training updated | Selected epoch | Current weight location |
| --- | --- | --- | --- | --- |
| `english-base` | `convaiinnovations/laya` (~421M parameters) | No task fine-tuning | None | Remote and Mac |
| `english-full` | English base | Encoder and decision head | 5 | Remote |
| `english-frozen` | English base | Decision head; encoder unchanged | 7 | Remote |
| `multilingual-base` | `convaiinnovations/laya-multilingual` (~322M parameters) | No task fine-tuning | None | Remote |
| `multilingual-full` | Multilingual base | Encoder and decision head | 8 | Remote |
| `multilingual-frozen` | Multilingual base | Decision head; encoder unchanged | 8 | Remote |

All six have local prediction results. Five of the six do not yet have local weights.
A frozen-encoder export is still a complete checkpoint, including the unchanged encoder.
Freezing saved training memory; it does not remove the encoder from inference.

The fine-tuned checkpoints all started from the original bases, not earlier Yelp or trial
checkpoints. The base revisions are:

- English: `7b928d828b7b0e022f929d9bd2e44165aa270148`
- Multilingual: `1720e3e3357cfe1e281542e223f8273b0890ca34`

Calibration is a separate probability adjustment, not another trained model. The main
evaluation has six raw model rows and four additional calibrated rows. An optional English
base calibration was also fitted remotely, but it was excluded from the untouched baseline.

## 5. Where the actual weights are

The English base is already on this Mac:

```text
model-serving/models/convaiinnovations--laya/
  7b928d828b7b0e022f929d9bd2e44165aa270148/
```

The following are remote paths on `vezilka-gpu-2`, under `/home/viktor.kostadinoski3/`,
as recorded by the completed evaluation. Their continued presence has not been rechecked
over SSH during this local inventory:

```text
laya-yelp/models/laya-english/                 English base
laya-yelp/models/laya-multilingual/            Multilingual base

laya-macedonian-sentiment/outputs/
  english-full/epoch-5/
  english-frozen/epoch-7/
  multilingual-full/epoch-8/
  multilingual-frozen/epoch-8/
```

Trials, nonselected epochs, and earlier Yelp runs may also remain remotely. They are not
additional selected sentiment models and have not been exhaustively inventoried here.
Remote commands are supplied for the user to execute; no SSH connection was made for this inventory.

To list all checkpoint folders, including old Yelp runs and trials, run this on the remote
machine. It is read-only and does not copy or load models:

```sh
find ~/laya-yelp/models ~/laya-yelp/outputs ~/laya-macedonian-sentiment/outputs \
  -type f -name rl_agent_config.json -printf '%h\n' | sort
```

Each listed folder with matching weights/tokenizer/encoder files is a checkpoint candidate.
The six selected paths above identify the experiment's actual comparison, regardless of
how many intermediate exports remain on disk.

A complete runnable checkpoint contains:

```text
checkpoint/
  model.safetensors       Learned numerical weights; the large file
  rl_agent_config.json    Architecture/runtime settings
  encoder/                Encoder configuration
  tokenizer/              Text-to-token rules
  questions.json          Saved task question, in our fine-tuned exports
```

`summary.json`, `predictions.jsonl`, and `logits.npz` are not checkpoints. They cannot answer
a new tweet without the weight files. Transfer the entire selected checkpoint folder when
you want local inference, rather than copying only `model.safetensors`.

## 6. Where the results are on this Mac

```text
model-training/outputs/macedonian-sentiment/
  final-test/
    report.json             Overall run status, settings, and provenance
    summary.json            Combined results and detailed per-model metrics
    summary.csv             Spreadsheet-friendly six raw + four calibrated rows
    <model-alias>/
      report.json           That model's results, selected epoch, and fingerprints
      predictions.jsonl     1,139 tweets: reference, predicted label, probabilities
      logits.npz            Raw scores, targets, IDs, and label order in NumPy format
  error-review/
    review.html             Offline comparison and human-review interface
    analysis.json           Verified class recalls and error counts
    review-notes.csv         Preserved copy of the 43 user review judgments
    review-notes-joined.json Each judgment linked to all six model predictions
    bootstrap-comparisons.json  Test-sampling uncertainty estimates
  local-smoke/
    report.json             One verified English-base prediction on the Mac GPU
    questions.json          Same exact question, saved with the check
    raw-temperature.json    Explicit T=1 settings; no calibration was fitted
```

The raw and calibrated rows share class predictions. Their differences are probability
metrics, not accuracy or macro-F1. The `model` column identifies the weight set, and
`variant` distinguishes raw from calibrated output.

`report.json` is a general filename whose meaning depends on its parent folder. A remote
training report has epoch history; a final-test model report has test metrics; a calibration
report has temperature-fitting diagnostics. Check the folder before interpreting one.

Currently missing from the local archive are the remote training histories/logs,
`selected-models.json`, the actual four calibration sidecar files and their fitting reports,
and the five absent checkpoints. The final summary records selected paths, hashes, fitted
temperatures, and metrics, but does not replace all those original artifacts.

## 7. Can these models run on this Mac?

Yes for local inference, with different levels of verification:

- This Mac has an ARM64 Apple Silicon processor and 18 GiB unified memory.
- PyTorch reports MPS support. MPS is PyTorch's Apple GPU backend; CUDA is NVIDIA's backend.
- The local environment is `model-serving/.venv`: Laya 0.3.29, PyTorch 2.14.1,
  Transformers 5.19.0. Remote evaluation used PyTorch 2.13.0+cu130.
- The stored English base completed one FP32 prediction on MPS, without CPU fallback,
  using the fixed 512/192 token budgets and T=1. Its probabilities matched the corresponding
  remote test example within the API's four-decimal reporting precision.
- Approximately 1.57 GiB of MPS tensor memory was allocated after that prediction. This
  is not a peak or total system-memory measurement, nor an inference-speed benchmark.
- The four fine-tuned exports have the same parameter shapes as their respective bases;
  freezing makes no architectural change. One model at a time should be practical on this
  machine. The multilingual model and four fine-tuned exports still require transfer and
  local verification before we claim they have actually run here.

The existing HTTP server can load a complete local checkpoint. From `model-serving/`:

```sh
USE_TF=0 .venv/bin/serve-laya \
  --checkpoint ./models/convaiinnovations--laya/7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --model-id laya-english-base-local \
  --device mps --max-len 512 --head-max-len 192 --port 8000
```

This starts a loopback HTTP service; it does not train or benchmark the model. No service
was left running by the local smoke check. For a downloaded fine-tune, change the checkpoint
path and give it a distinct model alias. Read [the serving guide](../../../model-serving/README.md)
for the request format and health endpoint.

Important confidence detail: the current HTTP server loads checkpoint-default temperatures
and has no CLI flag for a separate calibration file. The English base's shipped temperatures
are not the T=1 condition used in our final comparison. Fine-tuned exports were saved with
T=1; their fitted Macedonian temperatures live in separate remote sidecars. Laya's Python
API supports `calibration="/path/to/calibration.json"`; wiring explicit raw/calibrated modes
into the HTTP server is a separate, future serving change. Starting a server alone does
not reproduce the experiment's chosen confidence condition.

The training and final-evaluation scripts here explicitly require CUDA. Their remote
requirements should not be confused with the model's ability to perform MPS inference.

## 8. What Git saves

Dataset preparation was committed in `c3056be`. This folder versions the fine-tuning scripts,
tests, experiment guide, summary, and metric explanations. The root README links this
experiment from the repository index.

The completed remote run predates the local code review. Current source includes stricter
target validation, a Laya-version check for training, and rejection of optimized Python mode
(`-O` / `PYTHONOPTIMIZE`) so integrity assertions cannot be silently disabled. It also handles
invalid browser-stored review notes. These changes did not rerun the experiment or alter its
artifacts. Historical reports retain their original script fingerprints; a new run records
the hash of the source used for that run.

`model-training/outputs/`, `model-serving/models/`, downloaded datasets, and environments
are ignored. Committing the source code does not back up weights, datasets, or results.
Those need a separate artifact backup. Keep the remote selected checkpoints until the
local transfers have been verified.

## 9. Reading order

1. Understand the folders and distinguish data, instructions, weights, and results here.
2. Read the experiment summary for the completed experiment and its limitations.
3. Read `final-test/summary.csv` for the model comparison.
4. Learn the metrics from a real confusion matrix in [METRICS.md](METRICS.md): accuracy,
   per-class precision/recall/F1, macro-F1, then confidence metrics (cross-entropy, ECE, Brier).
5. Inspect individual predictions and the human error-review notes.

Section 10 below retains this experiment's detailed training/calibration/evaluation commands.
New model/task changes should follow a separately defined experiment.

## 10. Reproducing this specific experiment

The commands below are for Macedonian tweet sentiment only. Local source paths are relative
to the thesis repository root; remote commands run in `~/laya-macedonian-sentiment/`. The
remote filenames used in the completed run remain unchanged.


`model-training/experiments/macedonian-sentiment/train_sentiment.py` runs on the CUDA training machine with its existing Laya
0.3.29 environment; it does not import or change `research/`. Copy it to the remote experiment
directory as `train_full.py`. It expects the prepared sentiment release, `questions.json`,
and the two original-model validation reports produced during remote setup. Original models
and the Python environment are reused from `~/laya-yelp/`.

From the remote `~/laya-macedonian-sentiment/` directory:

```sh
USE_TF=0 ~/laya-yelp/.venv/bin/python -u train_full.py \
  --model english --epochs 8 --output outputs/english-full
```

Use `--model multilingual` for the other checkpoint and add `--freeze-encoder` for head-only
training. Give every run a new output directory. All runs start from the original checkpoint,
use FP32 supervised `soft-ce` loss, 512/192 token budgets, micro-batches of four with four
accumulation steps (effective batch 16), seed 42, and fixed option order. The head learning
rate is `1e-4`; full training also updates the encoder at `2.5e-5`. Eight epochs are an initial
common training budget, not a claim of optimal tuning for either architecture or freezing mode.

Validation runs after each epoch with temperature 1.0. Improving checkpoints are saved;
selection uses highest macro-F1, with lower cross-entropy breaking ties. The script preserves
training mode after validation and keeps a frozen encoder in evaluation mode. It checks all
encoder parameter fingerprints before/after training and reloads the selected checkpoint for
a final validation check. Checkpoint weights are exported in FP16 by Laya, so the report records
saved-checkpoint validation separately from the FP32 selection metrics. Calibration and test
files are not read by training.

`report.json` records source fingerprints, settings, versions, epoch metrics, and failure status.
Previous runs are never overwritten. Use tmux to protect a running job against SSH disconnects.
Saved model checkpoints do not include optimizer/scheduler state; this script does not offer
exact interrupted-run resumption. Training artifacts stay out of Git; script changes need a
separate commit when requested.

After fixing the six model selections in `selected-models.json`, copy
`model-training/experiments/macedonian-sentiment/calibrate_sentiment.py` to the same remote directory. Run one selected model
at a time, for example:

```sh
USE_TF=0 ~/laya-yelp/.venv/bin/python -u calibrate_sentiment.py --model english-full
```

Accepted model names are `english-base`, `english-full`, `english-frozen`,
`multilingual-base`, `multilingual-full`, and `multilingual-frozen`. The script checks the
selection, checkpoint fingerprints, shared question, and frozen release. It reads only the
856-example calibration cohort and fits Laya 0.3.29 temperature scaling, with the library's
temperature bounds of 0.5–5.0. All examples fit one Choice-type scalar; the library's
2,000-example floor prevents an additional option-count-specific fit on this cohort.

Each run creates a new `outputs/calibration/<model>/` directory containing a standard Laya
`calibration.json` sidecar, raw `logits.npz` with row IDs and label order, and `report.json`
with provenance, fitting diagnostics, boundary hits, and completion/failure status. No
checkpoint is rewritten. Temperature scaling preserves class predictions, accuracy, and
macro-F1. Before/after cross-entropy, ECE, and Brier scores here are fitting-cohort diagnostics;
independent calibration quality must be assessed later on the untouched test cohort. The
script does not perform test evaluation, abstention fitting, or further model selection.
Existing output directories are never overwritten; use `--output` with a new directory to
retry a failed run and `--batch-size` to reduce inference memory if needed.

The main experiment keeps both base models at temperature 1.0. A base-model calibration
file is an optional task-adapted confidence variant, not the untouched baseline; producing
such a file does not overwrite the original checkpoint.

When all four fine-tuned models have completed calibration, copy
`model-training/experiments/macedonian-sentiment/evaluate_sentiment_test.py` to the remote experiment directory and run:

```sh
USE_TF=0 ~/laya-yelp/.venv/bin/python -u evaluate_sentiment_test.py
```

This sequentially evaluates the six fixed checkpoints on all 1,139 original test tweets.
Before inference it verifies all selected checkpoints and all four calibration reports/files
against the frozen selection, question, dataset, and settings. It reads no train, validation,
or calibration examples and performs no fitting or selection. Both bases are evaluated only
at temperature 1.0; each fine-tuned checkpoint is evaluated at temperature 1.0 and at its
previously fitted temperature using the same raw logits. The reference classifier always
predicts negative, the training cohort's majority class fixed before test inspection.

`outputs/final-test/` contains per-model reports, raw logits, and per-tweet predictions,
plus a combined `summary.json` and `summary.csv` with six raw and four calibrated rows.
Reports include accuracy, macro-F1, class-wise F1, labelled confusion matrices,
cross-entropy/NLL, ECE, and Brier scores. Positive scalar temperature scaling is checked
to preserve predictions, accuracy, and macro-F1. Test confidence metrics can improve or
deteriorate; they must not be used to refit temperatures or choose different checkpoints.
All model variants use the common 512/192 token budgets, sequential option layout, FP32
inference, and fixed question. Checkpoint fingerprints are checked again after each run.
Partial progress and failures are recorded, but existing output directories are never
overwritten. If an operational failure requires a rerun, use `--output` with a new path
and keep the same fixed model/calibration protocol. No optimizer resumption is involved.

The evaluation checks run locally with the existing Laya-serving environment:

```sh
USE_TF=0 model-serving/.venv/bin/python -m unittest discover -s model-training/experiments/macedonian-sentiment/tests -v
```

These use synthetic records and mocked GPU/model loading with real Laya metric calculations
and temperature fitting. They check confusion/F1, positive-temperature invariance, malformed
targets, calibration-to-checkpoint binding, six raw/four calibrated output variants, unchanged
weights during evaluation, and refusal to overwrite results. Training-flow checks verify
epoch selection/reload, isolation from calibration/test data, and frozen-encoder mode restoration
after validation. Review-generation checks cover prediction integrity and embedded-text escaping.
They do not train or evaluate actual checkpoints on the test cohort.

After copying the remote `final-test/` outputs locally, build an offline review page with:

```sh
datasets/.venv/bin/python model-training/experiments/macedonian-sentiment/review_sentiment_results.py \
  --results model-training/outputs/macedonian-sentiment/final-test \
  --output model-training/outputs/macedonian-sentiment/error-review
```

The builder verifies prediction checksums, row alignment, and confusion matrices, then
creates `analysis.json` and a self-contained `review.html` in a new output directory. Open
the HTML directly to compare any two models, filter mistakes by class/confidence, and record
whether a reference label seems plausible, unclear, or questionable. Notes are stored in
the browser when supported and can be exported to CSV. They never relabel the frozen data
or alter the completed test results. Selected errors are diagnostic subsets, not representative
performance estimates. Any model changes motivated by this review require a separately
defined follow-up experiment.
