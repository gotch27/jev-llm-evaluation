# Model training workspace

This folder organizes individual training experiments. It is not a general-purpose training
framework. Currently, the only experiment maintained here is **Macedonian tweet sentiment
classification with Laya**.

## Layout

```text
model-training/
├── README.md                              Workspace index: this file
├── experiments/
│   └── macedonian-sentiment/
│       ├── README.md                      Dataset-specific setup and artifact guide
│       ├── SUMMARY.md                     Results and limitations of that experiment
│       ├── METRICS.md                      Worked examples using that experiment's results
│       ├── questions.json                 Its fixed question and sentiment options
│       ├── train_sentiment.py
│       ├── calibrate_sentiment.py
│       ├── evaluate_sentiment_test.py
│       ├── review_sentiment_results.py
│       └── tests/
└── outputs/
    └── macedonian-sentiment/               Local results/reviews; ignored by Git
```

## Current experiment

- [Experiment guide](experiments/macedonian-sentiment/README.md): files, data preparation,
  selected checkpoints, local/remote artifact locations, inference, and execution commands.
- [Experiment summary](experiments/macedonian-sentiment/SUMMARY.md): method, findings,
  limitations, and proposed follow-up work for this experiment.
- [Worked metrics](experiments/macedonian-sentiment/METRICS.md): explanation using the
  Macedonian sentiment confusion matrices and confidence results.

Those documents describe the named experiment. Their dataset versions, labels, sample
sizes, model selections, settings, and results are not defaults or findings for other tasks.

## Adding another experiment

Give a new dataset/task its own folder under `experiments/`, with its own README, task
definition, scripts/configuration, and documentation. Put its generated artifacts under a
separate `outputs/<experiment-name>/` directory. Shared datasets remain in `../datasets/`;
actual local inference checkpoints belong in `../model-serving/models/`.

Extract common code only when multiple experiments need it. There is no shared training
package or unused framework scaffold in this workspace.

The experiment files were initially moved together without changing the four scripts' contents.
Subsequent code review strengthened input checks and added regression tests. Remote scripts,
checkpoint locations, dataset releases, and completed-run artifacts remain unchanged; their
recorded script hashes identify the code that actually ran. Use the current local source
paths for future transfers.

Scripts and documentation are versioned in Git; outputs and model weights require a
separate artifact backup.
