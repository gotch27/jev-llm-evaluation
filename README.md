# Jev graduate thesis

One repository for thesis notes, shared datasets, Python research code, and independent local model serving.

- [Thesis home](vault/Thesis%20Home.md): direction, use cases, and research notes.
- [Research setup](research/README.md): Python environment and code organization.
- [Shared datasets](datasets/README.md): frozen versions used by benchmarking and remote fine-tuning.
- [Local model serving](model-serving/README.md): a separate environment for serving Laya checkpoints.
- [Model training workspace](model-training/README.md): index of experiment-specific training work.
- [Macedonian sentiment experiment](model-training/experiments/macedonian-sentiment/README.md): its scripts, data, checkpoints, results, and Mac inference.
- [Experiment summary](model-training/experiments/macedonian-sentiment/SUMMARY.md): the completed fine-tuning comparison and limitations.
- [Agent guide](AGENTS.md): collaboration and recording rules.

Open the repository root in your IDE. Read the Markdown documentation in `vault/`; relative links connect the notes and work in Markdown previews and repository browsers.

Notes, source code, experiment configurations, tests, and the dependency lockfile are versioned together. Credentials, downloaded datasets, and generated runs are ignored. Record dataset revisions and output locations in experiment notes so ignored artifacts remain traceable.

Data preparation and frozen JSONL releases live in `datasets/`. Each release records the original
source revision, exact prepared version, split checksums, and target schema. Training and research
consume the same releases; model instructions and training exports stay in their respective projects.

The research code includes pinned dataset adapters, structured System One and LLM clients,
resumable benchmarks, and evaluation reports. It can call Vercel or any TypeSafe-compatible HTTP
service. The separate model-serving project loads checkpoints; the two projects communicate only
over HTTP. Development runs are tracked separately from the final thesis evaluation.

## Training experiments

Training code and documentation live in named experiment folders under `model-training/experiments/`.
The current experiment is Macedonian tweet sentiment classification. Its scripts, experiment summary,
worked metric examples, and reproducibility commands are specific to that dataset and protocol;
the workspace is not a general-purpose training framework. Generated results remain separately
under `model-training/outputs/`.
