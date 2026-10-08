# Jev graduate thesis

One repository for thesis notes, Python research code, and independent local model serving.

- [Thesis home](vault/Thesis%20Home.md): direction, use cases, and research notes.
- [Research setup](research/README.md): Python environment and code organization.
- [Local model serving](model-serving/README.md): a separate environment for serving Laya checkpoints.
- [Agent guide](AGENTS.md): collaboration and recording rules.

Open the repository root in your IDE. Read the Markdown documentation in `vault/`; relative links connect the notes and work in Markdown previews and repository browsers.

Notes, source code, experiment configurations, tests, and the dependency lockfile are versioned together. Credentials, downloaded datasets, and generated runs are ignored. Record dataset revisions and output locations in experiment notes so ignored artifacts remain traceable.

The research code includes pinned dataset adapters, structured System One and LLM clients,
resumable benchmarks, and evaluation reports. It can call Vercel or any TypeSafe-compatible HTTP
service. The separate model-serving project loads checkpoints; the two projects communicate only
over HTTP. Development runs are tracked separately from the final thesis evaluation.
