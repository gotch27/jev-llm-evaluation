# Local model serving

This Python 3.12 uv project loads one Laya checkpoint and exposes TypeSafe-compatible
HTTP inference. It is independent of research/: no benchmark imports, Vercel calls,
or research credentials. Fine-tuning is a separate workflow.

## Setup and base checkpoint

Run from model-serving/:

```bash
uv sync --locked
uv run serve-laya --checkpoint convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --model-id laya-base --device mps --max-len 8192 --head-max-len 8192
```

The pinned base weights download into ignored models/. The server preloads before
binding to 127.0.0.1:8000. Use --device cpu for explicit CPU inference and --port
to run another checkpoint on another port. One worker serializes inference.

## Exported checkpoint

Place a complete Laya export in models/laya-banking77-ft-v1/ (weights, encoder,
tokenizer, and rl_agent_config.json), then run:

```bash
uv run serve-laya --checkpoint ./models/laya-banking77-ft-v1 \
  --model-id laya-banking77-ft-v1 --device mps --port 8001 \
  --max-len 8192 --head-max-len 8192
```

--model-id is an explicit request alias. Change it for every changed checkpoint.
Restart with the original checkpoint and settings when resuming a benchmark.
The raw response records a fingerprint of the checkpoint files for auditing.

## API

POST /v1/systemone accepts the TypeSafe state/questions/model payload. The model
must equal the configured alias. Questions and states pass to the loaded agent
without changing options or introducing automatic model routing. Choice, Noul,
and Score answers and usage are returned in TypeSafe-compatible form. GET /health
reports readiness and runtime provenance; the benchmark does not require this endpoint.

Runtime metadata includes the checkpoint source/revision/fingerprint, actual device
and dtype, package versions, token budgets, inference duration, and CPU fallback events.
No monetary cost is manufactured for local computation. Token limits, option loss
or collapse, and state truncation become explicit errors with raw diagnostics.

The base checkpoint defaults to a small token window. The 8192 overrides above
are intended for the full 77-option BANKING77 checks. They do not guarantee a
particular model accuracy; questions that still exceed the budget fail explicitly.

The pinned base export contains a Choice calibration temperature outside Laya's
supported range. Laya 0.3.29 warns and clamps it during loading. Treat the affected
confidence values as uncalibrated; keep the checkpoint and runtime versions fixed
when comparing results. The loader uses the published weights without training.

## Benchmarking

Leave this process running. In a second terminal, from research/:

```bash
uv run thesis-research benchmark --plan experiments/boolean_question_answering/benchmarks/boolq-local-smoke.toml --open-report
```

The server requires no authentication and binds to loopback by default. A protected
deployment can place it behind an authenticated reverse proxy; the benchmark supports
an explicitly named API-key environment variable for such endpoints.

## Checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

Offline tests inject a fake agent and do not download weights. Real local validation
records belong in ignored outputs/. No model is described as fine-tuned merely because
its base weights were loaded from a local directory.

Both wheel and source distributions explicitly exclude checkpoints and generated
artifacts; only source, tests, documentation, project metadata, and the lockfile ship.
