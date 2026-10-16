# Shared model clients

`thesis-model-clients` is the Python 3.12 model-calling library used by the thesis. Both the
benchmarking application and independent use cases import `thesis_model_clients`. It provides
hosted System One and LLM clients through Vercel AI Gateway, plus TypeSafe-compatible HTTP
clients for independently served models.

The library accepts JSON state and TypeSafe `Choice`, `Noul`, or `Score` questions. It returns
typed answers, per-question errors, routing information, latency, usage, cost, and raw call
records. Dataset selection, labels, benchmark plans, metrics, reports, and run directories belong
to callers. Importing the library does not load credentials or start model calls.

## Setup and use from another folder

From the repository root, prepare the library's own environment:

```sh
uv sync --project model-clients --locked
```

For a separate uv project in a sibling folder, add the local packages from that folder:

```sh
uv add --editable ../model-clients ../datasets
```

For a more deeply nested project, adjust those two paths. uv records them in that project's
`pyproject.toml`; commit its updated lockfile with the use case. The shared datasets package
supplies the verified image asset format and Pillow decoding. No training packages or parquet
extra are required by these clients. Research is not a dependency.

With the library installed in a project's environment, scripts in any of its folders can use
the same imports. They do not need `sys.path` changes or a benchmark TOML file.
Update existing client imports from `thesis_research.clients` to `thesis_model_clients`, and
image helper imports from `thesis_research.images` to `thesis_model_clients.images`.

Hosted clients read `AI_GATEWAY_API_KEY` from the process environment. A calling application
can explicitly load its own `.env` file; this library does not search for one. Self-hosted
clients use only an explicitly configured authentication variable, when needed.

```python
import asyncio

from thesis_model_clients import create_decision_client
from typesafe_sdk import Choice


async def main() -> None:
    questions = {
        "intent": Choice(
            instructions="Which intent best matches the message?",
            criteria={"card_arrival": None, "change_pin": None},
        )
    }
    async with create_decision_client(
        "system_one", "typesafe-ai/jev-1.14.0", "typesafe-ai", max_concurrency=1
    ) as client:
        result = await client.evaluate({"message": "When will my card arrive?"}, questions)
        print(result.answers)
        print(result.errors)
        print(result.usage.cost_usd)


asyncio.run(main())
```

A complete example prints a JSON record including diagnostics. With the credential in your
environment, run it from the repository root:

```sh
uv run --project model-clients model-clients/examples/classify_message.py
```

## Client selection and results

`create_decision_client(backend, model, provider, max_concurrency=...)` selects the same concrete
clients used by research. You can also instantiate the classes directly:

| Client | Endpoint | Settings |
| --- | --- | --- |
| `VercelSystemOneDecisionClient` | Vercel TypeSafe API | Exact model ID and pinned provider |
| `VercelLLMDecisionClient` | Vercel chat completions | Exact model ID, pinned provider, explicit `output_mode` |
| `TypeSafeHTTPDecisionClient` | Configured TypeSafe HTTP API root | Model ID, provider label, `base_url`, optional `api_key_env` |

For an LLM, choose `backend="llm"` and set `llm_output_mode="discrete"` or `"probabilities"`
in the factory. Native structured output is required. Optional `reasoning_effort` stays explicit.
For example:

```python
client = create_decision_client(
    "llm",
    "google/gemini-2.5-flash-lite",
    "google",
    max_concurrency=1,
    llm_output_mode="discrete",
)
```

For a local server, use `backend="system_one"`, `transport="self_hosted"`, and a `base_url`.
An optional `timeout_seconds` sets both the HTTP timeout and retry budget; the default is 120
seconds. Cloud credentials and Gateway routing options are not sent to local servers.

Always use clients as async context managers or close them with `await client.aclose()`.
Reuse a client across calls to reuse connections and enforce its shared concurrency bound.
System One sends all supplied questions in one request. LLMs send isolated requests per question.
Provider failures become entries in `result.errors`, allowing other answers to be retained.
Invalid local configuration or image inputs raise before sending the affected request.

`DecisionResult`, `DecisionError`, `CallRecord`, and `UsageTotals` are public contracts.
`decision_result_to_dict(result, answer_output_mode=...)` produces the same JSON-safe record
used by benchmarks. The output mode is `probabilities` by default; use `discrete` for discrete
LLM calls or `native_probabilities` for native System One records. Raw provider diagnostics
remain available in every mode. A missing cost or usage field stays `None`, while a reported
zero stays zero. Total latency includes queue time; individual call latency starts after
acquiring a concurrency slot.

Hosted clients preserve existing defaults: at most two transient retries, a 30-second retry
budget, and a 60-second HTTP timeout. Malformed LLM answers are not retried. Provider pinning,
question isolation, native probabilities, refusals, and raw response handling are shared with
the benchmark application. Supported hosted routes and recorded connectivity evidence are
listed in [the research documentation](../research/README.md#structured-model-clients).

## Images without a benchmark dataset

`load_image(Path(...))` verifies a local still PNG/JPEG and returns the shared `ImageAsset`.
It checks the existing size limits without resizing, rotating, or re-encoding the file.
It resolves the file path when loading, so the attachment survives a later directory change.
Pass the asset under `images` using the name of a top-level state field:

```python
from pathlib import Path

from thesis_model_clients import load_image
from typesafe_sdk import Noul

photo = load_image(Path("photo.png"))
result = await client.evaluate(
    {"photo": {"type": "image", **photo.metadata}},
    {"damaged": Noul(instructions="Does the product in `photo` show visible damage?")},
    images={"photo": photo},
)
```

Use this inside an async context with an image-enabled client. `image_capability(backend,
model, provider, transport="vercel")` exposes the same exact-route checks used by benchmarks.
Clients also apply those checks themselves. Unverified routes, including self-hosted image
routes, raise before inference. The model list, shared size limits, and live versus offline
transport evidence remain documented in [the image guide](../research/README.md#images-with-the-existing-question-types).
Image benchmarks additionally require frozen dataset releases and validate all participants
before running. Box remains unsupported.

## Layout and checks

- `src/thesis_model_clients/`: contracts, factory, HTTP clients, images, and result serialization.
- `examples/classify_message.py`: a direct call without benchmark infrastructure.
- `tests/`: offline transport, retry, concurrency, image, serialization, and independence checks.

From `model-clients/`:

```sh
uv run --locked pytest
uv run ruff check .
uv run ruff format --check .
uv build
```
