# Shared datasets

This folder owns data preparation and frozen datasets consumed by benchmarking and remote
fine-tuning. The `thesis-datasets` package uses Python 3.12 and has no model SDK, inference, or
training dependencies. Its optional `parquet` extra reads BoolQ's original files. Research depends
on this local package; model serving remains independent.

## Layout and identity

- `banking77/source/<source-revision>/`, `boolq/source/<source-revision>/`: verified official files.
- `macedonian-tweet-sentiment/source/<source-revision>/`: original MTEB Parquet splits,
  upstream dataset card, and source manifest with verified checksums and label counts.
- `yelp/`, `snips/`: existing source preparation and translation pipelines, raw files, source exports,
  translations, and private `.pipeline/` runtime state.
- `releases/<dataset-id>/<version>/`: immutable `manifest.json` and JSONL split files used by consumers.
- `src/thesis_datasets/`: shared source readers, publication, and verification.

Raw files, prepared exports, translations, and releases are ignored by Git. Code, tests, and tasks
are tracked. Transfer a complete release directory to the remote training machine, including its
manifest. Its full version ID is independent of its filesystem location.

The Macedonian tweet sentiment source is pinned to
`6ec78a685dad2e2b37af77264676e9dddfa7dc8e` from
`mteb/MacedonianTweetSentimentClassification`: 8,566 training tweets and 1,139 test tweets,
with labels `-1` (negative), `0` (neutral), and `1` (positive). Original files retain their text,
labels, row order, and upstream splits. The source audit confirms five exact texts shared by
training and test. The prepared release excludes those five training rows, preserves the
original test split, and allocates training/validation/calibration from the remaining pool.
This native Macedonian dataset requires no translation.

**Source revision** identifies the upstream commit. **Prepared version** is the SHA-256 of the
canonical manifest identity, including split checksums, schema, and preparation provenance.
Changing sampling, translations, records, targets, split assignment, or provenance produces a new
version. Publication is atomic and reproducible. An existing release is verified and reused;
modified releases fail rather than being repaired or overwritten. Consumers never follow `latest`.

Each UTF-8 JSONL record contains:

```json
{"id":"yelp-train-42","split":"train","inputs":{"text_en":"Original review","text_mk":"Превод"},"target":{"type":"score","level":3},"metadata":{"stars":4,"label":3}}
```

`inputs` contains only source fields available for model state. `target` is one of
`{"type":"choice","label":"intent"}`, `{"type":"noul","value":true}`, or
`{"type":"score","level":0}`. Score levels are zero-based and the manifest declares their
ordered descriptions. Original labels, source row positions, duplicate groups, translation
details, and review flags remain in `metadata`; they are not sent by the benchmark.

Yelp's existing 70/10/10/10 train/validation/calibration/test assignment is preserved. Original
test reviews stay in test. SNIPS retains its original `train`/`heldout` assignment; `heldout` came
from the upstream validation files, and original duplicate groups are preserved. Publication
does not resample, deduplicate, translate, or silently exclude flagged records. Translation
exports must be complete, match the original IDs, order, fields and splits, and identify the
same source checksum. Review flags remain available for inspection.

## Publish existing exports

Run `uv sync --locked` from `research/`, then return to the repository root for these commands.
The publisher only reads existing exports and creates releases; it makes no network requests.
Translation paths are explicit so a later retranslation cannot change the selected snapshot.

```sh
research/.venv/bin/thesis-datasets publish --dataset yelp \
  --source datasets/yelp/source/balanced-10000-seed-42-chars-40-3000 \
  --translation datasets/yelp/translated/anthropic--claude-haiku-5.5/f07358c57553/506abb5e65ef

research/.venv/bin/thesis-datasets publish --dataset snips \
  --source datasets/snips/source \
  --translation datasets/snips/translated/anthropic--claude-haiku-5.5/847620583733

research/.venv/bin/thesis-datasets publish --dataset banking77 \
  --revision 57ec275d8078af65b7731c2a98be812d844a6d6b

research/.venv/bin/thesis-datasets publish --dataset boolq \
  --revision 35b264d03638db9f4ce671b711558bf7ff0f80d5
```

Omit `--translation` to publish only original English Yelp/SNIPS inputs. Use `--root` to choose
a different release root, and `--cache-root` for official files at an explicitly chosen old root.
The command prints the release directory and full version ID. Source/translation paths above
describe the already prepared local exports; after cloning, restore them or transfer their frozen
release. For fresh official downloads, run the existing research `prepare` commands, which use
the shared source readers and also publish a release.

The migration made these snapshots available:

| Dataset | Prepared version | Splits and rows |
| --- | --- | --- |
| BANKING77 | `93a54246d2711c13dd8858802c523be746a74fe5f45b1bd3eadef5b0fef8a4b8` | train 10,003; test 3,080 |
| BoolQ | `fa3d7f52c73f211e8e6a0ffc8c76b273eb4392a135fa224f01c53b38d1792040` | train 9,427; validation 3,270 |
| Yelp bilingual | `9a58dbdebdc1d0bd32ef03f7db4ef3a8ac4a049d14a1335f12ae02d8f3129bf9` | train 7,000; validation 1,000; calibration 1,000; test 1,000 |
| SNIPS bilingual | `368e2749ab9afd3d2db9a36d0d5a9f9a5ccb3321a59edf5a7ac2e10d9334fd2c` | train 13,784; heldout 700 |
| Macedonian tweet sentiment | `e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc` | train 6,849; validation 856; calibration 856; test 1,139 |

## Prepare Macedonian sentiment for fine-tuning

The downloaded source lives under `macedonian-tweet-sentiment/source/<source-revision>/`.
Preparation reads this local snapshot without network or model calls. Run from `datasets/`:

```sh
uv sync --locked --extra parquet
uv run --locked --extra parquet thesis-datasets prepare \
  --dataset macedonian-tweet-sentiment --root . \
  --revision 6ec78a685dad2e2b37af77264676e9dddfa7dc8e --seed 42

uv run --locked --extra parquet thesis-datasets verify \
  --dataset macedonian-tweet-sentiment --root . \
  --version e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc
```

Each record retains its original text in `inputs.text_mk`, a Choice target of `negative`,
`neutral`, or `positive`, and metadata recording the original numeric label, source split,
one-based row number, and duplicate-group hash. IDs use the original split and row number;
they do not change when a training-pool record moves to validation or calibration.
The label order is negative, neutral, positive. Prompts and Laya-specific exports belong in
the training workflow, not these shared model-independent files.

Preparation groups texts using NFC Unicode normalization, case folding, and collapsed whitespace
solely for duplicate detection; the actual input text is never rewritten. All original test
records stay in their original order. Any training group present in test is excluded, and
training duplicates stay in one split. Contradictory training labels within a retained group
cause an error instead of a silent label choice. This is conservative duplicate detection,
not a fuzzy paraphrase or author/thread grouping check; the source has only text and labels.

The remaining pool is split 80/10/10 by class using seed 42 and SHA-256 ordering. Integer quotas
use largest-remainder rounding. Whole duplicate groups take precedence over exact quotas.
Rows retain their original source order inside each split. At the pinned revision there are
no within-split duplicate groups, and the class counts are:

| Split | Negative | Neutral | Positive | Total |
| --- | --- | --- | --- | --- |
| train | 3,738 | 1,024 | 2,087 | 6,849 |
| validation | 467 | 128 | 261 | 856 |
| calibration | 467 | 128 | 261 | 856 |
| test | 508 | 200 | 431 | 1,139 |

The manifest records source checksums, the preparation policy and seed, split checksums,
class counts, and IDs of the five excluded training rows. Repeating the same preparation
reuses the identical verified release; a different seed or policy creates a separate version.
Source files and existing releases are never overwritten. Transfer the entire release
directory, including `manifest.json`, to the remote machine. This step does not train models,
fit calibration parameters, or add benchmark configurations to `research/`.

## Use on a training machine

Install this package without the research project:

```sh
uv pip install ./datasets
thesis-datasets verify --root datasets --dataset yelp \
  --version 9a58dbdebdc1d0bd32ef03f7db4ef3a8ac4a049d14a1335f12ae02d8f3129bf9
```

Load the identical verified records and convert them to the format your training library needs:

```python
from pathlib import Path
from thesis_datasets import load_release

manifest, splits = load_release(
    Path("datasets"), "yelp",
    "9a58dbdebdc1d0bd32ef03f7db4ef3a8ac4a049d14a1335f12ae02d8f3129bf9",
)
train = splits["train"]
texts = [row["inputs"]["text_mk"] for row in train]
levels = [row["target"]["level"] for row in train]  # 0–4, equivalent to 1–5 stars
```

Record `manifest["version"]` in training provenance. Training-specific prompts, tokenization,
and export formats belong in the training project. Use validation for model selection,
calibration for calibration, and test for final evaluation. Existing remote training files are
not changed by this migration; confirm their IDs/text/targets/splits match before treating a
checkpoint as trained on this release.

## Verification and tests

Official source caches live in this project. The former `research/data/` folder, placeholder,
and compatibility links have been removed. Historical outputs and frozen configs are unchanged.
CLI commands default to `../datasets` when run from `research/`; `--data-dir` remains explicit
and supported.

```sh
cd datasets
uv sync --locked --extra parquet
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```
