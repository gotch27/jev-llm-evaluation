# Yelp Review Full → Macedonian

The default sample contains **10,000 unique complete reviews: 2,000 for each of five ratings**. The dataset and virtual environment are prepared on this Mac. No paid translation is launched by preparation or tests.

## Run

```sh
cd /Users/gorazdfilipovski/Documents/graduate-thesis/datasets/yelp
.venv/bin/python pipeline.py translate --total 10000
```

Default: **`anthropic/claude-haiku-5.5` through Vercel AI Gateway**. The script checks the current Gateway catalog for availability and structured-output support. Credentials come from `AI_GATEWAY_API_KEY`, then local `yelp/.env`, then your existing `../snips/.env`. No key is copied or logged. To create a separate private key file, use `.venv/bin/python pipeline.py configure`.

Optional small quality check before the full run:

```sh
.venv/bin/python pipeline.py translate --limit 25
```

This translates five training reviews per rating. Inspect their translations, then run the full command above; cached translations are reused. Pilot exports correctly remain partial against the full 10,000-row dataset. Haiku is the existing inexpensive default, not a demonstrated quality winner for this corpus.

Rerun the same command to resume. Completed batches are stored in SQLite by source-text hash and model/prompt/settings profile. Ctrl+C stops scheduling, finishes in-flight translation requests and writes partial outputs. Two reviews per request and three concurrent workers are the defaults. Transient network/rate-limit errors and malformed or unfinished responses retry up to five attempts with exponential backoff; `Retry-After` is respected. Authentication/payment errors stop scheduling further requests. Request IDs and exact translation IDs are checked before anything is cached. All selected reviews are translated in full, without truncation.

Retries can incur charges, and an interrupted/timed-out provider request may have been billed without a saved response. Resume avoids paying again for successfully cached reviews; it cannot guarantee exactly-once billing. Logs show batch progress and errors, without credentials or review text.

## Outputs

Source files are under `source/balanced-10000-seed-42-chars-40-3000/`:

| Split | Total | Per rating | Original Yelp source |
|---|---:|---:|---|
| train | 7,000 | 1,400 | train |
| validation | 1,000 | 200 | train |
| calibration | 1,000 | 200 | train |
| test | 1,000 | 200 | test |

Translated files are under `translated/<model>/<profile>/<sample>/`. `translated/latest.json` points to the last exported directory. Once complete, it contains `all.jsonl` and the four split files. Before completion, files have `.partial.jsonl` suffixes wherever reviews are missing. Existing complete split files stay available if other splits are still pending.

Each JSONL row contains:

```json
{"id":"yelp-train-123","text_en":"Original English review…","text_mk":"Македонски превод…","label":2,"stars":3,"split":"train"}
```

This abbreviated illustration omits additional provenance, duplicate hashes, model IDs, response ID, timestamp and review flags included in actual rows. **`label` is 0–4; `stars` is 1–5.** The translator sees only review text and temporary request IDs, never the gold rating or split. Ratings remain unchanged. Paragraph markers stored in the original data as literal backslash-n are decoded, and boundary whitespace is trimmed; raw source files remain available.

- `review.jsonl`: automatic flags for number changes, unexpected Cyrillic letters, no Cyrillic, unchanged text and suspicious output length. Flags are suggestions for manual review, not proof of an error. Inspect some unflagged examples too; these checks cannot certify sentiment preservation or translation quality.
- `pending.jsonl`: IDs, splits and ratings still missing translations.
- `summary.json`: progress, prompt/settings, source manifest, model provenance and reported token usage. Cost is a **cumulative profile-level base list-price estimate** including rejected responses, not an invoice; unknown usage, discounts and surcharges are excluded.
- `logs/pipeline.log`: run and retry logs.

```sh
.venv/bin/python pipeline.py status --total 10000
.venv/bin/python pipeline.py export --total 10000
```

`status` also exports current cached progress. Once the source and model catalog are saved, status/export work offline.

## Review flagged translations in a browser

Open `review/flagged-reviews.html` for an offline, self-contained review page. It includes every flagged row, English/Macedonian text side by side, filters, search, number highlights, review decisions, notes and suggested corrections. Nothing is sent to a service or written back to your dataset. Original translations remain visible even when a suggested correction is saved.

Decisions save in browser storage when available. Use **Export review decisions** to download a JSON backup and **Import saved decisions** to continue in another browser or after storage is cleared. Imports must match the exact source snapshot. Always export before closing; browser storage is not a permanent backup. Review corrections require a separate explicit step before being applied to the dataset.

Rebuild the HTML after exporting new translation results:

```sh
.venv/bin/python build_review.py
```

The HTML is a snapshot of `translated/latest.json` and its `review.jsonl`. It does not refresh automatically while translation runs. All flagged reviews are embedded locally; pagination changes only the visible list.

### Retry flags other than changed numbers

```sh
.venv/bin/python pipeline.py retranslate --exclude-flag check_numbers
# Preview the selection without sending paid requests:
.venv/bin/python pipeline.py retranslate --exclude-flag check_numbers --dry-run
```

This skips reviews flagged **only** for numbers. Reviews with another flag still qualify even when also flagged for numbers. Selection is frozen from the original export, and a more explicit prompt asks for complete reviews and standard Macedonian. Retranslations use a separate prompt/cache profile; completed replacements are skipped when resuming the same command. It does not repeatedly retry successful replacements just because they still have an automatic flag.

Original translations and cache entries remain intact. A separate combined export contains the full dataset, using new translations for selected reviews that finish and original translations elsewhere. `summary.json` reports dataset completeness separately from retranslation completeness; `retranslation-pending.jsonl` lists replacements still missing. `translated/latest.json` points to the combined export, and `status`/`export` respect that latest retranslation job for the same sample/model. Usage for the original pass and the retry pass is reported separately.

A retry may still raise flags, especially for preserved foreign names; inspect the results. Export any decisions from the old HTML before rebuilding it, because a changed translation snapshot has its own review state:

```sh
.venv/bin/python build_review.py
```

## Reproducibility and source

[Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) is the five-class review-rating benchmark constructed from the Yelp Dataset Challenge 2015. The original data has 650,000 training reviews (130,000 per rating) and 50,000 test reviews (10,000 per rating). Source revision is pinned to **`c1f9ee939b7d05667af864ee1cb066393154bf85`**. Both Parquet files (approximately 323 MB combined) are checked against the SHA-256 values published in the pinned repository metadata. The dataset card and metadata are saved in `raw/`. Its license field is `other`; consult the [linked Yelp dataset agreement](https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information) for the source terms.

Preparation scans both original files. Reviews equal after ignoring case and whitespace belong to one duplicate group. Groups with contradictory ratings are excluded. Every original test group is removed from the training/validation/calibration candidates, even if that test review will not be in our selected 1,000. Remaining groups are sampled by seeded SHA-256 rank per rating, keeping exactly one row from each group. The source manifest records counts, exclusions, raw hashes, sampling rules and split assignments. More complex near duplicates or shared authors/businesses may remain; source files expose only text and rating.

New source manifests record `preparation_version: 2`, which also trims boundary tabs and
paragraph breaks before duplicate matching. Existing saved source samples are reused with
their original manifest rather than silently resampled. To apply version 2 to an older sample,
prepare under a separate `--root` directory and review the new manifest before using it.

Defaults retain full texts of **40–3,000 English characters**. Longer and very short reviews are excluded rather than shortened. There are ample candidates per rating; preparation fails rather than silently producing an undersized sample. Character filtering favors shorter reviews and does not guarantee that Macedonian text fits Laya's token limit. Check its actual tokenizer before training. Translated English reviews are not naturally authored Macedonian, and dataset overlap with Laya has not been established.

For a different balanced size, e.g. `translate --total 20000`, the total must be a multiple of 50 so every split stays balanced. Changing size, seed or length limits creates a separate source sample/output directory; translations of overlapping identical texts reuse the existing profile cache.

## Installation and verification

Python 3.10+. The translation engine is shared with the adjacent `snips/pipeline.py`; keep the folders together. This pipeline does not modify the SNIPS translation engine. DuckDB runs locally with a 1 GB memory limit and two query threads; no GPU is needed for translation.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pipeline.py prepare --total 10000
.venv/bin/python -m unittest -v
```

Offline tests use synthetic provider responses and temporary Parquet files; they never send paid translation requests. This workflow prepares/ translates data only and does not start model training.
