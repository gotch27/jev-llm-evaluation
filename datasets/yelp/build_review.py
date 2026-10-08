#!/usr/bin/env python3
"""Build a self-contained, offline HTML page from the latest flagged Yelp export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(root=ROOT, source=None):
    root = Path(root).resolve()
    if source is None:
        source = Path(json.loads((root / "translated/latest.json").read_text())["directory"])
    source = Path(source).resolve()
    summary = json.loads((source / "summary.json").read_text())
    # JSON Lines use newline delimiters; splitlines() would also split valid U+2028/U+2029
    # characters inside a review string before the HTML escaping below can protect them.
    with (source / "review.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    fields = ("id", "split", "stars", "text_en", "text_mk", "review_flags", "duplicate_group")
    records = [{key: row[key] for key in fields} for row in rows]
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate source IDs in flagged export")
    fingerprint = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    payload = {"records": records, "fingerprint": fingerprint, "profile": summary["profile"],
               "source_sha256": summary["source_manifest"]["source_sha256"],
               "updated_at": summary["updated_at"], "translated_rows": summary["translated_rows"],
               "missing_rows": summary["missing_rows"], "model": summary["model"],
               "source_directory": str(source),
               "flag_counts": dict(Counter(flag for row in rows for flag in row["review_flags"]))}
    # JSON is embedded in a script-data element: never allow review text to close the element.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for literal, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                            ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        data = data.replace(literal, escape)
    template = (root / "review/template.html").read_text()
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("HTML template must contain exactly one data placeholder")
    output = root / "review/flagged-reviews.html"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(template.replace("__REVIEW_DATA__", data), encoding="utf-8")
    temporary.replace(output)
    print(f"Built {output}: {len(rows):,} flagged reviews, {summary['translated_rows']:,} translations")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Export directory; defaults to translated/latest.json")
    args = parser.parse_args()
    build(source=args.source)
