"""Publish and verify shared data snapshots without model calls."""

import argparse
import json
from pathlib import Path

from thesis_datasets.publish import publish_official_dataset, publish_text_dataset
from thesis_datasets.releases import load_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, publish, and verify shared datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare training splits from downloaded sources")
    prepare.add_argument("--dataset", choices=["macedonian-tweet-sentiment"], required=True)
    prepare.add_argument("--root", type=Path, default=Path("datasets"))
    prepare.add_argument(
        "--revision", help="Pinned source commit; defaults to the verified snapshot"
    )
    prepare.add_argument("--seed", type=int, default=42)
    publish = commands.add_parser("publish", help="Freeze existing prepared files; no downloads")
    publish.add_argument(
        "--dataset", choices=["banking77", "boolq", "yelp", "snips"], required=True
    )
    publish.add_argument("--root", type=Path, default=Path("datasets"))
    publish.add_argument("--source", type=Path, help="Yelp/SNIPS source export directory")
    publish.add_argument("--translation", type=Path, help="Explicit complete translation directory")
    publish.add_argument("--cache-root", type=Path, help="BANKING77/BoolQ source cache root")
    publish.add_argument("--revision", help="BANKING77/BoolQ exact source commit")
    verify = commands.add_parser("verify", help="Verify all records and hashes in a release")
    verify.add_argument("--root", type=Path, default=Path("datasets"))
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from thesis_datasets.sources.macedonian_sentiment import (
                REVISION,
                prepare_macedonian_sentiment,
            )

            directory = prepare_macedonian_sentiment(
                args.root, args.revision or REVISION, seed=args.seed
            )
            manifest, splits = load_release(args.root, args.dataset, directory.name)
            print(
                json.dumps(
                    {
                        "directory": str(directory.resolve()),
                        "dataset": args.dataset,
                        "version": manifest["version"],
                        "splits": {split: len(rows) for split, rows in splits.items()},
                        "label_counts": manifest["provenance"]["audit"]["label_counts"],
                        "excluded_training_rows": len(
                            manifest["provenance"]["audit"]["excluded_training_rows"]
                        ),
                    },
                    indent=2,
                )
            )
            return
        if args.command == "verify":
            manifest, splits = load_release(args.root, args.dataset, args.version)
            print(
                json.dumps(
                    {
                        "version": manifest["version"],
                        "splits": {split: len(rows) for split, rows in splits.items()},
                    },
                    indent=2,
                )
            )
            return
        if args.dataset in {"yelp", "snips"}:
            if args.source is None or args.cache_root is not None or args.revision is not None:
                raise ValueError("Yelp/SNIPS require --source; revision is read from its manifest")
            directory = publish_text_dataset(args.root, args.dataset, args.source, args.translation)
        else:
            if not args.revision or args.source is not None or args.translation is not None:
                raise ValueError("BANKING77/BoolQ require --revision and optionally --cache-root")
            directory = publish_official_dataset(
                args.root, args.dataset, args.revision, args.cache_root or args.root
            )
        print(
            json.dumps(
                {"directory": str(directory), "dataset": args.dataset, "version": directory.name},
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
