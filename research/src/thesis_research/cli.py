"""Prepare data, run structured predictions, and evaluate saved predictions."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from thesis_research.config import load_experiment_config
from thesis_research.datasets import load_prepared_banking77, prepare_banking77
from thesis_research.evaluation import run_classification_evaluation
from thesis_research.prediction_run import run_prediction_experiment

DEFAULT_CONFIG = Path("experiments/intent_classification/banking77.toml")


def main() -> None:
    """Parse command-line arguments and dispatch to the application functions."""
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "predict":
            run = asyncio.run(
                run_prediction_experiment(
                    args.config,
                    args.data_dir,
                    args.output_dir,
                    backend=args.backend,
                    model=args.model,
                    provider=args.provider,
                    max_concurrency=args.max_concurrency,
                    limit=args.limit,
                )
            )
            print(run)
            return
        if args.command == "evaluate":
            run = run_classification_evaluation(
                args.config,
                args.predictions,
                args.data_dir,
                args.output_dir,
            )
            print(run)
            return

        config = load_experiment_config(args.config)
        if args.command == "prepare":
            print(prepare_banking77(args.data_dir, config.dataset_revision))
            return
        _inspect_training_data(args.data_dir, config.dataset_revision, args.limit)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Download and verify BANKING77")
    _add_shared_arguments(prepare)

    inspect = commands.add_parser("inspect", help="Print training labels and examples")
    _add_shared_arguments(inspect)
    inspect.add_argument("--limit", type=int, default=5)

    predict = commands.add_parser("predict", help="Generate saved model predictions")
    _add_shared_arguments(predict)
    predict.add_argument("--backend", choices=("jev", "llm"), required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--provider")
    predict.add_argument("--max-concurrency", type=int, default=5)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))

    evaluate = commands.add_parser("evaluate", help="Score a saved prediction file")
    _add_shared_arguments(evaluate)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))


def _inspect_training_data(data: Path, revision: str, limit: int) -> None:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    labels, examples, _ = load_prepared_banking77(data, revision, "train")
    print(f"Training examples: {len(examples)}; labels: {len(labels)}")
    print(json.dumps(labels, ensure_ascii=False))
    for example in examples[:limit]:
        print(json.dumps(vars(example), ensure_ascii=False))


if __name__ == "__main__":
    main()
