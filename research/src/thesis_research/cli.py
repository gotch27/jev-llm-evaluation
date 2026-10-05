"""Prepare data, inspect examples, and run coordinated model benchmarks."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.config import load_task_config
from thesis_research.datasets import load_prepared_dataset, prepare_dataset, reference_class
from thesis_research.terminal import BenchmarkTerminal

DEFAULT_BENCHMARK_OUTPUT = Path("outputs/benchmarks")


def main() -> None:
    """Parse command-line arguments and dispatch to the application functions."""
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "benchmark":
            if args.retry_errors and args.resume is None:
                raise ValueError("--retry-errors requires --resume")
            with BenchmarkTerminal() as terminal:
                run = (
                    asyncio.run(
                        resume_benchmark(
                            args.resume,
                            args.data_dir,
                            observer=terminal,
                            runner_location=args.runner_location,
                            retry_errors=args.retry_errors,
                        )
                    )
                    if args.resume is not None
                    else asyncio.run(
                        run_benchmark(
                            args.plan,
                            args.data_dir,
                            args.output_dir,
                            observer=terminal,
                            runner_location=args.runner_location,
                        )
                    )
                )
            print(run)
            return

        config = load_task_config(args.config)
        if args.command == "prepare":
            print(
                prepare_dataset(
                    args.data_dir,
                    config.dataset.id,
                    config.dataset.revision,
                )
            )
            return
        _inspect_data(
            args.data_dir,
            config.dataset.id,
            config.dataset.revision,
            args.split,
            args.limit,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Download and verify the configured dataset")
    _add_shared_arguments(prepare)

    inspect = commands.add_parser("inspect", help="Print references and examples from one split")
    _add_shared_arguments(inspect)
    inspect.add_argument("--limit", type=int, default=5)
    inspect.add_argument("--split", default="train")

    benchmark = commands.add_parser("benchmark", help="Run or resume a coordinated benchmark")
    source = benchmark.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument("--resume", type=Path, metavar="RUN_DIRECTORY")
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))
    benchmark.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT)
    benchmark.add_argument(
        "--runner-location",
        help="Recorded location label such as local-mac-oslo or aws-us-east-1",
    )
    benchmark.add_argument(
        "--retry-errors",
        action="store_true",
        help="On resume, archive failed attempts and retry their example IDs",
    )
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))


def _inspect_data(
    data: Path,
    dataset_id: str,
    revision: str,
    split: str,
    limit: int,
) -> None:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    prepared = load_prepared_dataset(data, dataset_id, revision, split)
    distribution: dict[str, int] = {}
    for example in prepared.examples:
        key = reference_class(example.reference)
        distribution[key] = distribution.get(key, 0) + 1
    print(
        f"Dataset: {dataset_id}; split: {split}; examples: {len(prepared.examples)}; "
        f"reference type: {prepared.reference_schema.question_type}"
    )
    print(json.dumps(distribution, ensure_ascii=False))
    for example in prepared.examples[:limit]:
        print(json.dumps(asdict(example), ensure_ascii=False))


if __name__ == "__main__":
    main()
