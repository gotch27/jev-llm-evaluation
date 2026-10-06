"""Prepare data, inspect examples, run benchmarks and view saved reports."""

import argparse
import asyncio
import json
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from thesis_research.benchmark import resume_benchmark, run_benchmark
from thesis_research.config import load_task_config
from thesis_research.datasets import load_prepared_dataset, prepare_dataset, reference_class
from thesis_research.reporting import generate_dashboard
from thesis_research.terminal import BenchmarkTerminal

DEFAULT_BENCHMARK_OUTPUT = Path("outputs/benchmarks")


def main() -> None:
    """Parse command-line arguments and dispatch to the application functions."""
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "report":
            output = args.output_dir or Path("outputs/reports") / args.run.resolve().name
            if output.resolve().is_relative_to(args.run.resolve()):
                raise ValueError("Report exports must be outside the source benchmark directory")
            dashboard = generate_dashboard(args.run, output / "index.html")
            terminal = BenchmarkTerminal()
            terminal.show_results(args.run)
            terminal.visualization_completed(dashboard)
            if args.open_report:
                _open_dashboard(dashboard)
            print(dashboard)
            return
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
            if args.open_report:
                _open_dashboard(run / "report/index.html")
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
    benchmark.add_argument(
        "--open-report", action="store_true", help="Open the generated dashboard in your browser"
    )
    report = commands.add_parser("report", help="View a completed run without making model calls")
    report.add_argument("run", type=Path, metavar="RUN_DIRECTORY")
    report.add_argument("--output-dir", type=Path, help="Export directory outside the source run")
    report.add_argument("--open-report", action="store_true")
    return parser


def _open_dashboard(path: Path) -> None:
    """Open a local report when requested, preserving results if no browser is available."""
    try:
        opened = path.is_file() and webbrowser.open(path.resolve().as_uri())
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        print(
            f"Could not open dashboard automatically. Saved results: {path}",
            file=sys.stderr,
        )


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
