"""Select and preserve one BANKING77 cohort shared by every model."""

import random
from collections import defaultdict
from typing import Any, Sequence

from thesis_research.benchmark.types import CohortSpec
from thesis_research.datasets import Banking77Example


def select_banking77_cohort(
    examples: Sequence[Banking77Example],
    labels: Sequence[str],
    spec: CohortSpec,
) -> list[Banking77Example]:
    """Select all examples or a deterministic label-stratified subset.

    A stratified cohort first selects one example from every label, then
    continues round-robin across labels. The returned examples follow their
    original dataset order so every model receives the same stable sequence.
    """
    if spec.strategy == "all":
        return list(examples)
    if spec.size is None or spec.seed is None:
        raise ValueError(f"{spec.strategy} cohort requires both size and seed")
    if spec.size > len(examples):
        raise ValueError("Cohort size exceeds the configured dataset split")

    generator = random.Random(spec.seed)
    if spec.strategy == "random":
        chosen_ids = {example.id for example in generator.sample(list(examples), spec.size)}
        return [example for example in examples if example.id in chosen_ids]
    if spec.size < len(labels):
        raise ValueError(f"Stratified cohort size must be at least {len(labels)}")

    buckets: dict[str, list[Banking77Example]] = defaultdict(list)
    for example in examples:
        buckets[example.label].append(example)
    missing = [label for label in labels if not buckets[label]]
    if missing:
        raise ValueError(f"Dataset split has no examples for labels: {missing[:5]}")

    for label in labels:
        generator.shuffle(buckets[label])
    selected: list[Banking77Example] = []
    depth = 0
    while len(selected) < spec.size:
        added = False
        for label in labels:
            if depth < len(buckets[label]):
                selected.append(buckets[label][depth])
                added = True
                if len(selected) == spec.size:
                    break
        if not added:
            raise ValueError("Could not fill the requested stratified cohort")
        depth += 1
    order = {example.id: index for index, example in enumerate(examples)}
    return sorted(selected, key=lambda example: order[example.id])


def cohort_record(
    examples: Sequence[Banking77Example],
    *,
    dataset_revision: str,
    split: str,
    spec: CohortSpec,
) -> dict[str, Any]:
    """Serialize a frozen cohort with references and original messages."""
    return {
        "dataset_revision": dataset_revision,
        "split": split,
        "selection": spec.as_dict(),
        "count": len(examples),
        "examples": [
            {"id": example.id, "text": example.text, "reference_label": example.label}
            for example in examples
        ],
    }


def examples_from_cohort(
    value: object,
    dataset_examples: Sequence[Banking77Example],
    *,
    dataset_revision: str,
    split: str,
) -> list[Banking77Example]:
    """Validate a saved cohort against the pinned prepared dataset."""
    if not isinstance(value, dict):
        raise ValueError("Saved cohort must be a JSON object")
    if value.get("dataset_revision") != dataset_revision or value.get("split") != split:
        raise ValueError("Saved cohort does not match the benchmark dataset")
    records = value.get("examples")
    if not isinstance(records, list) or value.get("count") != len(records) or not records:
        raise ValueError("Saved cohort has an invalid example list")
    by_id = {example.id: example for example in dataset_examples}
    selected: list[Banking77Example] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Saved cohort contains a malformed example")
        example_id = record["id"]
        if example_id in seen or example_id not in by_id:
            raise ValueError(f"Saved cohort contains an invalid ID: {example_id}")
        example = by_id[example_id]
        if record.get("text") != example.text or record.get("reference_label") != example.label:
            raise ValueError(f"Saved cohort differs from the dataset for ID: {example_id}")
        seen.add(example_id)
        selected.append(example)
    return selected
