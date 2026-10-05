"""Select and preserve one labeled cohort shared by every model."""

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from thesis_research.benchmark.types import CohortSpec
from thesis_research.datasets import (
    ChoiceReference,
    DatasetExample,
    NoulReference,
    Reference,
    ReferenceSchema,
    ScoreReference,
    reference_class,
)
from thesis_research.tasks import StructuredTask, class_labels


def select_cohort(
    examples: Sequence[DatasetExample],
    schema: ReferenceSchema,
    spec: CohortSpec,
) -> list[DatasetExample]:
    """Select all rows or a deterministic random/stratified subset."""
    if spec.strategy == "all":
        return list(examples)
    if spec.size is None or spec.seed is None:
        raise ValueError(f"{spec.strategy} cohort requires both size and seed")
    if spec.size > len(examples):
        raise ValueError("Cohort size exceeds the configured dataset split")

    generator = random.Random(spec.seed)
    if spec.strategy == "random":
        selected_ids = {example.id for example in generator.sample(list(examples), spec.size)}
        return [example for example in examples if example.id in selected_ids]

    strata = class_labels(schema)
    if spec.size < len(strata):
        raise ValueError(f"Stratified cohort size must be at least {len(strata)}")
    buckets: dict[str, list[DatasetExample]] = defaultdict(list)
    for example in examples:
        buckets[reference_class(example.reference)].append(example)
    missing = [stratum for stratum in strata if not buckets[stratum]]
    if missing:
        raise ValueError(f"Dataset split has no examples for strata: {missing[:5]}")
    for stratum in strata:
        generator.shuffle(buckets[stratum])

    selected: list[DatasetExample] = []
    depth = 0
    while len(selected) < spec.size:
        added = False
        for stratum in strata:
            if depth < len(buckets[stratum]):
                selected.append(buckets[stratum][depth])
                added = True
                if len(selected) == spec.size:
                    break
        if not added:
            raise ValueError("Could not fill the requested stratified cohort")
        depth += 1
    order = {example.id: index for index, example in enumerate(examples)}
    return sorted(selected, key=lambda example: order[example.id])


def cohort_record(
    examples: Sequence[DatasetExample],
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    schema: ReferenceSchema,
    spec: CohortSpec,
    task: StructuredTask,
) -> dict[str, Any]:
    """Serialize the exact states and references of one frozen cohort."""
    return {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "question_type": schema.question_type,
        "selection": spec.as_dict(),
        "count": len(examples),
        "examples": [
            {
                "id": example.id,
                "state": task.build_state(example.state),
                "reference": _reference_record(example.reference),
            }
            for example in examples
        ],
    }


def examples_from_cohort(
    value: object,
    dataset_examples: Sequence[DatasetExample],
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    task: StructuredTask,
) -> list[DatasetExample]:
    """Validate frozen model states and typed references against prepared data."""
    if not isinstance(value, dict):
        raise ValueError("Saved cohort must be a JSON object")
    if value.get("dataset_revision") != dataset_revision or value.get("split") != split:
        raise ValueError("Saved cohort does not match the benchmark dataset")
    if value.get("dataset_id") != dataset_id:
        raise ValueError("Saved cohort dataset ID does not match the benchmark dataset")
    if value.get("question_type") != task.question_type:
        raise ValueError("Saved cohort question type does not match the task")
    records = value.get("examples")
    if not isinstance(records, list) or value.get("count") != len(records) or not records:
        raise ValueError("Saved cohort has an invalid example list")

    by_id = {example.id: example for example in dataset_examples}
    selected: list[DatasetExample] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Saved cohort contains a malformed example")
        example_id = record["id"]
        if example_id in seen or example_id not in by_id:
            raise ValueError(f"Saved cohort contains an invalid ID: {example_id}")
        example = by_id[example_id]
        if record.get("state") != task.build_state(example.state):
            raise ValueError(f"Saved cohort state differs for ID: {example_id}")
        if record.get("reference") != _reference_record(example.reference):
            raise ValueError(f"Saved cohort reference differs for ID: {example_id}")
        seen.add(example_id)
        selected.append(example)
    return selected


def _reference_record(reference: Reference) -> dict[str, Any]:
    if isinstance(reference, ChoiceReference):
        return {"type": "choice", "label": reference.label}
    if isinstance(reference, NoulReference):
        return {"type": "noul", "value": reference.value}
    assert isinstance(reference, ScoreReference)
    return {"type": "score", "level": reference.level}
