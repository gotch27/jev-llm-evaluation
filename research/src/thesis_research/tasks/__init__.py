"""Task definitions that turn dataset examples into structured model inputs."""

from thesis_research.tasks.structured import (
    StructuredTask,
    build_structured_task,
    class_labels,
)

__all__ = [
    "StructuredTask",
    "build_structured_task",
    "class_labels",
]
