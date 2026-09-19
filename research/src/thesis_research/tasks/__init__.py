"""Task definitions that turn dataset examples into structured model inputs."""

from thesis_research.tasks.intent_classification import (
    IntentClassificationTask,
    build_banking77_task,
)

__all__ = ["IntentClassificationTask", "build_banking77_task"]
