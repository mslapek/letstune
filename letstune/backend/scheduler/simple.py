from dataclasses import dataclass
from typing import Sequence

from letstune.backend import repo

__all__ = [
    "Task",
    "get_next_tasks",
]


@dataclass(kw_only=True, slots=True, frozen=True)
class Task:
    training_id: int


def get_next_tasks(
    trainings: Sequence[repo.Training],
) -> list[Task]:
    return [
        Task(training_id=t.training_id)
        for t in trainings
        if len(t.epochs) == 0 and t.error is None
    ]
