import math
import operator
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import letstune
from letstune.backend import repo

__all__ = [
    "Task",
    "Config",
    "get_next_tasks",
]


@dataclass(kw_only=True, slots=True, frozen=True)
class Task:
    training_id: int
    next_epoch: int
    duration: timedelta


@dataclass(kw_only=True, slots=True, frozen=True)
class Config:
    round_durations: Sequence[timedelta]
    trainings_reduction: float = 4.0

    def reduced_trainings_number(self, n: int) -> int:
        return math.ceil(n / self.trainings_reduction)


@dataclass(slots=True, frozen=True)
class _Training:
    training_id: int
    total_duration: timedelta
    next_epoch: int
    metric_value: float
    error: bool = False


def _normalize_training(
    metric: letstune.Metric,
    t: repo.Training,
) -> _Training:
    if t.error is not None:
        if metric.greater_is_better:
            metric_value = -math.inf
        else:
            metric_value = math.inf
    elif len(t.epochs) == 0:
        metric_value = 0.0
    else:
        metric_values = (e.metric_values[metric.name] for e in t.epochs)
        if metric.greater_is_better:
            metric_value = max(metric_values)
        else:
            metric_value = min(metric_values)

    return _Training(
        training_id=t.training_id,
        total_duration=sum(
            (e.duration for e in t.epochs),
            timedelta(),
        ),
        next_epoch=len(t.epochs),
        metric_value=metric_value,
        error=t.error is not None,
    )


def _normalize(
    metric: letstune.Metric,
    trainings: Sequence[repo.Training],
) -> list[_Training]:
    """Get sorted trainings."""

    ts = [_normalize_training(metric, t) for t in trainings]
    ts.sort(
        key=operator.attrgetter("metric_value"),
        reverse=metric.greater_is_better,
    )
    return ts


def get_next_tasks(
    config: Config,
    metric: letstune.Metric,
    trainings: Sequence[repo.Training],
) -> list[Task]:
    ts = _normalize(metric, trainings)
    return _get_next_tasks(config, ts)


def _get_next_tasks(
    config: Config,
    ts: list[_Training],
) -> list[Task]:
    total_round_duration = timedelta()
    for round_duration in config.round_durations:
        total_round_duration += round_duration

        tasks = []
        for t in ts:
            if t.total_duration < total_round_duration and not t.error:
                tasks.append(
                    Task(
                        training_id=t.training_id,
                        next_epoch=t.next_epoch,
                        duration=total_round_duration - t.total_duration,
                    )
                )

        if tasks:
            # some tasks must do the round
            return tasks

        # all tasks have already done the round.

        reduced_count = config.reduced_trainings_number(len(ts))
        while len(ts) > reduced_count:
            ts.pop()

    return []  # done!
