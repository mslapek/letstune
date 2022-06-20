from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, Sequence

import numpy as np
import pytest

import letstune.backend.runner
import letstune.backend.runner.simple
from letstune.backend import repo
from letstune.backend.repo import JSON, EpochStats, Training
from letstune.backend.scheduler.simple import Task
from letstune.metrics import MetricValues
from letstune.patch37 import dataclass

from .utils import HistoryLogger


class ModelParams(letstune.Params):
    alpha: int
    beta: str


@dataclass(frozen=True)
class Model:
    params: ModelParams


class SimpleRunner(letstune.backend.runner.simple.SimpleRunner[ModelParams]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.test_tasks: list[Task] = []
        self.time_iter: Iterator[datetime] = iter([])

    def _get_next_tasks(self, trainings: Sequence[repo.Training]) -> list[Task]:
        self.test_got_trainings = trainings
        return self.test_tasks

    def _time(self) -> datetime:
        return next(self.time_iter)


TRAININGS = (
    repo.Training(
        training_id=13,
        params='{"alpha": 13, "beta": 0.2}',
    ),
    repo.Training(
        training_id=16,
        params='{"alpha": 15, "beta": 0.3}',
        epochs=(
            repo.EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 0.5},
                start_time=datetime(2022, 10, 10, 10, 0, 0),
                end_time=datetime(2022, 10, 10, 11, 0, 0),
            ),
        ),
    ),
    repo.Training(
        training_id=20,
        params='{"alpha": 20, "beta": 0.5}',
        error="what's up?",
    ),
    repo.Training(
        training_id=660,
        params='{"alpha": 660, "beta": 0.5}',
    ),
)


class Repository(repo.Repository):
    def __init__(self, trainings: Sequence[Training]):
        self.trainings = trainings
        self.log: list[Any] = []

    def get_all_trainings(self) -> Sequence[Training]:
        return tuple(self.trainings)

    def add_training(self, training_id: int, params: JSON) -> None:
        raise RuntimeError

    def add_epoch(self, training_id: int, epoch_stats: EpochStats) -> None:
        self.log.append(("add_epoch", training_id, epoch_stats))

    def set_error(self, training_id: int, description: str) -> None:
        self.log.append(("set_error", training_id, description))


@dataclass(frozen=True)
class SimpleCheckpoint:
    training_id: int


class SimpleCheckpointFactory:
    def get_checkpoint(self, training_id: int) -> Any:
        return SimpleCheckpoint(training_id)


class Trainer(letstune.SimpleTrainer[ModelParams]):
    metric = letstune.Metric("accuracy")

    def __init__(self) -> None:
        self.log: list[Any] = []

    def train(self, params: ModelParams) -> tuple[Any, MetricValues]:
        self.log.append(("train", params))

        if 600 < params.alpha < 700:
            raise RuntimeError("today is a good day")

        return Model(params), {
            "accuracy": float(params.alpha),
            "f_score": -np.float32(params.alpha),  # type: ignore
        }

    def load_dataset(self, dataset: Any) -> None:
        self.log.append(("load_dataset", dataset))

    def save(self, checkpoint: Any, model: Any) -> None:
        self.log.append(("save", checkpoint, model))


@pytest.fixture
def dataset() -> object:
    return object()


@pytest.fixture
def trainer() -> Trainer:
    return Trainer()


@pytest.fixture
def checkpoint_factory() -> SimpleCheckpointFactory:
    return SimpleCheckpointFactory()


@pytest.fixture
def repository() -> Repository:
    return Repository(TRAININGS)


@pytest.fixture
def logger() -> HistoryLogger:
    return HistoryLogger()


@pytest.fixture
def runner(
    dataset: object,
    trainer: Trainer,
    checkpoint_factory: SimpleCheckpointFactory,
    repository: Repository,
    logger: HistoryLogger,
) -> SimpleRunner:
    runner = SimpleRunner(
        repository=repository,
        checkpoint_factory=checkpoint_factory,
        trainer=trainer,
        dataset=dataset,
        params_cls=ModelParams,
    )
    runner.logger = logger
    runner.time_iter = iter(
        [
            datetime(2022, 10, 1, 12, 0, 0),
            datetime(2022, 10, 1, 12, 30, 0),
            datetime(2022, 10, 1, 13, 0, 0),
            datetime(2022, 10, 1, 13, 45, 0),
        ]
    )

    return runner


def test_read_trainings_from_repository(runner: SimpleRunner) -> None:
    runner.run()

    assert runner.test_got_trainings == TRAININGS


def test_successful_tasks(
    runner: SimpleRunner,
    repository: Repository,
    dataset: object,
    logger: HistoryLogger,
) -> None:
    runner.test_tasks = [
        Task(training_id=13),
        Task(training_id=20),
    ]

    runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        ("train", ModelParams(alpha=13, beta=0.2)),
        (
            "save",
            SimpleCheckpoint(training_id=13),
            Model(params=ModelParams(alpha=13, beta=0.2)),
        ),
        ("train", ModelParams(alpha=20, beta=0.5)),
        (
            "save",
            SimpleCheckpoint(training_id=20),
            Model(params=ModelParams(alpha=20, beta=0.5)),
        ),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 13.0, "f_score": -13.0},
                start_time=datetime(2022, 10, 1, 12, 0),
                end_time=datetime(2022, 10, 1, 12, 30),
            ),
        ),
        (
            "add_epoch",
            20,
            EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 20.0, "f_score": -20.0},
                start_time=datetime(2022, 10, 1, 13, 0),
                end_time=datetime(2022, 10, 1, 13, 45),
            ),
        ),
    ]

    assert logger.history == [
        {"event": "round", "sub_event": "start", "tasks_count": 2},
        {"event": "train", "sub_event": "start", "training_id": 13},
        {
            "event": "train",
            "sub_event": "stop",
            "training_id": 13,
            "metric_value": 13.0,
        },
        {"event": "train", "sub_event": "start", "training_id": 20},
        {
            "event": "train",
            "sub_event": "stop",
            "training_id": 20,
            "metric_value": 20.0,
        },
        {"event": "round", "sub_event": "end", "tasks_count": 2},
    ]


def test_metric_values_normalization(
    runner: SimpleRunner,
    repository: Repository,
) -> None:
    runner.test_tasks = [
        Task(training_id=13),
        Task(training_id=20),
    ]

    runner.run()

    for event in repository.log:
        epoch_stats = event[2]
        assert isinstance(epoch_stats, EpochStats)

        assert all(type(v) == float for v in epoch_stats.metric_values.values())


def test_failed_tasks(
    runner: SimpleRunner,
    repository: Repository,
    dataset: object,
    logger: HistoryLogger,
) -> None:
    runner.test_tasks = [
        Task(training_id=13),
        Task(training_id=660),
    ]

    runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        ("train", ModelParams(alpha=13, beta=0.2)),
        (
            "save",
            SimpleCheckpoint(training_id=13),
            Model(params=ModelParams(alpha=13, beta=0.2)),
        ),
        ("train", ModelParams(alpha=660, beta=0.5)),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 13.0, "f_score": -13.0},
                start_time=datetime(2022, 10, 1, 12, 0),
                end_time=datetime(2022, 10, 1, 12, 30),
            ),
        ),
        (
            "set_error",
            660,
            "RuntimeError('today is a good day')",
        ),
    ]

    assert logger.history == [
        {"event": "round", "sub_event": "start", "tasks_count": 2},
        {"event": "train", "sub_event": "start", "training_id": 13},
        {
            "event": "train",
            "sub_event": "stop",
            "training_id": 13,
            "metric_value": 13.0,
        },
        {"event": "train", "sub_event": "start", "training_id": 660},
        {
            "event": "train",
            "sub_event": "stop",
            "training_id": 660,
            "status": "failed",
        },
        {"event": "round", "sub_event": "end", "tasks_count": 2},
    ]


def test_failed_tasks_with_error_passthrough(
    runner: SimpleRunner,
    repository: Repository,
    dataset: object,
    logger: HistoryLogger,
) -> None:
    runner.test_tasks = [
        Task(training_id=13),
        Task(training_id=660),
    ]
    runner.passthrough_errors = True

    with pytest.raises(RuntimeError, match="today is a good day"):
        runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        ("train", ModelParams(alpha=13, beta=0.2)),
        (
            "save",
            SimpleCheckpoint(training_id=13),
            Model(params=ModelParams(alpha=13, beta=0.2)),
        ),
        ("train", ModelParams(alpha=660, beta=0.5)),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 13.0, "f_score": -13.0},
                start_time=datetime(2022, 10, 1, 12, 0),
                end_time=datetime(2022, 10, 1, 12, 30),
            ),
        ),
    ]

    assert logger.history == [
        {"event": "round", "sub_event": "start", "tasks_count": 2},
        {"event": "train", "sub_event": "start", "training_id": 13},
        {
            "event": "train",
            "sub_event": "stop",
            "training_id": 13,
            "metric_value": 13.0,
        },
        {"event": "train", "sub_event": "start", "training_id": 660},
    ]


def test_original_trainer_is_not_touched(
    runner: SimpleRunner, trainer: Trainer
) -> None:
    runner.test_tasks = [
        Task(training_id=13),
        Task(training_id=20),
    ]

    runner.run()

    assert len(trainer.log) == 0


def test_repository_has_added_epoch_after_checkpoint_save(
    runner: SimpleRunner,
    trainer: Trainer,
    repository: Repository,
) -> None:
    unified_log: list[Any] = []
    repository.log = unified_log
    runner._trainer.log = unified_log  # type: ignore

    runner.test_tasks = [
        Task(training_id=13),
    ]

    runner.run()

    events = [e[0] for e in unified_log]
    assert events.index("save") < events.index("add_epoch")
