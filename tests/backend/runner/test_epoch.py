import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, Sequence

import pytest

import letstune.backend.runner
import letstune.backend.runner.epoch
from letstune.backend import repo
from letstune.backend.repo import JSON, EpochStats, Training
from letstune.backend.scheduler.epoch import Config, Task
from letstune.metrics import MetricValues

CONFIG = Config(round_durations=[timedelta(minutes=2), timedelta(minutes=5)])
METRIC = letstune.Metric("accuracy")


class ModelParams(letstune.Params):
    alpha: int
    beta: str


@dataclass(frozen=True)
class Model:
    params: ModelParams


class EpochRunner(letstune.backend.runner.epoch.EpochRunner[ModelParams]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tasks_iter: Iterator[Sequence[Task]] = iter([])
        self.time_iter: Iterator[datetime] = iter([])
        self.test_got_trainings: list[Any] = []

    def _get_next_tasks(self, trainings: Sequence[repo.Training]) -> list[Task]:
        self.test_got_trainings.append(trainings)
        return list(next(self.tasks_iter))

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
    repo.Training(
        training_id=770,
        params='{"alpha": 770, "beta": 0.5}',
    ),
    repo.Training(
        training_id=880,
        params='{"alpha": 880, "beta": 0.5}',
    ),
)


class IterableRepository(repo.Repository):
    def __init__(self, trainings_iter: Iterable[Sequence[Training]]) -> None:
        self.trainings_iter = iter(trainings_iter)
        self.log: list[Any] = []

    @staticmethod
    def const(trainings: Sequence[Training]) -> "IterableRepository":
        return IterableRepository(itertools.repeat(trainings))

    def get_all_trainings(self) -> Sequence[Training]:
        return tuple(next(self.trainings_iter))

    def add_training(self, training_id: int, params: JSON) -> None:
        raise RuntimeError

    def add_epoch(self, training_id: int, epoch_stats: EpochStats) -> None:
        self.log.append(("add_epoch", training_id, epoch_stats))

    def set_error(self, training_id: int, description: str) -> None:
        self.log.append(("set_error", training_id, description))


@dataclass(frozen=True)
class EpochCheckpoint:
    training_id: int
    epoch_id: int


class EpochCheckpointFactory:
    def get_checkpoint(self, training_id: int, epoch_id: int) -> Any:
        return EpochCheckpoint(training_id, epoch_id)


class Trainer(letstune.EpochTrainer[ModelParams]):
    metric = METRIC

    def __init__(self) -> None:
        self.log: list[Any] = []

    def create_model(self, params: ModelParams) -> None:
        self.log.append(("create_model", params))

        if 700 < params.alpha < 800:
            raise RuntimeError("hello")

        self.params = params

    def train_epoch(self, epoch: int) -> MetricValues:
        self.log.append(("train_epoch", epoch))

        if 600 < self.params.alpha < 700:
            raise RuntimeError("today is a good day")

        return {
            "accuracy": float(self.params.alpha),
            "f_score": float(epoch),
        }

    def load_dataset(self, dataset: Any) -> None:
        self.log.append(("load_dataset", dataset))

    def save(self, checkpoint: Any) -> None:
        self.log.append(("save", checkpoint))

    def load(self, checkpoint: Any, params: ModelParams) -> None:
        self.log.append(("load", checkpoint, params))

        if 800 < params.alpha < 900:
            raise RuntimeError("hi")

        self.params = params


@pytest.fixture
def dataset() -> object:
    return object()


@pytest.fixture
def trainer() -> Trainer:
    return Trainer()


@pytest.fixture
def checkpoint_factory() -> EpochCheckpointFactory:
    return EpochCheckpointFactory()


@pytest.fixture
def repository() -> IterableRepository:
    return IterableRepository.const(TRAININGS)


@pytest.fixture
def runner(
    dataset: object,
    trainer: Trainer,
    checkpoint_factory: EpochCheckpointFactory,
    repository: IterableRepository,
) -> EpochRunner:
    runner = EpochRunner(
        repository=repository,
        checkpoint_factory=checkpoint_factory,
        trainer=trainer,
        dataset=dataset,
        params_cls=ModelParams,
        config=CONFIG,
    )

    runner.tasks_iter = iter(
        [
            [
                Task(training_id=13, next_epoch=0, duration=timedelta(minutes=40)),
                Task(training_id=20, next_epoch=3, duration=timedelta(minutes=10)),
            ],
            [Task(training_id=13, next_epoch=100, duration=timedelta(minutes=20))],
            [],
        ]
    )
    runner.time_iter = iter(
        [
            # 2 epochs of training 13, round 0
            datetime(2022, 10, 1, 12, 0, 0),
            datetime(2022, 10, 1, 12, 30, 0),
            datetime(2022, 10, 1, 13, 0, 0),
            datetime(2022, 10, 1, 13, 45, 0),
            # 1 epoch of training 20, round 0
            datetime(2022, 10, 1, 14, 0, 0),
            datetime(2022, 10, 1, 14, 30, 0),
            # 3 epochs of training 13, round 1
            datetime(2022, 10, 1, 15, 0, 0),
            datetime(2022, 10, 1, 15, 10, 0),
            datetime(2022, 10, 1, 15, 20, 0),
            datetime(2022, 10, 1, 15, 25, 0),
            datetime(2022, 10, 1, 15, 30, 0),
            datetime(2022, 10, 1, 15, 40, 0),
        ]
    )

    return runner


def test_read_trainings_from_repository(
    dataset: object,
    trainer: Trainer,
    checkpoint_factory: EpochCheckpointFactory,
) -> None:
    trainings_seq = [
        (
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
        ),
        (
            repo.Training(
                training_id=64,
                params='{"alpha": 64, "beta": 0.2}',
            ),
            repo.Training(
                training_id=16,
                params='{"alpha": 16, "beta": 0.3}',
                epochs=(
                    repo.EpochStats(
                        epoch_id=0,
                        metric_values={"accuracy": 0.5},
                        start_time=datetime(2022, 10, 10, 10, 0, 0),
                        end_time=datetime(2022, 10, 10, 11, 0, 0),
                    ),
                    repo.EpochStats(
                        epoch_id=1,
                        metric_values={"accuracy": 0.5},
                        start_time=datetime(2022, 10, 12, 10, 0, 0),
                        end_time=datetime(2022, 10, 12, 11, 0, 0),
                    ),
                ),
            ),
        ),
    ]

    runner = EpochRunner(
        repository=IterableRepository(trainings_seq),
        checkpoint_factory=checkpoint_factory,
        trainer=trainer,
        dataset=dataset,
        params_cls=ModelParams,
        config=CONFIG,
    )
    runner.tasks_iter = iter(
        [
            [
                Task(training_id=13, next_epoch=0, duration=timedelta(minutes=10)),
            ],
            [],
        ]
    )
    runner.time_iter = iter(
        [
            datetime(2022, 10, 1, 12, 0, 0),
            datetime(2022, 10, 1, 12, 30, 0),
            datetime(2022, 10, 1, 13, 0, 0),
            datetime(2022, 10, 1, 13, 45, 0),
        ]
    )

    runner.run()

    assert runner._metric == METRIC
    assert runner._config == CONFIG
    assert runner.test_got_trainings == trainings_seq


def test_successful_tasks(
    runner: EpochRunner,
    repository: IterableRepository,
    dataset: object,
) -> None:
    runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        ("create_model", ModelParams(alpha=13, beta=0.2)),
        ("train_epoch", 0),
        ("save", EpochCheckpoint(training_id=13, epoch_id=0)),
        ("train_epoch", 1),
        ("save", EpochCheckpoint(training_id=13, epoch_id=1)),
        (
            "load",
            EpochCheckpoint(training_id=20, epoch_id=2),
            ModelParams(alpha=20, beta=0.5),
        ),
        ("train_epoch", 3),
        ("save", EpochCheckpoint(training_id=20, epoch_id=3)),
        (
            "load",
            EpochCheckpoint(training_id=13, epoch_id=99),
            ModelParams(alpha=13, beta=0.2),
        ),
        ("train_epoch", 100),
        ("save", EpochCheckpoint(training_id=13, epoch_id=100)),
        ("train_epoch", 101),
        ("save", EpochCheckpoint(training_id=13, epoch_id=101)),
        ("train_epoch", 102),
        ("save", EpochCheckpoint(training_id=13, epoch_id=102)),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 13.0, "f_score": 0.0},
                start_time=datetime(2022, 10, 1, 12, 0),
                end_time=datetime(2022, 10, 1, 12, 30),
            ),
        ),
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=1,
                metric_values={"accuracy": 13.0, "f_score": 1.0},
                start_time=datetime(2022, 10, 1, 13, 0),
                end_time=datetime(2022, 10, 1, 13, 45),
            ),
        ),
        (
            "add_epoch",
            20,
            EpochStats(
                epoch_id=3,
                metric_values={"accuracy": 20.0, "f_score": 3.0},
                start_time=datetime(2022, 10, 1, 14, 0),
                end_time=datetime(2022, 10, 1, 14, 30),
            ),
        ),
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=100,
                metric_values={"accuracy": 13.0, "f_score": 100.0},
                start_time=datetime(2022, 10, 1, 15, 0),
                end_time=datetime(2022, 10, 1, 15, 10),
            ),
        ),
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=101,
                metric_values={"accuracy": 13.0, "f_score": 101.0},
                start_time=datetime(2022, 10, 1, 15, 20),
                end_time=datetime(2022, 10, 1, 15, 25),
            ),
        ),
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=102,
                metric_values={"accuracy": 13.0, "f_score": 102.0},
                start_time=datetime(2022, 10, 1, 15, 30),
                end_time=datetime(2022, 10, 1, 15, 40),
            ),
        ),
    ]


def test_failed_tasks(
    runner: EpochRunner,
    repository: IterableRepository,
    dataset: object,
) -> None:
    runner.tasks_iter = iter(
        [
            [
                Task(training_id=13, next_epoch=1, duration=timedelta(minutes=2)),
                Task(training_id=660, next_epoch=1, duration=timedelta(minutes=2)),
                Task(training_id=770, next_epoch=0, duration=timedelta(minutes=2)),
                Task(training_id=880, next_epoch=1, duration=timedelta(minutes=2)),
            ],
            [],
        ]
    )
    runner.time_iter = (datetime(2022, 10, 10, i + 1, 0, 0) for i in range(12))

    runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        (
            "load",
            EpochCheckpoint(training_id=13, epoch_id=0),
            ModelParams(alpha=13, beta=0.2),
        ),
        ("train_epoch", 1),
        ("save", EpochCheckpoint(training_id=13, epoch_id=1)),
        (
            "load",
            EpochCheckpoint(training_id=660, epoch_id=0),
            ModelParams(alpha=660, beta=0.5),
        ),
        ("train_epoch", 1),
        ("create_model", ModelParams(alpha=770, beta=0.5)),
        (
            "load",
            EpochCheckpoint(training_id=880, epoch_id=0),
            ModelParams(alpha=880, beta=0.5),
        ),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=1,
                metric_values={"accuracy": 13.0, "f_score": 1.0},
                start_time=datetime(2022, 10, 10, 1, 0),
                end_time=datetime(2022, 10, 10, 2, 0),
            ),
        ),
        ("set_error", 660, "RuntimeError('today is a good day')"),
        ("set_error", 770, "RuntimeError('hello')"),
        ("set_error", 880, "RuntimeError('hi')"),
    ]


def test_failed_tasks_with_error_passthrough(
    runner: EpochRunner,
    repository: IterableRepository,
    dataset: object,
) -> None:
    runner.tasks_iter = iter(
        [
            [
                Task(training_id=13, next_epoch=1, duration=timedelta(minutes=2)),
                Task(training_id=660, next_epoch=1, duration=timedelta(minutes=2)),
                Task(training_id=16, next_epoch=1, duration=timedelta(minutes=2)),
            ],
            [],
        ]
    )
    runner.time_iter = (datetime(2022, 10, 10, i + 1, 0, 0) for i in range(4))
    runner.passthrough_errors = True

    with pytest.raises(RuntimeError, match="today is a good day"):
        runner.run()

    new_trainer = runner._trainer
    assert isinstance(new_trainer, Trainer)

    assert new_trainer.log == [
        ("load_dataset", dataset),
        (
            "load",
            EpochCheckpoint(training_id=13, epoch_id=0),
            ModelParams(alpha=13, beta=0.2),
        ),
        ("train_epoch", 1),
        ("save", EpochCheckpoint(training_id=13, epoch_id=1)),
        (
            "load",
            EpochCheckpoint(training_id=660, epoch_id=0),
            ModelParams(alpha=660, beta=0.5),
        ),
        ("train_epoch", 1),
    ]

    assert repository.log == [
        (
            "add_epoch",
            13,
            EpochStats(
                epoch_id=1,
                metric_values={"accuracy": 13.0, "f_score": 1.0},
                start_time=datetime(2022, 10, 10, 1, 0),
                end_time=datetime(2022, 10, 10, 2, 0),
            ),
        ),
    ]


def test_original_trainer_is_not_touched(
    runner: EpochRunner,
    trainer: Trainer,
) -> None:
    runner.run()

    assert len(trainer.log) == 0
