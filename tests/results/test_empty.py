from datetime import datetime, timedelta

import pytest

import letstune
from letstune.backend import repo
from letstune.backend.scheduler.epoch import Config
from letstune.results import epoch, simple

from . import utils
from .utils import ModelParams

METRIC = letstune.Metric("accuracy")


def test_simple_tuning_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="tuning must have at least one training"):
        _ = simple.build(
            metric=METRIC,
            checkpoint_factory=utils.SimpleCheckpointFactory(),
            params_cls=ModelParams,
            trainings=[],
        )


def test_epoch_tuning_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="tuning must have at least one training"):
        _ = epoch.build(
            metric=METRIC,
            checkpoint_factory=utils.EpochCheckpointFactory(),
            params_cls=ModelParams,
            config=Config(
                round_durations=[
                    timedelta(minutes=1),
                    timedelta(minutes=2),
                    timedelta(minutes=4),
                ],
            ),
            trainings=[],
        )


def test_simple_tuning_error_only_repr() -> None:
    tuning = simple.build(
        metric=METRIC,
        checkpoint_factory=utils.SimpleCheckpointFactory(),
        params_cls=ModelParams,
        trainings=[
            repo.Training(
                training_id=100,
                params="yoyo",
                epochs=[
                    repo.EpochStats(
                        epoch_id=1,
                        start_time=datetime(2022, 10, 10, 0, 0, 0),
                        end_time=datetime(2022, 10, 10, 10, 0, 0),
                        metric_values={"accuracy": 0.5},
                    )
                ],
                error="boom!",
            ),
            repo.Training(
                training_id=100,
                params="yoyo",
                epochs=[],
            ),
        ],
    )

    assert (
        repr(tuning)
        == "<TuningResults with 0 trainings and 2 errors; metric_value=???>"
    )


def test_epoch_tuning_error_only_repr() -> None:
    tuning = epoch.build(
        metric=METRIC,
        checkpoint_factory=utils.EpochCheckpointFactory(),
        params_cls=ModelParams,
        config=Config(round_durations=[]),
        trainings=[
            repo.Training(
                training_id=100,
                params="yoyo",
                epochs=[
                    repo.EpochStats(
                        epoch_id=1,
                        start_time=datetime(2022, 10, 10, 0, 0, 0),
                        end_time=datetime(2022, 10, 10, 10, 0, 0),
                        metric_values={"accuracy": 0.5},
                    )
                ],
                error="boom!",
            ),
            repo.Training(
                training_id=100,
                params="yoyo",
                epochs=[],
            ),
        ],
    )

    assert (
        repr(tuning)
        == "<TuningResults with 0 trainings and 2 errors; metric_value=???>"
    )
