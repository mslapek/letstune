import json
from datetime import datetime, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.backend import repo
from letstune.backend.scheduler.epoch import Config
from letstune.results import epoch

from . import utils
from .utils import ModelParams, OptimizerParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> epoch.TuningResults[ModelParams]:
    trainings = [
        repo.Training(
            training_id=100,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"),
                    zeta=1111.11,
                ).to_json()
            ),
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    start_time=datetime(2022, 10, 1, 10, 0, 0),
                    end_time=datetime(2022, 10, 1, 10, 2, 0),
                    metric_values={
                        "f_score": 0.1,
                        "accuracy": 0.2,
                    },
                ),
                repo.EpochStats(
                    epoch_id=1,
                    start_time=datetime(2022, 10, 1, 10, 2, 0),
                    end_time=datetime(2022, 10, 1, 10, 4, 0),
                    metric_values={
                        "f_score": 0.5,
                        "accuracy": 0.7,
                    },
                ),
                repo.EpochStats(
                    epoch_id=2,
                    start_time=datetime(2022, 10, 1, 10, 10, 0),
                    end_time=datetime(2022, 10, 1, 10, 11, 0),
                    metric_values={
                        "f_score": 0.3,
                        "accuracy": 0.5,
                    },
                ),
            ],
        ),
        repo.Training(
            training_id=50,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
                ).to_json()
            ),
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    start_time=datetime(2022, 10, 1, 15, 0, 0),
                    end_time=datetime(2022, 10, 1, 15, 2, 0),
                    metric_values={
                        "f_score": 0.1,
                        "accuracy": 0.2,
                    },
                ),
                repo.EpochStats(
                    epoch_id=1,
                    start_time=datetime(2022, 10, 1, 15, 2, 0),
                    end_time=datetime(2022, 10, 1, 15, 4, 0),
                    metric_values={
                        "f_score": 0.5,
                        "accuracy": 0.3,
                    },
                ),
                repo.EpochStats(
                    epoch_id=2,
                    start_time=datetime(2022, 10, 1, 15, 10, 0),
                    end_time=datetime(2022, 10, 1, 15, 11, 0),
                    metric_values={
                        "f_score": 0.8,
                        "accuracy": 0.6,
                    },
                ),
                repo.EpochStats(
                    epoch_id=3,
                    start_time=datetime(2022, 10, 1, 15, 12, 0),
                    end_time=datetime(2022, 10, 1, 15, 14, 0),
                    metric_values={
                        "f_score": 0.8,
                        "accuracy": 0.65,
                    },
                ),
            ],
        ),
        repo.Training(
            training_id=601,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"),
                    zeta=1111.11,
                ).to_json()
            ),
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    start_time=datetime(2022, 10, 1, 10, 0, 0),
                    end_time=datetime(2022, 10, 1, 10, 2, 0),
                    metric_values={
                        "f_score": 0.1,
                        "accuracy": 0.2,
                    },
                ),
                repo.EpochStats(
                    epoch_id=1,
                    start_time=datetime(2022, 10, 1, 10, 2, 0),
                    end_time=datetime(2022, 10, 1, 10, 4, 0),
                    metric_values={
                        "f_score": 0.5,
                        "accuracy": 0.7,
                    },
                ),
                repo.EpochStats(
                    epoch_id=2,
                    start_time=datetime(2022, 10, 1, 10, 10, 0),
                    end_time=datetime(2022, 10, 1, 10, 11, 0),
                    metric_values={
                        "f_score": 0.3,
                        "accuracy": 0.5,
                    },
                ),
            ],
            error="out of memory",
        ),
        repo.Training(
            training_id=600,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"),
                    zeta=1111.11,
                ).to_json()
            ),
            epochs=[],
        ),
    ]

    return epoch.build(
        metric=METRIC,
        checkpoint_factory=utils.EpochCheckpointFactory(),
        config=Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
                timedelta(minutes=4),
            ],
            trainings_reduction=2.0,
        ),
        params_cls=ModelParams,
        trainings=trainings,
    )


def test_len(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert len(tuning_results) == 2


def test_metric_value(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert tuning_results.metric_value == 0.7


def test_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert (
        repr(tuning_results)
        == "<TuningResults with 2 trainings and 2 errors; metric_value=0.7>"
    )


def test_get_item(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert isinstance(training, epoch.Training)
    assert training.training_id == 50


def test_get_item_slice(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    trainings = tuning_results[1:5]

    assert len(trainings) == 1
    assert isinstance(trainings, list)
    assert isinstance(trainings[0], epoch.Training)
    assert trainings[0].training_id == 50


def test_training(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert training.params == ModelParams(
        optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
    )
    assert training.start_time == datetime(2022, 10, 1, 15, 0, 0)
    assert training.end_time == datetime(2022, 10, 1, 15, 14, 0)

    assert len(training) == 4

    assert training.round == 1


def test_training_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert repr(training) == "<Training 50 with 4 epochs; metric_value=0.65>"


def test_get_epoch(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    assert isinstance(ep, epoch.Epoch)
    assert ep.epoch_id == 2


def test_epoch_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    assert repr(ep) == "<Epoch 2 from training 50; metric_value=0.6>"


def test_get_epoch_slice(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    epochs = training[1:3]

    assert len(epochs) == 2
    assert isinstance(epochs, list)
    assert isinstance(epochs[0], epoch.Epoch)
    assert epochs[0].epoch_id == 1


def test_best_epoch(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[0]
    ep = training.best_epoch

    assert isinstance(ep, epoch.Epoch)
    assert ep.epoch_id == 1


def test_last_epoch(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[0]
    ep = training.last_epoch

    assert isinstance(ep, epoch.Epoch)
    assert ep.epoch_id == 2


def test_epoch(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[0]
    ep = training[2]

    assert ep.epoch_id == 2

    assert ep.start_time == datetime(2022, 10, 1, 10, 10, 0)
    assert ep.end_time == datetime(2022, 10, 1, 10, 11, 0)
    assert ep.duration == timedelta(minutes=1)
    assert ep.cum_duration == timedelta(minutes=5)

    assert ep.metric_values == MappingProxyType(
        {
            "f_score": 0.3,
            "accuracy": 0.5,
        }
    )
    assert ep.metric_value == 0.5
    assert ep.cum_metric_value == 0.7


def test_duration(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert training.duration == timedelta(minutes=7)


def test_checkpoint(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    checkpoint = ep.checkpoint

    assert checkpoint == utils.EpochCheckpoint(training_id=50, epoch_id=2)


@pytest.mark.parametrize("i,expected_metric_value", [(0, 0.7), (1, 0.65)])
def test_training_metric_value(
    tuning_results: epoch.TuningResults[ModelParams],
    i: int,
    expected_metric_value: float,
) -> None:
    training = tuning_results[i]

    assert training.metric_value == expected_metric_value


def test_sorted_trainings(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    ids = [t.training_id for t in tuning_results]
    assert ids == [100, 50]


@pytest.mark.parametrize(
    "n,expected_ids",
    [
        (2, [100, 50]),
        (10, [100, 50]),
        (0, []),
    ],
)
def test_top_trainings(
    tuning_results: epoch.TuningResults[ModelParams],
    n: int,
    expected_ids: list[int],
) -> None:
    trainings = tuning_results[:n]

    ids = [t.training_id for t in trainings]
    assert ids == expected_ids


def test_metric(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert tuning_results.metric == METRIC


def test_errors(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert tuning_results.errors == (
        epoch.Error(
            training_id=601,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"),
                    zeta=1111.11,
                ).to_json()
            ),
            msg="out of memory",
        ),
        epoch.Error(
            training_id=600,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"),
                    zeta=1111.11,
                ).to_json()
            ),
            msg="no epochs",
        ),
    )


def test_to_df(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    df = tuning_results.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.dtypes.to_dict() == {
        "training_id": np.dtype("int64"),
        "params.optimizer.alpha": np.dtype("int64"),
        "params.optimizer.beta": np.dtype("float64"),
        "params.optimizer.gamma": np.dtype("object"),
        "params.zeta": np.dtype("float64"),
        "best_epoch_metrics.f_score": np.dtype("float64"),
        "best_epoch_metrics.accuracy": np.dtype("float64"),
        "last_epoch_metrics.f_score": np.dtype("float64"),
        "last_epoch_metrics.accuracy": np.dtype("float64"),
        "start_time": np.dtype("<M8[ns]"),
        "end_time": np.dtype("<M8[ns]"),
        "duration": np.dtype("<m8[ns]"),
        "epochs": np.dtype("int64"),
        "best_epoch": np.dtype("int64"),
        "round": np.dtype("int64"),
    }

    assert df.to_dict("list") == {
        "training_id": [100, 50],
        "best_epoch": [1, 3],
        "best_epoch_metrics.accuracy": [0.7, 0.65],
        "best_epoch_metrics.f_score": [0.5, 0.8],
        "duration": [
            Timedelta("0 days 00:05:00"),
            Timedelta("0 days 00:07:00"),
        ],
        "end_time": [
            Timestamp("2022-10-01 10:11:00"),
            Timestamp("2022-10-01 15:14:00"),
        ],
        "epochs": [3, 4],
        "last_epoch_metrics.accuracy": [0.5, 0.65],
        "last_epoch_metrics.f_score": [0.3, 0.8],
        "params.optimizer.alpha": [1, 3],
        "params.optimizer.beta": [3.3, 5.1],
        "params.optimizer.gamma": ["hello", "hi"],
        "params.zeta": [1111.11, -99.0],
        "round": [2, 1],
        "start_time": [
            Timestamp("2022-10-01 10:00:00"),
            Timestamp("2022-10-01 15:00:00"),
        ],
    }
