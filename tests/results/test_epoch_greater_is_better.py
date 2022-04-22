from datetime import datetime, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.results import epoch

from . import utils
from .utils import ModelParams, OptimizerParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> epoch.TuningResults[ModelParams]:
    builder: epoch.Builder[ModelParams] = epoch.Builder(
        metric=METRIC,
        checkpoint_factory=utils.EpochCheckpointFactory(),
        round_assigner=utils.RoundAssigner(
            {
                timedelta(minutes=2): 0,
                timedelta(minutes=4): 1,
                timedelta(minutes=5): 5,
                timedelta(minutes=7): 99,
            }
        ),
    )

    epoch_builder = builder.add_training(
        ModelParams(
            optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"), zeta=1111.11
        )
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 10, 0, 0),
        end_time=datetime(2022, 10, 1, 10, 2, 0),
        metric_values={
            "f_score": 0.1,
            "accuracy": 0.2,
        },
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 10, 2, 0),
        end_time=datetime(2022, 10, 1, 10, 4, 0),
        metric_values={
            "f_score": 0.5,
            "accuracy": 0.7,
        },
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 10, 10, 0),
        end_time=datetime(2022, 10, 1, 10, 11, 0),
        metric_values={
            "f_score": 0.3,
            "accuracy": 0.5,
        },
    )

    epoch_builder.build()

    epoch_builder = builder.add_training(
        ModelParams(
            optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
        ),
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 15, 0, 0),
        end_time=datetime(2022, 10, 1, 15, 2, 0),
        metric_values={
            "f_score": 0.1,
            "accuracy": 0.2,
        },
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 15, 2, 0),
        end_time=datetime(2022, 10, 1, 15, 4, 0),
        metric_values={
            "f_score": 0.5,
            "accuracy": 0.3,
        },
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 15, 10, 0),
        end_time=datetime(2022, 10, 1, 15, 11, 0),
        metric_values={
            "f_score": 0.8,
            "accuracy": 0.6,
        },
    )

    epoch_builder.add_epoch(
        start_time=datetime(2022, 10, 1, 15, 12, 0),
        end_time=datetime(2022, 10, 1, 15, 14, 0),
        metric_values={
            "f_score": 0.8,
            "accuracy": 0.65,
        },
    )

    epoch_builder.build()

    return builder.build()


def test_len(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert len(tuning_results) == 2


def test_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert (
        repr(tuning_results)
        == "<TuningResults with 2 trainings; best_training.metric_value=0.7>"
    )


def test_get_item(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert isinstance(training, epoch.Training)
    assert training.training_id == 1


def test_get_item_slice(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    trainings = tuning_results[1:5]

    assert len(trainings) == 1
    assert isinstance(trainings, list)
    assert isinstance(trainings[0], epoch.Training)
    assert trainings[0].training_id == 1


def test_training(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert training.params == ModelParams(
        optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
    )
    assert training.start_time == datetime(2022, 10, 1, 15, 0, 0)
    assert training.end_time == datetime(2022, 10, 1, 15, 14, 0)

    assert len(training) == 4


def test_training_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert repr(training) == "<Training 1 with 4 epochs; metric_value=0.65>"


def test_get_epoch(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    assert isinstance(ep, epoch.Epoch)
    assert ep.epoch_id == 2


def test_epoch_repr(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    assert repr(ep) == "<Epoch 2 from training 1; metric_value=0.6>"


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

    assert ep.round == 5


def test_duration(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert training.duration == timedelta(minutes=7)


def test_checkpoint(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]
    ep = training[2]

    checkpoint = ep.checkpoint

    assert checkpoint == utils.EpochCheckpoint(training_id=1, epoch_id=2)


@pytest.mark.parametrize("tuning_id,expected_metric_value", [(0, 0.7), (1, 0.65)])
def test_metric_value(
    tuning_results: epoch.TuningResults[ModelParams],
    tuning_id: int,
    expected_metric_value: float,
) -> None:
    training = tuning_results[tuning_id]

    assert training.metric_value == expected_metric_value


def test_best_training(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    best_training = tuning_results.best_training

    assert best_training.training_id == 0


def test_sorted_trainings(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    trainings = tuning_results.sorted_trainings()

    ids = [t.training_id for t in trainings]
    assert ids == [0, 1]


@pytest.mark.parametrize(
    "n,expected_ids",
    [
        (2, [0, 1]),
        (10, [0, 1]),
        (0, []),
    ],
)
def test_top_trainings(
    tuning_results: epoch.TuningResults[ModelParams],
    n: int,
    expected_ids: list[int],
) -> None:
    trainings = tuning_results.top_trainings(n)

    ids = [t.training_id for t in trainings]
    assert ids == expected_ids


def test_metric(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    assert tuning_results.metric == METRIC


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
        "round": [5, 99],
        "start_time": [
            Timestamp("2022-10-01 10:00:00"),
            Timestamp("2022-10-01 15:00:00"),
        ],
        "training_id": [0, 1],
    }
