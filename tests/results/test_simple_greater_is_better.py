from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.backend import repo
from letstune.results import simple

from . import utils
from .utils import ModelParams, OptimizerParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> simple.TuningResults[ModelParams]:
    return simple.build(
        metric=METRIC,
        checkpoint_factory=utils.SimpleCheckpointFactory(),
        params_cls=ModelParams,
        trainings=[
            repo.Training(
                training_id=4,
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
                            "f_score": 0.8,
                            "accuracy": 0.4,
                        },
                    )
                ],
            ),
            repo.Training(
                training_id=45,
                params=json.dumps(
                    ModelParams(
                        optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"),
                        zeta=-99.0,
                    ).to_json()
                ),
                epochs=[
                    repo.EpochStats(
                        epoch_id=0,
                        start_time=datetime(2022, 10, 1, 10, 1, 0),
                        end_time=datetime(2022, 10, 1, 10, 5, 0),
                        metric_values={
                            "f_score": 0.1,
                            "accuracy": 0.2,
                        },
                    )
                ],
            ),
            repo.Training(
                training_id=99,
                params=json.dumps(
                    ModelParams(
                        optimizer=OptimizerParams(
                            alpha=123, beta=567.567, gamma="magnificent"
                        ),
                        zeta=9.99,
                    ).to_json()
                ),
                epochs=[
                    repo.EpochStats(
                        epoch_id=0,
                        start_time=datetime(2022, 10, 1, 11, 0, 0),
                        end_time=datetime(2022, 10, 1, 12, 0, 0),
                        metric_values={
                            "f_score": 0.4,
                            "accuracy": 0.9,
                        },
                    )
                ],
                error="stack overflow",
            ),
            repo.Training(
                training_id=5,
                params=json.dumps(
                    ModelParams(
                        optimizer=OptimizerParams(alpha=123, beta=567.567, gamma="moo"),
                        zeta=9.99,
                    ).to_json()
                ),
                epochs=[
                    repo.EpochStats(
                        epoch_id=0,
                        start_time=datetime(2022, 10, 1, 11, 0, 0),
                        end_time=datetime(2022, 10, 1, 12, 0, 0),
                        metric_values={
                            "f_score": 0.4,
                            "accuracy": 0.3,
                        },
                    )
                ],
            ),
            repo.Training(
                training_id=9792,
                params=json.dumps(
                    ModelParams(
                        optimizer=OptimizerParams(
                            alpha=123, beta=567.567, gamma="keyboard"
                        ),
                        zeta=9.99,
                    ).to_json()
                ),
                epochs=[],
            ),
        ],
    )


def test_len(tuning_results: simple.TuningResults[ModelParams]) -> None:
    assert len(tuning_results) == 3


def test_get_item(tuning_results: simple.TuningResults[ModelParams]) -> None:
    training = tuning_results[1]

    assert isinstance(training, simple.Training)
    assert training.training_id == 5


def test_get_item_slice(tuning_results: simple.TuningResults[ModelParams]) -> None:
    trainings = tuning_results[1:5]

    assert len(trainings) == 2
    assert isinstance(trainings, list)
    assert isinstance(trainings[0], simple.Training)
    assert trainings[0].training_id == 5


def test_sorted_trainings(tuning_results: simple.TuningResults[ModelParams]) -> None:
    ids = [t.training_id for t in tuning_results]
    assert ids == [4, 5, 45]


def test_metric_value(tuning_results: simple.TuningResults[ModelParams]) -> None:
    assert tuning_results.metric_value == 0.4


def test_repr(tuning_results: simple.TuningResults[ModelParams]) -> None:
    assert (
        repr(tuning_results)
        == "<TuningResults with 3 trainings and 2 errors; metric_value=0.4>"
    )


def test_training(tuning_results: simple.TuningResults[ModelParams]) -> None:
    training = tuning_results[2]

    assert training.training_id == 45
    assert training.params == ModelParams(
        optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
    )
    assert training.start_time == datetime(2022, 10, 1, 10, 1, 0)
    assert training.end_time == datetime(2022, 10, 1, 10, 5, 0)
    assert training.metric_values == MappingProxyType(
        {
            "f_score": 0.1,
            "accuracy": 0.2,
        }
    )

    assert training.metric_value == 0.2


def test_duration(tuning_results: simple.TuningResults[ModelParams]) -> None:
    training = tuning_results[2]

    assert training.duration == timedelta(minutes=4)


def test_checkpoint(tuning_results: simple.TuningResults[ModelParams]) -> None:
    training = tuning_results[2]

    checkpoint = training.checkpoint

    assert checkpoint == utils.SimpleCheckpoint(training_id=45)


def test_training_metric_value(
    tuning_results: simple.TuningResults[ModelParams],
) -> None:
    training = tuning_results[2]

    assert training.metric_value == 0.2


@pytest.mark.parametrize(
    "n,expected_ids",
    [
        (2, [4, 5]),
        (10, [4, 5, 45]),
        (0, []),
    ],
)
def test_top_trainings(
    tuning_results: simple.TuningResults[ModelParams],
    n: int,
    expected_ids: list[int],
) -> None:
    trainings = tuning_results[:n]

    ids = [t.training_id for t in trainings]
    assert ids == expected_ids


def test_metric(tuning_results: simple.TuningResults[ModelParams]) -> None:
    assert tuning_results.metric == METRIC


def test_errors(tuning_results: simple.TuningResults[ModelParams]) -> None:
    errors = tuning_results.errors

    assert errors == (
        simple.Error(
            training_id=99,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(
                        alpha=123, beta=567.567, gamma="magnificent"
                    ),
                    zeta=9.99,
                ).to_json()
            ),
            msg="stack overflow",
        ),
        simple.Error(
            training_id=9792,
            params=json.dumps(
                ModelParams(
                    optimizer=OptimizerParams(
                        alpha=123, beta=567.567, gamma="keyboard"
                    ),
                    zeta=9.99,
                ).to_json()
            ),
            msg="no trainings",
        ),
    )


def test_to_df(tuning_results: simple.TuningResults[ModelParams]) -> None:
    df = tuning_results.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.dtypes.to_dict() == {
        "training_id": np.dtype("int64"),
        "params.optimizer.alpha": np.dtype("int64"),
        "params.optimizer.beta": np.dtype("float64"),
        "params.optimizer.gamma": np.dtype("object"),
        "params.zeta": np.dtype("float64"),
        "metrics.f_score": np.dtype("float64"),
        "metrics.accuracy": np.dtype("float64"),
        "start_time": np.dtype("<M8[ns]"),
        "end_time": np.dtype("<M8[ns]"),
        "duration": np.dtype("<m8[ns]"),
    }

    assert df.to_dict("list") == {
        "training_id": [4, 5, 45],
        "start_time": [
            Timestamp("2022-10-01 10:00:00"),
            Timestamp("2022-10-01 11:00:00"),
            Timestamp("2022-10-01 10:01:00"),
        ],
        "end_time": [
            Timestamp("2022-10-01 10:02:00"),
            Timestamp("2022-10-01 12:00:00"),
            Timestamp("2022-10-01 10:05:00"),
        ],
        "duration": [
            Timedelta("0 days 00:02:00"),
            Timedelta("0 days 01:00:00"),
            Timedelta("0 days 00:04:00"),
        ],
        "params.optimizer.alpha": [1, 123, 3],
        "params.optimizer.beta": [3.3, 567.567, 5.1],
        "params.optimizer.gamma": ["hello", "moo", "hi"],
        "params.zeta": [1111.11, 9.99, -99.0],
        "metrics.f_score": [0.8, 0.4, 0.1],
        "metrics.accuracy": [0.4, 0.3, 0.2],
    }
