import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.results import epoch, simple

from . import utils
from .utils import AnotherOptimizerParams, OptimizerParams, UnionModelParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> epoch.TuningResults[UnionModelParams]:
    builder: epoch.Builder[UnionModelParams] = epoch.Builder(
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
        UnionModelParams(
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
        UnionModelParams(
            optimizer=AnotherOptimizerParams(gamma="moo", epsilon=3.3), zeta=-99.0
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


def test_to_df(tuning_results: simple.TuningResults[UnionModelParams]) -> None:
    df = tuning_results.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.dtypes.to_dict() == {
        "training_id": np.dtype("int64"),
        "params.optimizer.type": np.dtype("object"),
        "params.optimizer.OptimizerParams.alpha": np.dtype("float64"),
        "params.optimizer.OptimizerParams.beta": np.dtype("float64"),
        "params.optimizer.OptimizerParams.gamma": np.dtype("object"),
        "params.optimizer.AnotherOptimizerParams.gamma": np.dtype("object"),
        "params.optimizer.AnotherOptimizerParams.epsilon": np.dtype("float64"),
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

    nan = math.nan

    expected_df = pd.DataFrame(
        {
            "training_id": [0, 1],
            "start_time": [
                Timestamp("2022-10-01 10:00:00"),
                Timestamp("2022-10-01 15:00:00"),
            ],
            "end_time": [
                Timestamp("2022-10-01 10:11:00"),
                Timestamp("2022-10-01 15:14:00"),
            ],
            "duration": [Timedelta("0 days 00:05:00"), Timedelta("0 days 00:07:00")],
            "epochs": [3, 4],
            "best_epoch": [1, 3],
            "round": [5, 99],
            "params.optimizer.OptimizerParams.alpha": [1.0, nan],
            "params.optimizer.OptimizerParams.beta": [3.3, nan],
            "params.optimizer.OptimizerParams.gamma": ["hello", nan],
            "params.optimizer.type": ["OptimizerParams", "AnotherOptimizerParams"],
            "params.zeta": [1111.11, -99.0],
            "best_epoch_metrics.f_score": [0.5, 0.8],
            "best_epoch_metrics.accuracy": [0.7, 0.65],
            "last_epoch_metrics.f_score": [0.3, 0.8],
            "last_epoch_metrics.accuracy": [0.5, 0.65],
            "params.optimizer.AnotherOptimizerParams.gamma": [nan, "moo"],
            "params.optimizer.AnotherOptimizerParams.epsilon": [nan, 3.3],
        }
    )

    expected_df = expected_df[list(df)]  # same column order

    pd.testing.assert_frame_equal(df, expected_df)
