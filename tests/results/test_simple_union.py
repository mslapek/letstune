import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.results import simple

from . import utils
from .utils import AnotherOptimizerParams, OptimizerParams, UnionModelParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> simple.TuningResults[UnionModelParams]:
    builder: simple.Builder[UnionModelParams] = simple.Builder(
        metric=METRIC, checkpoint_factory=utils.SimpleCheckpointFactory()
    )

    builder.add_training(
        UnionModelParams(
            optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"), zeta=1111.11
        ),
        start_time=datetime(2022, 10, 1, 10, 0, 0),
        end_time=datetime(2022, 10, 1, 10, 2, 0),
        metric_values={
            "f_score": 0.8,
            "accuracy": 0.4,
        },
    )

    builder.add_training(
        UnionModelParams(
            optimizer=OptimizerParams(alpha=3, beta=5.1, gamma="hi"), zeta=-99.0
        ),
        start_time=datetime(2022, 10, 1, 10, 1, 0),
        end_time=datetime(2022, 10, 1, 10, 5, 0),
        metric_values={
            "f_score": 0.1,
            "accuracy": 0.2,
        },
    )

    builder.add_training(
        UnionModelParams(
            optimizer=AnotherOptimizerParams(gamma="moo", epsilon=3.3), zeta=9.99
        ),
        start_time=datetime(2022, 10, 1, 11, 0, 0),
        end_time=datetime(2022, 10, 1, 12, 0, 0),
        metric_values={
            "f_score": 0.4,
            "accuracy": 0.3,
        },
    )

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
        "metrics.f_score": np.dtype("float64"),
        "metrics.accuracy": np.dtype("float64"),
        "start_time": np.dtype("<M8[ns]"),
        "end_time": np.dtype("<M8[ns]"),
        "duration": np.dtype("<m8[ns]"),
    }

    nan = math.nan

    expected_df = pd.DataFrame(
        {
            "duration": [
                Timedelta("0 days 00:02:00"),
                Timedelta("0 days 01:00:00"),
                Timedelta("0 days 00:04:00"),
            ],
            "end_time": [
                Timestamp("2022-10-01 10:02:00"),
                Timestamp("2022-10-01 12:00:00"),
                Timestamp("2022-10-01 10:05:00"),
            ],
            "metrics.accuracy": [0.4, 0.3, 0.2],
            "metrics.f_score": [0.8, 0.4, 0.1],
            "params.optimizer.AnotherOptimizerParams.epsilon": [nan, 3.3, nan],
            "params.optimizer.AnotherOptimizerParams.gamma": [nan, "moo", nan],
            "params.optimizer.OptimizerParams.alpha": [1.0, nan, 3.0],
            "params.optimizer.OptimizerParams.beta": [3.3, nan, 5.1],
            "params.optimizer.OptimizerParams.gamma": ["hello", nan, "hi"],
            "params.optimizer.type": [
                "OptimizerParams",
                "AnotherOptimizerParams",
                "OptimizerParams",
            ],
            "params.zeta": [1111.11, 9.99, -99.0],
            "start_time": [
                Timestamp("2022-10-01 10:00:00"),
                Timestamp("2022-10-01 11:00:00"),
                Timestamp("2022-10-01 10:01:00"),
            ],
            "training_id": [0, 2, 1],
        },
    )

    expected_df = expected_df[list(df)]  # same column order

    pd.testing.assert_frame_equal(df, expected_df)
