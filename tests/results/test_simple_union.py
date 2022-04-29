import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

import letstune
from letstune.backend import repo
from letstune.results import simple

from . import utils
from .utils import AnotherOptimizerParams, OptimizerParams, UnionModelParams

METRIC = letstune.Metric("accuracy")


@pytest.fixture
def tuning_results() -> simple.TuningResults[UnionModelParams]:
    return simple.build(
        metric=METRIC,
        checkpoint_factory=utils.SimpleCheckpointFactory(),
        params_cls=UnionModelParams,
        trainings=[
            repo.Training(
                training_id=101,
                params=json.dumps(
                    UnionModelParams(
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
                training_id=103,
                params=json.dumps(
                    UnionModelParams(
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
                training_id=102,
                params=json.dumps(
                    UnionModelParams(
                        optimizer=AnotherOptimizerParams(gamma="moo", epsilon=3.3),
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
                    ),
                ],
            ),
        ],
    )


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
            "training_id": [101, 102, 103],
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
        },
    )

    expected_df = expected_df[list(df)]  # same column order

    pd.testing.assert_frame_equal(df, expected_df)
