import json
from datetime import datetime, timedelta

import pytest

from letstune.backend import repo
from letstune.backend.scheduler.epoch import Config
from letstune.results import epoch

from . import utils
from .utils import ModelParams, OptimizerParams

METRIC = "accuracy"


@pytest.fixture
def tuning_results() -> epoch.TuningResults[ModelParams]:
    trainings = [
        repo.Training(
            training_id=i,
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
                        "accuracy": 0.0,
                    },
                ),
                repo.EpochStats(
                    epoch_id=1,
                    start_time=datetime(2022, 10, 1, 10, 2, 0),
                    end_time=datetime(2022, 10, 1, 10, 4, 0),
                    metric_values={
                        "f_score": 0.5,
                        "accuracy": 1 / (i + 1),
                    },
                ),
                repo.EpochStats(
                    epoch_id=2,
                    start_time=datetime(2022, 10, 1, 10, 10, 0),
                    end_time=datetime(2022, 10, 1, 10, 11, 0),
                    metric_values={
                        "f_score": 0.3,
                        "accuracy": 0.0,
                    },
                ),
            ],
            error="err" if i > 94 else None,
        )
        for i in range(100)
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
        ),
        params_cls=ModelParams,
        trainings=trainings,
    )


def test_rounds(tuning_results: epoch.TuningResults[ModelParams]) -> None:
    rounds = [t.round for t in tuning_results]

    assert rounds == [2] * 7 + [1] * 18 + [0] * (75 - 5)
