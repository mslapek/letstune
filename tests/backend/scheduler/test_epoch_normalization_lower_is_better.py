import math
from datetime import datetime, timedelta

from letstune import Metric
from letstune.backend import repo
from letstune.backend.scheduler.epoch import _normalize, _normalize_training, _Training

METRIC = Metric("mean_square_error")


def test_empty() -> None:
    ts = _normalize(METRIC, [])

    assert len(ts) == 0


def test_normalize_training() -> None:
    t = _normalize_training(
        METRIC,
        repo.Training(
            training_id=602,
            params="?????",
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    metric_values={"mean_square_error": 10.0},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"mean_square_error": 100.0},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
        ),
    )

    assert t == _Training(
        training_id=602,
        cum_duration=timedelta(minutes=4),
        next_epoch=2,
        metric_value=10.0,
    )


def test_normalize_empty_training() -> None:
    t = _normalize_training(
        METRIC,
        repo.Training(
            training_id=602,
            params="?????",
        ),
    )

    assert t == _Training(
        training_id=602,
        cum_duration=timedelta(),
        next_epoch=0,
        metric_value=0.0,
    )


def test_normalize_error_training() -> None:
    t = _normalize_training(
        METRIC,
        repo.Training(
            training_id=662,
            params="?????",
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    metric_values={"mean_square_error": 10.0},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"mean_square_error": 100.0},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
            error="",
        ),
    )

    assert t == _Training(
        training_id=662,
        cum_duration=timedelta(minutes=4),
        next_epoch=2,
        metric_value=math.inf,
        error=True,
    )


def test_sort() -> None:
    trainings = [
        repo.Training(
            training_id=602,
            params="?????",
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    metric_values={"mean_square_error": 10.0},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"mean_square_error": 100.0},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
        ),
        repo.Training(
            training_id=662,
            params="?????",
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    metric_values={"mean_square_error": 8.0},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"mean_square_error": 100.0},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
            error="",
        ),
        repo.Training(
            training_id=10,
            params="?????",
            epochs=[
                repo.EpochStats(
                    epoch_id=0,
                    metric_values={"mean_square_error": 80.0},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"mean_square_error": 50.0},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
        ),
    ]

    ts = _normalize(METRIC, trainings)

    assert ts == [
        _Training(
            training_id=602,
            cum_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=10.0,
        ),
        _Training(
            training_id=10,
            cum_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=50.0,
        ),
        _Training(
            training_id=662,
            cum_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=math.inf,
            error=True,
        ),
    ]
