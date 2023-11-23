import math
from datetime import datetime, timedelta

from letstune.backend import repo
from letstune.backend.scheduler.epoch import _normalize, _normalize_training, _Training

METRIC = "accuracy"


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
                    metric_values={"accuracy": 0.7},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"accuracy": 0.6},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
        ),
    )

    assert t == _Training(
        training_id=602,
        total_duration=timedelta(minutes=4),
        next_epoch=2,
        metric_value=0.7,
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
        total_duration=timedelta(),
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
                    metric_values={"accuracy": 0.7},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"accuracy": 0.6},
                    start_time=datetime(2022, 10, 10, 10, 6, 0),
                    end_time=datetime(2022, 10, 10, 10, 8, 0),
                ),
            ],
            error="",
        ),
    )

    assert t == _Training(
        training_id=662,
        total_duration=timedelta(minutes=4),
        next_epoch=2,
        metric_value=-math.inf,
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
                    metric_values={"accuracy": 0.7},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"accuracy": 0.6},
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
                    metric_values={"accuracy": 0.78},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"accuracy": 0.6},
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
                    metric_values={"accuracy": 0.4},
                    start_time=datetime(2022, 10, 10, 10, 0, 0),
                    end_time=datetime(2022, 10, 10, 10, 2, 0),
                ),
                repo.EpochStats(
                    epoch_id=1,
                    metric_values={"accuracy": 0.65},
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
            total_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=0.7,
        ),
        _Training(
            training_id=10,
            total_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=0.65,
        ),
        _Training(
            training_id=662,
            total_duration=timedelta(minutes=4),
            next_epoch=2,
            metric_value=-math.inf,
            error=True,
        ),
    ]
