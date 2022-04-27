from datetime import datetime

from letstune.backend import repo
from letstune.backend.scheduler.simple import Task, get_next_tasks


def test_empty() -> None:
    ts = get_next_tasks([])

    assert len(ts) == 0


def test_get_some_tasks() -> None:
    ts = get_next_tasks(
        [
            repo.Training(
                training_id=505,
                params="????",
                epochs=[],
            ),
            repo.Training(
                training_id=896,
                params="????",
                epochs=[
                    repo.EpochStats(
                        epoch_id=0,
                        metric_values={"accuracy": 0.7},
                        start_time=datetime(2022, 10, 10, 10, 0, 0),
                        end_time=datetime(2022, 10, 10, 10, 2, 0),
                    ),
                ],
            ),
            repo.Training(
                training_id=1505,
                params="????",
                epochs=[],
                error="",
            ),
            repo.Training(
                training_id=1896,
                params="????",
                epochs=[
                    repo.EpochStats(
                        epoch_id=0,
                        metric_values={"accuracy": 0.7},
                        start_time=datetime(2022, 10, 10, 10, 0, 0),
                        end_time=datetime(2022, 10, 10, 10, 2, 0),
                    ),
                ],
                error="",
            ),
            repo.Training(
                training_id=9595,
                params="????",
                epochs=[],
            ),
        ]
    )

    assert ts == [
        Task(training_id=505),
        Task(training_id=9595),
    ]
