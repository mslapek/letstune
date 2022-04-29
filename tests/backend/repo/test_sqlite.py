import operator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from letstune.backend import repo
from letstune.backend.repo.sqlite import SqliteRepo

TRAINING_67 = repo.Training(
    training_id=67,
    params='{"lambda": 67}',
    epochs=(
        repo.EpochStats(
            epoch_id=0,
            metric_values={"accuracy": 0.7, "f_score": 0.3},
            start_time=datetime(2030, 10, 10, 0, 10, 0),
            end_time=datetime(2030, 10, 10, 0, 10, 1),
        ),
        repo.EpochStats(
            epoch_id=2,
            metric_values={"accuracy": 0.9, "f_score": 0.8},
            start_time=datetime(2030, 10, 10, 0, 11, 0, tzinfo=timezone.utc),
            end_time=datetime(2030, 10, 10, 0, 11, 1, tzinfo=timezone.utc),
        ),
    ),
)

EXPECTED_TRAININGS = [
    repo.Training(training_id=1, params='{"lambda": 1}', epochs=tuple()),
    repo.Training(
        training_id=13,
        params='{"lambda": 13}',
        epochs=tuple(),
        error="shot down in flames",
    ),
    TRAINING_67,
]


def fill_repo(r: repo.Repository) -> None:
    for tid in [1, 67, 13]:
        r.add_training(tid, '{"lambda": ' + str(tid) + "}")

    for e in TRAINING_67.epochs:
        r.add_epoch(67, e)

    r.set_error(13, "shot down in flames")


@pytest.fixture
def sqlite_repo() -> repo.Repository:
    r = SqliteRepo(":memory:")
    fill_repo(r)
    return r


@pytest.fixture
def test_is_empty() -> None:
    r = SqliteRepo(":memory:")

    trainings = r.get_all_trainings()

    assert isinstance(trainings, tuple)
    assert len(trainings) == 0


def test_get_all_trainings(sqlite_repo: repo.Repository) -> None:
    trainings = sqlite_repo.get_all_trainings()

    assert isinstance(trainings, tuple)
    assert all(isinstance(t, repo.Training) for t in trainings)

    ts = sorted(trainings, key=operator.attrgetter("training_id"))

    assert ts == EXPECTED_TRAININGS


def test_get_training(sqlite_repo: repo.Repository) -> None:
    training = sqlite_repo.get_training(67)

    assert training == TRAINING_67


def test_get_missing_training(sqlite_repo: repo.Repository) -> None:
    with pytest.raises(ValueError):
        _ = sqlite_repo.get_training(999)


def test_doubled_training_id(sqlite_repo: repo.Repository) -> None:
    with pytest.raises(ValueError):
        sqlite_repo.add_training(67, "cosmos")

    assert sqlite_repo.get_training(67) == TRAINING_67


def test_double_epoch_id(sqlite_repo: repo.Repository) -> None:
    with pytest.raises(ValueError):
        sqlite_repo.add_epoch(
            67,
            repo.EpochStats(
                epoch_id=0,
                metric_values={"accuracy": 0.0, "f_score": 0.0},
                start_time=datetime(2040, 10, 10, 0, 11, 0),
                end_time=datetime(2040, 10, 10, 0, 11, 1),
            ),
        )

    assert sqlite_repo.get_training(67) == TRAINING_67


def test_db_reopen(tmp_path: Path) -> None:
    p = tmp_path / "stats.db"

    try:
        r = SqliteRepo(p)
        fill_repo(r)
    finally:
        r.close()

    try:
        r = SqliteRepo(p)
        trainings = r.get_all_trainings()

        ts = sorted(trainings, key=operator.attrgetter("training_id"))
        assert ts == EXPECTED_TRAININGS
    finally:
        r.close()
