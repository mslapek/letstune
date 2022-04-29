import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Sequence

from letstune.backend import repo
from letstune.backend.repo import JSON, EpochStats, Training

__all__ = ["SqliteRepo"]


class SqliteRepo(repo.Repository):
    def __init__(self, file: str | Path):
        self._connection = sqlite3.connect(str(file))
        _create_db(self._connection)

    def get_all_trainings(self) -> Sequence[Training]:
        cur = self._connection.cursor()
        try:
            all_training_epochs: dict[int, list[repo.EpochStats]] = {}

            for (
                training_id,
                epoch_id,
                metric_values,
                start_time,
                end_time,
            ) in cur.execute(
                "SELECT training_id, epoch_id,"
                "metric_values, "
                "start_time, end_time FROM epochs "
                "ORDER BY training_id, epoch_id"
            ):
                try:
                    training_epochs = all_training_epochs[training_id]
                except KeyError:
                    training_epochs = []
                    all_training_epochs[training_id] = training_epochs

                training_epochs.append(
                    repo.EpochStats(
                        epoch_id=epoch_id,
                        metric_values=json.loads(metric_values),
                        start_time=datetime.fromisoformat(start_time),
                        end_time=datetime.fromisoformat(end_time),
                    )
                )

            return tuple(
                repo.Training(
                    training_id=training_id,
                    params=params,
                    epochs=tuple(all_training_epochs.get(training_id, tuple())),
                    error=error,
                )
                for (training_id, params, error) in cur.execute(
                    "SELECT training_id, params, error FROM trainings"
                )
            )
        finally:
            cur.close()

    def add_training(self, training_id: int, params: JSON) -> None:
        cur = self._connection.cursor()
        try:
            cur.execute(
                "INSERT INTO trainings VALUES (?, ?, NULL)",
                (training_id, params),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("training already exists", e)
        finally:
            cur.close()

    def add_epoch(self, training_id: int, epoch_stats: EpochStats) -> None:
        cur = self._connection.cursor()
        try:
            cur.execute(
                "INSERT INTO epochs VALUES (?, ?, ?, ?, ?)",
                (
                    training_id,
                    epoch_stats.epoch_id,
                    json.dumps(epoch_stats.metric_values),
                    epoch_stats.start_time.isoformat(),
                    epoch_stats.end_time.isoformat(),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("epoch already exists", e)
        finally:
            cur.close()

    def set_error(self, training_id: int, description: str) -> None:
        cur = self._connection.cursor()
        try:
            cur.execute(
                "UPDATE trainings SET error = ? WHERE training_id = ?",
                (description, training_id),
            )
            self._connection.commit()
        finally:
            cur.close()

    def close(self) -> None:
        self._connection.close()


def _create_db(connection: sqlite3.Connection) -> None:
    cur = connection.cursor()
    try:
        cur.executescript(
            """
CREATE TABLE IF NOT EXISTS trainings (
    training_id INTEGER,
    params TEXT,
    error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS trainings_index
ON trainings(training_id);

CREATE TABLE IF NOT EXISTS epochs (
    training_id INTEGER,
    epoch_id INTEGER,
    metric_values TEXT,
    start_time TEXT,
    end_time TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS epochs_index
ON epochs(training_id, epoch_id);
        """
        )
        connection.commit()
    finally:
        cur.close()
