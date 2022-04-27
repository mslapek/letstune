from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

JSON = str


@dataclass(kw_only=True, slots=True, frozen=True)
class EpochStats:
    epoch_id: int
    metric_values: Mapping[str, float]
    start_time: datetime
    end_time: datetime

    @property
    def duration(self) -> timedelta:
        """Total duration of epoch."""
        return self.end_time - self.start_time


@dataclass(kw_only=True, slots=True, frozen=True)
class Training:
    training_id: int
    params: JSON
    epochs: Sequence[EpochStats]
    error: str | None = None


class Repository(ABC):
    @abstractmethod
    def get_all_trainings(self) -> Sequence[Training]:
        ...

    @abstractmethod
    def add_training(self, training_id: int, params: JSON) -> None:
        """Add a new training.

        Raises :class:`ValueError`, when given ``training_id`` is already
        occupied."""
        ...

    @abstractmethod
    def add_epoch(self, training_id: int, epoch_stats: EpochStats) -> int:
        """Add a new epoch.

        Raises :class:`ValueError`, when tuple
        ``(training_id, epoch_stats.epoch_id)`` is already
        occupied."""
        ...

    @abstractmethod
    def get_training(self, training_id: int) -> Training:
        ...

    @abstractmethod
    def set_error(self, training_id: int, description: str) -> None:
        ...

    def __close__(self) -> None:
        pass
