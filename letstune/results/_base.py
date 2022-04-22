import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Generic, Mapping, Sequence, TypeVar, overload

import pandas as pd

from letstune import Metric, Params

P = TypeVar("P", bound=Params)
T = TypeVar("T")


def freeze_metric_values(
    metric_values: Mapping[str, float]
) -> MappingProxyType[str, float]:
    return MappingProxyType({k: float(v) for k, v in metric_values.items()})


class SequenceProxy(Sequence[T]):
    __slots__: tuple[str, ...] = ()

    _sequence: tuple[T, ...]
    _metric: Metric

    @overload
    def __getitem__(self, indices: int) -> T:
        ...

    @overload
    def __getitem__(self, indices: slice) -> list[T]:
        ...

    def __getitem__(self, indices: int | slice) -> T | list[T]:
        if isinstance(indices, int):
            return self._sequence[indices]
        elif isinstance(indices, slice):
            return list(self._sequence[indices])
        else:
            raise TypeError("indices must be integers or slices")

    def __len__(self) -> int:
        return len(self._sequence)

    def _best(self) -> T:
        cmp = max if self._metric.greater_is_better else min
        return cmp(
            self,
            key=operator.attrgetter("metric_value"),
        )


@dataclass(init=False, repr=False, eq=False, slots=True, frozen=True)
class TrainingStats:
    metric_values: MappingProxyType[str, float]
    metric_value: float
    start_time: datetime
    end_time: datetime

    @property
    def duration(self) -> timedelta:
        """Total duration of training."""
        return self.end_time - self.start_time


class TuningResults(SequenceProxy[T], Generic[P, T]):
    @property
    def best_training(self) -> T:
        """The best training in the tuning."""
        return self._best()

    def sorted_trainings(self) -> list[T]:
        """Get a new list with sorted trainings, with the best at the top."""

        return sorted(
            self,
            key=operator.attrgetter("metric_value"),
            reverse=self.metric.greater_is_better,
        )

    def top_trainings(self, n: int = 5) -> list[T]:
        """Get ``n`` best trainings."""
        return self.sorted_trainings()[:n]

    def to_df(self) -> pd.DataFrame:
        """Get dataframe describing all trainings in the tuning.

        Columns correspond to fields from training objects.
        """

        return pd.json_normalize(
            [t._to_json() for t in self.sorted_trainings()]  # type: ignore
        )

    @property
    def metric(self) -> Metric:
        """Metric used in the tuning."""
        return self._metric

    def __repr__(self) -> str:
        return (
            f"<TuningResults with {len(self)} trainings; "
            f"best_training.metric_value="
            f"{self.best_training.metric_value}>"  # type: ignore
        )
