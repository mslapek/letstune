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


@dataclass(kw_only=True, slots=True, frozen=True)
class Error:
    training_id: int
    params: str
    msg: str


E = TypeVar("E", bound=Error)


class TuningResults(SequenceProxy[T], Generic[P, T, E]):
    _metric: Metric
    _errors: tuple[E, ...]

    def to_df(self) -> pd.DataFrame:
        """Get dataframe describing all trainings in the tuning.

        Columns correspond to fields from training objects.
        """

        return pd.json_normalize([t._to_json() for t in self])  # type: ignore

    @property
    def metric(self) -> Metric:
        """Metric used in the tuning."""
        return self._metric

    @property
    def metric_value(self) -> float:
        return self[0].metric_value  # type: ignore

    @property
    def errors(self) -> Sequence[E]:
        """Errors got during tuning."""
        return self._errors

    def __repr__(self) -> str:
        metric_value: str | float
        try:
            metric_value = self.metric_value
        except IndexError:
            metric_value = "???"

        return (
            f"<TuningResults with {len(self)} trainings; "
            f"metric_value={metric_value}>"
        )
