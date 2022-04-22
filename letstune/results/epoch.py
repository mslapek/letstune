"""Results of tuning using an epoch trainer.

:class:`TuningResults` has many :class:`Training` objects,
each representing training for a one :class:`Params` object.

Each :class:`Training` has many :class:`Epoch` objects,
each representing training for a one epoch.

Metric value
------------

In context of :class:`TuningResults` and :class:`Training` objects,
*metric value* means metric value from the *best epoch* (not last!).

Reporting
---------

:class:`TuningResults` provides useful reporting utilities,
like dataframe export.

The dataframe can be used for further analysis and visualization.

Model loading
-------------

To unpickle the best model::

    chk = tuning.best_training.best_epoch.checkpoint
    model = chk.load_pickle()

"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, Mapping, Protocol, TypeVar

from letstune import Metric, Params

from . import _base

P = TypeVar("P", bound=Params)

__all__ = [
    "CheckpointFactory",
    "Epoch",
    "Training",
    "TuningResults",
    "RoundAssigner",
    "TrainingBuilder",
    "Builder",
]


class CheckpointFactory(Protocol):
    """Type representing checkpoint factory.

    There will be large number of :class:`Training` instances.
    Factory allows to lazy initialize checkpoints.
    """

    def get_checkpoint(self, training_id: int, epoch_id: int) -> Any:
        ...


@dataclass(init=False, eq=False, slots=True, frozen=True)
class Epoch(_base.TrainingStats):
    """Single epoch from :class:`Training`."""

    training_id: int
    epoch_id: int
    cum_metric_value: float
    cum_duration: timedelta
    round: int
    _checkpoint_factory: CheckpointFactory

    @property
    def checkpoint(self) -> Any:
        """Checkpoint with fitted model during this training."""
        return self._checkpoint_factory.get_checkpoint(self.training_id, self.epoch_id)

    def __repr__(self) -> str:
        return (
            f"<Epoch {self.epoch_id} "
            f"from training {self.training_id}; "
            f"metric_value={self.metric_value}>"
        )


@dataclass(init=False, eq=False, slots=True, frozen=True)
class Training(_base.SequenceProxy[Epoch], Generic[P]):
    """Training of a model for given :class:`Params`.

    List of :class:`Epoch` objects, each representing one epoch.
    """

    training_id: int
    params: P
    _sequence: tuple[Epoch, ...]
    _metric: Metric

    @property
    def start_time(self) -> datetime:
        """Start time of the first epoch."""
        return self[0].start_time

    @property
    def end_time(self) -> datetime:
        """End time of the last epoch."""
        return self[-1].end_time

    @property
    def duration(self) -> timedelta:
        """Total duration of all epochs."""
        return sum((e.duration for e in self), start=timedelta())

    @property
    def best_epoch(self) -> Epoch:
        """The best epoch in the training."""
        return self._best()

    @property
    def last_epoch(self) -> Epoch:
        """Last epoch in the training."""
        return self[-1]

    @property
    def round(self) -> int:
        """Round survived by this training."""
        return self[-1].round

    @property
    def metric_value(self) -> float:
        """Metric value of the best epoch."""
        return self.best_epoch.metric_value

    def _to_json(self) -> dict[str, Any]:
        best_epoch = self.best_epoch
        last_epoch = self.last_epoch

        return {
            "training_id": self.training_id,
            "params": self.params._to_json(add_union_type=True),
            "best_epoch_metrics": dict(best_epoch.metric_values),
            "last_epoch_metrics": dict(last_epoch.metric_values),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "epochs": len(self),
            "best_epoch": best_epoch.epoch_id,
            "round": self.round,
        }

    def __repr__(self) -> str:
        return (
            f"<Training {self.training_id} "
            f"with {len(self)} epochs; "
            f"metric_value={self.metric_value}>"
        )


class TuningResults(_base.TuningResults[P, Training[P]]):
    """Results of epoch training.

    List of :class:`Training` objects, each representing one :class:`Params`.
    """

    pass


class RoundAssigner(Protocol):
    """Object assigning each epoch to a round, based on cumulative duration."""

    def assign_to_round(self, cum_duration: timedelta) -> int:
        ...


@dataclass(init=False, repr=False, eq=False, slots=True)
class TrainingBuilder(Generic[P]):
    """Builder of :class:`Training`. Only for *custom backend* developers.

    The builder is called repeatedly with ``add_epoch``.
    Finally, call ``build`` to add :class:`Training` to :class:`TuningResults`.
    """

    _builder: "Builder[P]"
    _params: P
    _training_id: int
    _epochs: list[Epoch]
    _cum_duration: timedelta
    _cum_metric_value: float

    def add_epoch(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        metric_values: Mapping[str, float],
    ) -> None:
        metric = self._builder._metric
        metric_value = metric_values[metric.name]

        if metric.greater_is_better:
            self._cum_metric_value = max(self._cum_metric_value, metric_value)
        else:
            self._cum_metric_value = min(self._cum_metric_value, metric_value)

        self._cum_duration += end_time - start_time

        ep = Epoch()
        object.__setattr__(ep, "training_id", self._training_id)
        object.__setattr__(ep, "epoch_id", len(self._epochs))
        object.__setattr__(ep, "start_time", start_time)
        object.__setattr__(ep, "end_time", end_time)
        object.__setattr__(
            ep, "metric_values", _base.freeze_metric_values(metric_values)
        )
        object.__setattr__(ep, "metric_value", metric_value)
        object.__setattr__(ep, "cum_metric_value", self._cum_metric_value)
        object.__setattr__(ep, "cum_duration", self._cum_duration)
        object.__setattr__(
            ep,
            "round",
            self._builder._round_assigner.assign_to_round(self._cum_duration),
        )
        object.__setattr__(ep, "_checkpoint_factory", self._builder._checkpoint_factory)

        self._epochs.append(ep)

    def build(self) -> None:
        if self._training_id == -1:
            raise RuntimeError

        if self._training_id != len(self._builder._trainings):
            raise RuntimeError

        if len(self._epochs) == 0:
            raise ValueError("training must have at least one epoch")

        t: Training[P] = Training()
        object.__setattr__(t, "training_id", self._training_id)
        object.__setattr__(t, "params", self._params)
        object.__setattr__(t, "_sequence", tuple(self._epochs))
        object.__setattr__(t, "_metric", self._builder._metric)

        self._builder._trainings.append(t)

        self._training_id = -1


class Builder(Generic[P]):
    """Builder of :class:`TuningResults`. Only for *custom backend* developers.

    For each training,
    the builder is called with ``add_training``, which returns :class:`TrainingBuilder`.

    After ``build`` was called on the training builder,
    next :class:`TrainingBuilder` can be created with
    ``add_training``.

    Finally, call ``build`` to get :class:`TuningResults`.
    """

    def __init__(
        self,
        *,
        metric: Metric,
        checkpoint_factory: CheckpointFactory,
        round_assigner: RoundAssigner,
    ):
        self._metric = metric
        self._checkpoint_factory = checkpoint_factory
        self._round_assigner = round_assigner

        self._trainings: list[Training[P]] = []

    def add_training(
        self,
        params: P,
    ) -> TrainingBuilder[P]:
        b: TrainingBuilder[P] = TrainingBuilder()

        b._builder = self
        b._params = params
        b._training_id = len(self._trainings)

        b._epochs = []

        b._cum_duration = timedelta()
        b._cum_metric_value = -math.inf if self._metric.greater_is_better else math.inf

        return b

    def build(self) -> TuningResults[P]:
        """Get new :class:`TuningResults`."""

        if len(self._trainings) == 0:
            raise ValueError("tuning must have at least one training")

        r: TuningResults[P] = TuningResults()
        r._sequence = tuple(self._trainings)
        r._metric = self._metric

        self._trainings.clear()

        return r
