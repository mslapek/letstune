"""Results of tuning using an epoch trainer.

:class:`TuningResults` has many :class:`Training` objects,
each representing training for a one :class:`letstune.Params` object.

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

    chk = tuning[0].best_epoch.checkpoint
    model = chk.load_pickle()

"""
import json
import math
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, Protocol, Sequence, TypeVar

from letstune import Metric, Params
from letstune.backend import repo
from letstune.backend.scheduler.epoch import Config

from . import _base

__all__ = [
    "CheckpointFactory",
    "Epoch",
    "Training",
    "TuningResults",
    "Error",
    "build",
]

P = TypeVar("P", bound=Params)


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
    total_metric_value: float
    total_duration: timedelta
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
    """Training of a model for given :class:`letstune.Params`.

    List of :class:`Epoch` objects, each representing one epoch.
    """

    training_id: int
    params: P
    round: int
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
        cmp = max if self._metric.greater_is_better else min
        return cmp(
            self,
            key=operator.attrgetter("metric_value"),
        )

    @property
    def last_epoch(self) -> Epoch:
        """Last epoch in the training."""
        return self[-1]

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


@dataclass(kw_only=True, slots=True, frozen=True)
class Error(_base.Error):
    pass


class TuningResults(_base.TuningResults[P, Training[P], Error]):
    """Results of epoch training.

    List of :class:`Training` objects, each representing one :class:`letstune.Params`.
    """


def _build_training(
    metric: Metric,
    checkpoint_factory: CheckpointFactory,
    params_cls: type[P],
    training: repo.Training,
) -> Training[P]:
    epochs = []
    total_metric_value = -math.inf if metric.greater_is_better else math.inf
    total_duration = timedelta()

    for e in training.epochs:
        metric_value = e.metric_values[metric.name]

        if metric.greater_is_better:
            total_metric_value = max(total_metric_value, metric_value)
        else:
            total_metric_value = min(total_metric_value, metric_value)

        total_duration += e.duration

        ep = Epoch()
        object.__setattr__(ep, "training_id", training.training_id)
        object.__setattr__(ep, "epoch_id", e.epoch_id)
        object.__setattr__(ep, "start_time", e.start_time)
        object.__setattr__(ep, "end_time", e.end_time)
        object.__setattr__(
            ep, "metric_values", _base.freeze_metric_values(e.metric_values)
        )
        object.__setattr__(ep, "metric_value", metric_value)
        object.__setattr__(ep, "total_metric_value", total_metric_value)
        object.__setattr__(ep, "total_duration", total_duration)
        object.__setattr__(ep, "_checkpoint_factory", checkpoint_factory)

        epochs.append(ep)

    if len(epochs) == 0:
        raise ValueError("training must have at least one epoch")

    t: Training[P] = Training()
    object.__setattr__(t, "training_id", training.training_id)
    object.__setattr__(t, "params", params_cls.from_json(json.loads(training.params)))
    object.__setattr__(t, "_sequence", tuple(epochs))
    object.__setattr__(t, "_metric", metric)

    return t


def _get_rounds(config: Config, trainings_count: int) -> list[int]:
    rounds = [0] * trainings_count
    n = len(rounds)

    for round in range(1, len(config.round_durations)):
        n = config.reduced_trainings_number(n)

        for i in range(n):
            rounds[i] = round

    return rounds


def build(
    *,
    metric: Metric,
    checkpoint_factory: CheckpointFactory,
    params_cls: type[P],
    trainings: Sequence[repo.Training],
    config: Config,
) -> TuningResults[P]:
    """Build :class:`TuningResults`. Only for *custom backend* developers."""

    if len(trainings) == 0:
        raise ValueError("tuning must have at least one training")

    valid_ts: list[Training[P]] = []
    errors: list[Error] = []
    for t in trainings:
        if t.error is None and len(t.epochs) > 0:
            valid_ts.append(
                _build_training(
                    metric,
                    checkpoint_factory,
                    params_cls,
                    t,
                )
            )
        else:
            error = t.error

            if error is None:
                if len(t.epochs) == 0:
                    error = "no epochs"
                else:
                    error = "unknown"

            errors.append(
                Error(
                    training_id=t.training_id,
                    params=t.params,
                    msg=error,
                )
            )

    valid_ts.sort(
        key=operator.attrgetter("metric_value"),
        reverse=metric.greater_is_better,
    )

    rounds = _get_rounds(config, len(trainings))
    for i, round in enumerate(rounds):
        if i >= len(valid_ts):
            break

        object.__setattr__(valid_ts[i], "round", round)

    r: TuningResults[P] = TuningResults()
    r._sequence = tuple(valid_ts)
    r._metric = metric
    r._errors = tuple(errors)

    return r
