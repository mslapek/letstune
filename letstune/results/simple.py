"""Results of tuning using a simple trainer.

:class:`TuningResults` has many :class:`Training` objects,
each representing training for a one :class:`letstune.Params` object.

Reporting
---------

:class:`TuningResults` provides useful reporting utilities,
like dataframe export.

The dataframe can be used for further analysis and visualization.

Model loading
-------------

To unpickle the best model::

    chk = tuning[0].checkpoint
    model = chk.load_pickle()

"""
import json
import operator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Sequence, TypeVar

from letstune import Metric, Params
from letstune.backend import repo

from . import _base

__all__ = [
    "CheckpointFactory",
    "Training",
    "Error",
    "TuningResults",
    "build",
]


P = TypeVar("P", bound=Params)


class CheckpointFactory(Protocol):
    """Type representing checkpoint factory.

    There will be large number of :class:`Training` instances.
    Factory allows to lazy initialize checkpoints.
    """

    def get_checkpoint(self, training_id: int) -> Any:
        ...


@dataclass(init=False, eq=False, slots=True, frozen=True)
class Training(_base.TrainingStats, Generic[P]):
    """Training of a model for given :class:`letstune.Params`."""

    params: P
    training_id: int
    _checkpoint_factory: CheckpointFactory

    @property
    def checkpoint(self) -> Any:
        """Checkpoint with fitted model during this training."""
        return self._checkpoint_factory.get_checkpoint(self.training_id)

    def _to_json(self) -> dict[str, Any]:
        return {
            "training_id": self.training_id,
            "params": self.params._to_json(add_union_type=True),
            "metrics": dict(self.metric_values),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return f"<Training {self.training_id}; metric_value={self.metric_value}>"


@dataclass(kw_only=True, slots=True, frozen=True)
class Error(_base.Error):
    pass


class TuningResults(_base.TuningResults[P, Training[P], Error]):
    """Results of simple training.

    List of :class:`Training` objects, each representing one :class:`letstune.Params`.
    """


def _build_training(
    metric: Metric,
    checkpoint_factory: CheckpointFactory,
    params_cls: type[P],
    training: repo.Training,
) -> Training[P]:
    epoch = training.epochs[0]

    t: Training[P] = Training()
    object.__setattr__(t, "training_id", training.training_id)
    object.__setattr__(t, "params", params_cls.from_json(json.loads(training.params)))
    object.__setattr__(t, "start_time", epoch.start_time)
    object.__setattr__(t, "end_time", epoch.end_time)
    object.__setattr__(
        t, "metric_values", _base.freeze_metric_values(epoch.metric_values)
    )
    object.__setattr__(t, "metric_value", epoch.metric_values[metric.name])
    object.__setattr__(t, "_checkpoint_factory", checkpoint_factory)

    return t


def build(
    *,
    metric: Metric,
    checkpoint_factory: CheckpointFactory,
    params_cls: type[P],
    trainings: Sequence[repo.Training],
) -> TuningResults[P]:
    """Build :class:`TuningResults`. Only for *custom backend* developers."""

    if len(trainings) == 0:
        raise ValueError("tuning must have at least one training")

    valid_ts: list[Training[P]] = []
    errors: list[Error] = []
    for t in trainings:
        if t.error is None and len(t.epochs) == 1:
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
                if len(t.epochs) != 1:
                    error = "no trainings"
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

    r: TuningResults[P] = TuningResults()
    r._sequence = tuple(valid_ts)
    r._metric = metric
    r._errors = tuple(errors)

    return r
