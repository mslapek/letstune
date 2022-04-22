"""Results of tuning using a simple trainer.

:class:`TuningResults` has many :class:`Training` objects,
each representing training for a one :class:`Params` object.

Reporting
---------

:class:`TuningResults` provides useful reporting utilities,
like dataframe export.

The dataframe can be used for further analysis and visualization.

Model loading
-------------

To unpickle the best model::

    chk = tuning.best_training.checkpoint
    model = chk.load_pickle()

"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Mapping, Protocol, TypeVar

from letstune import Metric, Params

from . import _base

__all__ = [
    "CheckpointFactory",
    "Training",
    "TuningResults",
    "Builder",
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
    """Training of a model for given :class:`Params`."""

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


class TuningResults(_base.TuningResults[P, Training[P]]):
    """Results of simple training.

    List of :class:`Training` objects, each representing one :class:`Params`.
    """

    pass


class Builder(Generic[P]):
    """Builder of :class:`TuningResults`. Only for *custom backend* developers.

    The builder is called repeatedly with ``add_training``.
    Finally, call ``build`` to get :class:`TuningResults`.
    """

    def __init__(
        self,
        *,
        metric: Metric,
        checkpoint_factory: CheckpointFactory,
    ):
        self._metric = metric
        self._checkpoint_factory = checkpoint_factory

        self._trainings: list[Training[P]] = []

    def add_training(
        self,
        params: P,
        *,
        start_time: datetime,
        end_time: datetime,
        metric_values: Mapping[str, float],
    ) -> None:
        t: Training[P] = Training()
        object.__setattr__(t, "training_id", len(self._trainings))
        object.__setattr__(t, "params", params)
        object.__setattr__(t, "start_time", start_time)
        object.__setattr__(t, "end_time", end_time)
        object.__setattr__(
            t, "metric_values", _base.freeze_metric_values(metric_values)
        )
        object.__setattr__(t, "metric_value", metric_values[self._metric.name])
        object.__setattr__(t, "_checkpoint_factory", self._checkpoint_factory)

        self._trainings.append(t)

    def build(self) -> TuningResults[P]:
        """Get new :class:`TuningResults`."""

        if len(self._trainings) == 0:
            raise ValueError("tuning must have at least one training")

        r: TuningResults[P] = TuningResults()
        r._sequence = tuple(self._trainings)
        r._metric = self._metric

        self._trainings.clear()

        return r
