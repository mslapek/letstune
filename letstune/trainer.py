"""Base classes used to describe your experiment.

*Trainer* is a class, which knows how to *train* a model!

:class:`SimpleTrainer` is good for sklearn-like models.

:class:`EpochTrainer` is perfect for neural networks and gradient boosting.

Metric values
-------------

Methods ``train`` and ``train_epoch`` always return *metric values*.

Metric values are represented by a dictionary with measured metric values,
for instance::

    metric_values = {
        "train_accuracy": 0.92,
        "valid_accuracy": 0.83,
        "train_f1_score": 0.76,
        "valid_f1_score": 0.66,
    }

Each trainer has a field ``metric`` of the type :class:`letstune.Metric`.

``metric.name`` declares, which metric value should be optimized by letstune.
Other metric values are stored only for an analysis after a tuning.

Serialization
-------------

All trainers should be pickleable *before* any call of ``load_dataset``.

After a call of ``load_dataset``, a trainer might be *not* pickleable.

Trainer reusability
-------------------

Trainers before ``load_dataset`` might be cloned by::

    new_trainer = pickle.loads(pickle.dumps(trainer))

``load_dataset`` can be called *at most once* on a trainer.

Other methods, including ``train``, ``load`` and ``create_model``,
can be called many times.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar

import numpy as np

import letstune

from .metrics import Metric, MetricValues
from .params import Params

__all__ = [
    "SimpleTrainer",
    "EpochTrainer",
]

P = TypeVar("P", bound=Params)


def _get_all_generic_params(obj: Any) -> Iterable[type]:
    for base in obj.__orig_bases__:
        yield from getattr(base, "__args__", tuple())


def _get_params(obj: Any) -> type[letstune.Params]:
    candidates = {
        t for t in _get_all_generic_params(obj) if issubclass(t, letstune.Params)
    }

    if len(candidates) > 1:
        raise ValueError(f"Expected one Params class in base, got {candidates}")

    if len(candidates) == 0:
        raise ValueError("Got no generic parameters in base")

    return next(iter(candidates))


class _BaseTrainer(ABC, Generic[P]):
    """Common features of :class:`SimpleTrainer` and :class:`EpochTrainer`."""

    @property
    @abstractmethod
    def metric(self) -> Metric:
        """Goal of a tuning.

        Methods ``train`` and ``train_epoch`` must return a dict containing
        name of this metric."""
        pass

    def get_random_params(self, rng: np.random.Generator) -> P:
        """Get random instance of the params.

        For a class inheriting from :class:`letstune.SimpleTrainer` [``P``]
        or :class:`letstune.EpochTrainer` [``P``],
        it returns a random instance of ``P``.
        """
        p = _get_params(type(self))
        return p.get_random_params(rng)  # type: ignore

    @abstractmethod
    def load_dataset(self, dataset: Any) -> None:
        """Load a dataset.

        ``dataset`` is a value passed-through by the function :func:`letstune.tune`.
        Usually a path to a directory with training data.

        It is expected to save the loaded dataset to ``self``, like in this example::

            def load_dataset(self, dataset):
                data = pd.read_csv(dataset)
                self.x = data[["bill_length_mm", "bill_depth_mm"]]
                self.y = data["species"]

        You can ignore the ``dataset`` parameter.
        """
        pass


class SimpleTrainer(_BaseTrainer[P]):
    """Base class for experiments *without* early-stopping.

    **Lifecycle**

    First, :meth:`load_dataset` is called with a dataset.
    It is expected to initialize fields related to the dataset.

    Then :meth:`train` is repeatedly called with various params.
    """

    @abstractmethod
    def train(self, params: P) -> tuple[Any, MetricValues]:
        """Train model parametrized by ``params`` and return it with metrics.

        Returns a tuple with ``(fitted_model, metric_values)``.

        Metric values are described in the documentation of
        ``letstune.trainer`` module.
        """
        pass

    def save(self, checkpoint: Any, model: Any) -> None:
        """Save a fitted model.

        ``checkpoint`` is an object passed from backend.
        Usually it has method ``save_pickle(model)``.

        ``model`` is the fitted model returned from ``train`` method.

        The default implementation pickles the model.
        """
        checkpoint.save_pickle(model)


class EpochTrainer(_BaseTrainer[P]):
    """Base class for experiments *with* early-stopping.

    An epoch trainer contains a currently trained model as a field;
    usually in ``self.model``.

    **Lifecycle**

    Training is epoch oriented.

    First, :meth:`load_dataset` is called with a dataset.
    It is expected to initialize fields related to the dataset.

    Then, model training for given params is performed:

    * First, :meth:`create_model` or :meth:`load` is called (but *NOT* both!).
    * :meth:`train_epoch` is repeatedly called.
    * Finally, :meth:`save` is called.

    The cycle, excluding :meth:`load_dataset`,
    might be repeated for different params.
    """

    def create_model(self, params: P) -> None:
        """Create a model parametrized by ``params``.

        Usually stores the model in ``self.model``.

        The default implementation uses ``params.create_model()``.
        """
        self.model = params.create_model()  # type: ignore

    @abstractmethod
    def train_epoch(self, epoch: int) -> MetricValues:
        """Train the model for a next epoch.

        Returns new metric values.

        Metric values are described in the documentation of
        ``letstune.trainer`` module.

        ``epoch`` is zero-indexed.
        """
        pass

    def save(self, checkpoint: Any) -> None:
        """Save the model in the current state.

        ``checkpoint`` is an object passed from backend.
        Usually it has method ``save_pickle(model)``.

        The default implementation pickles ``self.model``.
        """
        checkpoint.save_pickle(self.model)

    def load(self, checkpoint: Any, params: P) -> None:
        """Load the model from ``checkpoint``.

        ``checkpoint`` is an object passed from backend.
        Usually it has method ``load_pickle()``.

        Notice, that :meth:`create_model` might *NOT* be called
        before :meth:`load`.

        The default implementation unpickles to ``self.model``.
        """
        self.model = checkpoint.load_pickle()
