"""Module with class :class:`KerasTrainer`."""

from typing import Any, TypeVar

import numpy as np

import letstune

from .metrics import MetricValues
from .params import Params
from .rand import RandomParamsGenerator
from .trainer import EpochTrainer

__all__ = ["KerasTrainer"]

P = TypeVar("P", bound=Params)


class KerasTrainer(EpochTrainer[P]):
    """Trainer for Keras models *with* early-stopping.

    :class:`KerasTrainer` allows you to tune Keras models
    without creating own :class:`letstune.EpochTrainer` class.

    **Model**

    Model is created with ``params.create_model()``.
    The trainer is compatible with :class:`letstune.ModelParams`.

    You can manually define a method ``create_model`` in your params
    class.

    **Dataset**

    A :class:`KerasTrainer` can be passed a ``dataset``
    in two ways.

    **Training data only**::

        dataset = (x, y)

    Model is trained with ``model.fit(x, y)``.

    **Training and validation data**::

        dataset = (x, y, validation_data)

    where ``validation_data`` is usually ``(x_val, y_val)``.

    Model is trained with ``model.fit(x, y, validation_data=validation_data)``.

    **Custom dataset loading**

    You can create a new class, in order to give
    a custom ``load_dataset`` method::

        class DigitsTrainer(letstune.keras.KerasTrainer):
            def load_dataset(self, dataset: Any) -> None:
                self.x = ...
                self.y = ...
                self.validation_data = ...

    Model will be trained with::

        model.fit(self.x, self.y, validation_data=self.validation_data)

    This is useful for datasets, which are *not* numpy arrays,
    like ``tf.data.Dataset``.
    See "Serialization" in ``letstune.trainer`` module documentation.

    ``dataset`` can be used as a path to a directory with data.
    """

    x: Any = None
    y: Any = None
    validation_data: Any = None

    def __init__(
        self,
        params_cls: type[P] | RandomParamsGenerator[P],
        metric: letstune.Metric,
        *,
        create_model_kwargs: dict[str, Any] | None = None,
        fit_model_kwargs: dict[str, Any] | None = None,
    ):
        """Create :class:`KerasTrainer`.

        First parameter is the type of the used :class:`letstune.Params`.

        Instead of a type, you can pass any object with ``get_random_params``
        method.

        Second parameter is :class:`letstune.Metric`.
        If you are using ``validation_data``, don't forget to prefix
        metric name with ``val_``::

            metric = letstune.Metric("val_accuracy")

        **Additional keyword arguments**

        Dictionary ``create_model_kwargs`` has additional arguments, which will be
        used for ``params.create_model``::

            model = params.create_model(**create_model_kwargs)

        Notice, that :class:`letstune.ModelParams` has support for additional
        parameters in ``create_model``.

        Dictionary ``fit_model_kwargs`` has additional arguments, which will be
        used for ``model.fit``::

            model.fit(x, y, **fit_model_kwargs)

        With ``validation_data`` given::

            model.fit(x, y, validation_data=validation_data, **fit_model_kwargs)
        """

        self.__metric = metric
        self.__random_params_generator = params_cls
        self.__create_model_kwargs = create_model_kwargs or {}
        self.__fit_model_kwargs = fit_model_kwargs or {}

    @property
    def metric(self) -> letstune.Metric:
        return self.__metric

    def get_random_params(self, rng: np.random.Generator) -> P:
        return self.__random_params_generator.get_random_params(rng)

    def create_model(self, params: P) -> None:
        self.model = params.create_model(**self.__create_model_kwargs)  # type: ignore

    def train_epoch(self, epoch: int) -> MetricValues:
        fit_kwargs = self.__fit_model_kwargs.copy()

        for attr in ["x", "y", "validation_data"]:
            v = getattr(self, attr)
            if v is not None:
                fit_kwargs[attr] = v

        fit_kwargs["epochs"] = epoch + 1
        fit_kwargs["initial_epoch"] = epoch

        history = self.model.fit(**fit_kwargs)

        return {m: v[0] for m, v in history.history.items()}

    def load_dataset(self, dataset: Any) -> None:
        n = len(dataset)
        if n == 2:
            self.x, self.y = dataset
        elif n == 3:
            self.x, self.y, self.validation_data = dataset
        else:
            raise ValueError(
                f"wrong number of elements in dataset (expected 2 or 3, got {n})"
            )

    def save(self, checkpoint: Any) -> None:
        """Save the model in the current state.

        ``checkpoint`` is an object passed from backend.

        The default implementation saves ``self.model``
        with ``checkpoint.save_keras``.
        """

        checkpoint.save_keras(self.model)

    def load(self, checkpoint: Any, params: P) -> None:
        """Load the model from ``checkpoint``.

        ``checkpoint`` is an object passed from backend.

        Notice, that ``create_model`` might *NOT* be called
        before ``load``.

        The default implementation loads ``self.model``
        with ``checkpoint.load_keras``.
        """
        self.model = checkpoint.load_keras()
