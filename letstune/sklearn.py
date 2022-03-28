"""Module with class :class:`SklearnTrainer`."""

from typing import Any, Callable, TypeVar

import numpy as np

import letstune

from .metrics import MetricValues
from .params import Params
from .rand import RandomParamsGenerator

P = TypeVar("P", bound=Params)

__all__ = ["SklearnTrainer"]


class SklearnTrainer(letstune.SimpleTrainer[P]):
    """Trainer for sklearn-like models *without* early-stopping.

    :class:`SklearnTrainer` allows you to tune sklearn-like models
    without creating own :class:`SimpleTrainer` class.

    Model
    -----

    Model is created with ``params.create_model()``.
    The trainer is compatible with :class:`letstune.ModelParams`.

    You can manually define a method ``create_model`` in your params
    class::

        class MyParams(letstune.Params):
            l1_ratio: float = rand.uniform(0, 1)

            def create_model(self):
                return SGDClassifierParams(
                    l1_ratio=self.l1_ratio,
                    penalty="elasticnet",
                )

    Dataset
    -------

    A :class:`SklearnTrainer` can be passed a ``dataset``
    in two ways.

    **Training data only**::

        dataset = (X, y)

    Model is trained with ``model.fit(X, y)``.
    Metric ``train_score`` is calculated using ``(X, y)``.

    **Training and validation data**::

        dataset = (X_train, X_valid, y_train, y_valid)

    Model is trained with ``model.fit(X_train, y_train)``.
    Metric ``valid_score`` is calculated using ``(X_valid, y_valid)``.

    This is compatible with output from
    ``sklearn.model_selection.train_test_split``.

    ------

    You can force the calculation of ``train_score`` with
    ``return_train_score``::

        trainer = SklearnTrainer(...)
        trainer.return_train_score = True

    """

    X: Any = None
    y: Any = None
    X_valid: Any = None
    y_valid: Any = None
    return_train_score: bool = False

    def __init__(
        self,
        params_cls: type[P] | RandomParamsGenerator[P],
        *,
        create_model_kwargs: dict[str, Any] | None = None,
        fit_model_kwargs: dict[str, Any] | None = None,
        scorer: str | Callable[[Any, Any, Any], float] | None = None,
    ):
        """Create :class:`SklearnTrainer`.

        First parameter is the type of the used :class:`Params`.
        It is *the only required* parameter.

        Instead of the type, you can pass any object with ``get_random_params``
        method.

        Additional keyword arguments
        ----------------------------

        Dictionary ``create_model_kwargs`` has additional arguments, which will be
        used for ``params.create_model``::

            model = params.create_model(**create_model_kwargs)

        Notice, that :class:`ModelParams` has support for additional
        parameters in ``create_model``.

        Dictionary ``fit_model_kwargs`` has additional arguments, which will be
        used for ``model.fit``::

            model.fit(X, y, **fit_model_kwargs)

        Scorer
        ------

        By default ``model.score(X, y)`` is used for scoring.

        You can set ``scorer`` to a callable, which takes ``(model, X, y)``
        and returns a ``float``.

        ``scorer`` can be set to a string, then a scorer from
        ``sklearn.metrics.get_scorer`` will be used.

        """
        self._random_params_generator = params_cls
        self._create_model_kwargs = create_model_kwargs or {}
        self._fit_model_kwargs = fit_model_kwargs or {}
        self._scorer = scorer

    @property
    def metric(self) -> letstune.Metric:
        return letstune.Metric.sklearn(
            "valid_score" if self.X_valid is not None else "train_score"
        )

    def train(self, params: P) -> tuple[Any, MetricValues]:
        model = params.create_model(**self._create_model_kwargs)  # type: ignore
        model.fit(self.X, self.y, **self._fit_model_kwargs)
        scores = {}

        if self.X_valid is not None:
            scores["valid_score"] = self.score(model, self.X_valid, self.y_valid)

        if self.return_train_score or len(scores) == 0:
            scores["train_score"] = self.score(model, self.X, self.y)

        return model, scores

    def score(self, model: Any, X: Any, y: Any) -> float:
        """Evaluate score of the model."""
        if isinstance(self._scorer, str):
            import sklearn.metrics

            self._scorer = sklearn.metrics.get_scorer(self._scorer)

        if self._scorer is None:
            return model.score(X, y)  # type: ignore
        else:
            return self._scorer(model, X, y)  # type: ignore

    def load_dataset(self, dataset: Any) -> None:
        n = len(dataset)
        if n == 2:
            self.X, self.y = dataset
        elif n == 4:
            self.X, self.X_valid, self.y, self.y_valid = dataset
        else:
            raise ValueError(
                f"wrong number of elements in dataset (expected 2 or 4, got {n})"
            )

    def get_random_params(self, rng: np.random.Generator) -> P:
        return self._random_params_generator.get_random_params(rng)
