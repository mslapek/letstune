"""Trainers for sklearn-like models.

:class:`SklearnTrainer` trains model on a train dataset,
then evaluates metrics on a validation dataset.
Good companion for :func:`sklearn.model_selection.train_test_split`.

:class:`SklearnCVTrainer` evaluates model with k-fold cross-validation.
It's a great substitution for :class:`sklearn.model_selection.GridSearchCV`!
"""

from abc import ABC
from typing import Any, Callable, Iterable, Protocol, TypeVar

import numpy as np

import letstune

from .metrics import MetricValues
from .params import Params
from .rand import RandomParamsGenerator

__all__ = ["SklearnTrainer", "SklearnCVTrainer", "Splitter"]

P = TypeVar("P", bound=Params)


class _SklearnTrainerBase(letstune.SimpleTrainer[P], ABC):
    return_train_score: bool = False
    X: Any = None
    y: Any = None

    def __init__(
        self,
        params_cls: type[P] | RandomParamsGenerator[P],
        *,
        create_model_kwargs: dict[str, Any] | None = None,
        fit_model_kwargs: dict[str, Any] | None = None,
        scorer: str | Callable[[Any, Any, Any], float] | None = None,
    ):
        """Create :class:`SklearnTrainer`.

        First parameter is the type of the used :class:`letstune.Params`.
        It is *the only required* parameter.

        Instead of the type, you can pass any object with ``get_random_params``
        method.

        **Additional keyword arguments**

        Dictionary ``create_model_kwargs`` has additional arguments, which will be
        used for ``params.create_model``::

            model = params.create_model(**create_model_kwargs)

        Notice, that :class:`letstune.ModelParams` has support for additional
        parameters in ``create_model``.

        Dictionary ``fit_model_kwargs`` has additional arguments, which will be
        used for ``model.fit``::

            model.fit(X, y, **fit_model_kwargs)

        **Scorer**

        By default ``model.score(X, y)`` is used for scoring.

        You can set ``scorer`` to a callable, which takes ``(model, X, y)``
        and returns a ``float``.

        ``scorer`` can be set to a string, then a scorer from
        :func:`sklearn.metrics.get_scorer` will be used.

        """
        self.__random_params_generator = params_cls
        self.__create_model_kwargs = create_model_kwargs or {}
        self.__fit_model_kwargs = fit_model_kwargs or {}
        self.__scorer = scorer

    def _score(self, model: Any, X: Any, y: Any) -> float:
        """Evaluate score of the model."""
        if isinstance(self.__scorer, str):
            import sklearn.metrics

            self.__scorer = sklearn.metrics.get_scorer(self.__scorer)

        if self.__scorer is None:
            return model.score(X, y)  # type: ignore
        else:
            return self.__scorer(model, X, y)  # type: ignore

    def _create_model(self, params: P) -> Any:
        return params.create_model(**self.__create_model_kwargs)  # type: ignore

    def _fit_model(self, model: Any, X: Any, y: Any) -> None:
        model.fit(X, y, **self.__fit_model_kwargs)

    def get_random_params(self, rng: np.random.Generator) -> P:
        return self.__random_params_generator.get_random_params(rng)


class SklearnTrainer(_SklearnTrainerBase[P]):
    """Trainer for sklearn-like models *without* early-stopping.

    :class:`SklearnTrainer` allows you to tune sklearn-like models
    without creating own :class:`letstune.SimpleTrainer` class.

    **Model**

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

    **Dataset**

    A :class:`SklearnTrainer` consumes a ``dataset``
    with train/validation split::

        dataset = (X_train, X_valid, y_train, y_valid)

    This is compatible with output from
    :func:`sklearn.model_selection.train_test_split`.

    Model is trained with ``model.fit(X_train, y_train)``.

    §

    When ``trainer.optimize_train_score`` is ``True``, then validation
    data can be omitted::

        dataset = (X_train, y_train)

    **Metrics**

    By default, ``valid_score`` is optimized:

    >>> class MyParams(letstune.Params):
    ...     l1_ratio: float

    >>> trainer = SklearnTrainer(MyParams)
    >>> trainer.metric
    Metric(name='valid_score', greater_is_better=True)

    Metric ``valid_score`` is calculated using ``(X_valid, y_valid)``.

    ``train_score`` is not calculated.
    You can force the calculation of ``train_score`` with
    ``return_train_score``::

        trainer = SklearnTrainer(...)
        trainer.return_train_score = True

    Metric ``train_score`` is calculated using ``(X_train, y_train)``.

    §

    When ``trainer.optimize_train_score`` is ``True``,
    then ``train_score`` is optimized:

    >>> trainer = SklearnTrainer(MyParams)
    >>> trainer.optimize_train_score = True
    >>> trainer.metric
    Metric(name='train_score', greater_is_better=True)
    """

    X_valid: Any = None
    y_valid: Any = None
    optimize_train_score: bool = False

    @property
    def metric(self) -> letstune.Metric:
        return letstune.Metric(
            "train_score" if self.optimize_train_score else "valid_score"
        )

    def train(self, params: P) -> tuple[Any, MetricValues]:
        model = self._create_model(params)
        self._fit_model(model, self.X, self.y)
        scores = {}

        if self.X_valid is not None:
            scores["valid_score"] = self._score(model, self.X_valid, self.y_valid)

        if self.return_train_score or self.optimize_train_score or len(scores) == 0:
            scores["train_score"] = self._score(model, self.X, self.y)

        return model, scores

    def load_dataset(self, dataset: Any) -> None:
        n = len(dataset)

        if n == 4:
            self.X, self.X_valid, self.y, self.y_valid = dataset
            return

        if not self.optimize_train_score:
            raise ValueError(
                f"wrong number of elements in dataset (expected 4, got {n})"
            )

        if n == 2:
            self.X, self.y = dataset
        else:
            raise ValueError(
                f"wrong number of elements in dataset (expected 2 or 4, got {n})"
            )


class Splitter(Protocol):
    """Object performing CV splits.

    An example of such object is :class:`sklearn.model_selection.KFold`.
    """

    def split(X: Any, y: Any = None, groups: Any = None) -> Iterable[tuple[Any, Any]]:
        """Yield (``train_indices``, ``valid_indices``) for a given dataset."""
        pass


def _add_cv_scores_to_metrics(
    metrics: MetricValues, name: str, scores: list[float]
) -> None:
    metrics[f"mean_{name}_score"] = np.mean(scores)
    metrics[f"std_{name}_score"] = np.std(scores)

    for i, score in enumerate(scores):
        metrics[f"split{i}_{name}_score"] = score


class SklearnCVTrainer(_SklearnTrainerBase[P]):
    """Cross-validating trainer for sklearn-like models *without* early-stopping.

    :class:`SklearnCVTrainer` allows you to tune sklearn-like models
    without creating own :class:`letstune.SimpleTrainer` class.

    **Model**

    Model is created with ``params.create_model()``.
    See :class:`SklearnTrainer` for more details.

    **Dataset**

    Trainer accepts simple dataset::

        dataset = (X, y)

    Model is trained for each fold with ``model.fit(X_fold, y_fold)``.

    **Metrics**

    Trainer calculates score for validation data, including:

    * ``mean_valid_score``,
    * ``std_valid_score``,
    * ``split0_valid_score``, ``split1_valid_score``, ...

    You can enable the calculation of scores
    on training data (including ``mean_train_score``)
    with ``return_train_score``::

        trainer = SklearnCVTrainer(...)
        trainer.return_train_score = True

    """

    def __init__(
        self,
        params_cls: type[P] | RandomParamsGenerator[P],
        *,
        cv: int | Splitter | None = None,
        create_model_kwargs: dict[str, Any] | None = None,
        fit_model_kwargs: dict[str, Any] | None = None,
        scorer: str | Callable[[Any, Any, Any], float] | None = None,
    ):
        """Create :class:`SklearnCVTrainer`.

        First parameter is the type of the used :class:`letstune.Params`.
        It is *the only required* parameter.

        Instead of the type, you can pass any object with ``get_random_params``
        method.

        **Cross-validation**

        By default, the trainer performs 5-fold cross-validation
        *without* dataset shuffling.

        When ``cv`` is an integer, the trainer runs
        cross-validation with ``cv`` folds.
        The training is *without* dataset shuffling.

        ``cv`` can be set to a splitter - an object with method
        ``split(X, y)`` returning iterable of ``(train_indices, valid_indices)``.
        An example of such object is :class:`sklearn.model_selection.KFold`.


        **Additional keyword arguments**

        This class accepts the same keyword arguments as :class:`SklearnTrainer`.
        """
        super().__init__(
            params_cls,
            create_model_kwargs=create_model_kwargs,
            fit_model_kwargs=fit_model_kwargs,
            scorer=scorer,
        )
        if cv is None:
            cv = 5

        if isinstance(cv, int):
            import sklearn.model_selection

            cv = sklearn.model_selection.KFold(cv)

        self._cv: Splitter = cv  # type: ignore

    @property
    def metric(self) -> letstune.Metric:
        return letstune.Metric("mean_valid_score")

    def train(self, params: P) -> tuple[Any, MetricValues]:
        model = self._create_model(params)

        valid_scores = []
        train_scores = []

        for train_index, valid_index in self._cv.split(self.X, self.y):
            X_train, X_valid = self.X[train_index], self.X[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]

            self._fit_model(model, X_train, y_train)

            valid_scores.append(self._score(model, X_valid, y_valid))

            if self.return_train_score:
                train_scores.append(self._score(model, X_train, y_train))

        metrics: MetricValues = {}
        _add_cv_scores_to_metrics(metrics, "valid", valid_scores)

        if self.return_train_score:
            _add_cv_scores_to_metrics(metrics, "train", train_scores)

        return None, metrics

    def load_dataset(self, dataset: Any) -> None:
        n = len(dataset)
        if n == 2:
            self.X, self.y = dataset
        else:
            raise ValueError(
                f"wrong number of elements in dataset (expected 2, got {n})"
            )

    def final_train(self, best_params: P) -> Any:
        """
        Train model on the whole dataset.

        Returns fitted model.
        """
        model = self._create_model(best_params)
        self._fit_model(model, self.X, self.y)
        return model
