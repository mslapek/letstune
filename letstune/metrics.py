"""Module with class :class:`Metric` describing goal of a tuning."""

from __future__ import annotations

import dataclasses
import re
from typing import Literal

__all__ = [
    "Metric",
    "MetricValues",
]


MetricValues = dict[str, float]
"""Dictionary with metric values from a training."""


@dataclasses.dataclass(frozen=True)
class Metric:
    """Goal of a tuning.

    Training might return many metrics.
    String ``name`` declares, which one of these should be used.

    Boolean ``greater_is_better`` informs letstune how to choose
    the best parameters.

    >>> Metric("root_mean_square_error", greater_is_better=False)
    Metric(name='root_mean_square_error', greater_is_better=False)

    >>> Metric("train_accuracy", greater_is_better=True)
    Metric(name='train_accuracy', greater_is_better=True)
    """

    name: str
    greater_is_better: bool = dataclasses.field(kw_only=True)

    @classmethod
    def natural(cls, name: str) -> Metric:
        """Returns metric with automatically chosen ``greater_is_better``.

        Raises :class:`ValueError` when ``greater_is_better`` cannot be
        inferred.

        The rules are described below.

        Greater is better metrics
        -------------------------

        All *scores*:

        >>> Metric.natural("f1_score")
        Metric(name='f1_score', greater_is_better=True)

        >>> Metric.natural("foo_bar_score")
        Metric(name='foo_bar_score', greater_is_better=True)

        All *accuracy metrics*:

        >>> Metric.natural("top_k_categorical_accuracy")
        Metric(name='top_k_categorical_accuracy', greater_is_better=True)

        >>> Metric.natural("foo_bar_accuracy")
        Metric(name='foo_bar_accuracy', greater_is_better=True)

        >>> Metric.natural("acc")
        Metric(name='acc', greater_is_better=True)

        Miscellaneous:

        >>> Metric.natural("precision")
        Metric(name='precision', greater_is_better=True)

        Lower is better metrics
        -----------------------

        All *losses*:

        >>> Metric.natural("log_loss")
        Metric(name='log_loss', greater_is_better=False)

        >>> Metric.natural("foo_bar_loss")
        Metric(name='foo_bar_loss', greater_is_better=False)

        All *errors*:

        >>> Metric.natural("mean_absolute_error")
        Metric(name='mean_absolute_error', greater_is_better=False)

        >>> Metric.natural("foo_bar_error")
        Metric(name='foo_bar_error', greater_is_better=False)

        Miscellaneous:

        >>> Metric.natural("mae")
        Metric(name='mae', greater_is_better=False)

        Valid and train splits
        ----------------------

        Prefixes ``valid_`` and ``train_`` are ignored:

        >>> Metric.natural("valid_mean_absolute_error")
        Metric(name='valid_mean_absolute_error', greater_is_better=False)

        >>> Metric.natural("train_acc")
        Metric(name='train_acc', greater_is_better=True)

        Name normalization
        ------------------

        ``lower_case_with_underscores`` and ``CapitalizedWords``
        are supported:

        >>> Metric.natural("valid_mean_absolute_error")
        Metric(name='valid_mean_absolute_error', greater_is_better=False)

        >>> Metric.natural("ValidMeanAbsoluteError")
        Metric(name='ValidMeanAbsoluteError', greater_is_better=False)

        Negative metrics
        ----------------

        Metrics with ``neg`` in name are *not* supported:

        >>> Metric.natural("neg_mean_absolute_error")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Cannot infer greater_is_better for metric 'neg_mean_absolute_error'
        ...

        Bibliography
        ------------

        Inspired by metric names from:

        * `"Metrics and scoring: quantifying the quality of predictions" \
          <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
          from sklearn documentation
        * `"Metrics" <https://keras.io/api/metrics/>`_ from Keras documentation
        """  # noqa
        greater_is_better = _classify_metric(name)
        if isinstance(greater_is_better, str):
            msg = f"Cannot infer greater_is_better for metric {name!r}"

            if greater_is_better == "neg":
                msg += f"\nMaybe you should use Metric.sklearn({name!r})?"

            raise ValueError(msg)

        return Metric(name, greater_is_better=greater_is_better)

    @classmethod
    def sklearn(cls, name: str) -> Metric:
        """Returns metric, which represents sklearn scorer [#scorer]_ - i.e. metric with
        ``greater_is_better=True``.


        >>> Metric.sklearn("neg_root_mean_square_error")
        Metric(name='neg_root_mean_square_error', greater_is_better=True)

        Notice, that this function does *NOT* validate ``greater_is_better``:

        >>> Metric.sklearn("root_mean_square_error")
        Metric(name='root_mean_square_error', greater_is_better=True)

        .. [#scorer] See `scorer convention in sklearn documentation \
           <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`_.
        """
        return Metric(name, greater_is_better=True)


def _normalize(name: str) -> str:
    name = name.lower()
    name = re.sub("[^a-z0-9]+", "", name)
    return name


def _ends_with_suffix(name: str, suffixes: list[str]) -> bool:
    return any(name.endswith(s) for s in suffixes)


def _classify_metric(name: str) -> Literal[True, False, "err", "neg"]:
    name = _normalize(name)

    if "neg" in name:
        return "neg"

    for prefix in ["train", "valid"]:
        name = name.removeprefix(prefix)

    if _ends_with_suffix(name, ["accuracy", "acc", "score"]):
        return True

    if _ends_with_suffix(name, ["loss", "error", "crossentropy"]):
        return False

    if name in {
        "r2",
        "f1",
        "auc",
        "precision",
        "recall",
        "iou",
        "cosinesimilarity",
    }:
        return True

    if name in {
        "mae",
        "mse",
        "rmse",
    }:
        return False

    return "err"
