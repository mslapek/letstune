"""Module with class :class:`Metric` describing goal of a tuning."""

import dataclasses
import re

__all__ = [
    "Metric",
    "MetricValues",
]


MetricValues = dict[str, float]
"""Dictionary with metric values from a training."""


@dataclasses.dataclass(frozen=True, init=False, slots=True)
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

    **Automatic inference**

    When ``greater_is_better`` is not given, letstune will
    try to infer it from the ``name``.

    Raises :class:`ValueError` when ``greater_is_better`` cannot be
    inferred.

    The rules are described below.

    **Greater is better metrics**

    All *scores*:

    >>> Metric("f1_score")
    Metric(name='f1_score', greater_is_better=True)

    >>> Metric("foo_bar_score")
    Metric(name='foo_bar_score', greater_is_better=True)

    All *accuracy metrics*:

    >>> Metric("top_k_categorical_accuracy")
    Metric(name='top_k_categorical_accuracy', greater_is_better=True)

    >>> Metric("foo_bar_accuracy")
    Metric(name='foo_bar_accuracy', greater_is_better=True)

    >>> Metric("acc")
    Metric(name='acc', greater_is_better=True)

    **Lower is better metrics**

    All *losses*:

    >>> Metric("log_loss")
    Metric(name='log_loss', greater_is_better=False)

    >>> Metric("foo_bar_loss")
    Metric(name='foo_bar_loss', greater_is_better=False)

    All *errors*:

    >>> Metric("mean_absolute_error")
    Metric(name='mean_absolute_error', greater_is_better=False)

    >>> Metric("foo_bar_error")
    Metric(name='foo_bar_error', greater_is_better=False)

    Popular acronyms:

    >>> Metric("mae")
    Metric(name='mae', greater_is_better=False)

    **Negative metrics**

    Metrics with ``neg`` prefix have negated ``greater_is_better``:

    >>> Metric("neg_root_mean_square_error")
    Metric(name='neg_root_mean_square_error', greater_is_better=True)

    **Valid and train splits**

    Prefixes ``valid_`` and ``train_`` are ignored:

    >>> Metric("valid_mean_absolute_error")
    Metric(name='valid_mean_absolute_error', greater_is_better=False)

    >>> Metric("train_acc")
    Metric(name='train_acc', greater_is_better=True)

    **Name normalization**

    ``lower_case_with_underscores`` and ``CapitalizedWords``
    are supported:

    >>> Metric("valid_mean_absolute_error")
    Metric(name='valid_mean_absolute_error', greater_is_better=False)

    >>> Metric("ValidMeanAbsoluteError")
    Metric(name='ValidMeanAbsoluteError', greater_is_better=False)

    >>> Metric("valid_MeanAbsoluteError")
    Metric(name='valid_MeanAbsoluteError', greater_is_better=False)

    **Bibliography**

    Inspired by metric names from:

    * `"Metrics and scoring: quantifying the quality of predictions" \
      <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
      from sklearn documentation
    * `"Metrics" <https://keras.io/api/metrics/>`_ from Keras documentation
    """

    name: str
    greater_is_better: bool

    def __init__(self, name: str, *, greater_is_better: bool | None = None):
        object.__setattr__(self, "name", name)

        if greater_is_better is None:
            greater_is_better = _classify_metric(name)

        object.__setattr__(self, "greater_is_better", greater_is_better)


def _normalize(name: str) -> str:
    name = name.lower()
    name = re.sub("[^a-z0-9]+", "", name)
    return name


def _ends_with_suffix(name: str, suffixes: list[str]) -> bool:
    return any(name.endswith(s) for s in suffixes)


def _classify_metric(name: str) -> bool:
    original_name = name
    name = _normalize(name)

    for prefix in ["train", "valid", "val"]:
        name = name.removeprefix(prefix)

    neg = name.startswith("neg")
    if neg:
        name = name.removeprefix("neg")

    if _ends_with_suffix(name, ["accuracy", "acc", "score"]):
        return not neg

    if _ends_with_suffix(name, ["loss", "error", "crossentropy"]):
        return neg

    if name in {
        "mae",
        "mse",
        "rmse",
    }:
        return neg

    raise ValueError(f"cannot infer greater_is_better for metric {original_name!r}")
