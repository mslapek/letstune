"""Metric examples are based on:

* https://scikit-learn.org/stable/modules/model_evaluation.html
* https://keras.io/api/metrics/
"""

import pytest

from letstune import Metric

GREATER_IS_BETTER_EXAMPLES = [
    # accuracy
    "acc",
    "accuracy",
    "valid_accuracy",
    "train_accuracy",
    "binary_accuracy",
    "categorical_accuracy",
    "top_k_categorical_accuracy",
    "foobar_accuracy",
    "valid_foo_bar_accuracy",
    "BinaryAccuracy",
    # score
    "foobar_score",
    "valid_foo_bar_score",
    # miscellaneous
    "r2",
    "f1",
    "auc",
    "AUC",
    "valid_AUC",
    "precision",
    "TrainPrecision",
    "recall",
    "valid_recall",
    "iou",
    "cosine_similarity",
    "ValidCosineSimilarity",
]

LOWER_IS_BETTER_EXAMPLES = [
    # loss
    "loss",
    "valid_loss",
    "train_loss",
    # error
    "mean_absolute_error",
    "MeanAbsoluteError",
    "ValidMeanAbsoluteError",
    "valid_MeanAbsoluteError",
    "mean_square_error",
    "root_mean_squared_error",
    "valid_mean_square_error",
    "train_mean_square_error",
    "foobar_error",
    "train_foo_bar_error",
    # crossentropy
    "binary_crossentropy",
    "valid_foo_bar_crossentropy",
    # miscellaneous
    "mae",
    "mse",
    "rmse",
]

NEG_METRICS = [
    "neg_mean_square_error",
    "neg_accuracy",
    "valid_neg_mean_square_error",
]

UNKNOWN_METRICS = [
    "foo_bar",
    "train_f4",
    "valid_foo_bar",
    "tax_rate",
]


def test_instance_creation() -> None:
    m = Metric("accuracy", greater_is_better=True)

    assert m.name == "accuracy"
    assert m.greater_is_better


def test_greater_is_better_must_be_kwarg() -> None:
    with pytest.raises(TypeError):
        _ = Metric("accuracy", True)  # type: ignore

    with pytest.raises(TypeError):
        _ = Metric("accuracy")  # type: ignore


@pytest.mark.parametrize("name", GREATER_IS_BETTER_EXAMPLES)
def test_natural_greater_is_better_metric(name: str) -> None:
    m = Metric.natural(name)

    assert m.name == name
    assert m.greater_is_better


@pytest.mark.parametrize("name", LOWER_IS_BETTER_EXAMPLES)
def test_natural_lower_is_better_metric(name: str) -> None:
    m = Metric.natural(name)

    assert m.name == name
    assert not m.greater_is_better


@pytest.mark.parametrize("name", UNKNOWN_METRICS + NEG_METRICS)
def test_nonnatural_metric_raises_value_error(name: str) -> None:
    with pytest.raises(
        ValueError, match=f"Cannot infer greater_is_better for metric {name!r}"
    ):
        _ = Metric.natural(name)


@pytest.mark.parametrize("name", NEG_METRICS)
def test_nonnatural_metric_with_neg_suggests_metric_sklearn(name: str) -> None:
    with pytest.raises(
        ValueError, match=rf"Maybe you should use Metric\.sklearn\({name!r}\)\?"
    ):
        _ = Metric.natural(name)


@pytest.mark.parametrize(
    "name",
    GREATER_IS_BETTER_EXAMPLES
    + LOWER_IS_BETTER_EXAMPLES
    + UNKNOWN_METRICS
    + NEG_METRICS,
)
def test_sklearn_metric(name: str) -> None:
    m = Metric.sklearn(name)

    assert m.name == name
    assert m.greater_is_better
