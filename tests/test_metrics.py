import pytest

from letstune import Metric


def test_instance_creation() -> None:
    m = Metric("accuracy", greater_is_better=True)

    assert m.name == "accuracy"
    assert m.greater_is_better


def test_greater_is_better_must_be_kwarg() -> None:
    with pytest.raises(TypeError):
        _ = Metric("accuracy", True)  # type: ignore


def test_str() -> None:
    m = Metric("accuracy", greater_is_better=True)

    assert repr(m) == "Metric(name='accuracy', greater_is_better=True)"
    assert repr(m) == str(m)


@pytest.mark.parametrize(
    "name",
    [
        "f1_score",
        "foo_bar_score",
        "F1Score",
        "FooBarScore",
    ],
)
def test_score_is_greater_is_better(name: str) -> None:
    m = Metric(name)

    assert m.name == name
    assert m.greater_is_better


@pytest.mark.parametrize(
    "name",
    [
        "accuracy",
        "acc",
        "top_k_categorical_accuracy",
        "foo_bar_accuracy",
        "foo_bar_acc",
        "Accuracy",
        "Acc",
        "TopKCategoricalAccuracy",
        "FooBarAccuracy",
        "FooBarAcc",
    ],
)
def test_accuracy_is_greater_is_better(name: str) -> None:
    m = Metric(name)

    assert m.name == name
    assert m.greater_is_better


@pytest.mark.parametrize(
    "name",
    [
        "log_loss",
        "foo_bar_loss",
        "LogLoss",
        "FooBarLoss",
    ],
)
def test_loss_is_lower_is_better(name: str) -> None:
    m = Metric(name)

    assert m.name == name
    assert not m.greater_is_better


@pytest.mark.parametrize(
    "name",
    [
        "mean_absolute_error",
        "foo_bar_error",
        "MeanAbsoluteError",
        "FooBarError",
    ],
)
def test_error_is_lower_is_better(name: str) -> None:
    m = Metric(name)

    assert m.name == name
    assert not m.greater_is_better


@pytest.mark.parametrize(
    "name",
    [
        "mse",
        "rmse",
        "mae",
        "MSE",
        "RMSE",
        "MAE",
    ],
)
def test_popular_error_acronym_is_lower_is_better(name: str) -> None:
    m = Metric(name)

    assert m.name == name
    assert not m.greater_is_better


@pytest.mark.parametrize(
    "original_name",
    [
        "foo_bar_score",
        "foo_bar_accuracy",
        "foo_bar_acc",
        "foo_bar_loss",
        "foo_bar_error",
        "rmse",
        "FooBarScore",
        "FooBarAccuracy",
        "FooBarAcc",
        "FooBarLoss",
        "FooBarError",
        "RMSE",
    ],
)
def test_neg_negates_greater_is_better(original_name: str) -> None:
    neg_name = "neg_" + original_name

    assert Metric(neg_name).greater_is_better != Metric(original_name).greater_is_better


@pytest.mark.parametrize(
    "original_name",
    [
        "foo_bar_score",
        "foo_bar_accuracy",
        "neg_foo_bar_accuracy",
        "foo_bar_acc",
        "foo_bar_loss",
        "neg_foo_bar_loss",
        "foo_bar_error",
        "rmse",
        "FooBarScore",
        "FooBarAccuracy",
        "NegFooBarAccuracy",
        "FooBaracc",
        "FooBarLoss",
        "NegFooBarLoss",
        "FooBarError",
        "RMSE",
    ],
)
def test_valid_test_are_ignored(original_name: str) -> None:
    for prefix in ["train_", "valid_", "val_"]:
        new_name = prefix + original_name
        assert (
            Metric(new_name).greater_is_better
            == Metric(original_name).greater_is_better
        )


@pytest.mark.parametrize(
    "name",
    [
        "foo_bar",
        "FooBar",
        "train_f4",
        "valid_foo_bar",
        "tax_rate",
    ],
)
def test_unknown_metric_raises_value_error(name: str) -> None:
    with pytest.raises(
        ValueError, match=f"cannot infer greater_is_better for metric {name!r}"
    ):
        _ = Metric(name)
