from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune
from letstune import rand
from letstune.keras import KerasTrainer


@dataclass(frozen=True)
class History:
    history: dict[str, list[float]]


class MyFooModel:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = kwargs
        self.fit_kwargs: dict[str, Any] | None = None

    def fit(self, **kwargs: Any) -> History:
        self.fit_kwargs = kwargs
        return History({"qwerty": [13.33], "foobar": [99.0]})


class MyParams(letstune.Params):
    x: int = rand.oneof([45])  # type: ignore

    def create_model(self, **kwargs: Any) -> MyFooModel:
        return MyFooModel(
            **self.to_dict(),
            **kwargs,
        )


def test_get_random_params() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")

    params = trainer.get_random_params(np.random.default_rng(42))

    assert isinstance(params, MyParams)
    assert params.x == 45


def test_load_dataset() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")

    trainer.load_dataset(("array1", "array2"))

    assert trainer.x == "array1"
    assert trainer.y == "array2"
    assert trainer.validation_data is None


def test_load_dataset_with_valid() -> None:
    trainer = KerasTrainer(MyParams, "val_accuracy")

    trainer.load_dataset(("array1", "array2", "obj3"))

    assert trainer.x == "array1"
    assert trainer.y == "array2"
    assert trainer.validation_data == "obj3"


def test_load_dataset_with_invalid_argument() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")

    with pytest.raises(
        ValueError,
        match=r"wrong number of elements in dataset \(expected 2 or 3, got 4\)",
    ):
        trainer.load_dataset(("array1", "array2", "array3", "array4"))


def test_metric() -> None:
    trainer = KerasTrainer(MyParams, "foobar")

    assert trainer.metric == "foobar"


def test_create_model() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    trainer.create_model(params)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs is None


def test_create_model_kwargs() -> None:
    trainer = KerasTrainer(
        MyParams,
        "accuracy",
        create_model_kwargs={"color": "blue", "foo": 58},
    )
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    trainer.create_model(params)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12, "color": "blue", "foo": 58}
    assert model.fit_kwargs is None


def test_train_epoch() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)
    trainer.create_model(params)

    scores = trainer.train_epoch(5)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs == {
        "x": "array1",
        "y": "array2",
        "epochs": 6,
        "initial_epoch": 5,
        "verbose": 0,
    }

    assert scores == {"qwerty": 13.33, "foobar": 99.0}


def test_train_epoch_with_valid() -> None:
    trainer = KerasTrainer(MyParams, "accuracy")
    trainer.load_dataset(("array1", "array2", "obj3"))
    params = MyParams(x=12)
    trainer.create_model(params)

    scores = trainer.train_epoch(0)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs == {
        "x": "array1",
        "y": "array2",
        "validation_data": "obj3",
        "epochs": 1,
        "initial_epoch": 0,
        "verbose": 0,
    }

    assert scores == {"qwerty": 13.33, "foobar": 99.0}


def test_train_epoch_kwargs() -> None:
    trainer = KerasTrainer(
        MyParams,
        "accuracy",
        fit_model_kwargs={"color": "blue", "foo": 58},
    )
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)
    trainer.create_model(params)

    scores = trainer.train_epoch(5)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs == {
        "x": "array1",
        "y": "array2",
        "color": "blue",
        "foo": 58,
        "epochs": 6,
        "initial_epoch": 5,
        "verbose": 0,
    }

    assert scores == {"qwerty": 13.33, "foobar": 99.0}


def test_train_epoch_kwargs_with_valid() -> None:
    trainer = KerasTrainer(
        MyParams,
        "accuracy",
        fit_model_kwargs={"color": "blue", "foo": 58},
    )
    trainer.load_dataset(("array1", "array2", "obj3"))
    params = MyParams(x=12)
    trainer.create_model(params)

    scores = trainer.train_epoch(5)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs == {
        "x": "array1",
        "y": "array2",
        "validation_data": "obj3",
        "color": "blue",
        "foo": 58,
        "epochs": 6,
        "initial_epoch": 5,
        "verbose": 0,
    }

    assert scores == {"qwerty": 13.33, "foobar": 99.0}


def test_train_epoch_kwargs_with_custom_verbose() -> None:
    trainer = KerasTrainer(
        MyParams,
        "accuracy",
        fit_model_kwargs={"color": "blue", "foo": 58, "verbose": 123},
    )
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)
    trainer.create_model(params)

    scores = trainer.train_epoch(5)

    model = trainer.model
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_kwargs == {
        "x": "array1",
        "y": "array2",
        "color": "blue",
        "foo": 58,
        "epochs": 6,
        "initial_epoch": 5,
        "verbose": 123,
    }

    assert scores == {"qwerty": 13.33, "foobar": 99.0}


def test_save_calls_save_keras(mocker: MockerFixture) -> None:
    class Checkpoint:
        def save_keras(self, obj: Any) -> None:
            pass

    checkpoint = Checkpoint()
    save_keras_spy = mocker.spy(checkpoint, "save_keras")

    trainer = KerasTrainer(MyParams, "accuracy")
    model = object()
    trainer.model = model

    trainer.save(checkpoint)

    save_keras_spy.assert_called_once_with(model)


def test_load_calls_load_keras() -> None:
    model = object()

    class Checkpoint:
        def load_keras(self) -> Any:
            return model

    checkpoint = Checkpoint()
    trainer = KerasTrainer(MyParams, "accuracy")

    trainer.load(checkpoint, MyParams(x=999))

    assert trainer.model == model
