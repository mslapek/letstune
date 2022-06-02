from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune
from letstune import rand
from letstune.sklearn import SklearnTrainer


class MyFooModel:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = kwargs

        self.fit_args: tuple[Any, ...] = None
        self.fit_kwargs: dict[str, Any] = None

        self.score_args: tuple[Any, ...] = None
        self.score_args_history: list[tuple[Any, ...]] = []

        self._score_iterator = iter([11.11, 22.22])

    def fit(self, *args: Any, **kwargs: Any) -> None:
        self.fit_args = args
        self.fit_kwargs = kwargs

    def score(self, *args: Any) -> float:
        self.score_args = args
        self.score_args_history.append(args)
        return next(self._score_iterator)


class MyParams(letstune.Params):
    x: int = rand.oneof([45])  # type: ignore

    def create_model(self, **kwargs: Any) -> MyFooModel:
        return MyFooModel(
            **self.to_dict(),
            **kwargs,
        )


class MyScorer:
    def __init__(self) -> None:
        self.args_history: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> float:
        self.args_history.append(args)
        return 55.55


def test_get_random_params() -> None:
    trainer = SklearnTrainer(MyParams)

    params = trainer.get_random_params(np.random.default_rng(42))

    assert isinstance(params, MyParams)
    assert params.x == 45


def test_load_dataset() -> None:
    trainer = SklearnTrainer(MyParams)

    trainer.load_dataset(("array1", "array2", "array3", "array4"))

    assert trainer.X == "array1"
    assert trainer.X_valid == "array2"
    assert trainer.y == "array3"
    assert trainer.y_valid == "array4"


def test_load_dataset_optimize_train_score() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.optimize_train_score = True

    trainer.load_dataset(("array1", "array2", "array3", "array4"))

    assert trainer.X == "array1"
    assert trainer.X_valid == "array2"
    assert trainer.y == "array3"
    assert trainer.y_valid == "array4"


def test_load_dataset_train_only() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.optimize_train_score = True

    trainer.load_dataset(("array1", "array2"))

    assert trainer.X == "array1"
    assert trainer.y == "array2"
    assert trainer.X_valid is None
    assert trainer.y_valid is None


def test_load_dataset_with_invalid_argument() -> None:
    trainer = SklearnTrainer(MyParams)

    with pytest.raises(
        ValueError,
        match=r"wrong number of elements in dataset \(expected 4, got 3\)",
    ):
        trainer.load_dataset(("array1", "array2", "array3"))

    with pytest.raises(
        ValueError, match=r"wrong number of elements in dataset \(expected 4, got 2\)"
    ):
        trainer.load_dataset(("array1", "array2"))

    with pytest.raises(
        ValueError, match=r"wrong number of elements in dataset \(expected 4, got 1\)"
    ):
        trainer.load_dataset(("array1",))


def test_load_dataset_optimize_train_score_with_invalid_argument() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.optimize_train_score = True

    with pytest.raises(
        ValueError,
        match=r"wrong number of elements in dataset \(expected 2 or 4, got 3\)",
    ):
        trainer.load_dataset(("array1", "array2", "array3"))

    with pytest.raises(
        ValueError,
        match=r"wrong number of elements in dataset \(expected 2 or 4, got 1\)",
    ):
        trainer.load_dataset(("array1",))


def test_metric() -> None:
    trainer = SklearnTrainer(MyParams)

    assert trainer.metric == letstune.Metric("valid_score", greater_is_better=True)


def test_metric_optimize_train_score() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.optimize_train_score = True

    assert trainer.metric == letstune.Metric("train_score", greater_is_better=True)


def test_train() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.optimize_train_score = True
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array2")
    assert model.fit_kwargs == {}
    assert model.score_args == ("array1", "array2")

    assert scores == {"train_score": 11.11}


def test_train_with_valid() -> None:
    trainer = SklearnTrainer(MyParams)
    trainer.load_dataset(("array1", "array2", "array3", "array4"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array3")
    assert model.fit_kwargs == {}
    assert model.score_args == ("array2", "array4")

    assert scores == {"valid_score": 11.11}


@pytest.mark.parametrize("field_to_set", ["return_train_score", "optimize_train_score"])
def test_train_with_return_or_optimize_train_score(field_to_set: str) -> None:
    trainer = SklearnTrainer(MyParams)
    setattr(trainer, field_to_set, True)
    trainer.load_dataset(("array1", "array2", "array3", "array4"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array3")
    assert model.fit_kwargs == {}
    assert model.score_args_history == [("array2", "array4"), ("array1", "array3")]

    assert scores == {"valid_score": 11.11, "train_score": 22.22}


def test_train_with_custom_scorer() -> None:
    scorer = MyScorer()

    trainer = SklearnTrainer(MyParams, scorer=scorer)
    trainer.optimize_train_score = True
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array2")
    assert model.fit_kwargs == {}
    assert model.score_args is None

    assert scorer.args_history == [(model, "array1", "array2")]

    assert scores == {"train_score": 55.55}


def test_train_with_custom_scorer_valid() -> None:
    scorer = MyScorer()

    trainer = SklearnTrainer(MyParams, scorer=scorer)
    trainer.load_dataset(("array1", "array2", "array3", "array4"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array3")
    assert model.fit_kwargs == {}
    assert model.score_args is None

    assert scorer.args_history == [(model, "array2", "array4")]

    assert scores == {"valid_score": 55.55}


def test_train_with_custom_scorer_as_str(mocker: MockerFixture) -> None:
    scorer = MyScorer()

    def get_scorer(name: str) -> Any:
        if name == "foobar":
            return scorer
        else:
            raise ValueError

    mocker.patch("sklearn.metrics.get_scorer", get_scorer)

    trainer = SklearnTrainer(MyParams, scorer="foobar")
    trainer.optimize_train_score = True
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array2")
    assert model.fit_kwargs == {}
    assert model.score_args is None

    assert scorer.args_history == [(model, "array1", "array2")]

    assert scores == {"train_score": 55.55}


def test_create_model_kwargs() -> None:
    trainer = SklearnTrainer(MyParams, create_model_kwargs={"color": "blue", "foo": 58})
    trainer.optimize_train_score = True
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12, "color": "blue", "foo": 58}
    assert model.fit_args == ("array1", "array2")
    assert model.fit_kwargs == {}
    assert model.score_args == ("array1", "array2")

    assert scores == {"train_score": 11.11}


def test_fit_model_kwargs() -> None:
    trainer = SklearnTrainer(MyParams, fit_model_kwargs={"color": "blue", "foo": 58})
    trainer.optimize_train_score = True
    trainer.load_dataset(("array1", "array2"))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}
    assert model.fit_args == ("array1", "array2")
    assert model.fit_kwargs == {"color": "blue", "foo": 58}
    assert model.score_args == ("array1", "array2")

    assert scores == {"train_score": 11.11}
