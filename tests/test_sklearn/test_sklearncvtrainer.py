from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune
from letstune import rand
from letstune.sklearn import SklearnCVTrainer


class MyFooModel:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = kwargs

        self.fit_args_kwargs_history: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        self.score_args_history: list[tuple[Any, ...]] = []

        self._score_iterator = iter([11.11, 22.22, 33.33, 44.44, 55.55, 66.66])

    def fit(self, *args: Any, **kwargs: Any) -> None:
        self.fit_args_kwargs_history.append((args, kwargs))

    def score(self, *args: Any) -> float:
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
        self._score_iterator = iter([101.11, 202.22, 303.33])

    def __call__(self, *args: Any) -> float:
        self.args_history.append(args)
        return next(self._score_iterator)


def test_get_random_params() -> None:
    trainer = SklearnCVTrainer(MyParams)

    params = trainer.get_random_params(np.random.default_rng(42))

    assert isinstance(params, MyParams)
    assert params.x == 45


def test_load_dataset() -> None:
    trainer = SklearnCVTrainer(MyParams)

    trainer.load_dataset(("array1", "array2"))

    assert trainer.X == "array1"
    assert trainer.y == "array2"


def test_load_dataset_with_invalid_argument() -> None:
    trainer = SklearnCVTrainer(MyParams)

    with pytest.raises(
        ValueError,
        match=r"wrong number of elements in dataset \(expected 2, got 3\)",
    ):
        trainer.load_dataset(("array1", "array2", "array3"))


def test_metric() -> None:
    trainer = SklearnCVTrainer(MyParams)

    trainer.load_dataset(("array1", "array2"))

    assert trainer.metric == letstune.Metric("mean_valid_score", greater_is_better=True)


def test_train() -> None:
    created_models = []

    class MyParams2(letstune.Params):
        x: int = rand.oneof([45])  # type: ignore

        def create_model(self, **kwargs: Any) -> MyFooModel:
            nonlocal created_models

            m = MyFooModel(
                **self.to_dict(),
                **kwargs,
            )
            created_models.append(m)
            return m

    trainer = SklearnCVTrainer(MyParams2, cv=3)
    trainer.load_dataset((np.eye(6), 2.0 * np.arange(6)))
    params = MyParams2(x=12)

    model, scores = trainer.train(params)
    assert model is None

    assert len(created_models) == 1
    model = created_models[0]
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}

    assert len(model.fit_args_kwargs_history) == 3

    (args, kwargs) = model.fit_args_kwargs_history[1]
    (fit_X, fit_y) = args
    assert kwargs == {}

    np.testing.assert_equal(
        fit_X,
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            float,
        ),
    )
    np.testing.assert_equal(fit_y, np.array([0, 2, 8, 10], float))

    assert len(model.score_args_history) == 3
    (score_X, score_y) = model.score_args_history[1]

    np.testing.assert_equal(
        score_X,
        np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], float),
    )
    np.testing.assert_equal(
        score_y,
        np.array([4, 6], float),
    )

    expected_scores = [11.11, 22.22, 33.33]
    assert scores == {
        "mean_valid_score": np.mean(expected_scores),
        "std_valid_score": np.std(expected_scores),
        "split0_valid_score": 11.11,
        "split1_valid_score": 22.22,
        "split2_valid_score": 33.33,
    }


def test_default_5_fold_cv() -> None:
    trainer = SklearnCVTrainer(MyParams)
    trainer.load_dataset((np.eye(20), 2.0 * np.arange(20)))
    params = MyParams(x=12)

    model, scores = trainer.train(params)

    assert "split0_valid_score" in scores
    assert "split4_valid_score" in scores
    assert "split5_valid_score" not in scores


def test_train_with_return_train_score() -> None:
    trainer = SklearnCVTrainer(MyParams, cv=3)
    trainer.return_train_score = True
    trainer.load_dataset((np.eye(6), 2.0 * np.arange(6)))
    params = MyParams(x=12)

    _, scores = trainer.train(params)

    expected_valid_scores = [11.11, 33.33, 55.55]
    expected_train_scores = [22.22, 44.44, 66.66]
    assert scores == {
        "mean_valid_score": np.mean(expected_valid_scores),
        "std_valid_score": np.std(expected_valid_scores),
        "split0_valid_score": 11.11,
        "split1_valid_score": 33.33,
        "split2_valid_score": 55.55,
        "mean_train_score": np.mean(expected_train_scores),
        "std_train_score": np.std(expected_train_scores),
        "split0_train_score": 22.22,
        "split1_train_score": 44.44,
        "split2_train_score": 66.66,
    }


def test_train_with_custom_scorer() -> None:
    scorer = MyScorer()

    trainer = SklearnCVTrainer(MyParams, cv=3, scorer=scorer)
    trainer.load_dataset((np.eye(6), 2.0 * np.arange(6)))
    params = MyParams(x=12)

    model, scores = trainer.train(params)
    assert model is None

    assert len(scorer.args_history) == 3

    score_model, score_X, score_y = scorer.args_history[1]
    assert isinstance(score_model, MyFooModel)
    np.testing.assert_equal(
        score_X,
        np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], float),
    )
    np.testing.assert_equal(
        score_y,
        np.array([4, 6], float),
    )

    expected_scores = [101.11, 202.22, 303.33]
    assert scores == {
        "mean_valid_score": np.mean(expected_scores),
        "std_valid_score": np.std(expected_scores),
        "split0_valid_score": 101.11,
        "split1_valid_score": 202.22,
        "split2_valid_score": 303.33,
    }


def test_train_with_custom_scorer_as_str(mocker: MockerFixture) -> None:
    scorer = MyScorer()

    def get_scorer(name: str) -> Any:
        if name == "foobar":
            return scorer
        else:
            raise ValueError

    mocker.patch("sklearn.metrics.get_scorer", get_scorer)

    trainer = SklearnCVTrainer(MyParams, cv=3, scorer="foobar")
    trainer.load_dataset((np.eye(6), 2.0 * np.arange(6)))
    params = MyParams(x=12)

    model, scores = trainer.train(params)
    assert model is None

    expected_scores = [101.11, 202.22, 303.33]
    assert scores == {
        "mean_valid_score": np.mean(expected_scores),
        "std_valid_score": np.std(expected_scores),
        "split0_valid_score": 101.11,
        "split1_valid_score": 202.22,
        "split2_valid_score": 303.33,
    }


def test_create_model_kwargs(mocker: MockerFixture) -> None:
    trainer = SklearnCVTrainer(
        MyParams, cv=3, create_model_kwargs={"color": "blue", "foo": 58}
    )
    trainer.load_dataset((np.eye(6), 2 * np.arange(6)))
    params = MyParams(x=12)

    create_model_spy = mocker.spy(params, "create_model")

    _ = trainer.train(params)

    create_model_spy.assert_called_once_with(color="blue", foo=58)


def test_fit_model_kwargs() -> None:
    created_models = []

    class MyParams2(letstune.Params):
        x: int = rand.oneof([45])  # type: ignore

        def create_model(self, **kwargs: Any) -> MyFooModel:
            nonlocal created_models

            m = MyFooModel(
                **self.to_dict(),
                **kwargs,
            )
            created_models.append(m)
            return m

    trainer = SklearnCVTrainer(
        MyParams2, cv=3, fit_model_kwargs={"color": "blue", "foo": 58}
    )
    trainer.load_dataset((np.eye(6), 2 * np.arange(6)))
    params = MyParams2(x=12)

    _ = trainer.train(params)

    assert len(created_models) == 1
    model = created_models[0]
    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12}

    assert len(model.fit_args_kwargs_history) == 3

    (args, kwargs) = model.fit_args_kwargs_history[1]
    assert kwargs == {"color": "blue", "foo": 58}


def test_final_train() -> None:
    trainer = SklearnCVTrainer(
        MyParams,
        cv=3,
        fit_model_kwargs={"color": "blue", "foo": 58},
        create_model_kwargs={"lambda": 3.14, "name": "unknown"},
    )
    trainer.load_dataset((np.eye(6), 2 * np.arange(6)))
    params = MyParams(x=12)

    model = trainer.final_train(params)

    assert isinstance(model, MyFooModel)
    assert model.init_kwargs == {"x": 12, "lambda": 3.14, "name": "unknown"}
    assert len(model.fit_args_kwargs_history) == 1

    fit_args, fit_kwargs = model.fit_args_kwargs_history[0]
    assert fit_kwargs == {"color": "blue", "foo": 58}

    fit_X, fit_y = fit_args
    np.testing.assert_equal(fit_X, np.eye(6))
    np.testing.assert_equal(fit_y, 2 * np.arange(6))
