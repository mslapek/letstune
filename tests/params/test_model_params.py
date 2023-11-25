import pickle
import re
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune

from .utils import ConstantRandomParamsGenerator, assert_equal, assert_not_equal


class MyFooModel:
    def __init__(self, **kwargs: Any) -> None:
        self.called_kwargs = kwargs


alpha_gen = ConstantRandomParamsGenerator(45)
beta_gen = ConstantRandomParamsGenerator("hello")
gamma_gen = ConstantRandomParamsGenerator(3.3)
gens = [alpha_gen, beta_gen, gamma_gen]


class MyQwertyModelParams(letstune.ModelParams[MyFooModel]):
    model_cls = MyFooModel

    alpha: int = alpha_gen  # type: ignore
    beta: str = beta_gen  # type: ignore
    gamma: float = gamma_gen  # type: ignore

    def say_hello(self) -> str:
        return f"{self.beta}{self.alpha}"


@pytest.fixture
def params() -> MyQwertyModelParams:
    return MyQwertyModelParams(alpha=2, beta="qwerty", gamma=-10.0)


def test_instance_creation(params: MyQwertyModelParams) -> None:
    assert isinstance(params, MyQwertyModelParams)
    assert isinstance(params, letstune.Params)
    assert isinstance(params, letstune.ModelParams)
    assert params.alpha == 2
    assert params.beta == "qwerty"
    assert params.gamma == -10.0


def test_create_model(params: MyQwertyModelParams) -> None:
    model = params.create_model()

    assert isinstance(model, MyFooModel)
    assert model.called_kwargs == {"alpha": 2, "beta": "qwerty", "gamma": -10.0}


def test_create_model_with_args(params: MyQwertyModelParams) -> None:
    model = params.create_model(zeta="green", gamma=-999.0)

    assert isinstance(model, MyFooModel)
    assert model.called_kwargs == {
        "alpha": 2,
        "beta": "qwerty",
        "gamma": -999.0,
        "zeta": "green",
    }


def test_str(params: MyQwertyModelParams) -> None:
    assert str(params) == "MyQwertyModelParams(alpha=2, beta='qwerty', gamma=-10.0)"


def test_equality() -> None:
    p1 = MyQwertyModelParams(alpha=2, beta="qwerty", gamma=-10.0)
    p2 = MyQwertyModelParams(alpha=2, beta="qwerty", gamma=-10.0)

    assert_equal(p1, p2)


def test_inequality() -> None:
    p1 = MyQwertyModelParams(alpha=2, beta="qwerty", gamma=-10.0)
    p2 = MyQwertyModelParams(alpha=2, beta="berty", gamma=-10.0)

    assert_not_equal(p1, p2)


def test_get_random_params(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(42)
    spies = [mocker.spy(gen, "get_random_params") for gen in gens]

    p = MyQwertyModelParams.get_random_params(rng)

    for spy in spies:
        spy.assert_called_once_with(rng)

    assert isinstance(p, MyQwertyModelParams)
    assert isinstance(p, letstune.Params)
    assert p.alpha == 45
    assert p.beta == "hello"
    assert p.gamma == 3.3


def test_to_dict(params: MyQwertyModelParams) -> None:
    assert params.to_dict() == {"alpha": 2, "beta": "qwerty", "gamma": -10.0}


def test_to_json(params: MyQwertyModelParams) -> None:
    assert params.to_json() == {"alpha": 2, "beta": "qwerty", "gamma": -10.0}


def test_from_json(params: MyQwertyModelParams) -> None:
    assert (
        MyQwertyModelParams.from_json({"alpha": 2, "beta": "qwerty", "gamma": -10.0})
        == params
    )


def test_pickle(params: MyQwertyModelParams) -> None:
    s = pickle.dumps(params)

    p2 = pickle.loads(s)

    assert isinstance(p2, MyQwertyModelParams)
    assert p2 == params


def test_params_method(params: MyQwertyModelParams) -> None:
    v = params.say_hello()

    assert v == "qwerty2"


def test_has_no_dict(params: MyQwertyModelParams) -> None:
    assert not hasattr(params, "__dict__")


def test_unexpected_init_keyword_argument() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "MyQwertyModelParams.__init__() got "
            "an unexpected keyword argument 'zeta'"
        ),
    ):
        _ = MyQwertyModelParams(alpha=5, beta="red", gamma=3.14, zeta=99)


def test_unexpected_init_positional_argument() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Params.__init__() takes 1 positional argument but 4 were given"
        ),
    ):
        _ = MyQwertyModelParams(5, "red", 3.14)  # type: ignore


def test_missing_init_keyword_argument() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "MyQwertyModelParams.__init__() missing 1 "
            "required keyword-only argument: 'beta'"
        ),
    ):
        _ = MyQwertyModelParams(alpha=5, gamma=3.14)


def test_custom_create_model_is_forbidden() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            """cannot override create_model. """
            """Please use letstune.Params as the base:\n"""
            """class MyFooParams(letstune.Params):\n"""
            """    ...\n"""
        ),
    ):

        class MyFooParams(letstune.ModelParams[MyFooModel]):
            def create_model(self, x: int) -> MyFooModel:  # type: ignore
                return MyFooModel(x=x, y=8)

        _ = MyFooParams
