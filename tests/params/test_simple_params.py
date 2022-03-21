import pickle

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune

from .utils import ConstantRandomParamsGenerator

alpha_gen = ConstantRandomParamsGenerator(45)
beta_gen = ConstantRandomParamsGenerator("hello")
gamma_gen = ConstantRandomParamsGenerator(3.3)
gens = [alpha_gen, beta_gen, gamma_gen]


class SimpleBarParams(letstune.Params):
    alpha: int = alpha_gen  # type: ignore
    beta: str = beta_gen  # type: ignore
    gamma: float = gamma_gen  # type: ignore

    def say_hello(self) -> str:
        return f"{self.beta}{self.alpha}"


@pytest.fixture
def params() -> SimpleBarParams:
    return SimpleBarParams(alpha=2, beta="qwerty", gamma=-10.0)


def test_instance_creation(params: SimpleBarParams) -> None:
    assert isinstance(params, SimpleBarParams)
    assert isinstance(params, letstune.Params)
    assert params.alpha == 2
    assert params.beta == "qwerty"
    assert params.gamma == -10.0


def test_str(params: SimpleBarParams) -> None:
    assert str(params) == "SimpleBarParams(alpha=2, beta='qwerty', gamma=-10.0)"


def test_equality() -> None:
    p1 = SimpleBarParams(alpha=2, beta="qwerty", gamma=-10.0)
    p2 = SimpleBarParams(alpha=2, beta="qwerty", gamma=-10.0)

    assert p1 == p2
    assert not (p1 != p2)


def test_inequality() -> None:
    p1 = SimpleBarParams(alpha=2, beta="qwerty", gamma=-10.0)
    p2 = SimpleBarParams(alpha=2, beta="berty", gamma=-10.0)

    assert p1 != p2
    assert not (p1 == p2)


def test_get_random_params(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(42)
    spies = [mocker.spy(gen, "get_random_params") for gen in gens]

    p = SimpleBarParams.get_random_params(rng)

    for spy in spies:
        spy.assert_called_once_with(rng)

    assert isinstance(p, SimpleBarParams)
    assert isinstance(p, letstune.Params)
    assert p.alpha == 45
    assert p.beta == "hello"
    assert p.gamma == 3.3


def test_to_dict(params: SimpleBarParams) -> None:
    assert params.to_dict() == {"alpha": 2, "beta": "qwerty", "gamma": -10.0}


def test_to_json(params: SimpleBarParams) -> None:
    assert params.to_json() == {"alpha": 2, "beta": "qwerty", "gamma": -10.0}


def test_from_json(params: SimpleBarParams) -> None:
    assert (
        SimpleBarParams.from_json({"alpha": 2, "beta": "qwerty", "gamma": -10.0})
        == params
    )


def test_pickle(params: SimpleBarParams) -> None:
    s = pickle.dumps(params)

    p2 = pickle.loads(s)

    assert isinstance(p2, SimpleBarParams)
    assert p2 == params


def test_params_method(params: SimpleBarParams) -> None:
    v = params.say_hello()

    assert v == "qwerty2"


def test_unexpected_init_keyword_argument() -> None:
    with pytest.raises(
        TypeError,
        match=r"SimpleBarParams.__init__\(\) got an unexpected keyword argument 'zeta'",
    ):
        _ = SimpleBarParams(alpha=5, beta="red", gamma=3.14, zeta=99)


def test_unexpected_init_positional_argument() -> None:
    with pytest.raises(
        TypeError,
        match=r"Params.__init__\(\) takes 1 positional argument but 4 were given",
    ):
        _ = SimpleBarParams(5, "red", 3.14)  # type: ignore


def test_missing_init_keyword_argument() -> None:
    with pytest.raises(
        TypeError, match=r"SimpleBarParams.__init__\(\) missing keyword argument 'beta'"
    ):
        _ = SimpleBarParams(alpha=5, gamma=3.14)
