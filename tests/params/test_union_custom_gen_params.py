from typing import Iterable
from unittest.mock import MagicMock

import numpy as np
from pytest_mock import MockerFixture

import letstune

from .utils import ConstantRandomParamsGenerator

alpha_gen = ConstantRandomParamsGenerator(45)
beta_gen = ConstantRandomParamsGenerator("hello")
gamma_gen = ConstantRandomParamsGenerator(3.3)


class Nested1QwertyParams(letstune.Params):
    alpha: int = alpha_gen  # type: ignore
    beta: str = beta_gen  # type: ignore
    gamma: float = gamma_gen  # type: ignore


delta_gen = ConstantRandomParamsGenerator(90)
epsilon_gen = ConstantRandomParamsGenerator(-1)


class Nested2QwertyParams(letstune.Params):
    delta: int = delta_gen  # type: ignore
    epsilon: int = epsilon_gen  # type: ignore


nested_gens = [
    alpha_gen,
    beta_gen,
    gamma_gen,
    delta_gen,
    epsilon_gen,
]
zeta_gen = ConstantRandomParamsGenerator("blue")

qwerty_gen = ConstantRandomParamsGenerator(
    Nested2QwertyParams(
        delta=1290,
        epsilon=8483,
    )
)


class ParentFooParams(letstune.Params):
    qwerty_params: Nested1QwertyParams | Nested2QwertyParams = (
        qwerty_gen  # type: ignore
    )
    zeta: str = zeta_gen  # type: ignore


def test_get_random_params(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(42)

    def random_params_spy(gen: object) -> MagicMock:
        return mocker.spy(gen, "get_random_params")

    def get_spies(gens: Iterable[object]) -> list[MagicMock]:
        return [random_params_spy(gen) for gen in gens]

    nested_spies = get_spies(nested_gens)
    nested1_default_gen_spy = random_params_spy(Nested1QwertyParams)
    nested2_default_gen_spy = random_params_spy(Nested2QwertyParams)

    qwerty_spy = random_params_spy(qwerty_gen)
    zeta_spy = random_params_spy(zeta_gen)

    p = ParentFooParams.get_random_params(rng)

    for spy in nested_spies:
        spy.assert_not_called()

    nested1_default_gen_spy.assert_not_called()
    nested2_default_gen_spy.assert_not_called()

    zeta_spy.assert_called_once_with(rng)
    qwerty_spy.assert_called_once_with(rng)

    assert isinstance(p, ParentFooParams)
    assert isinstance(p, letstune.Params)

    qwerty_params = p.qwerty_params
    assert isinstance(qwerty_params, Nested2QwertyParams)
    assert isinstance(qwerty_params, letstune.Params)
    assert qwerty_params.delta == 1290
    assert qwerty_params.epsilon == 8483

    assert p.zeta == "blue"
