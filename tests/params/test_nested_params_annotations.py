from __future__ import annotations

import pickle
from typing import Iterable
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune

from .utils import ConstantRandomParamsGenerator

alpha_gen = ConstantRandomParamsGenerator(45)
beta_gen = ConstantRandomParamsGenerator("hello")
gamma_gen = ConstantRandomParamsGenerator(3.3)
nested1_gens = [alpha_gen, beta_gen, gamma_gen]


class Nested1QwertyParams(letstune.Params):
    alpha: int = alpha_gen  # type: ignore
    beta: str = beta_gen  # type: ignore
    gamma: float = gamma_gen  # type: ignore


delta_gen = ConstantRandomParamsGenerator(90)
epsilon_gen = ConstantRandomParamsGenerator(-1)
nested2_gens = [delta_gen, epsilon_gen]


class Nested2QwertyParams(letstune.Params):
    delta: int = delta_gen  # type: ignore
    epsilon: int = epsilon_gen  # type: ignore


nested2_gen = ConstantRandomParamsGenerator(Nested2QwertyParams(delta=4, epsilon=8))
zeta_gen = ConstantRandomParamsGenerator("blue")
parent_custom_gens = [nested2_gen, zeta_gen]


class ParentFooParams(letstune.Params):
    nested1_qwerty_params: Nested1QwertyParams
    nested2_qwerty_params: Nested2QwertyParams = nested2_gen  # type: ignore
    zeta: str = zeta_gen  # type: ignore


@pytest.fixture
def params() -> ParentFooParams:
    return ParentFooParams(
        nested1_qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        nested2_qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )


def test_instance_creation(params: ParentFooParams) -> None:
    assert isinstance(params, ParentFooParams)
    assert isinstance(params, letstune.Params)

    nested1_qwerty_params = params.nested1_qwerty_params
    assert isinstance(nested1_qwerty_params, Nested1QwertyParams)
    assert isinstance(nested1_qwerty_params, letstune.Params)
    assert nested1_qwerty_params.alpha == 1
    assert nested1_qwerty_params.beta == "hi"
    assert nested1_qwerty_params.gamma == -999.99

    nested2_qwerty_params = params.nested2_qwerty_params
    assert isinstance(nested2_qwerty_params, Nested2QwertyParams)
    assert isinstance(nested2_qwerty_params, letstune.Params)
    assert nested2_qwerty_params.delta == 73
    assert nested2_qwerty_params.epsilon == 1000

    assert params.zeta == "orange"


def test_str(params: ParentFooParams) -> None:
    assert (
        str(params) == "ParentFooParams("
        "nested1_qwerty_params=Nested1QwertyParams(alpha=1, beta='hi', gamma=-999.99), "
        "nested2_qwerty_params=Nested2QwertyParams(delta=73, epsilon=1000), "
        "zeta='orange')"
    )


def test_equality() -> None:
    p1 = ParentFooParams(
        nested1_qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        nested2_qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )
    p2 = ParentFooParams(
        nested1_qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        nested2_qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )

    assert p1 == p2
    assert not (p1 != p2)


def test_inequality() -> None:
    p1 = ParentFooParams(
        nested1_qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        nested2_qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )
    p2 = ParentFooParams(
        nested1_qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        nested2_qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=111,  # epsilon is different
        ),
        zeta="orange",
    )

    assert p1 != p2
    assert not (p1 == p2)


def test_get_random_params(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(42)

    def random_params_spy(gen: object) -> MagicMock:
        return mocker.spy(gen, "get_random_params")

    def get_spies(gens: Iterable[object]) -> list[MagicMock]:
        return [random_params_spy(gen) for gen in gens]

    nested1_spies = get_spies(nested1_gens)
    nested2_spies = get_spies(nested2_gens)
    parent_custom_spies = get_spies(parent_custom_gens)
    nested1_default_gen_spy = random_params_spy(Nested1QwertyParams)
    nested2_default_gen_spy = random_params_spy(Nested2QwertyParams)

    p = ParentFooParams.get_random_params(rng)

    for spy in nested1_spies:
        spy.assert_called_once_with(rng)

    nested1_default_gen_spy.assert_called_once_with(rng)

    for spy in nested2_spies:
        spy.assert_not_called()

    nested2_default_gen_spy.assert_not_called()

    for spy in parent_custom_spies:
        spy.assert_called_once_with(rng)

    assert isinstance(p, ParentFooParams)
    assert isinstance(p, letstune.Params)

    nested1_qwerty_params = p.nested1_qwerty_params
    assert isinstance(nested1_qwerty_params, Nested1QwertyParams)
    assert isinstance(nested1_qwerty_params, letstune.Params)
    assert nested1_qwerty_params.alpha == 45
    assert nested1_qwerty_params.beta == "hello"
    assert nested1_qwerty_params.gamma == 3.3

    nested2_qwerty_params = p.nested2_qwerty_params
    assert isinstance(nested2_qwerty_params, Nested2QwertyParams)
    assert isinstance(nested2_qwerty_params, letstune.Params)
    assert nested2_qwerty_params.delta == 4
    assert nested2_qwerty_params.epsilon == 8

    assert p.zeta == "blue"


def test_to_dict(params: ParentFooParams) -> None:
    assert params.to_dict() == {
        "nested1_qwerty_params": Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        "nested2_qwerty_params": Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        "zeta": "orange",
    }


def test_to_json(params: ParentFooParams) -> None:
    assert params.to_json() == {
        "nested1_qwerty_params": {
            "alpha": 1,
            "beta": "hi",
            "gamma": -999.99,
        },
        "nested2_qwerty_params": {
            "delta": 73,
            "epsilon": 1000,
        },
        "zeta": "orange",
    }


def test_from_json(params: ParentFooParams) -> None:
    assert (
        ParentFooParams.from_json(
            {
                "nested1_qwerty_params": {
                    "alpha": 1,
                    "beta": "hi",
                    "gamma": -999.99,
                },
                "nested2_qwerty_params": {
                    "delta": 73,
                    "epsilon": 1000,
                },
                "zeta": "orange",
            }
        )
        == params
    )


def test_pickle(params: ParentFooParams) -> None:
    s = pickle.dumps(params)

    p2 = pickle.loads(s)

    assert isinstance(p2, ParentFooParams)
    assert p2 == params
