from __future__ import annotations

import pickle
from typing import Any, Iterable
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

import letstune

from .utils import ConstantRandomParamsGenerator, assert_equal, assert_not_equal

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


zeta_gen = ConstantRandomParamsGenerator("blue")


class ParentFooParams(letstune.Params):
    qwerty_params: Nested1QwertyParams | Nested2QwertyParams
    zeta: str = zeta_gen  # type: ignore


def get_params_nested1() -> ParentFooParams:
    return ParentFooParams(
        qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        zeta="orange",
    )


def get_params_nested2() -> ParentFooParams:
    return ParentFooParams(
        qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )


def test_instance_creation_nested1() -> None:
    params_nested1 = get_params_nested1()
    assert isinstance(params_nested1, ParentFooParams)
    assert isinstance(params_nested1, letstune.Params)

    qwerty_params = params_nested1.qwerty_params
    assert isinstance(qwerty_params, Nested1QwertyParams)
    assert isinstance(qwerty_params, letstune.Params)
    assert qwerty_params.alpha == 1
    assert qwerty_params.beta == "hi"
    assert qwerty_params.gamma == -999.99

    assert params_nested1.zeta == "orange"


def test_instance_creation_nested2() -> None:
    params_nested2 = get_params_nested2()
    assert isinstance(params_nested2, ParentFooParams)
    assert isinstance(params_nested2, letstune.Params)

    qwerty_params = params_nested2.qwerty_params
    assert isinstance(qwerty_params, Nested2QwertyParams)
    assert isinstance(qwerty_params, letstune.Params)
    assert qwerty_params.delta == 73
    assert qwerty_params.epsilon == 1000

    assert params_nested2.zeta == "orange"


@pytest.mark.parametrize(
    "params, expected_str",
    [
        (
            get_params_nested1(),
            "ParentFooParams("
            "qwerty_params=Nested1QwertyParams(alpha=1, beta='hi', gamma=-999.99), "
            "zeta='orange')",
        ),
        (
            get_params_nested2(),
            "ParentFooParams("
            "qwerty_params=Nested2QwertyParams(delta=73, epsilon=1000), "
            "zeta='orange')",
        ),
    ],
)
def test_str(params: ParentFooParams, expected_str: str) -> None:
    assert str(params) == expected_str


def test_equality() -> None:
    p1 = ParentFooParams(
        qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        zeta="orange",
    )
    p2 = ParentFooParams(
        qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        zeta="orange",
    )

    assert_equal(p1, p2)


def test_inequality() -> None:
    p1 = ParentFooParams(
        qwerty_params=Nested1QwertyParams(
            alpha=1,
            beta="hi",
            gamma=-999.99,
        ),
        zeta="orange",
    )
    p2 = ParentFooParams(
        qwerty_params=Nested2QwertyParams(
            delta=73,
            epsilon=1000,
        ),
        zeta="orange",
    )

    assert_not_equal(p1, p2)


def test_get_random_params_nested1(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(42)

    def random_params_spy(gen: object) -> MagicMock:
        return mocker.spy(gen, "get_random_params")

    def get_spies(gens: Iterable[object]) -> list[MagicMock]:
        return [random_params_spy(gen) for gen in gens]

    nested1_spies = get_spies(nested1_gens)
    nested2_spies = get_spies(nested2_gens)
    zeta_spy = random_params_spy(zeta_gen)
    nested1_default_gen_spy = random_params_spy(Nested1QwertyParams)
    nested2_default_gen_spy = random_params_spy(Nested2QwertyParams)

    p = ParentFooParams.get_random_params(rng)

    assert (
        type(p.qwerty_params) is Nested1QwertyParams
    ), "change seed to get appropriate class"

    for spy in nested1_spies:
        spy.assert_called_once_with(rng)

    nested1_default_gen_spy.assert_called_once_with(rng)

    for spy in nested2_spies:
        spy.assert_not_called()

    nested2_default_gen_spy.assert_not_called()

    zeta_spy.assert_called_once_with(rng)

    assert isinstance(p, ParentFooParams)
    assert isinstance(p, letstune.Params)

    qwerty_params = p.qwerty_params
    assert isinstance(qwerty_params, Nested1QwertyParams)
    assert isinstance(qwerty_params, letstune.Params)
    assert qwerty_params.alpha == 45
    assert qwerty_params.beta == "hello"
    assert qwerty_params.gamma == 3.3

    assert p.zeta == "blue"


def test_get_random_params_nested2(mocker: MockerFixture) -> None:
    rng = np.random.default_rng(43)

    def random_params_spy(gen: object) -> MagicMock:
        return mocker.spy(gen, "get_random_params")

    def get_spies(gens: Iterable[object]) -> list[MagicMock]:
        return [random_params_spy(gen) for gen in gens]

    nested1_spies = get_spies(nested1_gens)
    nested2_spies = get_spies(nested2_gens)
    zeta_spy = random_params_spy(zeta_gen)
    nested1_default_gen_spy = random_params_spy(Nested1QwertyParams)
    nested2_default_gen_spy = random_params_spy(Nested2QwertyParams)

    p = ParentFooParams.get_random_params(rng)

    assert (
        type(p.qwerty_params) is Nested2QwertyParams
    ), "change seed to get appropriate class"

    for spy in nested1_spies:
        spy.assert_not_called()

    nested1_default_gen_spy.assert_not_called()

    for spy in nested2_spies:
        spy.assert_called_once_with(rng)

    nested2_default_gen_spy.assert_called_once_with(rng)

    zeta_spy.assert_called_once_with(rng)

    assert isinstance(p, ParentFooParams)
    assert isinstance(p, letstune.Params)

    qwerty_params = p.qwerty_params
    assert isinstance(qwerty_params, Nested2QwertyParams)
    assert isinstance(qwerty_params, letstune.Params)
    assert qwerty_params.delta == 90
    assert qwerty_params.epsilon == -1

    assert p.zeta == "blue"


@pytest.mark.parametrize(
    "params, expected_dict",
    [
        (
            get_params_nested1(),
            {
                "qwerty_params": Nested1QwertyParams(
                    alpha=1,
                    beta="hi",
                    gamma=-999.99,
                ),
                "zeta": "orange",
            },
        ),
        (
            get_params_nested2(),
            {
                "qwerty_params": Nested2QwertyParams(
                    delta=73,
                    epsilon=1000,
                ),
                "zeta": "orange",
            },
        ),
    ],
)
def test_to_dict(params: ParentFooParams, expected_dict: dict[str, Any]) -> None:
    assert params.to_dict() == expected_dict


@pytest.mark.parametrize(
    "params, expected_json",
    [
        (
            get_params_nested1(),
            {
                "qwerty_params": {
                    "Nested1QwertyParams": {
                        "alpha": 1,
                        "beta": "hi",
                        "gamma": -999.99,
                    }
                },
                "zeta": "orange",
            },
        ),
        (
            get_params_nested2(),
            {
                "qwerty_params": {
                    "Nested2QwertyParams": {
                        "delta": 73,
                        "epsilon": 1000,
                    }
                },
                "zeta": "orange",
            },
        ),
    ],
)
def test_to_json(params: ParentFooParams, expected_json: dict[str, Any]) -> None:
    assert params.to_json() == expected_json


@pytest.mark.parametrize(
    "params, input_json",
    [
        (
            get_params_nested1(),
            {
                "qwerty_params": {
                    "Nested1QwertyParams": {
                        "alpha": 1,
                        "beta": "hi",
                        "gamma": -999.99,
                    }
                },
                "zeta": "orange",
            },
        ),
        (
            get_params_nested2(),
            {
                "qwerty_params": {
                    "Nested2QwertyParams": {
                        "delta": 73,
                        "epsilon": 1000,
                    }
                },
                "zeta": "orange",
            },
        ),
    ],
)
def test_from_json(params: ParentFooParams, input_json: dict[str, Any]) -> None:
    assert ParentFooParams.from_json(input_json) == params


@pytest.mark.parametrize("params", [get_params_nested1(), get_params_nested2()])
def test_pickle(params: ParentFooParams) -> None:
    s = pickle.dumps(params)

    p2 = pickle.loads(s)

    assert isinstance(p2, ParentFooParams)
    assert p2 == params
