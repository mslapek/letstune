from __future__ import annotations

import pickle

import numpy as np
import pytest

import letstune

from .utils import assert_equal


class EmptyFooParams(letstune.Params):
    pass


@pytest.fixture
def params() -> EmptyFooParams:
    return EmptyFooParams()


def test_instance_creation(params: EmptyFooParams) -> None:
    assert isinstance(params, EmptyFooParams)
    assert isinstance(params, letstune.Params)


def test_str(params: EmptyFooParams) -> None:
    assert str(params) == "EmptyFooParams()"


def test_equality() -> None:
    p1 = EmptyFooParams()
    p2 = EmptyFooParams()

    assert_equal(p1, p2)


def test_get_random_params() -> None:
    rng = np.random.default_rng(42)

    p = EmptyFooParams.get_random_params(rng)

    assert isinstance(p, EmptyFooParams)


def test_to_dict(params: EmptyFooParams) -> None:
    assert params.to_dict() == {}


def test_to_json(params: EmptyFooParams) -> None:
    assert params.to_json() == {}


def test_from_json(params: EmptyFooParams) -> None:
    assert EmptyFooParams.from_json({}) == params


def test_pickle(params: EmptyFooParams) -> None:
    s = pickle.dumps(params)

    p2 = pickle.loads(s)

    assert isinstance(p2, EmptyFooParams)
    assert p2 == params
