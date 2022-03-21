import numpy as np
import pytest

import letstune
import letstune.params

from .utils import ConstantRandomParamsGenerator


class WithoutDefaultRandomGeneratorParams(letstune.Params):
    alpha: int = ConstantRandomParamsGenerator(5)  # type: ignore
    zeta: int


def test_no_default_random_generator() -> None:
    rng = np.random.default_rng(42)

    with pytest.raises(
        letstune.params.NoDefaultRandomGenerator,
        match="WithoutDefaultRandomGeneratorParams.+zeta",
    ):
        _ = WithoutDefaultRandomGeneratorParams.get_random_params(rng)


def test_no_other_params_subclasses() -> None:
    class Foo:
        pass

    with pytest.raises(TypeError, match="Foo"):

        class InvalidParams1(letstune.Params, Foo):
            x: int

        _ = InvalidParams1

    with pytest.raises(TypeError, match="Foo"):

        class InvalidParams2(Foo, letstune.Params):
            x: int

        _ = InvalidParams2


def test_no_other_model_params_subclasses() -> None:
    class Foo:
        pass

    class Model:
        def __init__(self, x: int):
            self.x = x

    with pytest.raises(TypeError, match="Foo"):

        class InvalidParams1(letstune.ModelParams[Model], Foo):
            x: int

        _ = InvalidParams1

    with pytest.raises(TypeError, match="Foo"):

        class InvalidParams2(Foo, letstune.ModelParams[Model]):
            x: int

        _ = InvalidParams2
