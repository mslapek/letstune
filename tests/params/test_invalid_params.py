import re
from itertools import product

import numpy as np
import pytest

import letstune
import letstune.params

from .utils import ConstantRandomParamsGenerator


class WithoutDefaultRandomGeneratorParams(letstune.Params):
    alpha: int = ConstantRandomParamsGenerator(5)  # type: ignore
    zeta: int


class Foo:
    pass


class SimpleParams(letstune.Params):
    alpha: int
    beta: str


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


def test_no_type_annotation() -> None:
    with pytest.raises(
        TypeError,
        match=r"alpha has no type annotation",
    ):

        class NoTypeAnnotationParams(letstune.Params):
            alpha = ConstantRandomParamsGenerator(5)
            beta: int

        _ = NoTypeAnnotationParams


def test_union_with_non_params() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("only Params are expected in type union (got NoneType)"),
    ):

        class NoneParams(letstune.Params):
            alpha: SimpleParams | None

        _ = NoneParams

    with pytest.raises(
        TypeError,
        match=re.escape("only Params are expected in type union (got str)"),
    ):

        class StrParams(letstune.Params):
            alpha: SimpleParams | str

        _ = StrParams


def test_invalid_params_type() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "invalid type of alpha "
            "(expected int, float, str, bool or Params, got list)"
        ),
    ):

        class ListParams(letstune.Params):
            alpha: list[int]

        _ = ListParams

    with pytest.raises(
        TypeError,
        match=re.escape(
            "invalid type of alpha "
            "(expected int, float, str, bool or Params, got Foo)"
        ),
    ):

        class FooParams(letstune.Params):
            alpha: Foo

        _ = FooParams


@pytest.mark.parametrize(
    [
        "method_name",
        "base",
        "decorator",
    ],
    product(
        [
            "__init__",
            "__repr__",
            "__str__",
            "__eq__",
            "to_json",
            "_to_json",
            "from_json",
            "to_dict",
            "__setattr__",
            "__getstate__",
            "__setstate__",
        ],
        ["letstune.Params", "letstune.ModelParams[Foo]"],
        ["", "@classmethod"],
    ),
)
def test_build_in_methods_are_final(
    method_name: str,
    base: str,
    decorator: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(f"cannot override {method_name}"),
    ):
        code = f"""
class {method_name}Params({base}):
    {decorator}
    def {method_name}(self):
        pass
"""
        exec(code)
