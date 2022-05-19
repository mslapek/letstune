"""Base classes :class:`Params` and :class:`ModelParams`
used to define hyper-parameters."""


__all__ = [
    "Params",
    "ModelParams",
    "NoDefaultRandomGeneratorError",
]

import dataclasses
import sys
from types import UnionType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

import numpy as np

from . import rand
from .rand import RandomParamsGenerator

SelfParams = TypeVar("SelfParams", bound="Params")
T = TypeVar("T")


class NoDefaultRandomGeneratorError(Exception):
    """Raised when :class:`Params`
    cannot derive default implementation of
    ``get_random_params`` for the given class."""

    def __init__(
        self,
        qualname: str,
        field_name: str | None = None,
    ) -> None:
        msg = f"class {qualname} doesn't have default random generator"
        if field_name is not None:
            msg += f" for field {field_name}"

        super().__init__(msg)


def _pop_annotations(dct: dict[str, Any]) -> dict[str, Any]:
    if "__annotations__" in dct:
        return dct.pop("__annotations__")  # type: ignore
    else:
        return {}


def _eval_annotations(
    dct: dict[str, Any], annotations: dict[str, Any]
) -> dict[str, type]:
    # compare with https://docs.python.org/3.10/howto/annotations.html
    globals_ = sys.modules[dct["__module__"]].__dict__

    return {
        field_name: eval(field_type, globals_)
        if isinstance(field_type, str)
        else field_type
        for field_name, field_type in annotations.items()
    }


def _validate_field_type(field_name: str, field_type: type) -> None:
    if isinstance(field_type, UnionType):
        for tt in field_type.__args__:
            if not issubclass(tt, Params):
                raise TypeError(
                    f"only Params are expected in "
                    f"type union (got {tt.__qualname__})"
                )
    else:
        valid_type = field_type in {int, float, str, bool} or issubclass(
            field_type, Params
        )

        if not valid_type:
            raise TypeError(
                f"invalid type of {field_name} "
                f"(expected int, float, str, bool or Params, "
                f"got {field_type.__qualname__})"
            )


def _random_generator_from_type(t: type[T]) -> RandomParamsGenerator[T]:
    if isinstance(t, RandomParamsGenerator):
        return t
    if t is bool:
        return rand.bools()  # type: ignore
    if isinstance(t, UnionType):
        return rand.oneof([_random_generator_from_type(tt) for tt in t.__args__])

    raise NoDefaultRandomGeneratorError(t.__qualname__)


def _pop_field(
    dct: dict[str, Any], field_name: str, field_type: type
) -> tuple[str, type, dataclasses.Field[Any]]:
    metadata = {}
    _validate_field_type(field_name, field_type)

    if field_name in dct:
        gen = dct.pop(field_name)
    else:
        try:
            gen = _random_generator_from_type(field_type)
        except NoDefaultRandomGeneratorError:
            gen = None

    if gen is not None:
        metadata["gen"] = gen

    return (field_name, field_type, dataclasses.field(metadata=metadata))


def _validate_forbidden_methods(
    name: str, dct: dict[str, Any], model_params: bool
) -> None:
    for k in dct:
        if k in {
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
        }:
            raise TypeError(f"cannot override {k}")
        elif model_params and k == "create_model":
            raise TypeError(
                f"""cannot override create_model. """
                f"""Please use letstune.Params as the base:\n"""
                f"""class {name}(letstune.Params):\n"""
                f"""    ...\n"""
            )


def _validate_leftover_generators(dct: dict[str, Any]) -> None:
    for k, v in dct.items():
        if isinstance(v, RandomParamsGenerator):
            raise TypeError(f"{k} has no type annotation")


def _assert_bases_are_valid(bases: tuple[Any, ...]) -> None:
    valid = len(bases) == 1

    if valid:
        base = bases[0]
        valid = base == Params or base == ModelParams

    if not valid:
        raise TypeError(
            f"only letstune.Params and letstune.ModelParams "
            f"are allowed as a base (got {bases})"
        )


def _is_class_from_letstune(name: str, dct: dict[str, Any]) -> bool:
    return dct["__module__"] == "letstune.params" and name in {
        "Params",
        "ModelParams",
    }


class _ParamsMeta(type):
    def __new__(
        mcs: type, name: str, bases: tuple[Any, ...], dct: dict[str, Any]
    ) -> Any:
        if _is_class_from_letstune(name, dct):
            return super().__new__(mcs, name, bases, dct)  # type: ignore

        _assert_bases_are_valid(bases)
        annotations = _eval_annotations(
            dct,
            _pop_annotations(dct),
        )
        _validate_forbidden_methods(name, dct, bases[0] == ModelParams)

        fields = [
            _pop_field(dct, field_name, field_type)
            for field_name, field_type in annotations.items()
        ]
        _validate_leftover_generators(dct)

        dataclass_cls = dataclasses.make_dataclass(
            "_" + name,
            fields,
            frozen=True,
            kw_only=True,
            slots=True,
        )
        bases = (*bases, dataclass_cls)

        dct["__slots__"] = tuple()

        return super().__new__(mcs, name, bases, dct)  # type: ignore


def _get_variant_name_from_json(json: Any) -> tuple[str, Any]:
    def raise_exception() -> None:
        raise ValueError(f"expected single-key dictionary (got {json!r})")

    single_key_dict = isinstance(json, dict) and len(json) == 1
    if not single_key_dict:
        raise_exception()

    (variant_name, json_value) = next(iter(json.items()))

    if not isinstance(variant_name, str):
        raise_exception()

    return variant_name, json_value


def _get_field_type_from_union(field_type: UnionType, variant_name: str) -> type:
    candidate: type
    for candidate in field_type.__args__:
        if candidate.__qualname__ == variant_name:
            return candidate

    raise ValueError(f"type name {variant_name!r} doesn't represent {field_type}")


class Params(metaclass=_ParamsMeta):
    """Base class used to define hyper-parameters.

    Example for neural network training digits recognition:

    >>> class DigitsTrainingParams(Params):
    ...     layer_number: int
    ...     learning_rate: float
    ...
    >>> params = DigitsTrainingParams(layer_number=2, learning_rate=0.1)
    >>> params
    DigitsTrainingParams(layer_number=2, learning_rate=0.1)

    **Random params generation**

    You can assign random generators to fields
    and use them to get random ``DigitsTrainingParams``:

    >>> from letstune import rand
    >>> import numpy as np
    ...
    >>> class DigitsTrainingParams(Params):
    ...     layer_number: int = rand.oneof([1, 2, 3])
    ...     learning_rate: float = rand.uniform(0.01, 0.1)
    ...
    >>> rng = np.random.default_rng(42)
    >>> DigitsTrainingParams.get_random_params(rng)  # doctest: +ELLIPSIS
    DigitsTrainingParams(layer_number=1, learning_rate=...)

    :class:`Params` can have defined parameter sampling in :meth:`get_random_params` method:

    >>> class DigitsTrainingParams(Params):
    ...     layer_number: int
    ...     learning_rate: float
    ...
    ...     @classmethod
    ...     def get_random_params(cls, rng):
    ...         return DigitsTrainingParams(
    ...             layer_number=rng.integers(1, 4),
    ...             learning_rate=rng.exponential(0.1),
    ...         )
    >>> rng = np.random.default_rng(42)
    >>> DigitsTrainingParams.get_random_params(rng)  # doctest: +ELLIPSIS
    DigitsTrainingParams(layer_number=1, learning_rate=...)

    **Nested params**

    You can compose :class:`Params` classes to express more
    sophisticated experiments.

    >>> class NeuralNetworkParams(Params):
    ...     layer_number: int = rand.oneof([1, 2, 3])
    ...     channels: int = rand.oneof([64, 128, 256])
    ...
    >>> class DigitsTrainingParams(Params):
    ...     neural_network: NeuralNetworkParams
    ...     learning_rate: float = rand.uniform(0.01, 0.1)
    >>> rng = np.random.default_rng(42)
    >>> DigitsTrainingParams.get_random_params(rng)  # doctest: +ELLIPSIS
    DigitsTrainingParams(neural_network=NeuralNetworkParams(layer_number=1, channels=256), learning_rate=...)

    **Union types**

    You can have different parameters for various kinds of your model.
    This way *letstune* can help you decide, which model family
    is the best for your problem.

    >>> class LassoParams(Params):
    ...     alpha: float = rand.uniform(0.01, 0.1)
    ...
    >>> class RandomForestParams(Params):
    ...     min_samples_split: int = rand.ints(1, 5)
    ...     max_features: str = rand.oneof(["sqrt", "log2"])
    ...
    >>> class ExperimentParams(Params):
    ...     model_params: LassoParams | RandomForestParams
    ...     pca_components: int = rand.oneof([16, 32, 64])
    ...
    >>> rng = np.random.default_rng(48)
    >>> ExperimentParams.get_random_params(rng)  # doctest: +ELLIPSIS
    ExperimentParams(model_params=LassoParams(alpha=...), pca_components=32)
    >>> ExperimentParams.get_random_params(rng)  # doctest: +ELLIPSIS
    ExperimentParams(model_params=RandomForestParams(min_samples_split=3, max_features='sqrt'), pca_components=64)

    Notice, that :meth:`get_random_params` returns various classes of ``model_params``.

    """  # noqa

    __slots__: tuple[str, ...] = tuple()

    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None:
            pass

    @classmethod
    def get_random_params(
        cls: type[SelfParams], rng: np.random.Generator
    ) -> SelfParams:
        """Get random instance of the params.

        **Default random generator**

        The default implementation chooses a random generator for each field:

        * If random generator is manually passed, it is used.
        * Otherwise, a random generator is deduced from the declared field type.
        * If there is no default random generator for the given type,
          then :class:`letstune.params.NoDefaultRandomGeneratorError` is raised.

        **Random generator from a type**

        If a given type has ``get_random_params`` method, then it is used.

        :class:`bool` is assigned ``True`` or ``False``, each with 50% probability.

        If a given type is a union, such as ``MyParams1 | MyParams2 | MyParams3``,
        then it
        is assigned::

            rand.oneof([MyParams1, MyParams2, MyParams3])

        """
        kwargs = {}

        for field in dataclasses.fields(cls):
            gen = field.metadata.get("gen")

            if gen is None:
                raise NoDefaultRandomGeneratorError(cls.__qualname__, field.name)

            kwargs[field.name] = gen.get_random_params(rng)

        return cls(**kwargs)

    def __field_to_json(
        self, field: dataclasses.Field[Any], add_union_type: bool
    ) -> Any:
        field_value = getattr(self, field.name)
        result: Any

        if isinstance(field_value, (int, float, str)):
            result = field_value
        else:
            result = field_value.to_json()

        if isinstance(field.type, UnionType):
            type_name = type(field_value).__qualname__
            d = {type_name: result}

            if add_union_type:
                d["type"] = type_name

            result = d

        return result

    @final
    def to_json(self) -> dict[str, Any]:
        """Converts given params to JSON.

        >>> class NeuralNetworkParams(Params):
        ...     layer_number: int
        ...     channels: int
        ...
        >>> class DigitsTrainingParams(Params):
        ...     neural_network: NeuralNetworkParams
        ...     learning_rate: float
        ...
        >>> params = DigitsTrainingParams(
        ...     neural_network=NeuralNetworkParams(
        ...         layer_number=5,
        ...         channels=256
        ...     ),
        ...     learning_rate=0.1,
        ... )
        ...
        >>> params.to_json()
        {'neural_network': {'layer_number': 5, 'channels': 256}, 'learning_rate': 0.1}

        In case of union types, it wraps object in a dict
        with exactly one key - the name of the used type.

        >>> class LassoParams(Params):
        ...     alpha: float
        ...
        >>> class RandomForestParams(Params):
        ...     min_samples_split: int
        ...     max_features: str
        ...
        >>> class ExperimentParams(Params):
        ...     model_params: LassoParams | RandomForestParams
        ...     pca_components: int
        ...
        >>> params = ExperimentParams(
        ...     model_params=LassoParams(alpha=0.7),
        ...     pca_components=4
        ... )
        >>> params.to_json()
        {'model_params': {'LassoParams': {'alpha': 0.7}}, 'pca_components': 4}

        >>> params = ExperimentParams(
        ...     model_params=RandomForestParams(
        ...         min_samples_split=4,
        ...         max_features="log2",
        ...     ),
        ...     pca_components=4
        ... )
        >>> import json
        >>> print(json.dumps(params.to_json(), indent=4))
        {
            "model_params": {
                "RandomForestParams": {
                    "min_samples_split": 4,
                    "max_features": "log2"
                }
            },
            "pca_components": 4
        }

        """  # noqa
        return self._to_json()

    @final
    def _to_json(self, *, add_union_type: bool = False) -> dict[str, Any]:
        return {
            field.name: self.__field_to_json(field, add_union_type)
            for field in dataclasses.fields(self)
        }

    @classmethod
    @final
    def from_json(cls: type[SelfParams], json: Any) -> SelfParams:
        """Creates params instance from JSON, which was produced by :meth:`to_json` method.

        >>> class NeuralNetworkParams(Params):
        ...     layer_number: int
        ...     channels: int
        ...
        >>> class DigitsTrainingParams(Params):
        ...     neural_network: NeuralNetworkParams
        ...     learning_rate: float

        >>> j = {
        ...     'neural_network': {
        ...         'layer_number': 5,
        ...         'channels': 256
        ...     },
        ...     'learning_rate': 0.1
        ... }
        >>> params = DigitsTrainingParams.from_json(j)
        >>> params
        DigitsTrainingParams(neural_network=NeuralNetworkParams(layer_number=5, channels=256), learning_rate=0.1)

        .. warning::
           :meth:`from_json` function *is not secure*.
           If you do not trust the input data, please verify schema
           of the input JSON before calling :meth:`from_json`.
        """  # noqa
        if not isinstance(json, dict):
            raise TypeError(f"expected dictionary (got {json=})")

        args = {}
        for field in dataclasses.fields(cls):
            json_value = json[field.name]
            field_type = field.type

            if isinstance(field.type, UnionType):
                variant_name, json_value = _get_variant_name_from_json(json_value)
                field_type = _get_field_type_from_union(field_type, variant_name)

            if hasattr(field_type, "from_json"):
                field_value = field_type.from_json(json_value)
            else:
                field_value = json_value

            args[field.name] = field_value

        return cls(**args)

    @final
    def to_dict(self) -> dict[str, Any]:
        """Converts given params to a dict.

        Notice, this is *not* a recursive operation.

        >>> class NeuralNetworkParams(Params):
        ...     layer_number: int
        ...     channels: int
        ...
        >>> class DigitsTrainingParams(Params):
        ...     neural_network: NeuralNetworkParams
        ...     learning_rate: float
        ...
        >>> params = DigitsTrainingParams(
        ...     neural_network=NeuralNetworkParams(
        ...         layer_number=5,
        ...         channels=256
        ...     ),
        ...     learning_rate=0.1,
        ... )
        ...
        >>> params.to_dict()
        {'neural_network': NeuralNetworkParams(layer_number=5, channels=256), 'learning_rate': 0.1}
        """  # noqa
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


M = TypeVar("M")


class ModelParams(Generic[M], Params):
    """Base class used to define hyper-parameters, which are used
    to create a model.

    Has all features of the :class:`Params` class.

    Additionally, it provides :meth:`create_model` method, which creates
    the model using given params.

    >>> class RandomForestRegressor:
    ...     '''sklearn-like model class.'''
    ...     def __init__(
    ...         self,
    ...         n_estimators: int = 100,
    ...         min_samples_split: int = 2,
    ...         max_features: str = "auto",
    ...     ):
    ...         self.n_estimators = n_estimators
    ...         self.min_samples_split = min_samples_split
    ...         self.max_features = max_features
    ...
    ...     def fit(self, X, y):
    ...         print(
    ...             f"fitted with {self.n_estimators=}\\n"
    ...             f"and {self.min_samples_split=}\\n"
    ...             f"and {self.max_features=}"
    ...         )

    >>> class RandomForestParams(ModelParams[RandomForestRegressor]):
    ...     min_samples_split: int
    ...     max_features: str

    >>> params = RandomForestParams(min_samples_split=8, max_features="log2")
    >>> params
    RandomForestParams(min_samples_split=8, max_features='log2')
    >>> isinstance(params, RandomForestParams)
    True

    >>> model = params.create_model()
    >>> isinstance(model, RandomForestRegressor)
    True
    >>> model.fit([1, 2, 3], [4, 5, 6])
    fitted with self.n_estimators=100
    and self.min_samples_split=8
    and self.max_features='log2'

    :meth:`create_model` can accept additional keyword arguments, which
    are passed to the model.

    >>> params = RandomForestParams(min_samples_split=8, max_features="log2")

    >>> model = params.create_model(n_estimators=500, min_samples_split=4)
    >>> model.fit([1, 2, 3], [4, 5, 6])
    fitted with self.n_estimators=500
    and self.min_samples_split=4
    and self.max_features='log2'

    """  # noqa

    __slots__: tuple[str, ...] = tuple()

    @final
    def create_model(self, **kwargs: Any) -> M:
        """For a class inheriting from :class:`ModelParams` [``M``],
        it returns a model of type ``M``.

        The model is created with arguments collected from the params.
        Additionally, it passes arguments given directly
        to the :meth:`create_model` method.

        For a class::

            class RandomForestParams(ModelParams[RandomForestRegressor]):
                min_samples_split: int
                max_features: str

            params = RandomForestParams(min_samples_split=8, max_features="log2")

        calling :meth:`create_model` method::

            model = params.create_model(n_estimators=500)

        is equivalent to::

            model = RandomForestParams(
                min_samples_split=params.min_samples_split,
                max_features=params.max_features,
                n_estimators=500,
            )

        Notice, that arguments passed to :meth:`create_model` have precedence
        over the arguments from ``params``.
        """
        m = type(self).__orig_bases__[0].__args__[0]  # type: ignore
        d = self.to_dict()
        d |= kwargs
        return m(**d)  # type: ignore
