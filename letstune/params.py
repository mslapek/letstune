"""Base classes :class:`Params` and :class:`ModelParams`
used to define hyper-parameters."""


__all__ = ["Params", "ModelParams", "NoDefaultRandomGenerator"]

import sys
from types import UnionType
from typing import Any, Generic, Mapping, TypeVar

import numpy as np

from . import rand
from .rand import RandomParamsGenerator

SelfParams = TypeVar("SelfParams", bound="Params")
T = TypeVar("T")


class NoDefaultRandomGenerator(Exception):
    """Raised when :class:`Params`
    cannot derive default implementation of
    ``get_random_params`` for the given class."""


def _ensure_annotations_exist(dct: dict[str, Any]) -> None:
    if "__annotations__" not in dct:
        dct["__annotations__"] = {}


def _eval_annotations(dct: dict[str, Any]) -> None:
    # compare with https://docs.python.org/3.10/howto/annotations.html
    globals_ = sys.modules[dct["__module__"]].__dict__
    annotations = dct["__annotations__"]

    for k, v in annotations.items():
        if isinstance(v, str):
            annotations[k] = eval(v, globals_)


def _get_slots(dct: dict[str, Any]) -> tuple[str, ...]:
    return tuple(dct["__annotations__"])


def _random_generator_from_type(t: type[T]) -> RandomParamsGenerator[T]:
    def get_error_msg() -> str:
        return f"Class {t.__qualname__} doesn't have default random generator"

    if isinstance(t, RandomParamsGenerator):
        return t
    if isinstance(t, UnionType):
        try:
            gens = [_random_generator_from_type(tt) for tt in t.__args__]
        except NoDefaultRandomGenerator as e:
            raise NoDefaultRandomGenerator(
                get_error_msg(),
                e,
            )

        return rand.oneof(gens)

    raise NoDefaultRandomGenerator(get_error_msg())


def _get_random_generators(
    dct: dict[str, Any]
) -> dict[str, RandomParamsGenerator[Any]]:
    annotations: Mapping[str, Any] = dct["__annotations__"]

    gens = {}

    for field_name, field_type in annotations.items():
        if field_name in dct:
            gen = dct[field_name]
        else:
            try:
                gen = _random_generator_from_type(field_type)
            except NoDefaultRandomGenerator as e:
                raise NoDefaultRandomGenerator(
                    f"Class {dct['__qualname__']} doesn't have "
                    f"default random generator "
                    f"for field {field_name}",
                    e,
                )

        gens[field_name] = gen

    return gens


def _delete_random_generators(dct: dict[str, Any]) -> None:
    for field in dct["__slots__"]:
        if field in dct:
            del dct[field]


def _assert_bases_are_valid(bases: tuple[Any, ...]) -> None:
    valid = len(bases) == 1

    if valid:
        base = bases[0]
        valid = base == Params or base == ModelParams

    if not valid:
        raise TypeError(
            f"Only letstune.Params and letstune.ModelParams "
            f"are allowed as a base, got {bases}"
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
        _ensure_annotations_exist(dct)
        _eval_annotations(dct)

        slots = _get_slots(dct)
        dct["__slots__"] = slots
        dct["__slots_set__"] = frozenset(slots)

        random_generators: NoDefaultRandomGenerator | dict[
            str, RandomParamsGenerator[Any]
        ]
        try:
            random_generators = _get_random_generators(dct)
        except NoDefaultRandomGenerator as e:
            random_generators = e
        dct["__random_generators__"] = random_generators

        _delete_random_generators(dct)

        cls = super().__new__(mcs, name, bases, dct)  # type: ignore

        return cls


def _assert_dict_contains_all_fields(self: object, kwargs: dict[str, Any]) -> None:
    field_names = self.__slots__
    missing_args = [k for k in field_names if k not in kwargs]
    if missing_args:
        raise TypeError(
            f"{type(self).__qualname__}.__init__() "
            f"missing keyword argument {missing_args[0]!r}"
        )


def _assert_field_is_expected(self: object, field_name: str) -> None:
    fields: frozenset[str] = self.__slots_set__  # type: ignore

    if field_name not in fields:
        raise TypeError(
            f"{type(self).__qualname__}.__init__() "
            f"got an unexpected keyword argument {field_name!r}"
        )


def _get_variant_name_from_json(json: Any) -> tuple[str, Any]:
    def raise_exception() -> None:
        raise ValueError(f"Expected single-key dictionary, got {json!r}")

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

    raise ValueError(f"Type name {variant_name!r} doesn't represent {field_type}")


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

    Random params generation
    ------------------------

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

    ``Params`` can have defined parameter sampling in ``get_random_params`` method:

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

    Nested params
    -------------

    You can compose ``Params`` classes to express more
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

    Union types
    -----------

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

    Notice, that ``get_random_params`` returns various classes of ``model_params``.

    """  # noqa

    def __init__(self, **kwargs: Any):
        _assert_dict_contains_all_fields(self, kwargs)

        for field_name, field_value in kwargs.items():
            _assert_field_is_expected(self, field_name)
            setattr(self, field_name, field_value)

    @classmethod
    def get_random_params(
        cls: type[SelfParams], rng: np.random.Generator
    ) -> SelfParams:
        """Get random instance of the params.

        Default random generator
        ------------------------

        The default implementation chooses a random generator for each field:

        * If random generator is manually passed, it is used.
        * Otherwise, a random generator is deduced from the declared field type.
        * If there is no default random generator for the given type,
          then :class:`NoDefaultRandomGenerator` is raised.

        Random generator from a type
        ----------------------------

        If a given type has ``get_random_params`` method, then it is used.

        If a given type is a union, such as ``MyParams1 | MyParams2 | MyParams3``,
        then it
        is assigned::

            rand.oneof([MyParams1, MyParams2, MyParams3])

        """
        random_generators: NoDefaultRandomGenerator | dict[
            str, RandomParamsGenerator[Any]
        ] = cls.__random_generators__  # type: ignore

        if isinstance(random_generators, Exception):
            raise random_generators

        return cls(
            **{f: g.get_random_params(rng) for f, g in random_generators.items()}
        )

    def __repr__(self) -> str:
        args = ", ".join(f"{f}={getattr(self, f)!r}" for f in self.__slots__)
        return f"{type(self).__qualname__}({args})"

    def __eq__(self, other: object) -> bool:
        if type(other) != type(self):
            return False

        return all(getattr(self, f) == getattr(other, f) for f in self.__slots__)

    def __field_to_json(self, field_name: str) -> Any:
        field_value = getattr(self, field_name)
        field_annotation = self.__annotations__[field_name]

        result: Any

        if isinstance(field_value, (int, float, str)):
            result = field_value
        else:
            result = field_value.to_json()

        if isinstance(field_annotation, UnionType):
            result = {type(field_value).__qualname__: result}

        return result

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
        return {
            field_name: self.__field_to_json(field_name)
            for field_name in self.__slots__
        }

    @classmethod
    def from_json(cls: type[SelfParams], json: Any) -> SelfParams:
        """Creates params instance from JSON, which was produced by ``to_json`` method.

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

        """  # noqa
        if not isinstance(json, dict):
            raise TypeError(f"Expected dictionary, got {json=}")

        args = {}
        for field_name, field_type in cls.__annotations__.items():
            json_value = json[field_name]

            if isinstance(field_type, UnionType):
                variant_name, json_value = _get_variant_name_from_json(json_value)
                field_type = _get_field_type_from_union(field_type, variant_name)

            if hasattr(field_type, "from_json"):
                field_value = field_type.from_json(json_value)
            else:
                field_value = json_value

            args[field_name] = field_value

        return cls(**args)

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
        return {f: getattr(self, f) for f in self.__slots__}


M = TypeVar("M")


class ModelParams(Generic[M], Params):
    """Base class used to define hyper-parameters, which are used
    to create a model.

    Has all features of the :class:`Params` class.

    Additionally, it provides ``create_model`` method, which creates
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

    ``create_model`` can accept additional keyword arguments, which
    are passed to the model.

    >>> params = RandomForestParams(min_samples_split=8, max_features="log2")

    >>> model = params.create_model(n_estimators=500, min_samples_split=4)
    >>> model.fit([1, 2, 3], [4, 5, 6])
    fitted with self.n_estimators=500
    and self.min_samples_split=4
    and self.max_features='log2'

    """  # noqa

    def create_model(self, **kwargs: Any) -> M:
        """For a class inheriting from ``ModelParams[M]``,
        it returns a model of type ``M``.

        The model is created with arguments collected from the params.
        Additionally, it passes arguments given directly to the ``create_model`` method.

        For a class::

            class RandomForestParams(ModelParams[RandomForestRegressor]):
                min_samples_split: int
                max_features: str

            params = RandomForestParams(min_samples_split=8, max_features="log2")

        calling ``create_model`` method::

            model = params.create_model(n_estimators=500)

        is equivalent to::

            model = RandomForestParams(
                min_samples_split=params.min_samples_split,
                max_features=params.max_features,
                n_estimators=500,
            )

        Notice, that arguments passed to ``create_model`` have precedence
        over the arguments from ``params``.
        """
        m = type(self).__orig_bases__[0].__args__[0]  # type: ignore
        d = self.to_dict()
        d |= kwargs
        return m(**d)  # type: ignore
