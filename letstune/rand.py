"""Random generators, which can be used inside :class:`letstune.Params`.

Random params generator
-----------------------

A :class:`RandomParamsGenerator` is an object with method ``get_random_params``,
which takes :class:`numpy.random.Generator` and returns a random value.

All functions in this module return a random generator.
Classes deriving from :class:`letstune.Params` are also random generators.

Function :func:`oneof` allows to compose many random generators.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

import letstune

__all__ = [
    "RandomParamsGenerator",
    "oneof",
    "ints",
    "bools",
    "uniform",
    "normal",
]


T = TypeVar("T", covariant=True)


@runtime_checkable
class RandomParamsGenerator(Protocol[T]):
    """An object with method ``get_random_params``,
    which takes :class:`numpy.random.Generator`
    and returns a random value of type ``T``.

    All functions in ``letstune.rand`` module return a :class:`RandomParamsGenerator`.
    Classes deriving from :class:`letstune.Params`
    are also a :class:`RandomParamsGenerator`.
    """

    def get_random_params(self, rng: np.random.Generator) -> T:
        ...


def oneof(
    candidates: Sequence[T | RandomParamsGenerator[T]],
    weights: Sequence[float | int] | None = None,
) -> RandomParamsGenerator[T]:
    """Returns a randomly chosen candidate.

    >>> gen = oneof([2, 4, 8])
    >>> rng = np.random.default_rng(42)
    >>> gen.get_random_params(rng)
    2
    >>> gen.get_random_params(rng)
    8

    **Weights**

    An optional parameter ``weights`` takes relative weights of
    each candidate.
    If present, candidates with higher weight will be chosen
    more often.

    **Random generators**

    If the chosen candidate has ``get_random_params`` method,
    then a result of this method is returned.
    Therefore, :func:`oneof` can be used with other random generators:

    >>> gen = oneof([uniform(1, 2), uniform(4, 5)])
    >>> rng = np.random.default_rng(131)
    >>> gen.get_random_params(rng)  # doctest: +ELLIPSIS
    1.60...
    >>> gen.get_random_params(rng)  # doctest: +ELLIPSIS
    4.53...

    :func:`oneof` can be used with classes *deriving* from :class:`letstune.Params`:

    >>> import letstune
    >>> from letstune import rand
    >>> class DigitsTrainingParams(letstune.Params):
    ...     layer_number: int = rand.oneof([1, 2, 3])
    ...     learning_rate: float = rand.uniform(0.01, 0.1)

    >>> rng = np.random.default_rng(42)
    >>> gen = oneof([DigitsTrainingParams])
    >>> gen.get_random_params(rng)  # doctest: +ELLIPSIS
    DigitsTrainingParams(layer_number=..., learning_rate=...)

    This allows to make samples from various model families::

        gen = oneof([LassoParams, RandomForestParams])

    There is an *exception*:
    if the chosen candidate is an *instance* of :class:`letstune.Params`,
    then it is just returned:

    >>> gen = oneof([DigitsTrainingParams(
    ...     layer_number=4,
    ...     learning_rate=0.001,
    ... )])

    >>> gen.get_random_params(rng)
    DigitsTrainingParams(layer_number=4, learning_rate=0.001)
    """
    if len(candidates) == 0:
        raise ValueError("got no candidates")

    p: np.ndarray[Any, Any] | None
    if weights is not None:
        if len(weights) != len(candidates):
            raise ValueError(
                f"expected length of weights to be "
                f"{len(candidates)}, got {len(weights)}"
            )

        p = np.array(weights, dtype=float)
        p /= p.sum()
    else:
        p = None

    return _OneOf(candidates, p)


@dataclass(frozen=True)
class _OneOf(Generic[T]):
    seq: Sequence[T | RandomParamsGenerator[T]]
    p: np.ndarray[Any, Any] | None = None

    def get_random_params(self, rng: np.random.Generator) -> T:
        i = rng.choice(len(self.seq), p=self.p)
        v = self.seq[i]

        if not isinstance(v, letstune.Params) and isinstance(v, RandomParamsGenerator):
            return v.get_random_params(rng)
        else:
            return v  # type: ignore


def ints(low: int, high: int) -> RandomParamsGenerator[int]:
    """Returns a random integer in the range ``low``, ``high``
    (both sides inclusive)."""
    if not (low <= high):
        raise ValueError("invalid range")

    return oneof(range(low, high + 1))


def bools(probability: float = 0.5) -> RandomParamsGenerator[bool]:
    """Returns ``True`` with given ``probability``."""
    if not 0 <= probability <= 1:
        raise ValueError("invalid probability")

    return oneof([False, True], [1 - probability, probability])


@dataclass(frozen=True)
class _Log:
    gen: RandomParamsGenerator[float]

    def get_random_params(self, rng: np.random.Generator) -> float:
        return float(np.exp(self.gen.get_random_params(rng)))


def uniform(
    low: float = 0.0, high: float = 1.0, *, log: bool = False
) -> RandomParamsGenerator[float]:
    """Returns a number from uniform distribution. [#uniform]_

    If ``log`` is ``True``, then the sample
    is from loguniform distribution. [#loguniform]_

    .. [#uniform] See `scipy.stats.uniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_.
    .. [#loguniform] See `scipy.stats.loguniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html>`_.
    """  # noqa
    valid = low <= high
    if log:
        valid &= low > 0

    if not valid:
        raise ValueError("invalid range")

    if low == high:
        return oneof([low])

    if log:
        return _Log(_Uniform(np.log(low), np.log(high)))

    return _Uniform(low, high)


@dataclass(frozen=True)
class _Uniform:
    low: float
    high: float

    def get_random_params(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.high)


def normal(
    median: float = 0.0, scale: float = 1.0, *, log: bool = False
) -> RandomParamsGenerator[float]:
    """Returns a number from normal distribution [#norm]_
    with given ``median`` [#mean]_ and ``scale`` (standard deviation).

    If ``log`` is ``True``, then the sample
    is from loguniform distribution. [#lognorm]_
    Then ``scale`` is the standard deviation of natural logarithm of the variable.

    .. [#norm] See `scipy.stats.norm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_.
    .. [#mean] Median of normal distribution is equal to its mean.
    .. [#lognorm] See `scipy.stats.lognorm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_.
       The distribution from scipy has a different parametrization then this function.
    """  # noqa
    if scale < 0:
        raise ValueError("invalid scale")

    if scale == 0:
        return oneof([median])

    if log:
        if median <= 0:
            raise ValueError("invalid median")

        return _Log(_Normal(np.log(median), scale))

    return _Normal(median, scale)


@dataclass(frozen=True)
class _Normal:
    median: float
    std: float

    def get_random_params(self, rng: np.random.Generator) -> float:
        return rng.normal(self.median, self.std)
