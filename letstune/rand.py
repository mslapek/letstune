from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T", covariant=True)


@runtime_checkable
class RandomParamsGenerator(Protocol[T]):
    def get_random_params(self, rng: np.random.Generator) -> T:
        ...


def oneof(
    candidates: Sequence[T | RandomParamsGenerator[T]],
) -> RandomParamsGenerator[T]:
    return _OneOf(candidates)


@dataclass(frozen=True)
class _OneOf(Generic[T]):
    seq: Sequence[T | RandomParamsGenerator[T]]

    def get_random_params(self, rng: np.random.Generator) -> T:
        i = rng.integers(len(self.seq))
        v = self.seq[i]

        if isinstance(v, RandomParamsGenerator):
            return v.get_random_params(rng)
        else:
            return v


def ints(low: int, to: int) -> RandomParamsGenerator[int]:
    return oneof(range(low, to + 1))


def uniform(low: float = 0.0, to: float = 1.0) -> RandomParamsGenerator[float]:
    return _Uniform(low, to)


@dataclass(frozen=True)
class _Uniform:
    low: float
    to: float

    def get_random_params(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.to)
