from __future__ import annotations

from typing import TypeVar

import numpy as np

from letstune import rand

MIN_P_VALUE = 0.1

T = TypeVar("T")


def get_samples(
    rng: np.random.Generator, gen: rand.RandomParamsGenerator[T], n: int = 1000
) -> list[T]:
    return [gen.get_random_params(rng) for _ in range(n)]
