from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def test_returns_int(rng: np.random.Generator) -> None:
    gen = rand.ints(5, 10)

    x = gen.get_random_params(rng)

    assert type(x) is int


def test_range(rng: np.random.Generator) -> None:
    gen = rand.ints(5, 10)

    samples = get_samples(rng, gen)

    _, frequencies = np.unique(samples, return_counts=True)
    _, p_value = stats.chisquare(frequencies)
    assert p_value > MIN_P_VALUE


def test_single_point(rng: np.random.Generator) -> None:
    gen = rand.ints(101, 101)

    samples = get_samples(rng, gen)

    assert all(s == 101 for s in samples)


def test_invalid_interval() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.ints(5, 4)
