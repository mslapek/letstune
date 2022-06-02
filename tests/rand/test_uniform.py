from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def test_returns_float(rng: np.random.Generator) -> None:
    gen = rand.uniform()

    x = gen.get_random_params(rng)

    assert type(x) is float


def test_default(rng: np.random.Generator) -> None:
    gen = rand.uniform()

    samples = get_samples(rng, gen)

    dist = stats.uniform()
    assert dist.median() == 0.5
    assert dist.ppf(0) == 0
    assert dist.ppf(1) == 1

    _, p_value = stats.kstest(samples, dist.cdf)

    assert p_value > MIN_P_VALUE


def test_custom_range(rng: np.random.Generator) -> None:
    gen = rand.uniform(6.0, 8.0)

    samples = get_samples(rng, gen)

    dist = stats.uniform(6.0, 2.0)
    assert dist.median() == 7.0
    assert dist.ppf(0) == 6.0
    assert dist.ppf(1) == 8.0

    _, p_value = stats.kstest(samples, dist.cdf)

    assert p_value > MIN_P_VALUE


def test_single_point(rng: np.random.Generator) -> None:
    gen = rand.uniform(7.5, 7.5)

    samples = get_samples(rng, gen)

    assert all(s == 7.5 for s in samples)


def test_invalid_range() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.uniform(1.0, 0.5)
