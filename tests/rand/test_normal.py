from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def test_returns_float(rng: np.random.Generator) -> None:
    gen = rand.normal()

    x = gen.get_random_params(rng)

    assert type(x) is float


def test_default(rng: np.random.Generator) -> None:
    gen = rand.normal()

    samples = get_samples(rng, gen)

    _, p_value = stats.kstest(samples, "norm")

    assert p_value > MIN_P_VALUE


def test_custom(rng: np.random.Generator) -> None:
    gen = rand.normal(5, 3)

    samples = get_samples(rng, gen)

    dist = stats.norm(5, 3)
    assert dist.median() == 5
    assert dist.std() == 3

    _, p_value = stats.kstest(samples, dist.cdf)

    assert p_value > MIN_P_VALUE


def test_single_point(rng: np.random.Generator) -> None:
    gen = rand.normal(7.5, 0)

    samples = get_samples(rng, gen)

    assert all(s == 7.5 for s in samples)


def test_invalid_scale() -> None:
    with pytest.raises(ValueError, match="invalid scale"):
        _ = rand.normal(1.0, -5.0)
