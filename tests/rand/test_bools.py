from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def get_frequencies(samples: list[bool]) -> list[int]:
    true_count = int(sum(samples))
    false_count = len(samples) - true_count
    frequencies = [false_count, true_count]
    return frequencies


def test_returns_bool(rng: np.random.Generator) -> None:
    gen = rand.bools()

    x = gen.get_random_params(rng)

    assert type(x) is bool


def test_default(rng: np.random.Generator) -> None:
    gen = rand.bools()

    samples = get_samples(rng, gen)

    frequencies = get_frequencies(samples)
    _, p_value = stats.chisquare(frequencies)
    assert p_value > MIN_P_VALUE


def test_custom_probability(rng: np.random.Generator) -> None:
    gen = rand.bools(0.7)

    samples = get_samples(rng, gen)

    frequencies = get_frequencies(samples)
    _, p_value = stats.chisquare(frequencies, [300, 700])
    assert p_value > MIN_P_VALUE


def test_zero_probability(rng: np.random.Generator) -> None:
    gen = rand.bools(0)

    samples = get_samples(rng, gen)

    assert all(s is False for s in samples)


def test_one_probability(rng: np.random.Generator) -> None:
    gen = rand.bools(1)

    samples = get_samples(rng, gen)

    assert all(s is True for s in samples)


def test_invalid_probability() -> None:
    with pytest.raises(ValueError, match="invalid probability"):
        _ = rand.bools(1.1)

    with pytest.raises(ValueError, match="invalid probability"):
        _ = rand.bools(-0.1)
