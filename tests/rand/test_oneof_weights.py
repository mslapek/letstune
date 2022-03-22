from typing import Any

import numpy as np
import pytest
from scipy import stats

from letstune import rand
from letstune.rand import RandomParamsGenerator

from .utils import MIN_P_VALUE, get_samples


def test_invalid_weight_count() -> None:
    colors = ["red", "green", "blue"]
    weights = [1, 1, 2, 2]

    with pytest.raises(ValueError, match="weights"):
        _ = rand.oneof(colors, weights)


def test_string_values(rng: np.random.Generator) -> None:
    colors = ["red", "green", "blue"]
    weights = [3, 1, 1]
    gen = rand.oneof(colors, weights=weights)

    samples = get_samples(rng, gen)

    frequencies = np.bincount([colors.index(c) for c in samples])
    _, p_value = stats.chisquare(frequencies, 1000 * np.array([3 / 5, 1 / 5, 1 / 5]))
    assert p_value > MIN_P_VALUE


def test_embedded_string_oneof(rng: np.random.Generator) -> None:
    warm_colors = ["red", "orange", "yellow"]
    cold_colors = ["blue", "violet"]
    colors = [*warm_colors, *cold_colors, "white"]

    gen: RandomParamsGenerator[str] = rand.oneof(
        [rand.oneof(warm_colors), rand.oneof(cold_colors), "white"], weights=[3, 2, 1]
    )

    samples = get_samples(rng, gen)

    frequencies = np.bincount([colors.index(c) for c in samples])
    _, p_value = stats.chisquare(frequencies)
    assert p_value > MIN_P_VALUE


def test_embedded_uniforms(rng: np.random.Generator) -> None:
    gen: RandomParamsGenerator[float] = rand.oneof(
        [
            rand.uniform(2, 5),
            rand.uniform(8, 10),
        ],
        weights=[5, 10],
    )

    samples = get_samples(rng, gen)

    def cdf(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cdf1 = stats.uniform.cdf(s, 2, 3)
        cdf2 = stats.uniform.cdf(s, 8, 2)
        return 1 / 3 * cdf1 + 2 / 3 * cdf2  # type: ignore

    _, p_value = stats.kstest(samples, cdf)
    assert p_value > MIN_P_VALUE
