from typing import Any

import numpy as np
import pytest
from scipy import stats

from letstune import rand
from letstune.rand import RandomParamsGenerator

from .utils import MIN_P_VALUE, get_samples


def test_no_candidates() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        _ = rand.oneof([])


def test_string_values(rng: np.random.Generator) -> None:
    colors = ["red", "green", "blue"]
    gen = rand.oneof(colors)

    samples = get_samples(rng, gen)

    frequencies = np.bincount([colors.index(c) for c in samples])
    _, p_value = stats.chisquare(frequencies)
    assert p_value > MIN_P_VALUE


def test_embedded_string_oneof(rng: np.random.Generator) -> None:
    warm_colors = ["red", "orange", "yellow"]
    cold_colors = ["blue", "violet"]
    colors = [*warm_colors, *cold_colors, "white"]

    gen: RandomParamsGenerator[str] = rand.oneof(
        [rand.oneof(warm_colors), rand.oneof(cold_colors), "white"]
    )

    samples = get_samples(rng, gen)

    frequencies = np.bincount([colors.index(c) for c in samples])
    _, p_value = stats.chisquare(
        frequencies, 1000 * np.array([1 / 9, 1 / 9, 1 / 9, 1 / 6, 1 / 6, 1 / 3])
    )
    assert p_value > MIN_P_VALUE


def test_embedded_uniforms(rng: np.random.Generator) -> None:
    gen: RandomParamsGenerator[float] = rand.oneof(
        [
            rand.uniform(2, 5),
            rand.uniform(8, 10),
        ]
    )

    samples = get_samples(rng, gen)

    def cdf(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cdf1 = stats.uniform.cdf(s, 2, 3)
        cdf2 = stats.uniform.cdf(s, 8, 2)
        return 0.5 * cdf1 + 0.5 * cdf2  # type: ignore

    _, p_value = stats.kstest(samples, cdf)
    assert p_value > MIN_P_VALUE
