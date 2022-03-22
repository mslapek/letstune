import math

import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def test_returns_float(rng: np.random.Generator) -> None:
    gen = rand.normal(0.1, 2.0, log=True)

    x = gen.get_random_params(rng)

    assert type(x) is float


def test_default() -> None:
    with pytest.raises(ValueError, match="invalid median"):
        _ = rand.normal(log=True)


@pytest.mark.parametrize(
    "median, scale",
    [
        (0.1, 2),
        (8.0, 0.5),
    ],
)
def test_custom_range(rng: np.random.Generator, median: float, scale: float) -> None:
    gen = rand.normal(median, scale, log=True)

    samples = get_samples(rng, gen)

    dist = stats.lognorm(scale, scale=median)
    # scale=median is NOT a bug!
    assert dist.median() == median

    for q in [0.05, 0.95]:
        assert dist.cdf(median * math.exp(scale * stats.norm.ppf(q))) == pytest.approx(
            q
        )

    _, p_value = stats.kstest(samples, dist.cdf)

    assert p_value > MIN_P_VALUE

    estimated_log_scale = np.std(np.log(samples))
    assert estimated_log_scale == pytest.approx(scale, rel=0.15)


def test_single_point(rng: np.random.Generator) -> None:
    gen = rand.normal(7.5, 0, log=True)

    samples = get_samples(rng, gen)

    assert all(s == pytest.approx(7.5) for s in samples)


def test_invalid_scale() -> None:
    with pytest.raises(ValueError, match="invalid scale"):
        _ = rand.normal(1.0, -5.0, log=True)


def test_invalid_median() -> None:
    with pytest.raises(ValueError, match="invalid median"):
        _ = rand.normal(-1.0, 0.2, log=True)

    with pytest.raises(ValueError, match="invalid median"):
        _ = rand.normal(0.0, 0.2, log=True)
