import numpy as np
import pytest
from scipy import stats

from letstune import rand

from .utils import MIN_P_VALUE, get_samples


def test_returns_float(rng: np.random.Generator) -> None:
    gen = rand.uniform(0.01, 0.1, log=True)

    x = gen.get_random_params(rng)

    assert type(x) is float


def test_default() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.uniform(log=True)


@pytest.mark.parametrize(
    "low, high, median",
    [
        (0.001, 10.0, 0.1),
        (2, 200, 20),
    ],
)
def test_custom_range(
    rng: np.random.Generator,
    low: float,
    high: float,
    median: float,
) -> None:
    gen = rand.uniform(low, high, log=True)

    samples = get_samples(rng, gen)

    dist = stats.loguniform(low, high)
    assert dist.median() == median
    assert dist.ppf(0) == low
    assert dist.ppf(1) == high

    _, p_value = stats.kstest(samples, dist.cdf)

    assert p_value > MIN_P_VALUE


def test_single_point(rng: np.random.Generator) -> None:
    gen = rand.uniform(7.5, 7.5, log=True)

    samples = get_samples(rng, gen)

    assert all(s == 7.5 for s in samples)


def test_invalid_range() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.uniform(1.0, 0.2, log=True)


def test_non_positive_range() -> None:
    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.uniform(-1.0, 0.2, log=True)

    with pytest.raises(ValueError, match="invalid range"):
        _ = rand.uniform(0.0, 0.2, log=True)
