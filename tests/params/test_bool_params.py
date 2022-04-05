import numpy as np
import pytest
from scipy import stats

import letstune

from .utils import ConstantRandomParamsGenerator

beta_gen = ConstantRandomParamsGenerator(True)


class SimpleBarParams(letstune.Params):
    alpha: bool
    beta: bool = beta_gen  # type: ignore


@pytest.fixture
def params() -> SimpleBarParams:
    return SimpleBarParams(alpha=True, beta=False)


def test_instance_creation(params: SimpleBarParams) -> None:
    assert isinstance(params, SimpleBarParams)
    assert isinstance(params, letstune.Params)
    assert params.alpha is True
    assert params.beta is False


def test_get_random_params() -> None:
    rng = np.random.default_rng(42)

    samples = [SimpleBarParams.get_random_params(rng) for _ in range(1000)]

    for s in samples:
        assert isinstance(s, SimpleBarParams)
        assert s.beta is True

    true_count = int(sum(s.alpha for s in samples))
    false_count = len(samples) - true_count

    _, p_value = stats.chisquare([false_count, true_count])
    assert p_value > 0.1
