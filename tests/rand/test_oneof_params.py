import numpy as np

import letstune
from letstune import rand

alpha_gen = rand.oneof([45])
beta_gen = rand.oneof(["hello"])
gamma_gen = rand.oneof([3.3])
gens = [alpha_gen, beta_gen, gamma_gen]


class SimpleBarParams(letstune.Params):
    alpha: int = alpha_gen  # type: ignore
    beta: str = beta_gen  # type: ignore
    gamma: float = gamma_gen  # type: ignore


def test_params_values(rng: np.random.Generator) -> None:
    params = SimpleBarParams(alpha=1, beta="green", gamma=11.11)
    gen = rand.oneof([params])

    sample = gen.get_random_params(rng)

    assert sample == params


def test_random_params(rng: np.random.Generator) -> None:
    gen = rand.oneof([SimpleBarParams])

    sample = gen.get_random_params(rng)

    assert sample == SimpleBarParams(
        alpha=45,
        beta="hello",
        gamma=3.3,
    )
