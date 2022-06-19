from typing import Any, Sequence

import letstune
from letstune.backend import repo
from letstune.backend.facade import _deduplicate_params, _fill_with_params
from letstune.backend.repo import JSON, EpochStats, Training


def test_deduplicate_params() -> None:
    class MyParams(letstune.Params):
        alpha: int
        beta: float

    ps = [
        MyParams(alpha=1, beta=3.3),
        MyParams(alpha=2, beta=3.3),
        MyParams(alpha=1, beta=3.3),
        MyParams(alpha=2, beta=3.3 + 1e-10),
    ]

    ds = _deduplicate_params(ps)

    assert ds == [
        MyParams(alpha=1, beta=3.3),
        MyParams(alpha=2, beta=3.3),
        MyParams(alpha=2, beta=3.3 + 1e-10),
    ]


def test_fill_with_params() -> None:
    class Repository(repo.Repository):
        def __init__(self) -> None:
            self.log: list[Any] = []

        def get_all_trainings(self) -> Sequence[Training]:
            return [
                repo.Training(
                    training_id=i,
                    params="aaa",
                )
                for i in range(12)
            ]

        def add_training(self, training_id: int, params: JSON) -> None:
            self.log.append((training_id, params))

        def add_epoch(self, training_id: int, epoch_stats: EpochStats) -> None:
            raise RuntimeError

        def set_error(self, training_id: int, description: str) -> None:
            raise RuntimeError

    repository = Repository()

    class MyParams(letstune.Params):
        alpha: int

    k = 600

    class Generator:
        def get_random_params(self, rng: Any) -> MyParams:
            nonlocal k
            k += 1

            if k == 614:
                return MyParams(alpha=613)  # duplicated 613
            else:
                return MyParams(alpha=k)

    generator = Generator()

    result = _fill_with_params(repository, generator, 15)

    assert result == MyParams
    assert k == 615
    assert repository.log == [
        (12, '{"alpha": 613}'),
        (13, '{"alpha": 615}'),
    ]
