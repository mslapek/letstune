from typing import Any, Generic, TypeVar

import numpy as np

import letstune
from letstune import SimpleTrainer, rand
from letstune.metrics import MetricValues


class SomeParams(letstune.Params):
    x: int = rand.oneof([13])  # type: ignore


def test_inherit_one() -> None:
    P = TypeVar("P", bound=letstune.Params)

    class MyBaseTrainer(SimpleTrainer[P]):
        metric = letstune.Metric.natural("loss")

        def train(self, params: P) -> tuple[Any, MetricValues]:
            return object(), {"loss": 0.5}

        def load_dataset(self, dataset: Any) -> None:
            pass

    class MyDerivedTrainer(MyBaseTrainer[SomeParams]):
        pass

    rng = np.random.default_rng(42)
    trainer = MyDerivedTrainer()

    params = trainer.get_random_params(rng)

    assert isinstance(params, SomeParams)
    assert params.x == 13


def test_inherit_many() -> None:
    P = TypeVar("P", bound=letstune.Params)

    class MyBaseTrainer(SimpleTrainer[P]):
        metric = letstune.Metric.natural("loss")

        def train(self, params: P) -> tuple[Any, MetricValues]:
            return object(), {"loss": 0.5}

        def load_dataset(self, dataset: Any) -> None:
            pass

    class SomeClass1:
        def hello1(self) -> None:
            pass

    class SomeClass2:
        def hello2(self) -> None:
            pass

    class MyDerivedTrainer(SomeClass1, MyBaseTrainer[SomeParams], SomeClass2):
        pass

    rng = np.random.default_rng(42)
    trainer = MyDerivedTrainer()

    params = trainer.get_random_params(rng)

    assert isinstance(params, SomeParams)
    assert params.x == 13


def test_inherit_many_with_generics() -> None:
    P = TypeVar("P", bound=letstune.Params)
    A = TypeVar("A")
    B = TypeVar("B")

    class MyBaseTrainer(SimpleTrainer[P]):
        metric = letstune.Metric.natural("loss")

        def train(self, params: P) -> tuple[Any, MetricValues]:
            return object(), {"loss": 0.5}

        def load_dataset(self, dataset: Any) -> None:
            pass

    class SomeClass1(Generic[A]):
        def hello1(self) -> None:
            pass

    class SomeClass2(Generic[B]):
        def hello2(self) -> None:
            pass

    class MyDerivedTrainer(
        SomeClass1[int], MyBaseTrainer[SomeParams], SomeClass2[float]
    ):
        pass

    rng = np.random.default_rng(42)
    trainer = MyDerivedTrainer()

    params = trainer.get_random_params(rng)

    assert isinstance(params, SomeParams)
    assert params.x == 13
