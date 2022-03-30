from typing import Any

import numpy as np
from pytest_mock import MockerFixture

import letstune
from letstune import SimpleTrainer, rand
from letstune.metrics import MetricValues


class SomeParams(letstune.Params):
    x: int = rand.oneof([13])  # type: ignore


class QwertyTrainer(SimpleTrainer[SomeParams]):
    metric = letstune.Metric("accuracy")

    def train(self, params: SomeParams) -> tuple[Any, MetricValues]:
        return object(), {"accuracy": 0.8}

    def load_dataset(self, dataset: Any) -> None:
        pass


def test_get_random_params() -> None:
    rng = np.random.default_rng(42)
    trainer = QwertyTrainer()

    params = trainer.get_random_params(rng)

    assert isinstance(params, SomeParams)
    assert params == SomeParams(x=13)


def test_save_pickles_model(mocker: MockerFixture) -> None:
    class Checkpoint:
        def save_pickle(self, obj: Any) -> None:
            pass

    checkpoint = Checkpoint()
    save_pickle_spy = mocker.spy(checkpoint, "save_pickle")

    trainer = QwertyTrainer()
    model = object()

    trainer.save(checkpoint, model)

    save_pickle_spy.assert_called_once_with(model)
