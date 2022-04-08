from typing import Any

import numpy as np
from pytest_mock import MockerFixture

import letstune
from letstune import EpochTrainer, rand
from letstune.metrics import MetricValues


class Model:
    def __init__(self, x: int):
        self.x = x


class SomeParams(letstune.Params):
    x: int = rand.oneof([13])  # type: ignore

    def create_model(self) -> Model:
        return Model(self.x)


class QwertyTrainer(EpochTrainer[SomeParams]):
    metric = letstune.Metric("accuracy")

    def train_epoch(self, epoch: int) -> MetricValues:
        return {}

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
    model = Model(894)
    trainer.model = model

    trainer.save(checkpoint)

    save_pickle_spy.assert_called_once_with(model)


def test_load_unpickles_model(mocker: MockerFixture) -> None:
    model = Model(1010)

    class Checkpoint:
        def load_pickle(self) -> Model:
            return model

    checkpoint = Checkpoint()
    load_pickle_spy = mocker.spy(checkpoint, "load_pickle")

    trainer = QwertyTrainer()

    trainer.load(checkpoint, SomeParams(x=999))

    load_pickle_spy.assert_called_once_with()
    assert trainer.model == model


def test_create_model_calls_params_create_model() -> None:
    create_model_call_count = 0

    class SomeParams2(letstune.Params):
        x: int = rand.oneof([13])  # type: ignore

        def create_model(self) -> Model:
            nonlocal create_model_call_count
            create_model_call_count += 1
            return Model(self.x)

    class QwertyTrainer2(EpochTrainer[SomeParams2]):
        metric = letstune.Metric("accuracy")

        def train_epoch(self, epoch: int) -> MetricValues:
            return {}

        def load_dataset(self, dataset: Any) -> None:
            pass

    params = SomeParams2(x=913)
    trainer = QwertyTrainer2()

    trainer.create_model(params)

    assert create_model_call_count == 1
    assert isinstance(trainer.model, Model)
    assert trainer.model.x == 913
