"""Training of MNIST digits classifier with Keras.

Parameter specification is split into many classes.
The example features *type union*, which
allows comparing different optimizer algorithms.

This example is based on :class:`letstune.keras.KerasTrainer`.

It does the same thing as example ``advanced_mnist_epochtrainer``,
but is more concise thanks to the :class:`letstune.keras.KerasTrainer`
helper.

Based on `Getting started with KerasTuner \
<https://keras.io/guides/keras_tuner/getting_started/>`_.
"""

from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import letstune
import letstune.keras
from letstune import rand


class DenseParams(letstune.ModelParams[layers.Dense]):
    units: int = rand.oneof(range(32, 512 + 1, 32))  # type: ignore
    activation: str = rand.oneof(["relu", "tanh"])  # type: ignore


class AdamParams(letstune.ModelParams[keras.optimizers.Adam]):
    learning_rate: float = rand.uniform(1e-4, 1e-2, log=True)  # type: ignore
    epsilon: float = rand.uniform(1e-8, 1e1, log=True)  # type: ignore


class AdadeltaParams(letstune.ModelParams[keras.optimizers.Adadelta]):
    pass


class MNISTParams(letstune.Params):
    dense: DenseParams
    optimizer: AdamParams | AdadeltaParams
    dropout: bool

    def create_model(self) -> keras.Model:
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(self.dense.create_model())

        if self.dropout:
            model.add(layers.Dropout(rate=0.25))

        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer=self.optimizer.create_model(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class MNISTTrainer(letstune.keras.KerasTrainer[MNISTParams]):
    def __init__(self) -> None:
        super().__init__(MNISTParams, letstune.Metric("val_accuracy"))

    def load_dataset(self, dataset: Any) -> None:
        (x, y), _ = keras.datasets.mnist.load_data()

        x_train = x[:-10000]
        x_val = x[-10000:]
        y_train = y[:-10000]
        y_val = y[-10000:]

        self.x = np.expand_dims(x_train, -1).astype("float32") / 255.0
        x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0

        num_classes = 10
        self.y = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)

        self.validation_data = (x_val, y_val)


trainer = MNISTTrainer()
trainer.load_dataset(None)

params = MNISTParams(
    dense=DenseParams(units=64, activation="relu"),
    optimizer=AdamParams(learning_rate=1e-3, epsilon=1e-7),
    dropout=True,
)
trainer.create_model(params)

metrics1 = trainer.train_epoch(0)
metrics2 = trainer.train_epoch(1)


def test_model_has_all_metrics() -> None:
    for m in [metrics1, metrics2]:
        assert set(m) == {"accuracy", "loss", "val_accuracy", "val_loss"}


def test_model_has_good_metrics() -> None:
    assert metrics1["val_accuracy"] > 0.90
    assert metrics2["val_accuracy"] > 0.94
