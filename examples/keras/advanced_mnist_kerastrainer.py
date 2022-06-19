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
from datetime import timedelta
from pathlib import Path
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

tuning = letstune.tune(
    trainer,
    8,
    results_dir=Path.home() / "ltexamples/keras/advanced_mnist_kerastrainer",
    rounds=[
        timedelta(seconds=2),
        timedelta(seconds=4),
    ],
)
print(f" DONE: {tuning}")

model = tuning[0].best_epoch.checkpoint.load_keras()
print(f"MODEL: {model}")
