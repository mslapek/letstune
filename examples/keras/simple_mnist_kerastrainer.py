from __future__ import annotations

"""Training of MNIST digits classifier with Keras.

Only one parameter is tuned: ``units`` in ``layers.Dense``.

This example is based on :class:`letstune.keras.KerasTrainer`.

It does the same thing as example ``simple_mnist_epochtrainer``,
but is more concise thanks to the :class:`letstune.keras.KerasTrainer`
helper.

Based on `Getting started with KerasTuner \
<https://keras.io/guides/keras_tuner/getting_started/>`_.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import letstune
import letstune.keras
from letstune import rand

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class MNISTParams(letstune.Params):
    units: int = rand.oneof(range(32, 512 + 1, 32))  # type: ignore

    def create_model(self) -> keras.Model:
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=self.units,
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


trainer = letstune.keras.KerasTrainer(
    MNISTParams,
    letstune.Metric("val_accuracy"),
)

tuning = letstune.tune(
    trainer,
    16,
    dataset=(x_train, y_train, (x_val, y_val)),
    results_dir=Path.home() / "ltexamples/keras/simple_mnist_kerastrainer",
    training_maximum_duration=timedelta(seconds=30),
)
print(f" DONE: {tuning}")

model = tuning[0].best_epoch.checkpoint.load_keras()
print(f"MODEL: {model}")
