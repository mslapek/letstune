"""Training of MNIST digits classifier with Keras.

Parameter specification is split into many classes.
The example features *type union*, which
allows comparing different optimizer algorithms.

This example is based on :class:`letstune.EpochTrainer`.

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


class MNISTTrainer(letstune.EpochTrainer[MNISTParams]):
    metric = letstune.Metric("val_accuracy")

    def load_dataset(self, dataset: Any) -> None:
        (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x[:-10000]
        x_val = x[-10000:]
        y_train = y[:-10000]
        y_val = y[-10000:]

        self.x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
        self.x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
        self.x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

        self.num_classes = 10
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        # default implementation of create_model
        # sets self.model to params.create_model()

        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=epoch + 1,
            initial_epoch=epoch,
        )

        return {m: v[0] for m, v in history.history.items()}

    def save(self, checkpoint: Any) -> None:
        checkpoint.save_keras(self.model)

    def load(self, checkpoint: Any, params: MNISTParams) -> None:
        self.model = checkpoint.load_keras()


trainer = MNISTTrainer()

tuning = letstune.tune(
    trainer,
    8,
    results_dir=Path.home() / "ltexamples/keras/advanced_mnist_epochtrainer",
    rounds=[
        timedelta(seconds=2),
        timedelta(seconds=4),
    ],
)
print(f" DONE: {tuning}")

model = tuning[0].best_epoch.checkpoint.load_keras()
print(f"MODEL: {model}")
