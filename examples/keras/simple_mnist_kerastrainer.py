"""Training of MNIST digits classifier with Keras.

Only one parameter is tuned: ``units`` in ``layers.Dense``.

This example is based on :class:`letstune.keras.KerasTrainer`.

It does the same thing as example ``simple_mnist_epochtrainer``,
but is more concise thanks to the :class:`letstune.keras.KerasTrainer`
helper.

Based on `Getting started with KerasTuner \
<https://keras.io/guides/keras_tuner/getting_started/>`_.
"""

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
trainer.load_dataset((x_train, y_train, (x_val, y_val)))

params = MNISTParams(units=64)
trainer.create_model(params)

metrics1 = trainer.train_epoch(0)
metrics2 = trainer.train_epoch(1)


def test_model_has_all_metrics() -> None:
    for m in [metrics1, metrics2]:
        assert set(m) == {"accuracy", "loss", "val_accuracy", "val_loss"}


def test_model_has_good_metrics() -> None:
    assert metrics1["val_accuracy"] > 0.90
    assert metrics2["val_accuracy"] > 0.94
