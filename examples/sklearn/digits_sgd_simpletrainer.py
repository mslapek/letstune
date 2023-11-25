"""Training of digits classifier with ``SGDClassifier`` from sklearn.

This example is based on :class:`letstune.SimpleTrainer`.
"""
from pathlib import Path
from typing import Any

import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import letstune
from letstune import rand
from letstune.trainer import MetricValues

X, y = sklearn.datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class SGDClassifierParams(letstune.ModelParams[SGDClassifier]):
    average: bool
    l1_ratio: float = rand.uniform(0, 1)  # type: ignore
    alpha: float = rand.uniform(1e-2, 1e0, log=True)  # type: ignore


class DigitsTrainer(letstune.SimpleTrainer[SGDClassifierParams]):
    params_cls = SGDClassifierParams
    metric = "accuracy"

    def load_dataset(self, dataset: Any) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = dataset

    def train(self, params: SGDClassifierParams) -> tuple[Any, MetricValues]:
        model = params.create_model(
            loss="hinge",
            penalty="elasticnet",
            fit_intercept=True,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)

        accuracy = model.score(self.X_test, self.y_test)

        return model, {"accuracy": accuracy}


trainer = DigitsTrainer()

tuning = letstune.tune(
    trainer,
    16,
    dataset=(X_train, X_test, y_train, y_test),
    results_dir=Path.home() / "ltexamples/digits_sgd_simpletrainer",
)
print(f" DONE: {tuning}")

model = tuning[0].checkpoint.load_pickle()
print(f"MODEL: {model}")
