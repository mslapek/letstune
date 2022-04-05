"""Training of digits classifier with ``SGDClassifier`` from sklearn.

This example is based on :class:`letstune.SimpleTrainer`.
"""

from typing import Any

import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import letstune
from letstune import rand
from letstune.metrics import MetricValues

X, y = sklearn.datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class SGDClassifierParams(letstune.ModelParams[SGDClassifier]):
    average: bool
    l1_ratio: float = rand.uniform(0, 1)  # type: ignore
    alpha: float = rand.uniform(1e-2, 1e0, log=True)  # type: ignore


class DigitsTrainer(letstune.SimpleTrainer[SGDClassifierParams]):
    metric = letstune.Metric("accuracy")

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

trainer.load_dataset((X_train, X_test, y_train, y_test))

params = SGDClassifierParams(
    alpha=0.15,
    average=False,
    l1_ratio=0.033,
)
model, metrics = trainer.train(params)


def test_model_has_all_metrics() -> None:
    assert set(metrics) == {"accuracy"}


def test_model_has_correct_valid_score() -> None:
    accuracy = model.score(X_test, y_test)

    assert accuracy == metrics["accuracy"]
    assert accuracy > 0.90
