"""Training of digits classifier with ``SGDClassifier`` from sklearn.

This example is based on :class:`letstune.sklearn.SklearnTrainer`.

It does the same thing as example ``digits_sgd_simpletrainer``,
but is more concise thanks to the :class:`letstune.sklearn.SklearnTrainer`
helper.
"""

import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import letstune.sklearn
from letstune import rand

X, y = sklearn.datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


class SGDClassifierParams(letstune.ModelParams[SGDClassifier]):
    average: bool
    l1_ratio: float = rand.uniform(0, 1)  # type: ignore
    alpha: float = rand.uniform(1e-2, 1e0, log=True)  # type: ignore


trainer: letstune.sklearn.SklearnTrainer[
    SGDClassifierParams
] = letstune.sklearn.SklearnTrainer(
    SGDClassifierParams,
    create_model_kwargs=dict(
        loss="hinge",
        penalty="elasticnet",
        fit_intercept=True,
        random_state=42,
    ),
)
trainer.return_train_score = True

trainer.load_dataset((X_train, X_test, y_train, y_test))

params = SGDClassifierParams(
    alpha=0.15,
    average=False,
    l1_ratio=0.033,
)
model, metrics = trainer.train(params)


def test_model_has_all_metrics() -> None:
    assert set(metrics) == {"valid_score", "train_score"}


def test_model_has_correct_valid_score() -> None:
    accuracy = model.score(X_test, y_test)

    assert accuracy == metrics["valid_score"]
    assert accuracy > 0.90


def test_model_has_correct_train_score() -> None:
    accuracy = model.score(X_train, y_train)

    assert accuracy == metrics["train_score"]
    assert accuracy > 0.90
