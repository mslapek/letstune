from __future__ import annotations

"""Training of digits classifier with ``SGDClassifier`` from sklearn.

This example is based on :class:`letstune.sklearn.SklearnTrainer`.

It does the same thing as example ``digits_sgd_simpletrainer``,
but is more concise thanks to the :class:`letstune.sklearn.SklearnTrainer`
helper.
"""
from pathlib import Path

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

tuning = letstune.tune(
    trainer,
    16,
    dataset=(X_train, X_test, y_train, y_test),
    results_dir=Path.home() / "ltexamples/digits_sgd_sklearntrainer",
)
print(f" DONE: {tuning}")

model = tuning[0].checkpoint.load_pickle()
print(f"MODEL: {model}")
