from __future__ import annotations

"""Training of digits classifier with ``SGDClassifier``
from sklearn with cross-validation.

This example is based on :class:`letstune.sklearn.SklearnCVTrainer`.

It does the same thing as example ``digits_sgd_simpletrainer_cv``,
but is more concise thanks to the :class:`letstune.sklearn.SklearnCVTrainer`
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


trainer: letstune.sklearn.SklearnCVTrainer[
    SGDClassifierParams
] = letstune.sklearn.SklearnCVTrainer(
    SGDClassifierParams,
    cv=sklearn.model_selection.StratifiedKFold(5),
    create_model_kwargs=dict(
        loss="hinge",
        penalty="elasticnet",
        fit_intercept=True,
        random_state=42,
    ),
)

tuning = letstune.tune(
    trainer,
    16,
    dataset=(X_train, y_train),
    results_dir=Path.home() / "ltexamples/digits_sgd_sklearncvtrainer",
)
print(f"  DONE TUNING: {tuning}")

trainer.load_dataset((X_train, y_train))

final_model = trainer.final_train(tuning[0].params)
print(f"DONE TRAINING: {final_model}")
