"""Training of digits classifier with ``SGDClassifier``
from sklearn with cross-validation.

This example is based on :class:`letstune.sklearn.SklearnCVTrainer`.

It does the same thing as example ``digits_sgd_simpletrainer_cv``,
but is more concise thanks to the :class:`letstune.sklearn.SklearnCVTrainer`
helper.
"""

import numpy as np
import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import letstune
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

trainer.load_dataset((X_train, y_train))

params = SGDClassifierParams(
    alpha=0.15,
    average=False,
    l1_ratio=0.033,
)
model, metrics = trainer.train(params)

final_model = trainer.final_train(params)


def test_model_is_none() -> None:
    assert model is None


def test_final_model_is_sgd_classifier() -> None:
    assert isinstance(final_model, SGDClassifier)


def test_model_has_all_metrics() -> None:
    assert set(metrics) == {
        "mean_valid_score",
        "split0_valid_score",
        "split1_valid_score",
        "split2_valid_score",
        "split3_valid_score",
        "split4_valid_score",
        "std_valid_score",
    }


def test_model_has_correct_valid_scores() -> None:
    scores = sklearn.model_selection.cross_val_score(
        params.create_model(
            loss="hinge",
            penalty="elasticnet",
            fit_intercept=True,
            random_state=42,
        ),
        X_train,
        y_train,
    )

    assert metrics["mean_valid_score"] == np.mean(scores)
    assert metrics["std_valid_score"] == np.std(scores)

    assert metrics["mean_valid_score"] > 0.90
