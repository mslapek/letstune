<div align="center">
  <img src="https://raw.githubusercontent.com/mslapek/letstune/main/img/logo.svg"><br>
</div>

-----------------

# letstune

*Hyper-parameter tuning for the masses!*

![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg?style=flat-square)
[![Documentation Status](https://readthedocs.org/projects/letstune/badge/?version=latest&style=flat-square)](https://letstune.readthedocs.io/en/latest/?badge=latest)
[![PyPI wheel](https://img.shields.io/pypi/wheel/letstune?color=orange&label=pip&style=flat-square)](https://pypi.org/project/letstune/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat-square)](https://pycqa.github.io/isort/)

[![Lint and test workflow](https://github.com/mslapek/letstune/actions/workflows/linttest.yml/badge.svg)](https://github.com/mslapek/letstune/actions/workflows/linttest.yml)

* [Documentation](https://letstune.readthedocs.io/en/latest/)
* [PyPI Package](https://pypi.org/project/letstune/)
* [Examples](examples)

Machine Learning algorithms have **many parameters**, which are expected to be
chosen by a user - like the number of layers or learning rate.

It requires **a lot of trial and error**.

_letstune_ **automatically tries** various parameter configurations and
gives you back the best model.

## How it differs from `GridSearchCV`?

_letstune_ will give you **a better model** in **a shorter time** in comparison
to the classical hyperparameter tuning algorithms.

1. Generate random parameters
2. Evaluate each parameter with **a small time budget**
3. **Drop low-performers** automatically, only good-performers will stay in the pool

The 3rd point is the distinguishing feature of _letstune_ - other algorithms
dutifully train weak models - without a good reason.

## Ergonomics

Common tasks in _letstune_ are realized with Python one-liners:

The best model:

```python
model = tuning[0].best_epoch.checkpoint.load_pickle()
```

Pandas summary dataframe with **parameters** and **metric values**:

```python
df = tuning.to_df()
```

## Great! How to use it?

Install with **pip**:

```
pip install letstune
```

First, define your **parameters**:

```python
import letstune
from letstune import Params, rand

class SGDClassifierParams(Params):
    model_cls = SGDClassifier

    average: bool
    l1_ratio: float = rand.uniform(0, 1)
    alpha: float = rand.uniform(1e-2, 1e0, log=True)
```

Then define a **trainer**.
**Trainer** is an object, which knows how to **train** a model!

```python
class DigitsTrainer(letstune.SimpleTrainer):
    params_cls = SGDClassifierParams
    metric = "accuracy"

    def load_dataset(self, dataset):
        self.X_train, self.X_test, self.y_train, self.y_test = dataset

    def train(self, params):
        # params has type SGDClassifierParams

        # letstune provides method create_model
        # returning SGDClassifier
        model = params.create_model(
            loss="hinge",
            penalty="elasticnet",
            fit_intercept=True,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)

        accuracy = model.score(self.X_test, self.y_test)

        return model, {"accuracy": accuracy}


trainer = DigitsTrainer()  # new instance!
```

Neural networks and gradient boosting trainings
can be based on `letstune.EpochTrainer`,
which has `train_epoch` method.

Finally, **let's tune**!

```python
tuning = letstune.tune(
    trainer,
    16,  # number of tested random parameters
    dataset=(X_train, X_test, y_train, y_test),
    results_dir="digits_tuning",
)
```

**Our model** is ready to use:

```python
model = tuning[0].checkpoint.load_pickle()
```

Don't forget to check out [examples directory](examples)! 👀

Documentation is [here](https://letstune.readthedocs.io/en/latest/)!

## Additionally

Works with your favourite ML library 🐍 - it's **library agnostic**!

**Resumes work** from the point, where program was stopped.

Permissive **business-friendly** MIT license.

## References

*A System for Massively Parallel Hyperparameter Tuning* by Li et al.;
[arXiv:1810.05934](https://arxiv.org/abs/1810.05934)

Overview of various hyperparameter-tuning algorithms.
_letstune_ implements a variant of Successive Halving.

## Contributing

Issues are tracked on [GitHub](https://github.com/mslapek/letstune/issues).

## Changelog

Please see [CHANGELOG.md](CHANGELOG.md).
