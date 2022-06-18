from datetime import timedelta
from pathlib import Path
from typing import Any

import sklearn.model_selection
import xgboost as xgb
from sklearn.datasets import make_classification

import letstune
from letstune import rand

NUM_CLASSES = 3


class BoosterParams(letstune.Params):
    max_depth: int = rand.ints(2, 6)  # type: ignore
    eta: float = rand.uniform(0.01, 0.2, log=True)  # type: ignore

    def create_model(self, **kwargs: Any) -> xgb.Booster:
        return xgb.Booster(  # type: ignore
            {
                **self.to_dict(),
                "objective": "multi:softmax",
                "num_class": NUM_CLASSES,
            },
            **kwargs,
        )


class ClassificationTrainer(letstune.EpochTrainer[BoosterParams]):
    metric = letstune.Metric("valid_accuracy")

    def load_dataset(self, dataset: Any) -> None:
        X, y = make_classification(
            n_samples=2000,
            n_informative=5,
            n_classes=NUM_CLASSES,
            random_state=42,
        )

        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
            X,
            y,
            test_size=0.33,
            random_state=42,
        )

        self.d_train = xgb.DMatrix(data=X_train, label=y_train)
        self.d_valid = xgb.DMatrix(data=X_valid, label=y_valid)

        self.y_valid = y_valid

    def create_model(self, params: BoosterParams) -> None:
        self.model = params.create_model(
            cache=[self.d_train, self.d_valid],
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.update(
            self.d_train,
            epoch,
        )
        y_valid_pred = self.model.predict(self.d_valid)
        return {
            "valid_accuracy": sklearn.metrics.accuracy_score(
                self.y_valid, y_valid_pred
            ),
        }

    def load(self, checkpoint: Any, params: BoosterParams) -> None:
        self.model = checkpoint.load_xgboost(
            lambda model_file: params.create_model(
                cache=[self.d_train, self.d_valid],
                model_file=model_file,
            )
        )

    def save(self, checkpoint: Any) -> None:
        checkpoint.save_xgboost(self.model)


trainer = ClassificationTrainer()

tuning = letstune.tune(
    trainer,
    8,
    results_dir=Path.home() / "ltexamples/xgboost/classification_epochtrainer",
    rounds=[
        timedelta(seconds=2),
        timedelta(seconds=4),
    ],
)
print(f" DONE: {tuning}")

model = tuning[0].best_epoch.checkpoint.load_xgboost()
print(f"MODEL: {model}")
